"""
High-level extraction coordination and orchestration.

Coordinates multi-source variant extraction with harmonization,
manages execution planning, and generates comprehensive reports.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from datetime import datetime
from tqdm import tqdm
import uuid

from ..models.analysis import DataType, AnalysisRequest
from ..models.harmonization import ExtractionPlan, HarmonizationStats
from ..models.probe_validation import ProbeSelectionReport
from ..core.config import Settings
from ..utils.paths import PgenFileSet
from .extractor import VariantExtractor
from .transformer import GenotypeTransformer
from .output import TrawFormatter
from .harmonizer import HarmonizationEngine
from .probe_selector import ProbeSelector
from .probe_recommender import ProbeRecommendationEngine

logger = logging.getLogger(__name__)


def extract_single_file_process_worker(
    file_path: str,
    data_type: str, 
    snp_list_ids: List[str],
    snp_list_df: pd.DataFrame,
    settings: 'Settings'
) -> pd.DataFrame:
    """
    Process-isolated extraction worker for ProcessPoolExecutor.
    
    Creates fresh instances of all required objects within the process
    to ensure complete isolation and avoid serialization issues.
    
    Args:
        file_path: Path to PLINK file to extract
        data_type: Data type (NBA, WGS, IMPUTED)
        snp_list_ids: List of variant IDs to extract
        snp_list_df: SNP list DataFrame
        settings: Settings object
        
    Returns:
        DataFrame with extracted and harmonized genotypes
    """
    try:
        # Import here to avoid circular imports in multiprocessing
        from .extractor import VariantExtractor
        
        # Create fresh instances in this process
        extractor = VariantExtractor(settings)
        
        # Perform extraction 
        df = extractor.extract_single_file_harmonized(file_path, snp_list_ids, snp_list_df)
        
        if df.empty:
            return df
        
        # Add metadata tagging
        df['data_type'] = data_type
        df['source_file'] = file_path
        
        # Parse ancestry from file path
        if data_type in ["NBA", "IMPUTED"]:
            ancestry = _parse_ancestry_from_path(file_path, settings.ANCESTRIES)
            if ancestry:
                df['ancestry'] = ancestry
        
        return df
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Process worker failed for {file_path}: {e}")
        return pd.DataFrame()


def _parse_ancestry_from_path(file_path: str, ancestries: List[str]) -> Optional[str]:
    """Helper function to parse ancestry from file path."""
    path_parts = file_path.split(os.sep)
    for part in reversed(path_parts):
        if part in ancestries:
            return part
    return None


class ExtractionCoordinator:
    """Coordinates multi-source variant extraction with harmonization."""
    
    def __init__(
        self, 
        extractor: VariantExtractor, 
        transformer: GenotypeTransformer,
        settings: Settings
    ):
        self.extractor = extractor
        self.transformer = transformer
        self.settings = settings
        self.formatter = TrawFormatter()
        self.harmonization_engine = HarmonizationEngine(settings)
    
    def load_snp_list(self, file_path: str) -> pd.DataFrame:
        """
        Load and validate SNP list from file.
        
        Args:
            file_path: Path to SNP list file (CSV)
            
        Returns:
            Validated SNP list DataFrame
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"SNP list file not found: {file_path}")
            
            # Load file
            snp_list = pd.read_csv(file_path)
            logger.info(f"Loaded {len(snp_list)} variants from SNP list: {file_path}")
            
            # Validate and normalize
            validated_snp_list = self._validate_snp_list(snp_list)
            
            return validated_snp_list
            
        except Exception as e:
            logger.error(f"Failed to load SNP list from {file_path}: {e}")
            raise
    
    
    def _validate_snp_list(self, snp_list: pd.DataFrame) -> pd.DataFrame:
        """Validate SNP list format and content."""
        if snp_list.empty:
            raise ValueError("SNP list is empty")
        
        # Check for required columns
        if 'hg38' in snp_list.columns:
            # Extract coordinates from hg38 format
            coords = snp_list['hg38'].str.split(':', expand=True)
            if coords.shape[1] >= 4:
                snp_list['chromosome'] = coords[0].str.replace('chr', '').str.upper()
                snp_list['position'] = pd.to_numeric(coords[1], errors='coerce')
                snp_list['ref'] = coords[2].str.upper().str.strip()
                snp_list['alt'] = coords[3].str.upper().str.strip()
                
                # Create variant_id if not present
                if 'variant_id' not in snp_list.columns:
                    snp_list['variant_id'] = snp_list['hg38']
            else:
                raise ValueError("Invalid hg38 coordinate format in SNP list")
        
        # Ensure required columns exist
        required_cols = ['chromosome', 'position', 'ref', 'alt']
        missing_cols = [col for col in required_cols if col not in snp_list.columns]
        if missing_cols:
            raise ValueError(f"SNP list missing required columns: {missing_cols}")
        
        # Remove invalid rows
        initial_count = len(snp_list)
        snp_list = snp_list.dropna(subset=required_cols)
        
        # Validate chromosome values
        valid_chroms = set([str(i) for i in range(1, 23)] + ['X', 'Y', 'MT'])
        snp_list = snp_list[snp_list['chromosome'].isin(valid_chroms)]
        
        # Validate position values
        snp_list = snp_list[snp_list['position'] > 0]
        
        # Validate alleles (allow any DNA sequence)
        # Check that alleles contain only valid DNA characters
        dna_pattern = r'^[ATCG]+$'
        snp_list = snp_list[
            snp_list['ref'].str.match(dna_pattern, na=False) &
            snp_list['alt'].str.match(dna_pattern, na=False)
        ]
        
        final_count = len(snp_list)
        if final_count < initial_count:
            logger.warning(f"Filtered SNP list from {initial_count} to {final_count} variants")
        
        if final_count == 0:
            raise ValueError("No valid variants remain after validation")
        
        logger.info(f"Validated SNP list with {final_count} variants")
        return snp_list.reset_index(drop=True)
    
    
    def plan_extraction(
        self, 
        snp_list_ids: List[str], 
        data_types: List[DataType],
        ancestries: Optional[List[str]] = None
    ) -> ExtractionPlan:
        """
        Create extraction plan for specified variants and data types.
        
        Args:
            snp_list_ids: List of variant IDs to extract
            data_types: List of data types to extract from
            ancestries: Optional list of ancestries to include
            
        Returns:
            ExtractionPlan with file paths and metadata
        """
        plan = ExtractionPlan(
            snp_list_ids=snp_list_ids,
            expected_total_variants=len(snp_list_ids)
        )
        
        total_files = 0
        
        for data_type in data_types:
            files = []
            source_ancestries = []
            
            if data_type == DataType.WGS:
                # Single WGS file
                wgs_path = self.settings.get_wgs_path() + ".pgen"
                if os.path.exists(wgs_path):
                    files.append(wgs_path)
                
            elif data_type == DataType.NBA:
                # NBA files by ancestry
                available_ancestries = ancestries or self.settings.list_available_ancestries("NBA")
                for ancestry in available_ancestries:
                    nba_path = self.settings.get_nba_path(ancestry) + ".pgen"
                    if os.path.exists(nba_path):
                        files.append(nba_path)
                        source_ancestries.append(ancestry)
                
            elif data_type == DataType.IMPUTED:
                # Imputed files by ancestry and chromosome
                available_ancestries = ancestries or self.settings.list_available_ancestries("IMPUTED")
                for ancestry in available_ancestries:
                    available_chroms = self.settings.list_available_chromosomes(ancestry)
                    for chrom in available_chroms:
                        imputed_path = self.settings.get_imputed_path(ancestry, chrom) + ".pgen"
                        if os.path.exists(imputed_path):
                            files.append(imputed_path)
                            if ancestry not in source_ancestries:
                                source_ancestries.append(ancestry)
            
            if files:
                plan.add_data_source(data_type.value, files, source_ancestries)
                total_files += len(files)
        
        # Sample counts will be determined during actual extraction
        
        # Estimate execution time (rough)
        # Base time + per-file time + per-variant time
        base_time = 2  # minutes
        per_file_time = 0.5  # minutes per file
        per_variant_time = 0.01  # minutes per variant
        
        estimated_time = base_time + (total_files * per_file_time) + (len(snp_list_ids) * per_variant_time)
        plan.estimated_duration_minutes = estimated_time
        
        logger.info(f"Created extraction plan: {total_files} files, {len(snp_list_ids)} variants, ~{estimated_time:.1f} min")
        
        return plan
    
    def execute_harmonized_extraction(
        self, 
        plan: ExtractionPlan, 
        snp_list_df: pd.DataFrame,
        parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute harmonized extraction according to plan using ProcessPool.
        
        Args:
            plan: ExtractionPlan with file paths and variants
            snp_list_df: SNP list DataFrame for harmonization
            parallel: Whether to run extractions in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary mapping data_type to combined DataFrame with harmonized genotypes
        """
        start_time = time.time()
        logger.info(f"Starting harmonized extraction for {len(plan.snp_list_ids)} variants")
        
        # Use settings default if not provided
        if max_workers is None:
            max_workers = self.settings.max_workers
        
        if parallel:
            return self._execute_with_process_pool(plan, snp_list_df, max_workers)
        else:
            # Execute sequentially
            results_by_datatype = {}
            for data_type in plan.data_types:
                try:
                    files = plan.get_files_for_data_type(data_type)
                    result_df = self._extract_data_type(data_type, plan.snp_list_ids, files, snp_list_df)
                    if not result_df.empty:
                        # Remove duplicates within this data type
                        result_df = result_df.drop_duplicates(
                            subset=['snp_list_id', 'variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'], 
                            keep='first'
                        )
                        results_by_datatype[data_type] = result_df
                        logger.info(f"Completed extraction for {data_type}: {len(result_df)} variants")
                except Exception as e:
                    logger.error(f"Failed extraction for {data_type}: {e}")
            
            execution_time = time.time() - start_time
            total_variants = sum(len(df) for df in results_by_datatype.values())
            logger.info(f"Completed harmonized extraction in {execution_time:.1f}s: {total_variants} total variants across {len(results_by_datatype)} data types")
            
            return results_by_datatype
    
    def _execute_with_process_pool(
        self, 
        plan: ExtractionPlan, 
        snp_list_df: pd.DataFrame, 
        max_workers: int
    ) -> Dict[str, pd.DataFrame]:
        """ProcessPool-based parallel extraction."""
        start_time = time.time()
        
        # Flatten all files across all data types into single task list
        all_tasks = []
        for data_type in plan.data_types:
            files = plan.get_files_for_data_type(data_type)
            for file_path in files:
                all_tasks.append((file_path, data_type))
        
        # Handle case where no files are found
        if not all_tasks:
            logger.warning("No files found for extraction")
            return {}
        
        # Calculate optimal process count
        optimal_workers = self._calculate_optimal_workers(len(all_tasks), max_workers)
        logger.info(f"Starting ProcessPool extraction: {len(all_tasks)} files, {optimal_workers} processes")
        
        all_results = []
        failed_extractions = []
        
        with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(
                    extract_single_file_process_worker,
                    file_path,
                    data_type,
                    plan.snp_list_ids,
                    snp_list_df,
                    self.settings
                ): (file_path, data_type)
                for file_path, data_type in all_tasks
            }
            
            # Collect results with progress tracking
            pbar = tqdm(total=len(all_tasks), desc="Extracting variants")
            with pbar:
                for future in as_completed(future_to_task):
                    file_path, data_type = future_to_task[future]
                    try:
                        result_df = future.result()
                        if not result_df.empty:
                            all_results.append(result_df)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Process extraction failed for {file_path}: {e}")
                        failed_extractions.append((file_path, data_type, str(e)))
                        pbar.update(1)
        
        # Log failures
        if failed_extractions:
            logger.warning(f"Failed to extract {len(failed_extractions)} files:")
            for file_path, data_type, error in failed_extractions:
                logger.warning(f"  {data_type}: {file_path} - {error}")
        
        # Group results by data type
        results_by_datatype = {}
        
        # Group ProcessPool results by data_type
        wgs_results = [df for df in all_results if (df['data_type'] == 'WGS').all()]
        nba_results = [df for df in all_results if (df['data_type'] == 'NBA').all()]
        imputed_results = [df for df in all_results if (df['data_type'] == 'IMPUTED').all()]
        
        # Combine within each data type (only deduplicating within same data type)
        if wgs_results:
            wgs_combined = pd.concat(wgs_results, ignore_index=True)
            # Remove duplicates within WGS only
            wgs_combined = wgs_combined.drop_duplicates(
                subset=['snp_list_id', 'variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'], 
                keep='first'
            )
            # Reorder columns to ensure metadata comes before samples
            wgs_combined = self._reorder_dataframe_columns(wgs_combined)
            results_by_datatype['WGS'] = wgs_combined
            logger.info(f"WGS combined: {len(wgs_combined)} variants")
        
        if nba_results:
            nba_combined = pd.concat(nba_results, ignore_index=True)
            # Remove duplicates within NBA only
            nba_combined = nba_combined.drop_duplicates(
                subset=['snp_list_id', 'variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'], 
                keep='first'
            )
            # Reorder columns to ensure metadata comes before samples
            nba_combined = self._reorder_dataframe_columns(nba_combined)
            results_by_datatype['NBA'] = nba_combined
            logger.info(f"NBA combined: {len(nba_combined)} variants")
        
        if imputed_results:
            imputed_combined = pd.concat(imputed_results, ignore_index=True)
            # Remove duplicates within IMPUTED only
            imputed_combined = imputed_combined.drop_duplicates(
                subset=['snp_list_id', 'variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'], 
                keep='first'
            )
            # Reorder columns to ensure metadata comes before samples
            imputed_combined = self._reorder_dataframe_columns(imputed_combined)
            results_by_datatype['IMPUTED'] = imputed_combined
            logger.info(f"IMPUTED combined: {len(imputed_combined)} variants")
        
        execution_time = time.time() - start_time
        total_variants = sum(len(df) for df in results_by_datatype.values())
        logger.info(f"ProcessPool extraction completed in {execution_time:.1f}s: {total_variants} total variants across {len(results_by_datatype)} data types")
        
        return results_by_datatype
    
    def _reorder_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder DataFrame columns to ensure metadata columns come before sample columns.
        Also normalizes sample IDs to consistent format.
        
        Args:
            df: DataFrame with potentially mixed column order
            
        Returns:
            DataFrame with metadata columns first, then normalized sample columns
        """
        # Define metadata columns in desired order
        METADATA_COLUMNS = [
            'chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
            'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
            'pgen_a1', 'pgen_a2', 'data_type', 'source_file', 'ancestry'
        ]
        
        # Optional metadata that might be added later
        OPTIONAL_METADATA = ['rsid', 'locus', 'snp_name']
        
        # Separate metadata from sample columns
        metadata_present = [col for col in METADATA_COLUMNS if col in df.columns]
        optional_present = [col for col in OPTIONAL_METADATA if col in df.columns]
        
        # All remaining columns are sample columns
        all_metadata = set(metadata_present + optional_present)
        sample_columns = [col for col in df.columns if col not in all_metadata]
        
        # Normalize sample IDs and create column mapping
        column_mapping = {}
        normalized_sample_columns = []
        seen_normalized = set()
        
        for col in sample_columns:
            normalized_col = self._normalize_sample_id(col)
            
            # Handle potential duplicates after normalization
            if normalized_col in seen_normalized:
                logger.warning(f"Duplicate sample ID after normalization: {col} -> {normalized_col}")
                # Keep the first occurrence, skip duplicates
                continue
            
            column_mapping[col] = normalized_col
            normalized_sample_columns.append(normalized_col)
            seen_normalized.add(normalized_col)
        
        # Sort normalized sample columns for consistency
        normalized_sample_columns = sorted(normalized_sample_columns)
        
        # Final order: required metadata + optional metadata + normalized samples
        ordered_columns = metadata_present + optional_present + normalized_sample_columns
        
        # Create new DataFrame with renamed columns
        df_renamed = df.copy()
        df_renamed = df_renamed.rename(columns=column_mapping)
        
        # Reorder the DataFrame
        return df_renamed[ordered_columns]
    
    def _normalize_sample_id(self, sample_id: str) -> str:
        """
        Normalize sample IDs to consistent format.
        
        Handles:
        - WGS duplicated IDs: BBDP_000005_BBDP_000005 -> BBDP_000005
        - NBA/IMPUTED prefixes: 0_BBDP_000005 -> BBDP_000005
        
        Args:
            sample_id: Original sample ID
            
        Returns:
            Normalized sample ID
        """
        # Remove '0_' prefix from NBA/IMPUTED data
        if sample_id.startswith('0_'):
            sample_id = sample_id[2:]
        
        # Fix WGS duplicated IDs: BBDP_000005_BBDP_000005 -> BBDP_000005
        if '_' in sample_id:
            parts = sample_id.split('_')
            # Check if it's a duplicated pattern (at least 4 parts, first half == second half)
            if len(parts) >= 4 and len(parts) % 2 == 0:
                mid = len(parts) // 2
                first_half = '_'.join(parts[:mid])
                second_half = '_'.join(parts[mid:])
                if first_half == second_half:
                    sample_id = first_half
        
        return sample_id
    
    def _calculate_optimal_workers(self, total_files: int, max_workers: int) -> int:
        """Calculate optimal process count based on system resources and settings."""
        return self.settings.get_optimal_workers(total_files)
    
    def _extract_data_type(
        self, 
        data_type: str, 
        snp_list_ids: List[str], 
        files: List[str],
        snp_list_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Extract variants from a specific data type using the provided file list."""
        all_dfs = []
        
        for file_path in files:
            try:
                # Extract from individual file
                df = self.extractor.extract_single_file_harmonized(file_path, snp_list_ids, snp_list_df)
                if not df.empty:
                    df['data_type'] = data_type
                    
                    # Extract ancestry from file path for NBA and IMPUTED
                    if data_type in ["NBA", "IMPUTED"]:
                        # Parse ancestry from file path (e.g., /path/AAC/AAC_release10_vwb.pgen -> AAC)
                        path_parts = file_path.split(os.sep)
                        ancestry = None
                        for part in reversed(path_parts):
                            if part in self.settings.ANCESTRIES:
                                ancestry = part
                                break
                        if ancestry:
                            df['ancestry'] = ancestry
                    
                    all_dfs.append(df)
                    
            except Exception as e:
                logger.error(f"Failed to extract from {file_path}: {e}")
        
        # Combine all results
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            # Reorder columns to ensure metadata comes before samples
            return self._reorder_dataframe_columns(combined_df)
        else:
            return pd.DataFrame()
    
    
    def _generate_harmonization_summary_from_plan(self, plan: ExtractionPlan) -> Dict[str, Any]:
        """
        Generate harmonization summary from extraction plan.
        
        Args:
            plan: ExtractionPlan with file information
            
        Returns:
            Summary dictionary with statistics
        """
        summary = {
            "total_variants": len(plan.snp_list_ids),
            "unique_variants": len(plan.snp_list_ids),
            "extraction_timestamp": datetime.now().isoformat(),
            "harmonization_actions": {},
            "by_data_type": {},
            "by_ancestry": {},
            "by_chromosome": {},
            "harmonized_files_available": False,
            "export_method": "realtime_harmonization"
        }
        
        return summary
    
    def export_pipeline_results(
        self, 
        plan: ExtractionPlan,
        output_dir: str,
        base_name: Optional[str] = None,
        formats: List[str] = ['parquet'],
        snp_list: Optional[pd.DataFrame] = None,
        harmonization_summary: Optional[Dict[str, Any]] = None,
        max_workers: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Export pipeline results for all data types.
        
        Args:
            plan: ExtractionPlan with file information
            output_dir: Output directory
            base_name: Base name for output files
            formats: List of output formats
            snp_list: Original SNP list for metadata
            harmonization_summary: Harmonization summary statistics
            
        Returns:
            Dictionary mapping format to output file path
        """
        if base_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"harmonized_variants_{timestamp}"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        output_files = {}
        actual_counts = {'total_samples': 0, 'total_variants': 0, 'by_data_type': {}}
        
        # Always use traditional DataFrame-based export
        if True:  # Always use traditional approach
            logger.info("Using traditional DataFrame-based export with separate data type outputs")
            # Extract using traditional method - now returns dict by data type
            extracted_by_datatype = self.execute_harmonized_extraction(plan, snp_list, parallel=True, max_workers=max_workers)
            
            # Export separate files for each data type
            for data_type, df in extracted_by_datatype.items():
                if df.empty:
                    logger.warning(f"No variants extracted for {data_type}, skipping export")
                    continue
                
                # Update harmonization summary with actual counts from this data type
                current_summary = harmonization_summary.copy() if harmonization_summary else {}
                sample_cols = self.formatter._get_sample_columns(df)
                actual_sample_count = len(sample_cols)
                actual_variant_count = len(df)
                
                # Aggregate actual counts (use max sample count across data types since samples are consistent)
                actual_counts['total_samples'] = max(actual_counts['total_samples'], actual_sample_count)
                actual_counts['total_variants'] += actual_variant_count
                actual_counts['by_data_type'][data_type] = {
                    'variants': actual_variant_count,
                    'samples': actual_sample_count
                }
                
                current_summary['total_samples'] = actual_sample_count
                current_summary['total_variants'] = actual_variant_count
                current_summary['data_type'] = data_type
                
                # Create data-type specific base name
                datatype_base_name = f"{base_name}_{data_type}"
                
                # Export for this data type
                datatype_output_files = self.export_results(
                    df=df,
                    output_dir=output_dir,
                    base_name=datatype_base_name,
                    formats=formats,
                    snp_list=snp_list,
                    harmonization_summary=current_summary
                )
                
                # Add data type prefix to output file keys
                for file_type, file_path in datatype_output_files.items():
                    output_files[f"{data_type}_{file_type}"] = file_path
                
                logger.info(f"Exported {data_type}: {len(datatype_output_files)} files, {actual_variant_count} variants")
            
            return output_files, actual_counts
        
        # This section is no longer needed since we always use traditional extraction above
        
        # Additional metadata files removed (QC report and harmonization report)
        # All necessary information is available in variant summary and pipeline results
        
        logger.info(f"Exported {len(output_files)} files using native PLINK approach")
        return output_files
    
    
    
    
    def export_results(
        self, 
        df: pd.DataFrame, 
        output_dir: str,
        base_name: Optional[str] = None,
        formats: List[str] = ['parquet'],
        snp_list: Optional[pd.DataFrame] = None,
        harmonization_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export extraction results in specified formats.
        
        Args:
            df: Harmonized genotype DataFrame
            output_dir: Output directory
            base_name: Base name for output files
            formats: List of output formats
            snp_list: Original SNP list for metadata
            harmonization_summary: Harmonization summary statistics
            
        Returns:
            Dictionary mapping format to output file path
        """
        if base_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"harmonized_variants_{timestamp}"
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export in multiple formats
        output_files = self.formatter.export_multiple_formats(
            df=df,
            output_dir=output_dir,
            base_name=base_name,
            formats=formats,
            snp_list=snp_list,
            harmonization_stats=harmonization_summary
        )
        
        # QC report removed - same information available in variant summary
        
        logger.info(f"Exported results to {len(output_files)} files in {output_dir}")
        return output_files
    
    def run_full_extraction_pipeline(
        self,
        snp_list_path: str,
        data_types: List[DataType],
        output_dir: str,
        ancestries: Optional[List[str]] = None,
        parallel: bool = True,
        max_workers: Optional[int] = None,
        output_name: Optional[str] = None,
        enable_probe_selection: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete extraction pipeline from SNP list to final output.

        Args:
            snp_list_path: Path to SNP list file
            data_types: Data types to extract from
            output_dir: Output directory
            ancestries: Optional ancestry filter
            parallel: Use parallel processing
            max_workers: Maximum parallel workers
            output_name: Custom name for output files (default: auto-generated)
            enable_probe_selection: Enable probe quality analysis and selection (default: True)

        Returns:
            Dictionary with pipeline results and metadata
        """
        pipeline_start = time.time()
        
        # Use custom output name if provided, otherwise generate automatic name
        if output_name:
            job_id = output_name
            logger.info(f"Starting full extraction pipeline with custom name: {job_id}")
        else:
            job_id = f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Starting full extraction pipeline: {job_id}")
        
        results = {
            'job_id': job_id,
            'start_time': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'summary': {},
            'errors': []
        }
        
        try:
            # Step 1: Load and validate SNP list
            logger.info("Step 1: Loading SNP list")
            snp_list = self.load_snp_list(snp_list_path)
            snp_list_ids = snp_list['snp_name'].tolist() if 'snp_name' in snp_list.columns else snp_list.index.tolist()
            
            # Step 2: Create extraction plan
            logger.info("Step 2: Creating extraction plan")
            plan = self.plan_extraction(snp_list_ids, data_types, ancestries)
            
            # Step 3: Generate harmonization summary
            logger.info("Step 3: Generating harmonization summary")
            harmonization_summary = self._generate_harmonization_summary_from_plan(plan)
            
            # Step 4: Export results using real-time extraction
            logger.info("Step 4: Exporting results")
            output_files, actual_counts = self.export_pipeline_results(
                plan=plan,
                output_dir=output_dir,
                base_name=job_id,
                formats=['parquet'],  # Only parquet format - contains all data
                snp_list=snp_list,
                harmonization_summary=harmonization_summary,
                max_workers=max_workers
            )
            
            # Update harmonization summary with actual counts
            updated_summary = harmonization_summary.copy() if harmonization_summary else {}
            updated_summary.update({
                'total_samples': actual_counts['total_samples'],
                'total_variants': actual_counts['total_variants'],
                'by_data_type': actual_counts['by_data_type']
            })
            
            # Pipeline completed successfully
            results['success'] = True
            results['output_files'] = output_files
            results['summary'] = updated_summary
            # Include only essential plan information, excluding estimates
            plan_info = {
                'snp_list_ids': plan.snp_list_ids,
                'data_sources': plan.data_sources,
                'created_at': plan.created_at
            }
            results['extraction_plan'] = plan_info
            
            pipeline_time = time.time() - pipeline_start
            results['execution_time_seconds'] = pipeline_time
            results['end_time'] = datetime.now().isoformat()
            
            logger.info(f"Pipeline completed successfully in {pipeline_time:.1f}s: {job_id}")

            # Run probe selection postprocessing if enabled
            if enable_probe_selection:
                logger.info("🔬 Running probe selection analysis...")
                probe_selection_results = self.run_probe_selection_postprocessing(
                    output_dir=output_dir,
                    output_name=job_id,
                    data_types=data_types
                )
                if probe_selection_results:
                    results['output_files'].update(probe_selection_results)
                    logger.info("✅ Probe selection analysis completed")
                else:
                    logger.warning("⚠️ Probe selection analysis did not generate results")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now().isoformat()
            results['execution_time_seconds'] = time.time() - pipeline_start
        
        # Save pipeline results
        try:
            results_path = os.path.join(output_dir, f"{job_id}_pipeline_results.json")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            import json
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            results['pipeline_results_file'] = results_path
            
        except Exception as e:
            logger.error(f"Failed to save pipeline results: {e}")

        return results

    def run_probe_selection_postprocessing(
        self,
        output_dir: str,
        output_name: str,
        data_types: List[DataType]
    ) -> Optional[Dict[str, str]]:
        """
        Run probe selection analysis on existing parquet files.

        Args:
            output_dir: Directory containing parquet files
            output_name: Base name for files (e.g., 'release10')
            data_types: Data types to analyze

        Returns:
            Dictionary mapping output file types to file paths, or None if failed
        """
        try:
            # Check if NBA and WGS data are available (required for probe selection)
            if DataType.NBA not in data_types or DataType.WGS not in data_types:
                logger.warning("Probe selection requires both NBA and WGS data types")
                return None

            nba_parquet_path = os.path.join(output_dir, f"{output_name}_NBA.parquet")
            wgs_parquet_path = os.path.join(output_dir, f"{output_name}_WGS.parquet")

            # Verify files exist
            if not os.path.exists(nba_parquet_path):
                logger.warning(f"NBA parquet file not found: {nba_parquet_path}")
                return None
            if not os.path.exists(wgs_parquet_path):
                logger.warning(f"WGS parquet file not found: {wgs_parquet_path}")
                return None

            logger.info(f"Running probe selection analysis: NBA={nba_parquet_path}, WGS={wgs_parquet_path}")

            # Initialize probe selector and recommender
            probe_selector = ProbeSelector(self.settings)
            probe_recommender = ProbeRecommendationEngine(strategy="consensus")

            # Run probe analysis
            probe_analysis_by_mutation, summary = probe_selector.analyze_probes(
                nba_parquet_path=nba_parquet_path,
                wgs_parquet_path=wgs_parquet_path
            )

            if not probe_analysis_by_mutation:
                logger.warning("No multiple-probe mutations found for analysis")
                return None

            # Extract mutation metadata from NBA data for recommendations
            nba_df = pd.read_parquet(nba_parquet_path)
            mutation_metadata = {}
            for _, row in nba_df.iterrows():
                snp_list_id = row['snp_list_id']
                if snp_list_id not in mutation_metadata:
                    mutation_metadata[snp_list_id] = {
                        'snp_list_id': snp_list_id,
                        'chromosome': row['chromosome'],
                        'position': row['position'],
                        'wgs_cases': 0  # Will be filled by actual WGS data analysis
                    }

            # Generate recommendations
            mutation_analyses, methodology_comparison = probe_recommender.recommend_probes(
                probe_analysis_by_mutation=probe_analysis_by_mutation,
                mutation_metadata=mutation_metadata
            )

            # Create comprehensive report
            report = ProbeSelectionReport(
                job_id=output_name,
                summary=summary,
                probe_comparisons=mutation_analyses,
                methodology_comparison=methodology_comparison
            )

            # Save JSON report
            report_path = os.path.join(output_dir, f"{output_name}_probe_selection.json")
            with open(report_path, 'w') as f:
                import json
                json.dump(report.model_dump(), f, indent=2, default=str)

            logger.info(f"Probe selection report saved: {report_path}")
            logger.info(f"Analyzed {summary.mutations_with_multiple_probes} mutations with multiple probes")
            logger.info(f"Method agreement rate: {methodology_comparison.agreement_rate:.3f}")

            return {
                "probe_selection_report": report_path
            }

        except Exception as e:
            logger.error(f"Probe selection analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None