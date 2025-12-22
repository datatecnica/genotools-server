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
        
        # Parse ancestry from file path (WGS, NBA, and IMPUTED are all split by ancestry)
        if data_type in ["NBA", "IMPUTED", "WGS"]:
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
                # WGS files by ancestry and chromosome (like IMPUTED)
                available_ancestries = ancestries or self.settings.list_available_ancestries("WGS")
                for ancestry in available_ancestries:
                    available_chroms = self.settings.list_available_chromosomes(ancestry, data_type="WGS")
                    for chrom in available_chroms:
                        wgs_path = self.settings.get_wgs_path(ancestry, chrom) + ".pgen"
                        if os.path.exists(wgs_path):
                            files.append(wgs_path)
                            if ancestry not in source_ancestries:
                                source_ancestries.append(ancestry)
                if files:
                    logger.info(f"Found {len(files)} WGS files across {len(source_ancestries)} ancestries")

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

            elif data_type == DataType.EXOMES:
                # EXOMES: Per-chromosome files
                try:
                    available_chroms = self.settings.list_available_exomes_chromosomes()
                    for chrom in available_chroms:
                        exomes_path = self.settings.get_exomes_path(chrom) + ".pgen"
                        if os.path.exists(exomes_path):
                            files.append(exomes_path)
                    if files:
                        logger.info(f"Found {len(files)} EXOMES files")
                except ValueError as e:
                    # Handle releases where EXOMES is not available
                    logger.info(f"EXOMES not available: {e}")

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
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
        """
        Execute harmonized extraction according to plan using ProcessPool.

        Args:
            plan: ExtractionPlan with file paths and variants
            snp_list_df: SNP list DataFrame for harmonization
            parallel: Whether to run extractions in parallel
            max_workers: Maximum parallel workers

        Returns:
            Tuple of (results_by_datatype, extraction_stats)
            - results_by_datatype: Dict mapping data_type to combined DataFrame
            - extraction_stats: Dict with per-data-type extraction outcomes
        """
        start_time = time.time()
        logger.info(f"Starting harmonized extraction for {len(plan.snp_list_ids)} variants")

        # Use settings default if not provided
        if max_workers is None:
            max_workers = self.settings.max_workers

        if parallel:
            return self._execute_with_process_pool(plan, snp_list_df, max_workers)
        else:
            # Execute sequentially (simplified tracking - no per-file granularity)
            results_by_datatype = {}
            extraction_stats = {}
            for data_type in plan.data_types:
                files = plan.get_files_for_data_type(data_type)
                extraction_stats[data_type] = {
                    'planned_files': len(files),
                    'successful': 0,
                    'empty': 0,
                    'failed': 0,
                    'successful_files': [],
                    'empty_files': [],
                    'failed_files': []
                }
                try:
                    result_df = self._extract_data_type(data_type, plan.snp_list_ids, files, snp_list_df)
                    if not result_df.empty:
                        # Remove duplicates within this data type
                        result_df = result_df.drop_duplicates(
                            subset=['snp_list_id', 'variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'],
                            keep='first'
                        )
                        results_by_datatype[data_type] = result_df
                        extraction_stats[data_type]['successful'] = len(files)
                        logger.info(f"Completed extraction for {data_type}: {len(result_df)} variants")
                    else:
                        extraction_stats[data_type]['empty'] = len(files)
                except Exception as e:
                    logger.error(f"Failed extraction for {data_type}: {e}")
                    extraction_stats[data_type]['failed'] = len(files)

            execution_time = time.time() - start_time
            total_variants = sum(len(df) for df in results_by_datatype.values())
            logger.info(f"Completed harmonized extraction in {execution_time:.1f}s: {total_variants} total variants across {len(results_by_datatype)} data types")

            return results_by_datatype, extraction_stats
    
    def _execute_with_process_pool(
        self,
        plan: ExtractionPlan,
        snp_list_df: pd.DataFrame,
        max_workers: int
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]]]:
        """
        ProcessPool-based parallel extraction with outcome tracking.

        Returns:
            Tuple of (results_by_datatype, extraction_stats)
            - results_by_datatype: Dict mapping data_type to combined DataFrame
            - extraction_stats: Dict with per-data-type extraction outcomes
        """
        start_time = time.time()

        # Flatten all files across all data types into single task list
        all_tasks = []
        planned_files_by_type = {}  # Track planned files per data type
        for data_type in plan.data_types:
            files = plan.get_files_for_data_type(data_type)
            planned_files_by_type[data_type] = len(files)
            for file_path in files:
                all_tasks.append((file_path, data_type))

        # Handle case where no files are found
        if not all_tasks:
            logger.warning("No files found for extraction")
            return {}, {}

        # Calculate optimal process count
        optimal_workers = self._calculate_optimal_workers(len(all_tasks), max_workers)
        logger.info(f"Starting ProcessPool extraction: {len(all_tasks)} files, {optimal_workers} processes")

        all_results = []

        # Track outcomes per data type: successful (non-empty), empty, failed (exception)
        extraction_outcomes = {}
        for data_type in plan.data_types:
            extraction_outcomes[data_type] = {
                'successful': [],  # Files that returned non-empty DataFrame
                'empty': [],       # Files that returned empty DataFrame (no matching variants)
                'failed': []       # Files that threw exceptions
            }

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
                            extraction_outcomes[data_type]['successful'].append(file_path)
                        else:
                            # Empty result - file processed but no matching variants
                            extraction_outcomes[data_type]['empty'].append(file_path)
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Process extraction failed for {file_path}: {e}")
                        extraction_outcomes[data_type]['failed'].append({
                            'file': file_path,
                            'error': str(e)
                        })
                        pbar.update(1)

        # Build extraction stats summary
        extraction_stats = {}
        for data_type in plan.data_types:
            outcomes = extraction_outcomes[data_type]
            extraction_stats[data_type] = {
                'planned_files': planned_files_by_type[data_type],
                'successful': len(outcomes['successful']),
                'empty': len(outcomes['empty']),
                'failed': len(outcomes['failed']),
                'successful_files': outcomes['successful'],
                'empty_files': outcomes['empty'],
                'failed_files': outcomes['failed']
            }

        # Log extraction summary
        logger.info("\n" + "=" * 60)
        logger.info("EXTRACTION SUMMARY")
        logger.info("=" * 60)
        has_failures = False
        for data_type, stats in extraction_stats.items():
            status = "✅" if stats['failed'] == 0 else "❌"
            if stats['failed'] > 0:
                has_failures = True
            logger.info(f"  {data_type}: {stats['successful']}/{stats['planned_files']} successful, "
                       f"{stats['empty']} empty, {stats['failed']} failed {status}")
        logger.info("=" * 60 + "\n")

        # Log detailed failures if any
        if has_failures:
            logger.warning("FAILED EXTRACTIONS (see failed_files.json for retry):")
            for data_type, stats in extraction_stats.items():
                if stats['failed'] > 0:
                    logger.warning(f"  {data_type}:")
                    for failure in stats['failed_files'][:5]:  # Show first 5
                        logger.warning(f"    - {failure['file']}: {failure['error']}")
                    if len(stats['failed_files']) > 5:
                        logger.warning(f"    ... and {len(stats['failed_files']) - 5} more")
        
        # Group results by data type
        results_by_datatype = {}
        
        # Group ProcessPool results by data_type
        wgs_results = [df for df in all_results if (df['data_type'] == 'WGS').all()]
        nba_results = [df for df in all_results if (df['data_type'] == 'NBA').all()]
        imputed_results = [df for df in all_results if (df['data_type'] == 'IMPUTED').all()]
        exomes_results = [df for df in all_results if (df['data_type'] == 'EXOMES').all()]

        # Combine within each data type (only deduplicating within same data type)
        if wgs_results:
            # Merge WGS results across ancestries (WGS is split by ancestry+chromosome like IMPUTED)
            # Using _merge_ancestry_results ensures sample columns from all ancestries are preserved
            wgs_combined = self._merge_ancestry_results(wgs_results, 'WGS')
            # Reorder columns to ensure metadata comes before samples
            wgs_combined = self._reorder_dataframe_columns(wgs_combined)
            results_by_datatype['WGS'] = wgs_combined
            logger.info(f"WGS combined: {len(wgs_combined)} variants")
        
        if nba_results:
            # Merge NBA results across ancestries instead of simple concat
            nba_combined = self._merge_ancestry_results(nba_results, 'NBA')
            # Reorder columns to ensure metadata comes before samples
            nba_combined = self._reorder_dataframe_columns(nba_combined)
            results_by_datatype['NBA'] = nba_combined
            logger.info(f"NBA combined: {len(nba_combined)} variants")
        
        if imputed_results:
            # Merge IMPUTED results across ancestries instead of simple concat
            imputed_combined = self._merge_ancestry_results(imputed_results, 'IMPUTED')
            # Reorder columns to ensure metadata comes before samples
            imputed_combined = self._reorder_dataframe_columns(imputed_combined)
            results_by_datatype['IMPUTED'] = imputed_combined
            logger.info(f"IMPUTED combined: {len(imputed_combined)} variants")

        if exomes_results:
            # EXOMES is split by chromosome only (not by ancestry) - all samples are in each chromosome file
            # Simple concat is correct here since each variant appears only once (from its chromosome file)
            exomes_combined = pd.concat(exomes_results, ignore_index=True)
            exomes_combined = exomes_combined.drop_duplicates(
                subset=['snp_list_id', 'variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'],
                keep='first'
            )
            exomes_combined = self._reorder_dataframe_columns(exomes_combined)
            results_by_datatype['EXOMES'] = exomes_combined
            logger.info(f"EXOMES combined: {len(exomes_combined)} variants")

        execution_time = time.time() - start_time
        total_variants = sum(len(df) for df in results_by_datatype.values())
        logger.info(f"ProcessPool extraction completed in {execution_time:.1f}s: {total_variants} total variants across {len(results_by_datatype)} data types")

        return results_by_datatype, extraction_stats
    
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
        # Note: 'ancestry' removed - not meaningful for multi-ancestry merged data
        METADATA_COLUMNS = [
            'chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
            'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
            'pgen_a1', 'pgen_a2', 'data_type', 'source_file',
            'maf_corrected', 'original_alt_af'  # MAF correction columns
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

        Note: FID_IID normalization (e.g., AUTH_FAM000001_AUTH_000091 -> AUTH_000091)
        is now handled at extraction time in extractor.py using the psam file.

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

    def _merge_ancestry_results(self, result_dfs: List[pd.DataFrame], data_type: str) -> pd.DataFrame:
        """
        Merge results from different ancestries for the same data type.

        Uses a two-phase approach for efficiency:
        1. Concat within ancestry (same samples, different variants from different chromosomes)
        2. Merge across ancestries (different samples, potentially overlapping variants)

        This is more efficient than sequential merging of all DataFrames because:
        - Concat is O(n) row stacking vs O(n²) for repeated merges
        - Only ~11 ancestry merges needed vs 75-150 file merges

        Args:
            result_dfs: List of DataFrames from different ancestry/chromosome files
            data_type: The data type being merged (NBA/IMPUTED)

        Returns:
            Merged DataFrame with all samples having genotypes for all variants
        """
        if not result_dfs:
            return pd.DataFrame()

        if len(result_dfs) == 1:
            # Single file - no merge needed
            df = result_dfs[0].copy()
            if 'ancestry' in df.columns:
                df = df.drop(columns=['ancestry'])
            return df

        # Define key columns for merging (variant identifier columns)
        merge_keys = ['snp_list_id', 'variant_id', 'chromosome', 'position',
                      'counted_allele', 'alt_allele']

        # Metadata columns that should be consistent across ancestries
        # Note: 'ancestry' column is intentionally excluded because:
        # - Merged data contains samples from multiple ancestries
        # - Each sample's ancestry is implicit in its ID (cohort prefix)
        # - A single 'ancestry' value would be misleading for multi-ancestry data
        metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                         'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                         'pgen_a1', 'pgen_a2', 'data_type', 'source_file',
                         'maf_corrected', 'original_alt_af']

        # ===== Phase 1: Group DataFrames by ancestry =====
        ancestry_groups = {}
        for df in result_dfs:
            # Try to extract ancestry from source_file first, then ancestry column
            if 'source_file' in df.columns and df['source_file'].notna().any():
                source = df['source_file'].iloc[0]
                ancestry = self._extract_ancestry_from_path(source)
            elif 'ancestry' in df.columns and df['ancestry'].notna().any():
                ancestry = df['ancestry'].iloc[0]
            else:
                ancestry = 'unknown'

            if ancestry not in ancestry_groups:
                ancestry_groups[ancestry] = []
            ancestry_groups[ancestry].append(df)

        logger.debug(f"Grouped {len(result_dfs)} {data_type} files into {len(ancestry_groups)} ancestry groups: {list(ancestry_groups.keys())}")

        # ===== Phase 2: Concat within each ancestry (same samples, different variants) =====
        ancestry_dfs = {}
        for ancestry, dfs in ancestry_groups.items():
            if len(dfs) == 1:
                combined = dfs[0].copy()
            else:
                # Simple concat - these have the same samples, just stacking variant rows
                combined = pd.concat(dfs, ignore_index=True)
                # Deduplicate variants within ancestry
                combined = combined.drop_duplicates(subset=merge_keys, keep='first')

            # Drop ancestry column (will be implicit in sample IDs)
            if 'ancestry' in combined.columns:
                combined = combined.drop(columns=['ancestry'])

            ancestry_dfs[ancestry] = combined
            logger.debug(f"  {ancestry}: {len(combined)} variants from {len(dfs)} chromosome files")

        # ===== Phase 3: Merge across ancestries (different samples, overlapping variants) =====
        ancestry_list = list(ancestry_dfs.keys())
        merged_df = ancestry_dfs[ancestry_list[0]].copy()

        for ancestry in ancestry_list[1:]:
            df = ancestry_dfs[ancestry]
            # Get sample columns from this DataFrame (columns not in metadata or merge keys)
            df_sample_cols = [col for col in df.columns
                              if col not in metadata_cols and col not in merge_keys]

            # Select only merge keys and new sample columns for merging
            cols_to_merge = merge_keys + df_sample_cols
            df_to_merge = df[cols_to_merge].copy()

            # Merge on variant identifiers
            merged_df = pd.merge(
                merged_df,
                df_to_merge,
                on=merge_keys,
                how='outer',  # Use outer join to keep all variants
                suffixes=('', '_dup')  # Avoid column name conflicts
            )

            # Combine duplicate columns: take non-NaN value from either column
            # This handles variants that exist in multiple ancestries where metadata
            # columns (like source_file) might differ
            dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
            for dup_col in dup_cols:
                base_col = dup_col[:-4]  # Remove '_dup' suffix
                if base_col in merged_df.columns:
                    # Combine: use base_col where not NaN, otherwise use dup_col
                    merged_df[base_col] = merged_df[base_col].combine_first(merged_df[dup_col])
            # Now drop the _dup columns
            if dup_cols:
                merged_df = merged_df.drop(columns=dup_cols)

        # Final deduplication (shouldn't be needed after proper merge, but safety net)
        merged_df = merged_df.drop_duplicates(
            subset=merge_keys,
            keep='first'
        )

        logger.info(f"Merged {len(result_dfs)} {data_type} files ({len(ancestry_groups)} ancestries) into {len(merged_df)} variants")

        return merged_df

    def _extract_ancestry_from_path(self, path: str) -> str:
        """
        Extract ancestry code from file path.

        Handles paths like:
        - /path/AAC/chr1_AAC_release11.pgen
        - /path/to/file_AFR_something.pgen

        Args:
            path: File path string

        Returns:
            Ancestry code (e.g., 'AAC', 'AFR') or 'unknown' if not found
        """
        if not path:
            return 'unknown'
        for ancestry in self.settings.ANCESTRIES:
            if f'/{ancestry}/' in path or f'_{ancestry}_' in path:
                return ancestry
        return 'unknown'

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
            "by_chromosome": {}
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
    ) -> Tuple[Dict[str, str], Dict[str, Any], Dict[str, Dict[str, Any]]]:
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
            Tuple of (output_files, actual_counts, extraction_stats)
            - output_files: Dict mapping format to output file path
            - actual_counts: Dict with total samples/variants counts
            - extraction_stats: Dict with per-data-type extraction outcomes
        """
        if base_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"harmonized_variants_{timestamp}"

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        output_files = {}
        actual_counts = {'total_samples': 0, 'total_variants': 0, 'by_data_type': {}}

        logger.info("Using traditional DataFrame-based export with separate data type outputs")
        # Extract using traditional method - now returns (dict by data type, extraction_stats)
        extracted_by_datatype, extraction_stats = self.execute_harmonized_extraction(
            plan, snp_list, parallel=True, max_workers=max_workers
        )

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

        return output_files, actual_counts, extraction_stats
    
    
    
    
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
        enable_locus_reports: bool = True,
        enable_coverage_profiling: bool = True,
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
            enable_locus_reports: Enable locus report generation (default: True)
            enable_coverage_profiling: Enable SNP coverage profiling (default: True)

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
            output_files, actual_counts, extraction_stats = self.export_pipeline_results(
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

            # Add extraction stats to results (simplified version for JSON)
            extraction_stats_summary = {}
            for data_type, stats in extraction_stats.items():
                extraction_stats_summary[data_type] = {
                    'planned_files': stats['planned_files'],
                    'successful': stats['successful'],
                    'empty': stats['empty'],
                    'failed': stats['failed'],
                    # Include failed file paths (without full details) for reference
                    'failed_files': [f['file'] if isinstance(f, dict) else f for f in stats['failed_files']]
                }
            results['extraction_stats'] = extraction_stats_summary

            # Save failed files JSON if there are any failures
            has_failures = any(stats['failed'] > 0 for stats in extraction_stats.values())
            if has_failures:
                failed_files_data = {
                    'job_name': job_id,
                    'timestamp': datetime.now().isoformat(),
                    'failed_files': {}
                }
                for data_type, stats in extraction_stats.items():
                    if stats['failed'] > 0:
                        failed_files_data['failed_files'][data_type] = [
                            f['file'] if isinstance(f, dict) else f for f in stats['failed_files']
                        ]

                failed_files_path = os.path.join(output_dir, f"{job_id}_failed_files.json")
                import json
                with open(failed_files_path, 'w') as f:
                    json.dump(failed_files_data, f, indent=2)
                results['failed_files_path'] = failed_files_path
                logger.warning(f"⚠️ Some extractions failed. Failed files saved to: {failed_files_path}")

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

            if enable_locus_reports:
                logger.info("📊 Running locus report generation...")
                locus_report_results = self.run_locus_report_postprocessing(
                    output_dir=output_dir,
                    output_name=job_id,
                    data_types=data_types
                )
                if locus_report_results:
                    results['output_files'].update(locus_report_results)
                    logger.info("✅ Locus report generation completed")
                else:
                    logger.warning("⚠️ Locus report generation did not generate results")

            # Run coverage profiling if enabled
            if enable_coverage_profiling:
                logger.info("📈 Running coverage profiling...")
                coverage_results = self.run_coverage_profiling_postprocessing(
                    output_dir=output_dir,
                    output_name=job_id,
                    data_types=data_types,
                    max_workers=max_workers
                )
                if coverage_results:
                    results['output_files'].update(coverage_results)
                    logger.info("✅ Coverage profiling completed")
                else:
                    logger.warning("⚠️ Coverage profiling did not generate results")

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

    def retry_failed_extraction(
        self,
        failed_files_json_path: str,
        snp_list_path: str,
        output_dir: str,
        max_workers: Optional[int] = None,
        enable_probe_selection: bool = True,
        enable_locus_reports: bool = True
    ) -> Dict[str, Any]:
        """
        Retry extraction for previously failed files.

        Loads existing parquet results, runs extraction only on failed files,
        and merges new results with existing data.

        Args:
            failed_files_json_path: Path to the failed_files.json from a previous run
            snp_list_path: Path to SNP list file
            output_dir: Output directory (should be same as original run)
            max_workers: Maximum parallel workers
            enable_probe_selection: Enable probe quality analysis after retry
            enable_locus_reports: Enable locus report generation after retry

        Returns:
            Dictionary with retry results and metadata
        """
        import json

        pipeline_start = time.time()
        logger.info(f"Starting retry extraction from: {failed_files_json_path}")

        # Load failed files JSON
        with open(failed_files_json_path, 'r') as f:
            failed_files_data = json.load(f)

        job_name = failed_files_data.get('job_name', 'retry')
        failed_files_by_type = failed_files_data.get('failed_files', {})

        if not failed_files_by_type:
            logger.info("No failed files to retry")
            return {
                'success': True,
                'job_id': job_name,
                'message': 'No failed files to retry',
                'retried_files': 0
            }

        results = {
            'job_id': job_name,
            'start_time': datetime.now().isoformat(),
            'success': False,
            'output_files': {},
            'retry_stats': {},
            'errors': []
        }

        try:
            # Load SNP list
            logger.info("Loading SNP list...")
            snp_list = self.load_snp_list(snp_list_path)
            snp_list_ids = snp_list['snp_name'].tolist() if 'snp_name' in snp_list.columns else snp_list.index.tolist()

            total_retried = 0
            total_succeeded = 0

            # Process each data type with failed files
            for data_type_str, failed_file_paths in failed_files_by_type.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Retrying {len(failed_file_paths)} failed {data_type_str} files...")
                logger.info(f"{'='*60}")

                # Load existing parquet for this data type
                existing_parquet_path = os.path.join(output_dir, f"{job_name}_{data_type_str}.parquet")
                existing_df = None
                if os.path.exists(existing_parquet_path):
                    existing_df = pd.read_parquet(existing_parquet_path)
                    logger.info(f"Loaded existing {data_type_str} data: {len(existing_df)} variants")
                else:
                    logger.warning(f"No existing parquet found at {existing_parquet_path}")

                # Run extraction on failed files only
                retry_results = []
                retry_outcomes = {'successful': [], 'empty': [], 'failed': []}

                optimal_workers = self._calculate_optimal_workers(len(failed_file_paths), max_workers or self.settings.max_workers)

                with ProcessPoolExecutor(max_workers=optimal_workers) as executor:
                    future_to_file = {
                        executor.submit(
                            extract_single_file_process_worker,
                            file_path,
                            data_type_str,
                            snp_list_ids,
                            snp_list,
                            self.settings
                        ): file_path
                        for file_path in failed_file_paths
                    }

                    pbar = tqdm(total=len(failed_file_paths), desc=f"Retrying {data_type_str}")
                    with pbar:
                        for future in as_completed(future_to_file):
                            file_path = future_to_file[future]
                            try:
                                result_df = future.result()
                                if not result_df.empty:
                                    retry_results.append(result_df)
                                    retry_outcomes['successful'].append(file_path)
                                else:
                                    retry_outcomes['empty'].append(file_path)
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"Retry failed for {file_path}: {e}")
                                retry_outcomes['failed'].append({'file': file_path, 'error': str(e)})
                                pbar.update(1)

                # Log retry results for this data type
                logger.info(f"\n{data_type_str} Retry Results:")
                logger.info(f"  Successful: {len(retry_outcomes['successful'])}")
                logger.info(f"  Empty: {len(retry_outcomes['empty'])}")
                logger.info(f"  Still failing: {len(retry_outcomes['failed'])}")

                results['retry_stats'][data_type_str] = {
                    'attempted': len(failed_file_paths),
                    'successful': len(retry_outcomes['successful']),
                    'empty': len(retry_outcomes['empty']),
                    'failed': len(retry_outcomes['failed']),
                    'still_failing': [f['file'] if isinstance(f, dict) else f for f in retry_outcomes['failed']]
                }

                total_retried += len(failed_file_paths)
                total_succeeded += len(retry_outcomes['successful'])

                # Merge new results with existing data
                if retry_results:
                    new_df = pd.concat(retry_results, ignore_index=True)

                    if data_type_str in ['NBA', 'IMPUTED']:
                        # Merge across ancestries
                        new_df = self._merge_ancestry_results(retry_results, data_type_str)

                    new_df = self._reorder_dataframe_columns(new_df)

                    if existing_df is not None:
                        # Merge new results with existing (outer join on variant keys)
                        merge_keys = ['snp_list_id', 'variant_id', 'chromosome', 'position',
                                      'counted_allele', 'alt_allele']

                        # Get sample columns from new data
                        metadata_cols = set(merge_keys + ['(C)M', 'COUNTED', 'ALT', 'harmonization_action',
                                                          'pgen_a1', 'pgen_a2', 'data_type', 'source_file',
                                                          'maf_corrected', 'original_alt_af'])
                        new_sample_cols = [col for col in new_df.columns if col not in metadata_cols]

                        # Select columns to merge
                        cols_to_merge = merge_keys + new_sample_cols
                        new_df_to_merge = new_df[cols_to_merge]

                        # Merge with existing
                        merged_df = pd.merge(
                            existing_df,
                            new_df_to_merge,
                            on=merge_keys,
                            how='outer',
                            suffixes=('', '_new')
                        )

                        # For overlapping sample columns, prefer new data (non-NaN)
                        for col in new_sample_cols:
                            new_col = f"{col}_new"
                            if new_col in merged_df.columns:
                                merged_df[col] = merged_df[new_col].combine_first(merged_df[col])
                                merged_df = merged_df.drop(columns=[new_col])

                        final_df = merged_df
                    else:
                        final_df = new_df

                    # Reorder and save
                    final_df = self._reorder_dataframe_columns(final_df)
                    output_path = os.path.join(output_dir, f"{job_name}_{data_type_str}.parquet")
                    final_df.to_parquet(output_path, index=False)
                    logger.info(f"Saved merged {data_type_str} results: {len(final_df)} variants to {output_path}")
                    results['output_files'][f"{data_type_str}_parquet"] = output_path

            # Save updated failed files JSON if there are still failures
            still_failing = {dt: stats['still_failing'] for dt, stats in results['retry_stats'].items()
                           if stats['still_failing']}
            if still_failing:
                updated_failed_path = os.path.join(output_dir, f"{job_name}_failed_files.json")
                updated_failed_data = {
                    'job_name': job_name,
                    'timestamp': datetime.now().isoformat(),
                    'failed_files': still_failing,
                    'retry_history': [failed_files_json_path]
                }
                with open(updated_failed_path, 'w') as f:
                    json.dump(updated_failed_data, f, indent=2)
                logger.warning(f"⚠️ {sum(len(v) for v in still_failing.values())} files still failing. Updated: {updated_failed_path}")
                results['failed_files_path'] = updated_failed_path

            results['success'] = True
            results['retried_files'] = total_retried
            results['succeeded_files'] = total_succeeded

            pipeline_time = time.time() - pipeline_start
            results['execution_time_seconds'] = pipeline_time
            results['end_time'] = datetime.now().isoformat()

            logger.info(f"\n{'='*60}")
            logger.info(f"RETRY COMPLETE: {total_succeeded}/{total_retried} files succeeded")
            logger.info(f"{'='*60}")

            # Run postprocessing if enabled
            data_types = [DataType[dt] for dt in failed_files_by_type.keys()]

            if enable_probe_selection:
                logger.info("🔬 Running probe selection analysis...")
                probe_results = self.run_probe_selection_postprocessing(
                    output_dir=output_dir,
                    output_name=job_name,
                    data_types=data_types
                )
                if probe_results:
                    results['output_files'].update(probe_results)

            if enable_locus_reports:
                logger.info("📊 Running locus report generation...")
                locus_results = self.run_locus_report_postprocessing(
                    output_dir=output_dir,
                    output_name=job_name,
                    data_types=data_types
                )
                if locus_results:
                    results['output_files'].update(locus_results)

        except Exception as e:
            logger.error(f"Retry failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['errors'].append(str(e))
            results['end_time'] = datetime.now().isoformat()
            results['execution_time_seconds'] = time.time() - pipeline_start

        # Save retry results
        try:
            retry_results_path = os.path.join(output_dir, f"{job_name}_retry_results.json")
            with open(retry_results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            results['retry_results_file'] = retry_results_path
        except Exception as e:
            logger.error(f"Failed to save retry results: {e}")

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

    def run_locus_report_postprocessing(
        self,
        output_dir: str,
        output_name: str,
        data_types: List[DataType]
    ) -> Optional[Dict[str, str]]:
        """
        Run locus report generation on existing parquet files.

        Args:
            output_dir: Directory containing parquet files
            output_name: Base name for files (e.g., 'release10')
            data_types: Data types to analyze

        Returns:
            Dictionary mapping output file types to file paths, or None if failed
        """
        try:
            from .locus_report_generator import LocusReportGenerator

            # Build parquet file map
            parquet_files = {}
            for data_type in data_types:
                parquet_path = os.path.join(output_dir, f"{output_name}_{data_type.name}.parquet")
                if os.path.exists(parquet_path):
                    parquet_files[data_type.name] = parquet_path
                else:
                    logger.warning(f"{data_type.name} parquet file not found: {parquet_path}")

            if not parquet_files:
                logger.warning("No parquet files found for locus report generation")
                return None

            logger.info(f"Running locus report generation with data types: {list(parquet_files.keys())}")

            # Check for probe selection file
            probe_selection_path = os.path.join(output_dir, f"{output_name}_probe_selection.json")
            if os.path.exists(probe_selection_path):
                logger.info(f"Probe selection file found: {probe_selection_path}")
                logger.info("NBA locus reports will be filtered to use only selected probes")
            else:
                logger.info("No probe selection file found, all variants will be included")
                probe_selection_path = None

            # Initialize generator with probe selection
            generator = LocusReportGenerator(self.settings, probe_selection_path=probe_selection_path)

            # Generate independent reports for each available data type
            data_types_to_generate = list(parquet_files.keys())

            if not data_types_to_generate:
                logger.warning("No data types available for locus report generation")
                return None

            logger.info(f"Generating locus reports for data types: {data_types_to_generate}")

            # Generate reports
            output_files = generator.generate_reports(
                parquet_files=parquet_files,
                output_dir=output_dir,
                job_name=output_name,
                data_types=data_types_to_generate
            )

            logger.info(f"Locus report generation complete. Generated {len(output_files)} files")

            return output_files

        except Exception as e:
            logger.error(f"Locus report generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def run_coverage_profiling_postprocessing(
        self,
        output_dir: str,
        output_name: str,
        data_types: List[DataType],
        max_workers: int = None
    ) -> Optional[Dict[str, str]]:
        """
        Run SNP coverage profiling against source pvar files.

        This analyzes which SNP list variants exist in each data type's source files,
        providing coverage statistics by locus and variant.

        Args:
            output_dir: Directory for output files
            output_name: Base name for files (e.g., 'release10')
            data_types: Data types to analyze
            max_workers: Number of parallel workers (default: auto-detect)

        Returns:
            Dictionary mapping output file types to file paths, or None if failed
        """
        try:
            from .coverage_profiler import run_coverage_profiling

            # Get base path and release from settings
            base_path = Path(self.settings.mnt_path) / "gp2tier2_vwb" / f"release{self.settings.release}"
            release = self.settings.release

            # Convert data types to string list
            data_type_names = [dt.name for dt in data_types]

            logger.info(f"Running coverage profiling for data types: {data_type_names}")
            logger.info(f"Base path: {base_path}")
            logger.info(f"Release: {release}")

            # Run coverage profiling
            output_files = run_coverage_profiling(
                snp_list_path=str(self.settings.snp_list_path),
                output_dir=output_dir,
                base_path=str(base_path),
                release=release,
                ancestry="EUR",  # Use EUR as reference (variants same across ancestries)
                max_workers=max_workers or self.settings.max_workers,
                output_name=output_name,
                data_types=data_type_names
            )

            if output_files:
                logger.info(f"Coverage profiling complete. Generated {len(output_files)} files")
            else:
                logger.warning("Coverage profiling returned no output files")

            return output_files

        except Exception as e:
            logger.error(f"Coverage profiling failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None