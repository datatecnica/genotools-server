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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
import uuid

from ..models.analysis import DataType, AnalysisRequest
from ..models.harmonization import ExtractionPlan, HarmonizationStats
from ..core.config import Settings
from ..utils.paths import PgenFileSet
from .extractor import VariantExtractor
from .transformer import GenotypeTransformer
from .output import TrawFormatter
from .cache import CacheBuilder

logger = logging.getLogger(__name__)


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
        self.cache_builder = CacheBuilder(settings)
    
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
    
    def validate_snp_list(self, snp_list: pd.DataFrame) -> bool:
        """
        Validate SNP list format and return validation status.
        
        Args:
            snp_list: SNP list DataFrame
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_snp_list(snp_list)
            return True
        except Exception as e:
            logger.error(f"SNP list validation failed: {e}")
            return False
    
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
        estimated_samples = 0
        
        for data_type in data_types:
            files = []
            source_ancestries = []
            
            if data_type == DataType.WGS:
                # Single WGS file
                wgs_path = self.settings.get_wgs_path() + ".pgen"
                if os.path.exists(wgs_path):
                    files.append(wgs_path)
                    estimated_samples += 50000  # Estimate
                
            elif data_type == DataType.NBA:
                # NBA files by ancestry
                available_ancestries = ancestries or self.settings.list_available_ancestries("NBA")
                for ancestry in available_ancestries:
                    nba_path = self.settings.get_nba_path(ancestry) + ".pgen"
                    if os.path.exists(nba_path):
                        files.append(nba_path)
                        source_ancestries.append(ancestry)
                        estimated_samples += 5000  # Estimate per ancestry
                
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
                
                # Estimate samples for imputed (same samples across chromosomes)
                estimated_samples += len(source_ancestries) * 10000  # Estimate per ancestry
            
            if files:
                plan.add_data_source(data_type.value, files, source_ancestries)
                total_files += len(files)
        
        plan.expected_total_samples = estimated_samples
        
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
        parallel: bool = True,
        max_workers: int = 4
    ) -> pd.DataFrame:
        """
        Execute harmonized extraction according to plan.
        
        Args:
            plan: ExtractionPlan with file paths and variants
            parallel: Whether to run extractions in parallel
            max_workers: Maximum parallel workers
            
        Returns:
            Combined DataFrame with harmonized genotypes
        """
        start_time = time.time()
        logger.info(f"Starting harmonized extraction for {len(plan.snp_list_ids)} variants")
        
        all_results = []
        
        if parallel and len(plan.data_types) > 1:
            # Execute data types in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_data_type = {
                    executor.submit(
                        self._extract_data_type,
                        data_type,
                        plan.snp_list_ids,
                        plan.get_files_for_data_type(data_type)
                    ): data_type
                    for data_type in plan.data_types
                }
                
                for future in as_completed(future_to_data_type):
                    data_type = future_to_data_type[future]
                    try:
                        result_df = future.result()
                        if not result_df.empty:
                            all_results.append(result_df)
                            logger.info(f"Completed extraction for {data_type}: {len(result_df)} variants")
                    except Exception as e:
                        logger.error(f"Failed extraction for {data_type}: {e}")
        else:
            # Execute sequentially
            for data_type in plan.data_types:
                try:
                    files = plan.get_files_for_data_type(data_type)
                    result_df = self._extract_data_type(data_type, plan.snp_list_ids, files)
                    if not result_df.empty:
                        all_results.append(result_df)
                        logger.info(f"Completed extraction for {data_type}: {len(result_df)} variants")
                except Exception as e:
                    logger.error(f"Failed extraction for {data_type}: {e}")
        
        # Merge results
        if all_results:
            combined_df = self.extractor.merge_harmonized_genotypes(all_results)
        else:
            combined_df = pd.DataFrame()
        
        execution_time = time.time() - start_time
        logger.info(f"Completed harmonized extraction in {execution_time:.1f}s: {len(combined_df)} variants")
        
        return combined_df
    
    def _extract_data_type(
        self, 
        data_type: str, 
        snp_list_ids: List[str], 
        files: List[str]
    ) -> pd.DataFrame:
        """Extract variants from a specific data type using the provided file list."""
        all_dfs = []
        
        for file_path in files:
            try:
                # Extract from individual file
                df = self.extractor.extract_single_file_harmonized(file_path, snp_list_ids)
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
            return pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def generate_harmonization_summary(
        self, 
        results: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate comprehensive harmonization summary.
        
        Args:
            results: Combined extraction results
            
        Returns:
            Summary dictionary with statistics
        """
        if results.empty:
            return {
                "total_variants": 0,
                "total_samples": 0,
                "summary": "No variants extracted"
            }
        
        # Get sample columns
        sample_cols = self.formatter._get_sample_columns(results)
        
        summary = {
            "total_variants": len(results),
            "total_samples": len(sample_cols),
            "unique_variants": results['snp_list_id'].nunique() if 'snp_list_id' in results.columns else len(results),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        # Harmonization action counts
        if 'harmonization_action' in results.columns:
            action_counts = results['harmonization_action'].value_counts().to_dict()
            summary['harmonization_actions'] = action_counts
            
            # Calculate harmonization rates
            total_with_action = sum(action_counts.values())
            for action, count in action_counts.items():
                summary[f'rate_{action.lower()}'] = count / total_with_action if total_with_action > 0 else 0
        
        # Data type breakdown
        if 'data_type' in results.columns:
            summary['by_data_type'] = results['data_type'].value_counts().to_dict()
        
        # Ancestry breakdown
        if 'ancestry' in results.columns:
            ancestry_counts = results['ancestry'].value_counts().to_dict()
            summary['by_ancestry'] = ancestry_counts
        
        # Chromosome breakdown
        if 'chromosome' in results.columns:
            summary['by_chromosome'] = results['chromosome'].value_counts().to_dict()
        
        # Quality metrics
        if sample_cols:
            missing_rates = []
            alt_freqs = []
            
            for _, row in results.iterrows():
                # Calculate per-variant statistics
                genotypes = [row[col] for col in sample_cols if pd.notna(row[col])]
                if genotypes:
                    # Convert to numeric, handling string genotypes
                    try:
                        genotypes = pd.to_numeric(genotypes, errors='coerce')
                        genotypes = np.array(genotypes)
                        valid_gts = genotypes[~np.isnan(genotypes)]
                        
                        # Missing rate
                        missing_rate = 1 - (len(valid_gts) / len(genotypes))
                        missing_rates.append(missing_rate)
                        
                        # Allele frequency
                        if len(valid_gts) > 0:
                            alt_count = np.sum(valid_gts == 2) * 2 + np.sum(valid_gts == 1)
                            total_alleles = len(valid_gts) * 2
                            alt_freq = alt_count / total_alleles if total_alleles > 0 else 0
                            alt_freqs.append(alt_freq)
                    except Exception as e:
                        logger.warning(f"Failed to calculate statistics for variant: {e}")
                        continue
            
            if missing_rates:
                summary['quality_metrics'] = {
                    'mean_missing_rate': np.mean(missing_rates),
                    'max_missing_rate': np.max(missing_rates),
                    'variants_high_missing': np.sum(np.array(missing_rates) > 0.1)
                }
            
            if alt_freqs:
                summary['allele_frequency_metrics'] = {
                    'mean_alt_freq': np.mean(alt_freqs),
                    'rare_variants': np.sum(np.array(alt_freqs) < 0.01),
                    'common_variants': np.sum(np.array(alt_freqs) > 0.05)
                }
        
        # Transformation summary (only if harmonization metadata is available)
        if 'harmonization_action' in results.columns and 'genotype_transform' in results.columns:
            transform_summary = self.transformer.get_transformation_summary(results)
            summary['transformation_summary'] = transform_summary
        else:
            summary['transformation_summary'] = {
                "note": "Transformation summary not available - harmonization metadata not present in final output"
            }
        
        return summary
    
    def _generate_harmonization_summary_from_plan(self, plan: ExtractionPlan) -> Dict[str, Any]:
        """
        Generate harmonization summary from extraction plan without full DataFrame extraction.
        
        Args:
            plan: ExtractionPlan with file information
            
        Returns:
            Summary dictionary with statistics
        """
        summary = {
            "total_variants": 0,
            "total_samples": 0,
            "unique_variants": 0,
            "extraction_timestamp": datetime.now().isoformat(),
            "harmonization_actions": {},
            "by_data_type": {},
            "by_ancestry": {},
            "by_chromosome": {}
        }
        
        all_harmonization_records = []
        
        # Collect harmonization info from all files in the plan
        for data_type in plan.data_types:
            files = plan.get_files_for_data_type(data_type)
            data_type_variants = 0
            
            for file_path in files:
                try:
                    cache_df = self.extractor._load_harmonization_cache(file_path)
                    if not cache_df.empty:
                        # Filter cache for requested variants
                        plan_df = self.extractor._get_extraction_plan(plan.snp_list_ids, cache_df)
                        if not plan_df.empty:
                            all_harmonization_records.append(plan_df)
                            data_type_variants += len(plan_df)
                            
                            # Count by ancestry
                            if 'ancestry' in plan_df.columns:
                                for ancestry in plan_df['ancestry'].dropna().unique():
                                    summary['by_ancestry'][ancestry] = summary['by_ancestry'].get(ancestry, 0) + len(plan_df[plan_df['ancestry'] == ancestry])
                            
                            # Count by chromosome
                            if 'chromosome' in plan_df.columns:
                                for chrom in plan_df['chromosome'].astype(str).unique():
                                    summary['by_chromosome'][chrom] = summary['by_chromosome'].get(chrom, 0) + len(plan_df[plan_df['chromosome'].astype(str) == chrom])
                except Exception as e:
                    logger.error(f"Error processing harmonization cache for {file_path}: {e}")
            
            if data_type_variants > 0:
                summary['by_data_type'][data_type] = data_type_variants
        
        # Combine all harmonization records
        if all_harmonization_records:
            combined_df = pd.concat(all_harmonization_records, ignore_index=True)
            
            # Remove duplicates based on snp_list_id
            combined_df = combined_df.drop_duplicates(subset=['snp_list_id'], keep='first')
            
            summary['total_variants'] = len(combined_df)
            summary['unique_variants'] = combined_df['snp_list_id'].nunique()
            
            # Harmonization action counts
            if 'harmonization_action' in combined_df.columns:
                action_counts = combined_df['harmonization_action'].value_counts().to_dict()
                summary['harmonization_actions'] = action_counts
                
                # Calculate harmonization rates
                total_with_action = sum(action_counts.values())
                for action, count in action_counts.items():
                    summary[f'rate_{action.lower()}'] = count / total_with_action if total_with_action > 0 else 0
            
            # Get actual sample counts from PLINK files
            actual_sample_count = 0
            sample_counts_by_file = {}
            
            for data_type in plan.data_types:
                files = plan.get_files_for_data_type(data_type)
                for file_path in files:
                    try:
                        # Get base path by removing .pgen extension
                        base_path = file_path.replace('.pgen', '') if file_path.endswith('.pgen') else file_path
                        pgen_set = PgenFileSet(
                            base_path=base_path,
                            pgen_file=f"{base_path}.pgen",
                            pvar_file=f"{base_path}.pvar", 
                            psam_file=f"{base_path}.psam"
                        )
                        file_sample_count = pgen_set.get_sample_count()
                        sample_counts_by_file[file_path] = file_sample_count
                        # For multiple files, we don't sum them since they're separate analyses
                        # We take the max as they should all have the same samples for NBA
                        actual_sample_count = max(actual_sample_count, file_sample_count)
                    except Exception as e:
                        logger.warning(f"Could not get sample count from {file_path}: {e}")
            
            # Use actual sample count if available, otherwise fall back to estimate
            summary['total_samples'] = actual_sample_count if actual_sample_count > 0 else plan.expected_total_samples
            
            # Add harmonized files info
            harmonized_files_available = False
            for data_type in plan.data_types:
                files = plan.get_files_for_data_type(data_type)
                for file_path in files:
                    cache_df = self.extractor._load_harmonization_cache(file_path)
                    if not cache_df.empty and 'harmonized_file_path' in cache_df.columns:
                        harmonized_pgen = cache_df['harmonized_file_path'].iloc[0]
                        if harmonized_pgen and os.path.exists(harmonized_pgen):
                            harmonized_files_available = True
                            break
                if harmonized_files_available:
                    break
            
            summary['harmonized_files_available'] = harmonized_files_available
            summary['export_method'] = 'native_plink' if harmonized_files_available else 'traditional'
        
        return summary
    
    def export_results_with_native_plink(
        self, 
        plan: ExtractionPlan,
        output_dir: str,
        base_name: Optional[str] = None,
        formats: List[str] = ['traw', 'parquet'],
        snp_list: Optional[pd.DataFrame] = None,
        harmonization_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Export results using PLINK's native export when harmonized files are available.
        
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
        
        # Check if we can use native PLINK export
        can_use_native_export = False
        harmonized_files = []
        
        # Check each data source for harmonized files
        for data_type in plan.data_types:
            files = plan.get_files_for_data_type(data_type)
            for file_path in files:
                cache_df = self.extractor._load_harmonization_cache(file_path)
                if not cache_df.empty and 'harmonized_file_path' in cache_df.columns:
                    harmonized_pgen = cache_df['harmonized_file_path'].iloc[0]
                    if harmonized_pgen and os.path.exists(harmonized_pgen):
                        harmonized_files.append(harmonized_pgen)
                        can_use_native_export = True
        
        if can_use_native_export and 'traw' in formats:
            logger.info("Using PLINK native TRAW export from harmonized files")
            try:
                # Use PLINK2 to export TRAW directly from harmonized files
                if len(harmonized_files) == 1:
                    # Single file export
                    native_traw_path = self._export_native_traw_single(
                        harmonized_files[0], output_dir, base_name, plan.snp_list_ids
                    )
                    if native_traw_path:
                        output_files['traw'] = native_traw_path
                        logger.info(f"Generated native TRAW file: {native_traw_path}")
                else:
                    # Multi-file export (need to merge)
                    native_traw_path = self._export_native_traw_merged(
                        harmonized_files, output_dir, base_name, plan.snp_list_ids
                    )
                    if native_traw_path:
                        output_files['traw'] = native_traw_path
                        logger.info(f"Generated merged native TRAW file: {native_traw_path}")
                        
            except Exception as e:
                logger.error(f"Failed to generate native TRAW export: {e}")
                logger.info("Falling back to traditional DataFrame-based export")
                can_use_native_export = False
        
        # If native export failed or not available, fall back to traditional approach
        if not can_use_native_export or 'traw' not in output_files:
            logger.info("Using traditional DataFrame-based export")
            # Extract using traditional method
            extracted_df = self.execute_harmonized_extraction(plan, parallel=True, max_workers=4)
            
            # Use traditional export
            traditional_output_files = self.export_results(
                df=extracted_df,
                output_dir=output_dir,
                base_name=base_name,
                formats=formats,
                snp_list=snp_list,
                harmonization_summary=harmonization_summary
            )
            output_files.update(traditional_output_files)
            return output_files
        
        # For other formats, still need to extract DataFrame
        if any(fmt in formats for fmt in ['parquet', 'csv', 'json']):
            extracted_df = self.execute_harmonized_extraction(plan, parallel=True, max_workers=4)
            
            # Export other formats using traditional methods
            other_formats = [fmt for fmt in formats if fmt != 'traw']
            if other_formats:
                other_output_files = self.formatter.export_multiple_formats(
                    df=extracted_df,
                    output_dir=output_dir,
                    base_name=base_name,
                    formats=other_formats,
                    snp_list=snp_list,
                    harmonization_stats=harmonization_summary
                )
                output_files.update(other_output_files)
        
        # Always generate additional metadata files
        try:
            # Create QC report
            qc_path = os.path.join(output_dir, f"{base_name}_qc_report.json")
            
            # If we have extracted_df, use it; otherwise create minimal QC report
            if 'extracted_df' in locals() and not extracted_df.empty:
                self.formatter.create_qc_report(extracted_df, qc_path)
            else:
                # Create basic QC report from harmonization summary
                basic_qc = harmonization_summary.copy() if harmonization_summary else {}
                basic_qc['generation_timestamp'] = datetime.now().isoformat()
                basic_qc['export_method'] = 'native_plink'
                
                with open(qc_path, 'w') as f:
                    import json
                    json.dump(basic_qc, f, indent=2, default=str)
            
            output_files['qc_report'] = qc_path
            
            # Harmonization report
            if harmonization_summary:
                report_path = os.path.join(output_dir, f"{base_name}_harmonization_report.json")
                self.formatter.write_harmonization_report(harmonization_summary, report_path)
                output_files['harmonization_report'] = report_path
                
        except Exception as e:
            logger.error(f"Failed to create additional files: {e}")
        
        logger.info(f"Exported {len(output_files)} files using native PLINK approach")
        return output_files
    
    def _export_native_traw_single(
        self, 
        harmonized_pgen_path: str, 
        output_dir: str, 
        base_name: str,
        snp_list_ids: List[str]
    ) -> Optional[str]:
        """Export TRAW from single harmonized PLINK file."""
        try:
            # Remove .pgen extension
            harmonized_base = harmonized_pgen_path.replace('.pgen', '')
            output_path = os.path.join(output_dir, f"{base_name}.traw")
            
            # Create variant list file
            with tempfile.TemporaryDirectory() as temp_dir:
                variant_file = os.path.join(temp_dir, "variants.txt")
                with open(variant_file, 'w') as f:
                    for var_id in snp_list_ids:
                        f.write(f"{var_id}\n")
                
                # PLINK2 command to export TRAW
                temp_output = os.path.join(temp_dir, "export")
                cmd = [
                    'plink2',
                    '--pfile', harmonized_base,
                    '--extract', variant_file,
                    '--export', 'A-transpose',
                    '--out', temp_output
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                
                if result.returncode == 0:
                    temp_traw = f"{temp_output}.traw"
                    if os.path.exists(temp_traw):
                        # Move to final location
                        import shutil
                        shutil.move(temp_traw, output_path)
                        return output_path
                    else:
                        logger.error("TRAW file was not created by PLINK2")
                        return None
                else:
                    logger.error(f"PLINK2 export failed: {result.stderr}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in native TRAW export: {e}")
            return None
    
    def _export_native_traw_merged(
        self, 
        harmonized_files: List[str], 
        output_dir: str, 
        base_name: str,
        snp_list_ids: List[str]
    ) -> Optional[str]:
        """Export and merge TRAW from multiple harmonized PLINK files."""
        try:
            output_path = os.path.join(output_dir, f"{base_name}.traw")
            
            # Export each file separately then merge
            traw_files = []
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create variant list file
                variant_file = os.path.join(temp_dir, "variants.txt")
                with open(variant_file, 'w') as f:
                    for var_id in snp_list_ids:
                        f.write(f"{var_id}\n")
                
                # Export each harmonized file
                for i, harmonized_pgen in enumerate(harmonized_files):
                    harmonized_base = harmonized_pgen.replace('.pgen', '')
                    temp_output = os.path.join(temp_dir, f"export_{i}")
                    
                    cmd = [
                        'plink2',
                        '--pfile', harmonized_base,
                        '--extract', variant_file,
                        '--export', 'A-transpose',
                        '--out', temp_output
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    
                    if result.returncode == 0:
                        temp_traw = f"{temp_output}.traw"
                        if os.path.exists(temp_traw):
                            traw_files.append(temp_traw)
                        else:
                            logger.warning(f"TRAW file not created for {harmonized_pgen}")
                    else:
                        logger.error(f"PLINK2 export failed for {harmonized_pgen}: {result.stderr}")
                
                # Merge TRAW files
                if traw_files:
                    self._merge_traw_files(traw_files, output_path)
                    return output_path
                else:
                    logger.error("No TRAW files were successfully created")
                    return None
                    
        except Exception as e:
            logger.error(f"Error in merged TRAW export: {e}")
            return None
    
    def _merge_traw_files(self, traw_files: List[str], output_path: str) -> None:
        """Merge multiple TRAW files into one."""
        try:
            # Read all TRAW files
            dfs = []
            for traw_file in traw_files:
                df = pd.read_csv(traw_file, sep='\t', low_memory=False)
                dfs.append(df)
            
            # Concatenate all DataFrames
            if dfs:
                merged_df = pd.concat(dfs, ignore_index=True)
                
                # Remove duplicates based on SNP column
                merged_df = merged_df.drop_duplicates(subset=['SNP'], keep='first')
                
                # Write merged TRAW
                merged_df.to_csv(output_path, sep='\t', index=False, na_rep='NA')
                logger.info(f"Merged {len(traw_files)} TRAW files into {output_path}")
            else:
                logger.error("No TRAW files to merge")
                
        except Exception as e:
            logger.error(f"Error merging TRAW files: {e}")
            raise
    
    def export_results(
        self, 
        df: pd.DataFrame, 
        output_dir: str,
        base_name: Optional[str] = None,
        formats: List[str] = ['traw', 'parquet'],
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
        
        # Create QC report
        try:
            qc_path = os.path.join(output_dir, f"{base_name}_qc_report.json")
            self.formatter.create_qc_report(df, qc_path)
            output_files['qc_report'] = qc_path
        except Exception as e:
            logger.error(f"Failed to create QC report: {e}")
        
        logger.info(f"Exported results to {len(output_files)} files in {output_dir}")
        return output_files
    
    def run_full_extraction_pipeline(
        self, 
        snp_list_path: str,
        data_types: List[DataType],
        output_dir: str,
        ancestries: Optional[List[str]] = None,
        output_formats: List[str] = ['traw', 'parquet'],
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Run complete extraction pipeline from SNP list to final output.
        
        Args:
            snp_list_path: Path to SNP list file
            data_types: Data types to extract from
            output_dir: Output directory
            ancestries: Optional ancestry filter
            output_formats: Output formats
            parallel: Use parallel processing
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary with pipeline results and metadata
        """
        pipeline_start = time.time()
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
            snp_list_ids = snp_list['variant_id'].tolist() if 'variant_id' in snp_list.columns else snp_list.index.tolist()
            
            # Step 2: Create extraction plan
            logger.info("Step 2: Creating extraction plan")
            plan = self.plan_extraction(snp_list_ids, data_types, ancestries)
            
            # Step 3: Check for harmonized files and generate summary from cache
            logger.info("Step 3: Generating harmonization summary from cache")
            harmonization_summary = self._generate_harmonization_summary_from_plan(plan)
            
            # Step 4: Export results (using native PLINK export when possible)
            logger.info("Step 4: Exporting results")
            output_files = self.export_results_with_native_plink(
                plan=plan,
                output_dir=output_dir,
                base_name=job_id,
                formats=output_formats,
                snp_list=snp_list,
                harmonization_summary=harmonization_summary
            )
            
            # Pipeline completed successfully
            results['success'] = True
            results['output_files'] = output_files
            results['summary'] = harmonization_summary
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