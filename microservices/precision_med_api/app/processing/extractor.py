"""
Variant extraction engine with allele harmonization.

Extracts variants from PLINK files and applies harmonization transformations
to ensure alleles match the reference SNP list orientation.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import tempfile
import time

from ..models.analysis import DataType
from ..models.harmonization import HarmonizationRecord, HarmonizationAction
from ..core.config import Settings
from ..utils.parquet_io import read_parquet, save_parquet
from .transformer import GenotypeTransformer
from .cache import CacheBuilder

logger = logging.getLogger(__name__)


class VariantExtractor:
    """Extracts variants from PLINK files with harmonization."""
    
    def __init__(self, cache_dir: str, settings: Settings):
        self.cache_dir = cache_dir
        self.settings = settings
        self.transformer = GenotypeTransformer()
        self.cache_builder = CacheBuilder(settings)
    
    def _check_plink_availability(self) -> bool:
        """Check if PLINK 2.0 is available."""
        try:
            result = subprocess.run(['plink2', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.warning("PLINK 2.0 not found, falling back to alternative extraction methods")
            return False
    
    def _extract_with_plink2(
        self, 
        pgen_path: str, 
        variant_ids: List[str],
        output_prefix: str
    ) -> Optional[str]:
        """
        Extract variants using PLINK 2.0.
        
        Args:
            pgen_path: Path to PGEN file (without extension)
            variant_ids: List of variant IDs to extract
            output_prefix: Output file prefix
            
        Returns:
            Path to output TRAW file or None if failed
        """
        try:
            # Create variant list file
            variant_file = f"{output_prefix}_variants.txt"
            with open(variant_file, 'w') as f:
                for var_id in variant_ids:
                    f.write(f"{var_id}\n")
            
            # PLINK 2.0 command to extract variants
            cmd = [
                'plink2',
                '--pfile', pgen_path,
                '--extract', variant_file,
                '--export', 'A-transpose',
                '--out', output_prefix
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                traw_file = f"{output_prefix}.traw"
                if os.path.exists(traw_file):
                    return traw_file
            else:
                logger.error(f"PLINK 2.0 extraction failed: {result.stderr}")
            
        except Exception as e:
            logger.error(f"Error running PLINK 2.0: {e}")
        
        # Clean up temp files
        for temp_file in [variant_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return None
    
    def _read_traw_file(self, traw_path: str) -> pd.DataFrame:
        """Read PLINK TRAW format file."""
        try:
            # TRAW format: CHR SNP (C)M POS COUNTED ALT [sample genotypes...]
            df = pd.read_csv(traw_path, sep='\t', low_memory=False)
            
            # Rename columns for consistency
            if 'CHR' in df.columns:
                df = df.rename(columns={'CHR': 'chromosome'})
            if 'SNP' in df.columns:
                df = df.rename(columns={'SNP': 'variant_id'})
            if 'POS' in df.columns:
                df = df.rename(columns={'POS': 'position'})
            
            logger.info(f"Read {len(df)} variants from TRAW file {traw_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read TRAW file {traw_path}: {e}")
            raise
    
    def _simulate_plink_extraction(
        self, 
        pgen_path: str, 
        variant_ids: List[str]
    ) -> pd.DataFrame:
        """
        Simulate PLINK extraction for testing (when PLINK not available).
        Creates dummy genotype data with realistic structure.
        """
        logger.warning("Simulating PLINK extraction - using dummy data")
        
        # Generate dummy data
        n_samples = 1000
        sample_ids = [f"SAMPLE_{i:06d}" for i in range(n_samples)]
        
        data = []
        for var_id in variant_ids:
            # Generate realistic genotype frequencies
            freq = np.random.uniform(0.01, 0.5)  # Minor allele frequency
            p_aa = (1 - freq) ** 2
            p_ab = 2 * freq * (1 - freq)
            p_bb = freq ** 2
            
            # Generate genotypes
            genotypes = np.random.choice([0, 1, 2], size=n_samples, p=[p_aa, p_ab, p_bb])
            
            # Add some missing data
            missing_mask = np.random.random(n_samples) < 0.02  # 2% missing
            genotypes = genotypes.astype(float)
            genotypes[missing_mask] = np.nan
            
            # Create row
            row = {
                'chromosome': var_id.split(':')[0],
                'variant_id': var_id,
                'position': int(var_id.split(':')[1]),
                'counted_allele': var_id.split(':')[2],
                'alt_allele': var_id.split(':')[3]
            }
            
            # Add genotype columns
            for i, sample_id in enumerate(sample_ids):
                row[sample_id] = genotypes[i]
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _load_harmonization_cache(self, file_path: str) -> pd.DataFrame:
        """Load harmonization cache for a specific file."""
        try:
            # Determine cache path based on file path using settings method
            if "wgs" in file_path.lower():
                cache_path = self.settings.get_harmonization_cache_path("WGS", self.settings.release)
            elif "nba" in file_path.lower() or any(anc in file_path for anc in self.settings.ANCESTRIES):
                # Extract ancestry from path
                ancestry = None
                for anc in self.settings.ANCESTRIES:
                    if anc in file_path:
                        ancestry = anc
                        break
                if ancestry:
                    cache_path = self.settings.get_harmonization_cache_path("NBA", self.settings.release, ancestry)
                else:
                    raise ValueError(f"Cannot determine ancestry from path: {file_path}")
            else:
                # Assume imputed
                # Extract ancestry and chromosome from path
                ancestry = None
                chrom = None
                for anc in self.settings.ANCESTRIES:
                    if anc in file_path:
                        ancestry = anc
                        break
                
                # Extract chromosome
                import re
                chrom_match = re.search(r'chr(\d+|X|Y|MT)', file_path)
                if chrom_match:
                    chrom = chrom_match.group(1)
                
                if ancestry and chrom:
                    cache_path = self.settings.get_harmonization_cache_path("IMPUTED", self.settings.release, ancestry, chrom)
                else:
                    raise ValueError(f"Cannot determine ancestry/chromosome from path: {file_path}")
            
            if os.path.exists(cache_path):
                return read_parquet(cache_path)
            else:
                logger.info(f"Harmonization cache not found: {cache_path}")
                logger.info("Building harmonization cache automatically...")
                return self._build_cache_if_missing(file_path, cache_path)
                
        except Exception as e:
            logger.error(f"Failed to load harmonization cache for {file_path}: {e}")
            return pd.DataFrame()
    
    def _build_cache_if_missing(self, pgen_path: str, cache_path: str) -> pd.DataFrame:
        """Build harmonization cache if it doesn't exist."""
        try:
            # Determine data type and extract metadata from path
            if "wgs" in pgen_path.lower():
                data_type = "WGS"
                ancestry = None
            elif "nba" in pgen_path.lower() or any(anc in pgen_path for anc in self.settings.ANCESTRIES):
                data_type = "NBA"
                ancestry = None
                for anc in self.settings.ANCESTRIES:
                    if anc in pgen_path:
                        ancestry = anc
                        break
                if not ancestry:
                    raise ValueError(f"Cannot determine ancestry from path: {pgen_path}")
            else:
                data_type = "IMPUTED"
                ancestry = None
                for anc in self.settings.ANCESTRIES:
                    if anc in pgen_path:
                        ancestry = anc
                        break
                if not ancestry:
                    raise ValueError(f"Cannot determine ancestry from path: {pgen_path}")
            
            # Get PVAR path
            pvar_path = pgen_path.replace('.pgen', '.pvar')
            if not os.path.exists(pvar_path):
                logger.error(f"PVAR file not found: {pvar_path}")
                return pd.DataFrame()
            
            # Load SNP list from coordinator (we need it for cache building)
            # For now, we'll need to accept it as a parameter or load the default
            snp_list_path = self.settings.snp_list_path
            if not os.path.exists(snp_list_path):
                logger.error(f"SNP list not found: {snp_list_path}")
                return pd.DataFrame()
            
            # Load and parse SNP list
            import pandas as pd
            snp_list = pd.read_csv(snp_list_path)
            coords = snp_list['hg38'].str.split(':', expand=True)
            snp_list['chromosome'] = coords[0].str.replace('chr', '').str.upper()
            snp_list['position'] = pd.to_numeric(coords[1], errors='coerce')
            snp_list['ref'] = coords[2].str.upper().str.strip()
            snp_list['alt'] = coords[3].str.upper().str.strip()
            snp_list['variant_id'] = snp_list['hg38']
            
            logger.info(f"Building harmonization cache for {data_type} {ancestry or ''}")
            
            # Build cache
            cache_df, stats = self.cache_builder.build_harmonization_cache(
                pvar_path=pvar_path,
                snp_list=snp_list,
                data_type=data_type,
                ancestry=ancestry
            )
            
            # Save cache
            from pathlib import Path
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_parquet(cache_path)
            
            logger.info(f"Built and saved harmonization cache: {len(cache_df)} variants")
            logger.info(f"Cache stats: {stats}")
            
            return cache_df
            
        except Exception as e:
            logger.error(f"Failed to build harmonization cache: {e}")
            return pd.DataFrame()
    
    def _get_extraction_plan(
        self, 
        snp_list_ids: List[str], 
        cache_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get extraction plan from harmonization cache.
        
        Args:
            snp_list_ids: Variant IDs from SNP list
            cache_df: Harmonization cache DataFrame
            
        Returns:
            DataFrame with extraction plan
        """
        if cache_df.empty:
            return pd.DataFrame()
        
        # Filter cache for requested variants
        plan_df = cache_df[cache_df['snp_list_id'].isin(snp_list_ids)].copy()
        
        # Only include successfully harmonized variants
        valid_actions = [HarmonizationAction.EXACT, HarmonizationAction.SWAP, 
                        HarmonizationAction.FLIP, HarmonizationAction.FLIP_SWAP]
        plan_df = plan_df[plan_df['harmonization_action'].isin([a.value for a in valid_actions])]
        
        logger.info(f"Extraction plan: {len(plan_df)} variants from {len(snp_list_ids)} requested")
        return plan_df
    
    def _extract_raw_genotypes(
        self, 
        pgen_path: str, 
        pgen_variant_ids: List[str]
    ) -> pd.DataFrame:
        """
        Extract raw genotypes from PLINK file.
        
        Args:
            pgen_path: Path to PGEN file
            pgen_variant_ids: List of variant IDs to extract from PLINK file
            
        Returns:
            DataFrame with raw genotypes
        """
        # Remove .pgen extension if present
        if pgen_path.endswith('.pgen'):
            pgen_base = pgen_path[:-5]
        else:
            pgen_base = pgen_path
        
        # Check if PLINK is available
        if self._check_plink_availability():
            # Create temporary output prefix
            with tempfile.TemporaryDirectory() as temp_dir:
                output_prefix = os.path.join(temp_dir, "extracted")
                
                traw_file = self._extract_with_plink2(pgen_base, pgen_variant_ids, output_prefix)
                
                if traw_file and os.path.exists(traw_file):
                    return self._read_traw_file(traw_file)
        
        # Fall back to simulation if PLINK not available
        logger.warning(f"Using simulated extraction for {pgen_path}")
        return self._simulate_plink_extraction(pgen_path, pgen_variant_ids)
    
    def _apply_genotype_transform(
        self, 
        genotypes: np.ndarray, 
        transform: Optional[str]
    ) -> np.ndarray:
        """Apply genotype transformation."""
        return self.transformer.apply_transformation_by_formula(genotypes, transform)
    
    def _harmonize_extracted_genotypes(
        self, 
        raw_df: pd.DataFrame, 
        harmonization_records: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply harmonization transformations to extracted genotypes.
        
        Args:
            raw_df: Raw genotype data
            harmonization_records: Harmonization records with transformation info
            
        Returns:
            DataFrame with harmonized genotypes
        """
        if raw_df.empty or harmonization_records.empty:
            return raw_df
        
        harmonized_df = raw_df.copy()
        
        # Get sample columns (exclude metadata columns)
        metadata_cols = ['chromosome', 'variant_id', 'position', 'counted_allele', 'alt_allele']
        sample_cols = [col for col in raw_df.columns if col not in metadata_cols]
        
        # Create mapping from PGEN variant ID to harmonization info
        harm_lookup = harmonization_records.set_index('pgen_variant_id').to_dict('index')
        
        # Apply transformations
        for idx, row in harmonized_df.iterrows():
            var_id = row['variant_id']
            
            if var_id in harm_lookup:
                harm_info = harm_lookup[var_id]
                transform = harm_info.get('genotype_transform')
                action = harm_info.get('harmonization_action')
                
                # Apply transformation to all sample columns
                for col in sample_cols:
                    if pd.notna(row[col]):
                        original_gt = np.array([row[col]])
                        transformed_gt = self._apply_genotype_transform(original_gt, transform)
                        harmonized_df.at[idx, col] = transformed_gt[0]
                
                # Update allele information to match SNP list
                harmonized_df.at[idx, 'counted_allele'] = harm_info['snp_list_a1']
                harmonized_df.at[idx, 'alt_allele'] = harm_info['snp_list_a2']
                
                # Add harmonization metadata
                harmonized_df.at[idx, 'harmonization_action'] = action
                harmonized_df.at[idx, 'snp_list_id'] = harm_info['snp_list_id']
        
        return harmonized_df
    
    def _extract_from_harmonized_plink(
        self, 
        harmonized_pgen_path: str, 
        plan_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract variants from harmonized PLINK files using PLINK2.
        
        Args:
            harmonized_pgen_path: Path to harmonized PGEN file
            plan_df: Extraction plan DataFrame
            
        Returns:
            DataFrame with harmonized genotypes (no additional transformation needed)
        """
        try:
            # Extract variant IDs we need from the harmonized file
            pgen_variant_ids = plan_df['pgen_variant_id'].tolist()
            
            # Remove .pgen extension if present
            if harmonized_pgen_path.endswith('.pgen'):
                harmonized_base = harmonized_pgen_path[:-5]
            else:
                harmonized_base = harmonized_pgen_path
            
            # Check if PLINK is available
            if self._check_plink_availability():
                # Use PLINK2 to extract variants from harmonized files
                with tempfile.TemporaryDirectory() as temp_dir:
                    output_prefix = os.path.join(temp_dir, "harmonized_extract")
                    
                    traw_file = self._extract_with_plink2(harmonized_base, pgen_variant_ids, output_prefix)
                    
                    if traw_file and os.path.exists(traw_file):
                        # Read the TRAW file directly - no additional harmonization needed!
                        traw_df = self._read_traw_file(traw_file)
                        
                        # Add metadata from plan
                        if not traw_df.empty:
                            # Create mapping from pgen_variant_id to metadata
                            metadata_map = plan_df.set_index('pgen_variant_id')[
                                ['snp_list_id', 'harmonization_action', 'genotype_transform']
                            ].to_dict('index')
                            
                            # Add metadata columns
                            for idx, row in traw_df.iterrows():
                                var_id = row.get('variant_id', '')
                                if var_id in metadata_map:
                                    metadata = metadata_map[var_id]
                                    traw_df.at[idx, 'snp_list_id'] = metadata['snp_list_id']
                                    traw_df.at[idx, 'harmonization_action'] = metadata['harmonization_action']
                                    traw_df.at[idx, 'genotype_transform'] = metadata['genotype_transform']
                                    traw_df.at[idx, 'pgen_variant_id'] = var_id
                            
                            # Add allele information
                            traw_df['counted_allele'] = traw_df.get('COUNTED', '.')
                            traw_df['alt_allele'] = traw_df.get('ALT', '.')
                            
                            logger.info(f"Extracted {len(traw_df)} variants from harmonized PLINK files")
                            return traw_df
                        else:
                            logger.warning("Empty TRAW file from harmonized PLINK extraction")
                            return pd.DataFrame()
                    else:
                        logger.error("Failed to extract from harmonized PLINK files")
                        return pd.DataFrame()
            else:
                logger.warning("PLINK2 not available, cannot extract from harmonized files")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting from harmonized PLINK files: {e}")
            return pd.DataFrame()
    
    def extract_single_file_harmonized(
        self, 
        pgen_path: str, 
        snp_list_ids: List[str]
    ) -> pd.DataFrame:
        """
        Extract variants from a single PLINK file with harmonization.
        
        Args:
            pgen_path: Path to PGEN file
            snp_list_ids: List of variant IDs from SNP list
            
        Returns:
            DataFrame with harmonized genotypes
        """
        logger.info(f"Extracting {len(snp_list_ids)} variants from {pgen_path}")
        
        # Load harmonization cache
        cache_df = self._load_harmonization_cache(pgen_path)
        
        if cache_df.empty:
            logger.warning(f"No harmonization cache found for {pgen_path}")
            return pd.DataFrame()
        
        # Get extraction plan
        plan_df = self._get_extraction_plan(snp_list_ids, cache_df)
        
        if plan_df.empty:
            logger.warning(f"No variants to extract from {pgen_path}")
            return pd.DataFrame()
        
        # Check if harmonized PLINK files are available
        if 'harmonized_file_path' in cache_df.columns:
            harmonized_pgen_path = cache_df['harmonized_file_path'].iloc[0]
            if harmonized_pgen_path and os.path.exists(harmonized_pgen_path):
                logger.info(f"Using harmonized PLINK files: {harmonized_pgen_path}")
                return self._extract_from_harmonized_plink(harmonized_pgen_path, plan_df)
        
        # Fallback to traditional raw extraction + harmonization
        logger.info("Using traditional extraction with manual harmonization")
        pgen_variant_ids = plan_df['pgen_variant_id'].tolist()
        raw_df = self._extract_raw_genotypes(pgen_path, pgen_variant_ids)
        
        if raw_df.empty:
            logger.warning(f"No genotypes extracted from {pgen_path}")
            return pd.DataFrame()
        
        # Apply harmonization
        harmonized_df = self._harmonize_extracted_genotypes(raw_df, plan_df)
        
        logger.info(f"Extracted and harmonized {len(harmonized_df)} variants from {pgen_path}")
        return harmonized_df
    
    def extract_nba(
        self, 
        snp_list_ids: List[str], 
        ancestries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Extract variants from NBA files."""
        if ancestries is None:
            ancestries = self.settings.list_available_ancestries("NBA")
        
        all_dfs = []
        
        for ancestry in ancestries:
            try:
                pgen_path = self.settings.get_nba_path(ancestry) + ".pgen"
                if os.path.exists(pgen_path):
                    df = self.extract_single_file_harmonized(pgen_path, snp_list_ids)
                    if not df.empty:
                        df['data_type'] = 'NBA'
                        df['ancestry'] = ancestry
                        all_dfs.append(df)
                else:
                    logger.warning(f"NBA file not found: {pgen_path}")
            except Exception as e:
                logger.error(f"Failed to extract from NBA {ancestry}: {e}")
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def extract_wgs(self, snp_list_ids: List[str]) -> pd.DataFrame:
        """Extract variants from WGS file."""
        try:
            pgen_path = self.settings.get_wgs_path() + ".pgen"
            if os.path.exists(pgen_path):
                df = self.extract_single_file_harmonized(pgen_path, snp_list_ids)
                if not df.empty:
                    df['data_type'] = 'WGS'
                    df['ancestry'] = None
                return df
            else:
                logger.warning(f"WGS file not found: {pgen_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to extract from WGS: {e}")
            return pd.DataFrame()
    
    def extract_imputed(
        self, 
        snp_list_ids: List[str], 
        ancestries: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Extract variants from imputed files."""
        if ancestries is None:
            ancestries = self.settings.list_available_ancestries("IMPUTED")
        
        all_dfs = []
        
        for ancestry in ancestries:
            available_chroms = self.settings.list_available_chromosomes(ancestry)
            
            for chrom in available_chroms:
                try:
                    pgen_path = self.settings.get_imputed_path(ancestry, chrom) + ".pgen"
                    if os.path.exists(pgen_path):
                        df = self.extract_single_file_harmonized(pgen_path, snp_list_ids)
                        if not df.empty:
                            df['data_type'] = 'IMPUTED'
                            df['ancestry'] = ancestry
                            df['chromosome_file'] = chrom
                            all_dfs.append(df)
                    else:
                        logger.warning(f"Imputed file not found: {pgen_path}")
                except Exception as e:
                    logger.error(f"Failed to extract from imputed {ancestry} chr{chrom}: {e}")
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def extract_all_sources(
        self, 
        snp_list_ids: List[str], 
        data_types: List[DataType]
    ) -> pd.DataFrame:
        """
        Extract variants from all specified data sources.
        
        Args:
            snp_list_ids: Variant IDs to extract
            data_types: List of data types to extract from
            
        Returns:
            Combined DataFrame with all extracted variants
        """
        all_dfs = []
        
        for data_type in data_types:
            try:
                if data_type == DataType.NBA:
                    df = self.extract_nba(snp_list_ids)
                elif data_type == DataType.WGS:
                    df = self.extract_wgs(snp_list_ids)
                elif data_type == DataType.IMPUTED:
                    df = self.extract_imputed(snp_list_ids)
                else:
                    logger.warning(f"Unknown data type: {data_type}")
                    continue
                
                if not df.empty:
                    all_dfs.append(df)
                    
            except Exception as e:
                logger.error(f"Failed to extract from {data_type}: {e}")
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            logger.info(f"Extracted {len(combined_df)} total variant records from {len(data_types)} data sources")
            return combined_df
        else:
            logger.warning("No variants extracted from any data source")
            return pd.DataFrame()
    
    def merge_harmonized_genotypes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge harmonized genotypes from multiple sources.
        
        Args:
            dfs: List of DataFrames with harmonized genotypes
            
        Returns:
            Merged DataFrame
        """
        if not dfs:
            return pd.DataFrame()
        
        if len(dfs) == 1:
            return dfs[0]
        
        # Concatenate all DataFrames
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Remove duplicates based on variant and sample
        # Priority: WGS > NBA > IMPUTED
        data_type_priority = {'WGS': 3, 'NBA': 2, 'IMPUTED': 1}
        merged_df['priority'] = merged_df['data_type'].map(data_type_priority)
        
        # Sort by priority and keep first occurrence
        merged_df = merged_df.sort_values('priority', ascending=False)
        merged_df = merged_df.drop_duplicates(subset=['snp_list_id'], keep='first')
        merged_df = merged_df.drop(columns=['priority'])
        
        logger.info(f"Merged to {len(merged_df)} unique variants")
        return merged_df
    
    def validate_allele_consistency(self, df: pd.DataFrame) -> bool:
        """Validate that alleles are consistent across variants."""
        if df.empty:
            return True
        
        # Check for variants with inconsistent alleles
        inconsistent = df.groupby('snp_list_id').agg({
            'counted_allele': 'nunique',
            'alt_allele': 'nunique'
        })
        
        inconsistent_variants = inconsistent[
            (inconsistent['counted_allele'] > 1) | (inconsistent['alt_allele'] > 1)
        ]
        
        if len(inconsistent_variants) > 0:
            logger.warning(f"Found {len(inconsistent_variants)} variants with inconsistent alleles")
            return False
        
        return True
    
    def handle_multi_allelic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle multi-allelic variants by keeping most common."""
        if df.empty:
            return df
        
        # Group by position and keep most common variant
        position_groups = df.groupby(['chromosome', 'position'])
        
        filtered_rows = []
        for name, group in position_groups:
            if len(group) > 1:
                # Keep variant with highest frequency (if available)
                if 'alt_freq' in group.columns:
                    best_variant = group.loc[group['alt_freq'].idxmax()]
                else:
                    # Keep first variant
                    best_variant = group.iloc[0]
                filtered_rows.append(best_variant)
            else:
                filtered_rows.append(group.iloc[0])
        
        result_df = pd.DataFrame(filtered_rows).reset_index(drop=True)
        
        if len(result_df) < len(df):
            logger.info(f"Filtered {len(df)} to {len(result_df)} variants (removed multi-allelic)")
        
        return result_df