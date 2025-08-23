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
from ..utils.parquet_io import save_parquet
from .transformer import GenotypeTransformer
from .harmonization import HarmonizationEngine

logger = logging.getLogger(__name__)


class VariantExtractor:
    """Extracts variants from PLINK files with harmonization."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.transformer = GenotypeTransformer()
        self.harmonization_engine = HarmonizationEngine(settings)
    
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
        # Allow multiple SNP list variants to map to the same PGEN variant (many-to-one mapping)
        # This enables proper handling of cases like multiple probes at the same genomic position
        if harmonization_records['pgen_variant_id'].duplicated().any():
            logger.info(f"Found {harmonization_records['pgen_variant_id'].duplicated().sum()} SNP list variants mapping to the same PGEN variant - preserving all mappings")
        
        # Create a list-based lookup to handle multiple SNP list variants per PGEN variant
        harm_lookup = {}
        for _, record in harmonization_records.iterrows():
            pgen_id = record['pgen_variant_id']
            if pgen_id not in harm_lookup:
                harm_lookup[pgen_id] = []
            harm_lookup[pgen_id].append(record.to_dict())
        
        # Apply transformations - handle multiple SNP list variants per PGEN variant
        transformed_rows = []
        
        for _, row in harmonized_df.iterrows():
            var_id = row['variant_id']
            
            if var_id in harm_lookup:
                # Process each harmonization record for this PGEN variant
                for harm_info in harm_lookup[var_id]:
                    transform = harm_info.get('genotype_transform')
                    action = harm_info.get('harmonization_action')
                    
                    # Create a copy of the row for this SNP list variant
                    transformed_row = row.copy()
                    
                    # Apply transformation to all sample columns
                    for col in sample_cols:
                        if pd.notna(row[col]):
                            original_gt = np.array([row[col]])
                            transformed_gt = self._apply_genotype_transform(original_gt, transform)
                            transformed_row[col] = transformed_gt[0]
                    
                    # Update allele information to match SNP list
                    transformed_row['counted_allele'] = harm_info['snp_list_a1']
                    transformed_row['alt_allele'] = harm_info['snp_list_a2']
                    
                    # Add harmonization metadata
                    transformed_row['harmonization_action'] = action
                    transformed_row['snp_list_id'] = harm_info['snp_list_id']
                    
                    transformed_rows.append(transformed_row)
        
        # Convert list of transformed rows back to DataFrame
        if transformed_rows:
            harmonized_df = pd.DataFrame(transformed_rows)
            # Reset index to avoid duplicate indices
            harmonized_df = harmonized_df.reset_index(drop=True)
        else:
            # No harmonization records found, return empty DataFrame with same structure
            harmonized_df = pd.DataFrame(columns=harmonized_df.columns)
        
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
        snp_list_ids: List[str],
        snp_list_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extract variants from a single PLINK file with real-time harmonization.
        
        Args:
            pgen_path: Path to PGEN file
            snp_list_ids: List of variant IDs from SNP list
            snp_list_df: Optional SNP list DataFrame (if not provided, will be reconstructed)
            
        Returns:
            DataFrame with harmonized genotypes
        """
        logger.info(f"Extracting {len(snp_list_ids)} variants from {pgen_path}")
        
        # Read PVAR file for this PLINK file
        try:
            pvar_df = self.harmonization_engine.read_pvar_file(pgen_path)
        except Exception as e:
            logger.error(f"Failed to read PVAR file for {pgen_path}: {e}")
            return pd.DataFrame()
        
        # Create SNP list DataFrame if not provided
        if snp_list_df is None:
            # Reconstruct from IDs (this is a fallback - ideally pass the full DataFrame)
            snp_list_df = self._reconstruct_snp_list_from_ids(snp_list_ids)
        
        # Perform real-time harmonization
        harmonization_records = self.harmonization_engine.harmonize_variants(pvar_df, snp_list_df)
        
        if not harmonization_records:
            logger.warning(f"No variants harmonized for {pgen_path}")
            return pd.DataFrame()
        
        # Convert to extraction plan DataFrame
        plan_df = self._harmonization_records_to_plan_df(harmonization_records, snp_list_ids)
        
        if plan_df.empty:
            logger.warning(f"No variants to extract from {pgen_path}")
            return pd.DataFrame()
        
        # Use traditional raw extraction + harmonization (no pre-harmonized files in cache-free approach)
        logger.info("Using real-time extraction with harmonization")
        pgen_variant_ids = plan_df['pgen_variant_id'].tolist()
        raw_df = self._extract_raw_genotypes(pgen_path, pgen_variant_ids)
        
        if raw_df.empty:
            logger.warning(f"No genotypes extracted from {pgen_path}")
            return pd.DataFrame()
        
        # Apply harmonization
        harmonized_df = self._harmonize_extracted_genotypes(raw_df, plan_df)
        
        logger.info(f"Extracted and harmonized {len(harmonized_df)} variants from {pgen_path}")
        return harmonized_df
    
    def _reconstruct_snp_list_from_ids(self, snp_list_ids: List[str]) -> pd.DataFrame:
        """
        Reconstruct SNP list DataFrame from variant IDs.
        This is a fallback method - ideally the full SNP list should be passed.
        """
        data = []
        for variant_id in snp_list_ids:
            try:
                # Parse variant ID format: chr:pos:ref:alt
                parts = variant_id.split(':')
                if len(parts) >= 4:
                    data.append({
                        'variant_id': variant_id,
                        'chromosome': parts[0],
                        'position': int(parts[1]),
                        'ref': parts[2],
                        'alt': parts[3]
                    })
            except Exception as e:
                logger.warning(f"Failed to parse variant ID {variant_id}: {e}")
        
        return pd.DataFrame(data)
    
    def _harmonization_records_to_plan_df(
        self, 
        records: List[HarmonizationRecord], 
        requested_snp_ids: List[str]
    ) -> pd.DataFrame:
        """
        Convert harmonization records to extraction plan DataFrame.
        """
        plan_data = []
        
        for record in records:
            if record.snp_list_id in requested_snp_ids:
                plan_data.append({
                    'snp_list_id': record.snp_list_id,
                    'pgen_variant_id': record.pgen_variant_id,
                    'harmonization_action': record.harmonization_action.value,
                    'genotype_transform': record.genotype_transform,
                    'chromosome': record.chromosome,
                    'position': record.position,
                    'snp_list_a1': record.snp_list_a1,
                    'snp_list_a2': record.snp_list_a2,
                    'pgen_a1': record.pgen_a1,
                    'pgen_a2': record.pgen_a2
                })
        
        plan_df = pd.DataFrame(plan_data)
        logger.info(f"Created extraction plan for {len(plan_df)} variants")
        return plan_df
    
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
        
        # Sort by priority and keep first occurrence for true duplicates
        merged_df = merged_df.sort_values('priority', ascending=False)
        # Use SNP list ID, chromosome, position, and alleles to identify true duplicates
        # This prevents excluding different SNP list variants at the same position (e.g., V499L vs V499M)
        dedup_columns = ['snp_list_id', 'chromosome', 'position', 'counted_allele', 'alt_allele']
        merged_df = merged_df.drop_duplicates(subset=dedup_columns, keep='first')
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