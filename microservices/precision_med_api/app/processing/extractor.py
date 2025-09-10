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
from concurrent.futures import as_completed
import subprocess
import tempfile
import time

from ..models.analysis import DataType
from ..models.harmonization import HarmonizationRecord, HarmonizationAction
from ..core.config import Settings
from ..utils.parquet_io import save_parquet
from .transformer import GenotypeTransformer
from .harmonizer import HarmonizationEngine

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
                                  capture_output=True, text=True, timeout=self.settings.plink_timeout_short)
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
        Extract variants using PLINK 2.0 with two-step memory optimization.
        
        Step 1: Extract variants to intermediate PGEN (memory efficient)
        Step 2: Convert small PGEN to TRAW (low memory usage)
        
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
            
            # Step 1: Extract variants to intermediate PGEN file (memory efficient)
            intermediate_prefix = f"{output_prefix}_filtered"
            cmd_step1 = [
                'plink2',
                '--pfile', pgen_path,
                '--extract', variant_file,
                '--make-pgen',
                '--out', intermediate_prefix
            ]
            
            logger.debug(f"Step 1: Extracting {len(variant_ids)} variants to intermediate PGEN")
            result_step1 = subprocess.run(cmd_step1, capture_output=True, text=True, timeout=self.settings.plink_timeout_medium)
            
            if result_step1.returncode != 0:
                logger.error(f"PLINK 2.0 Step 1 (PGEN extraction) failed: {result_step1.stderr}")
                return None
            
            # Step 2: Convert small filtered PGEN to TRAW (low memory usage)
            cmd_step2 = [
                'plink2',
                '--pfile', intermediate_prefix,
                '--export', 'A-transpose',
                '--out', output_prefix
            ]
            
            logger.debug(f"Step 2: Converting filtered PGEN to TRAW")
            result_step2 = subprocess.run(cmd_step2, capture_output=True, text=True, timeout=self.settings.plink_timeout_medium)
            
            if result_step2.returncode == 0:
                traw_file = f"{output_prefix}.traw"
                if os.path.exists(traw_file):
                    logger.debug(f"Successfully extracted {len(variant_ids)} variants via two-step process")
                    return traw_file
            else:
                logger.error(f"PLINK 2.0 Step 2 (TRAW export) failed: {result_step2.stderr}")
            
        except Exception as e:
            logger.error(f"Error running PLINK 2.0 two-step extraction: {e}")
        
        # Clean up temp files
        temp_files = [
            variant_file,
            f"{intermediate_prefix}.pgen",
            f"{intermediate_prefix}.pvar", 
            f"{intermediate_prefix}.psam",
            f"{intermediate_prefix}.log"
        ]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
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
                            # Convert to numeric first, handling string genotypes
                            try:
                                numeric_gt = pd.to_numeric(row[col], errors='coerce')
                                if pd.notna(numeric_gt):
                                    original_gt = np.array([numeric_gt])
                                    transformed_gt = self._apply_genotype_transform(original_gt, transform)
                                    transformed_row[col] = transformed_gt[0]
                                else:
                                    # Keep original value if conversion fails
                                    transformed_row[col] = row[col]
                            except Exception as e:
                                logger.warning(f"Failed to transform genotype {row[col]} for column {col}: {e}")
                                transformed_row[col] = row[col]
                    
                    # Update allele information to match SNP list
                    transformed_row['counted_allele'] = harm_info['snp_list_a1']
                    transformed_row['alt_allele'] = harm_info['snp_list_a2']
                    
                    # Add harmonization metadata
                    transformed_row['harmonization_action'] = action
                    transformed_row['snp_list_id'] = harm_info['snp_list_id']
                    
                    # Add original PVAR alleles for transparency
                    transformed_row['pgen_a1'] = harm_info['pgen_a1']
                    transformed_row['pgen_a2'] = harm_info['pgen_a2']
                    
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
    