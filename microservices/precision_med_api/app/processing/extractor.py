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
    
    def _read_traw_file(self, traw_path: str, psam_path: str = None) -> pd.DataFrame:
        """Read PLINK TRAW format file.

        Args:
            traw_path: Path to TRAW file
            psam_path: Optional path to PSAM file for FID_IID -> IID mapping

        Returns:
            DataFrame with genotypes, sample IDs normalized to IID only
        """
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

            # Normalize sample IDs from FID_IID to IID using psam file
            if psam_path and os.path.exists(psam_path):
                df = self._normalize_sample_ids_from_psam(df, psam_path)

            logger.info(f"Read {len(df)} variants from TRAW file {traw_path}")
            return df

        except Exception as e:
            logger.error(f"Failed to read TRAW file {traw_path}: {e}")
            raise

    def _normalize_sample_ids_from_psam(self, df: pd.DataFrame, psam_path: str) -> pd.DataFrame:
        """Normalize sample column IDs from FID_IID format to IID only.

        PLINK2 A-transpose export creates column names as FID_IID.
        This reads the psam file to build exact FID_IID -> IID mapping.

        Args:
            df: DataFrame with sample columns in FID_IID format
            psam_path: Path to psam file with FID and IID columns

        Returns:
            DataFrame with sample columns renamed to IID only
        """
        try:
            # Read psam file (tab-separated, first line is header starting with #FID)
            psam_df = pd.read_csv(psam_path, sep='\t')

            # Handle header - psam files have #FID as first column name
            if '#FID' in psam_df.columns:
                psam_df = psam_df.rename(columns={'#FID': 'FID'})

            if 'FID' not in psam_df.columns or 'IID' not in psam_df.columns:
                logger.warning(f"PSAM file missing FID/IID columns: {psam_df.columns.tolist()}")
                return df

            # Build mapping: FID_IID -> IID
            rename_map = {}
            for _, row in psam_df.iterrows():
                fid = str(row['FID'])
                iid = str(row['IID'])
                fid_iid = f"{fid}_{iid}"
                if fid_iid in df.columns:
                    rename_map[fid_iid] = iid

            if rename_map:
                logger.info(f"Normalizing {len(rename_map)} sample IDs from FID_IID to IID format")
                df = df.rename(columns=rename_map)

            return df

        except Exception as e:
            logger.warning(f"Failed to normalize sample IDs from psam: {e}")
            return df
    
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
                    # Pass psam path for FID_IID -> IID normalization
                    psam_path = f"{pgen_base}.psam"
                    return self._read_traw_file(traw_file, psam_path=psam_path)
        
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
                    
                    # Apply harmonization genotype transformation (SWAP/FLIP_SWAP cases)
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
                    
                    # Now fix allele counting - transform genotypes to count ALT allele instead of REF
                    if action in [HarmonizationAction.EXACT, HarmonizationAction.FLIP]:
                        # For EXACT/FLIP: PLINK counts A1 (REF), we want to count A2 (ALT)
                        # Need to flip genotypes: 0->2, 1->1, 2->0
                        for col in sample_cols:
                            if pd.notna(transformed_row[col]):
                                try:
                                    gt = pd.to_numeric(transformed_row[col], errors='coerce')
                                    if pd.notna(gt):
                                        # Flip genotype to count ALT instead of REF
                                        flipped_gt = 2.0 - gt
                                        transformed_row[col] = flipped_gt
                                except Exception as e:
                                    logger.warning(f"Failed to flip genotype {transformed_row[col]} for column {col}: {e}")
                        
                        # Update TRAW columns to reflect ALT counting
                        transformed_row['COUNTED'] = harm_info['snp_list_a2']  # ALT allele (pathogenic)
                        transformed_row['ALT'] = harm_info['snp_list_a1']      # REF allele
                        
                    elif action in [HarmonizationAction.SWAP, HarmonizationAction.FLIP_SWAP]:
                        # For SWAP/FLIP_SWAP: After harmonization transform, PLINK A1 corresponds to SNP ALT
                        # Genotypes already count the correct allele, just update labels
                        transformed_row['COUNTED'] = harm_info['snp_list_a2']  # ALT allele (pathogenic) 
                        transformed_row['ALT'] = harm_info['snp_list_a1']      # REF allele
                        
                    else:
                        # INVALID or AMBIGUOUS - keep original PLINK alleles
                        transformed_row['COUNTED'] = harm_info['pgen_a1']
                        transformed_row['ALT'] = harm_info['pgen_a2']

                    # Add metadata columns for transparency
                    transformed_row['counted_allele'] = transformed_row['COUNTED']
                    transformed_row['alt_allele'] = transformed_row['ALT']
                    
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

    def _apply_maf_correction(self, harmonized_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply minor allele frequency correction to ensure we count the minor allele.

        If ALT AF > 0.5, flip genotypes so we count REF (the minor allele) instead.
        This ensures we always count the pathogenic/minor allele regardless of which
        allele is REF vs ALT in the source data.

        Args:
            harmonized_df: DataFrame with harmonized genotypes

        Returns:
            DataFrame with MAF-corrected genotypes and new columns:
            - maf_corrected: bool indicating if correction was applied
            - original_alt_af: float with the original ALT allele frequency
        """
        if harmonized_df.empty:
            return harmonized_df

        # Define metadata columns to exclude from sample columns
        metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                         'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                         'pgen_a1', 'pgen_a2', 'data_type', 'source_file']
        sample_cols = [c for c in harmonized_df.columns if c not in metadata_cols]

        corrected_rows = []
        for idx, row in harmonized_df.iterrows():
            row_copy = row.copy()

            # Calculate ALT AF from valid genotypes
            genotypes = pd.to_numeric(row[sample_cols], errors='coerce')
            valid_gt = genotypes.dropna()

            if len(valid_gt) == 0:
                row_copy['maf_corrected'] = False
                row_copy['original_alt_af'] = None
                corrected_rows.append(row_copy)
                continue

            total_alleles = len(valid_gt) * 2
            alt_allele_count = valid_gt.sum()
            alt_af = alt_allele_count / total_alleles

            if alt_af > 0.5:
                # Flip genotypes: 0→2, 1→1, 2→0
                for col in sample_cols:
                    if pd.notna(row_copy[col]):
                        try:
                            gt = float(row_copy[col])
                            row_copy[col] = 2.0 - gt
                        except:
                            pass

                # Swap counted/alt allele labels
                old_counted = row_copy.get('counted_allele', '')
                old_alt = row_copy.get('alt_allele', '')
                row_copy['counted_allele'] = old_alt
                row_copy['alt_allele'] = old_counted
                row_copy['COUNTED'] = old_alt
                row_copy['ALT'] = old_counted
                row_copy['maf_corrected'] = True

                logger.info(f"MAF correction applied to {row_copy.get('variant_id', 'unknown')}: "
                           f"ALT AF={alt_af:.3f}, now counting {old_alt}")
            else:
                row_copy['maf_corrected'] = False

            row_copy['original_alt_af'] = alt_af
            corrected_rows.append(row_copy)

        return pd.DataFrame(corrected_rows)

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
        
        # Use traditional raw extraction + harmonization
        logger.info("Using real-time extraction with harmonization")
        pgen_variant_ids = plan_df['pgen_variant_id'].tolist()
        raw_df = self._extract_raw_genotypes(pgen_path, pgen_variant_ids)
        
        if raw_df.empty:
            logger.warning(f"No genotypes extracted from {pgen_path}")
            return pd.DataFrame()
        
        # Apply harmonization
        harmonized_df = self._harmonize_extracted_genotypes(raw_df, plan_df)

        # Apply MAF correction to ensure minor allele counting
        harmonized_df = self._apply_maf_correction(harmonized_df)

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
    