"""
New variant harmonization engine based on proven old version logic.

This implementation fixes issues with the current harmonization engine by:
1. Pre-computing all variant representations (exact, swap, flip, flip_swap)
2. Using exact variant matching instead of position-based lookup with quality scoring
3. Preserving all valid matches instead of selecting "best" matches
4. Providing deterministic, reproducible results
"""

import os
import pandas as pd
from typing import List, Dict, Any
import logging

from ..models.harmonization import (
    HarmonizationRecord, 
    HarmonizationAction
)
from ..core.config import Settings
from .cache import AlleleHarmonizer

logger = logging.getLogger(__name__)


class HarmonizationEngine:
    """Improved harmonization engine based on proven old version logic."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.harmonizer = AlleleHarmonizer()
        
        # Complement map for strand flipping
        self.complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    def read_pvar_file(self, pgen_path: str) -> pd.DataFrame:
        """
        Read PVAR file for a PLINK file.
        
        Args:
            pgen_path: Path to .pgen file
            
        Returns:
            DataFrame with PVAR data
        """
        pvar_path = pgen_path.replace('.pgen', '.pvar')
        
        if not os.path.exists(pvar_path):
            raise FileNotFoundError(f"PVAR file not found: {pvar_path}")
        
        try:
            # Read PVAR file (tab-separated)
            df = pd.read_csv(pvar_path, sep='\t', low_memory=False, dtype=str)
            
            # Handle column names (remove # prefix if present)
            df.columns = [col.lstrip('#') for col in df.columns]
            
            # Standardize column names
            expected_cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT']
            if not all(col in df.columns for col in expected_cols):
                raise ValueError(f"PVAR file missing required columns: {expected_cols}. Found: {list(df.columns)}")
            
            # Clean and standardize data
            df['CHROM'] = df['CHROM'].astype(str).str.strip().str.replace('chr', '').str.upper()
            df['POS'] = pd.to_numeric(df['POS'], errors='coerce')
            df['ID'] = df['ID'].astype(str).str.strip()
            df['REF'] = df['REF'].astype(str).str.strip().str.upper()
            df['ALT'] = df['ALT'].astype(str).str.strip().str.upper()
            
            # Remove invalid rows
            df = df.dropna(subset=['CHROM', 'POS', 'REF', 'ALT'])
            
            logger.info(f"Read {len(df)} variants from {pvar_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read PVAR file {pvar_path}: {e}")
            raise
    
    def _create_variant_lookup(self, snp_list: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Create comprehensive variant lookup with all possible representations.
        
        This is the key improvement from the old version - we pre-compute all 
        possible variant orientations to enable exact matching.
        
        Args:
            snp_list: SNP list DataFrame with chromosome, position, ref, alt columns
            
        Returns:
            Dictionary mapping exact variant IDs to variant info and match type
        """
        variant_lookup = {}
        
        logger.info(f"Creating variant lookup for {len(snp_list)} SNP list variants")
        
        for _, snp_row in snp_list.iterrows():
            chrom = str(snp_row['chromosome']).strip()
            pos = str(snp_row['position']).strip()
            ref = str(snp_row['ref']).strip().upper()
            alt = str(snp_row['alt']).strip().upper()
            
            # Create base variant info
            base_info = {
                'snp_row': snp_row,
                'snp_list_id': snp_row.get('variant_id', f"{chrom}:{pos}:{ref}:{alt}"),
                'chromosome': chrom,
                'position': int(pos),
                'snp_list_a1': ref,
                'snp_list_a2': alt
            }
            
            # 1. Exact match: REF:ALT -> REF:ALT
            exact_id = f"{chrom}:{pos}:{ref}:{alt}"
            variant_lookup[exact_id] = {
                **base_info,
                'match_type': 'EXACT',
                'harmonization_action': HarmonizationAction.EXACT,
                'genotype_transform': None
            }
            
            # 2. Swap match: REF:ALT -> ALT:REF (alleles swapped)
            swap_id = f"{chrom}:{pos}:{alt}:{ref}"
            variant_lookup[swap_id] = {
                **base_info,
                'match_type': 'SWAP', 
                'harmonization_action': HarmonizationAction.SWAP,
                'genotype_transform': '2-x'  # 0->2, 1->1, 2->0
            }
            
            # 3. Flip match: REF:ALT -> FLIP(REF):FLIP(ALT) (strand flipped)
            ref_flip = self.complement_map.get(ref, ref)
            alt_flip = self.complement_map.get(alt, alt)
            
            flip_id = f"{chrom}:{pos}:{ref_flip}:{alt_flip}"
            variant_lookup[flip_id] = {
                **base_info,
                'match_type': 'FLIP',
                'harmonization_action': HarmonizationAction.FLIP, 
                'genotype_transform': None  # Same genotypes, just flipped strand
            }
            
            # 4. Flip+Swap match: REF:ALT -> FLIP(ALT):FLIP(REF) (both operations)
            flip_swap_id = f"{chrom}:{pos}:{alt_flip}:{ref_flip}"
            variant_lookup[flip_swap_id] = {
                **base_info,
                'match_type': 'FLIP_SWAP',
                'harmonization_action': HarmonizationAction.FLIP_SWAP,
                'genotype_transform': '2-x'  # Both strand flip and allele swap
            }
        
        logger.info(f"Created variant lookup with {len(variant_lookup)} total variant representations")
        return variant_lookup
    
    def _create_pvar_variants(self, pvar_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create standardized variant representations from PVAR data.
        
        Args:
            pvar_df: PVAR DataFrame
            
        Returns:
            List of variant dictionaries with exact variant IDs
        """
        pvar_variants = []
        
        for _, pvar_row in pvar_df.iterrows():
            chrom = str(pvar_row['CHROM']).strip()
            pos = str(pvar_row['POS']).strip()
            ref = str(pvar_row['REF']).strip().upper()
            alt = str(pvar_row['ALT']).strip().upper()
            
            # Create exact variant ID for matching
            exact_variant_id = f"{chrom}:{pos}:{ref}:{alt}"
            
            pvar_variants.append({
                'exact_variant_id': exact_variant_id,
                'pvar_row': pvar_row,
                'chromosome': chrom,
                'position': int(pos),
                'pgen_variant_id': str(pvar_row['ID']).strip(),
                'pgen_a1': ref,
                'pgen_a2': alt
            })
        
        return pvar_variants
    
    def harmonize_variants(
        self, 
        pvar_df: pd.DataFrame, 
        snp_list: pd.DataFrame
    ) -> List[HarmonizationRecord]:
        """
        Harmonize variants between PVAR and SNP list using exact matching.
        
        This is the main improvement: instead of position-based lookup with quality scoring,
        we use exact variant matching with all pre-computed representations.
        
        Args:
            pvar_df: DataFrame from PVAR file
            snp_list: Normalized SNP list DataFrame
            
        Returns:
            List of harmonization records for ALL valid matches (no filtering)
        """
        logger.info(f"Starting harmonization: {len(pvar_df)} PVAR variants vs {len(snp_list)} SNP list variants")
        
        # Step 1: Create comprehensive variant lookup (key improvement from old version)
        variant_lookup = self._create_variant_lookup(snp_list)
        
        # Step 2: Create PVAR variant representations
        pvar_variants = self._create_pvar_variants(pvar_df)
        
        # Step 3: Find exact matches (no quality scoring!)
        records = []
        matched_count = 0
        
        for pvar_variant in pvar_variants:
            exact_id = pvar_variant['exact_variant_id']
            
            # Check if this exact variant exists in our lookup
            if exact_id in variant_lookup:
                match_info = variant_lookup[exact_id]
                
                # Create harmonization record
                record = HarmonizationRecord(
                    snp_list_id=match_info['snp_list_id'],
                    pgen_variant_id=pvar_variant['pgen_variant_id'],
                    chromosome=match_info['chromosome'],
                    position=match_info['position'],
                    snp_list_a1=match_info['snp_list_a1'],
                    snp_list_a2=match_info['snp_list_a2'],
                    pgen_a1=pvar_variant['pgen_a1'],
                    pgen_a2=pvar_variant['pgen_a2'],
                    harmonization_action=match_info['harmonization_action'],
                    genotype_transform=match_info['genotype_transform'],
                    file_path="",  # Set by caller
                    data_type="",  # Set by caller  
                    ancestry=None  # Set by caller
                )
                
                records.append(record)
                matched_count += 1
                
                # Log detailed match info for debugging
                logger.debug(f"Matched: {exact_id} -> {match_info['match_type']} -> {match_info['snp_list_id']}")
        
        logger.info(f"Harmonization complete: {matched_count} matches found")
        
        # Log summary by harmonization action
        if records:
            action_counts = {}
            for record in records:
                action = record.harmonization_action.value
                action_counts[action] = action_counts.get(action, 0) + 1
            
            logger.info("Harmonization breakdown:")
            for action, count in action_counts.items():
                logger.info(f"  {action}: {count} variants")
        
        return records