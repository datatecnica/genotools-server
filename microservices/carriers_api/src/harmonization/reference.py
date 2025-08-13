"""
Reference allele management.
"""

import pandas as pd
from typing import List, Dict, Optional, Set
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReferenceManager:
    """
    Manages reference allele information and SNP lists.
    """
    
    def __init__(self, storage):
        self.storage = storage
        self._reference_cache: Dict[str, pd.DataFrame] = {}
    
    async def load_snplist(self, snplist_path: str) -> pd.DataFrame:
        """
        Load and validate SNP list file.
        
        Expected columns:
        - hg38: Variant in chr:pos:ref:alt format
        - locus: Gene/locus name
        - snp_name: Variant name
        - rsid: dbSNP ID (optional)
        - hg19: hg19 coordinates (optional)
        """
        if snplist_path in self._reference_cache:
            return self._reference_cache[snplist_path].copy()
        
        # Load SNP list
        snp_df = await self.storage.read_csv(snplist_path, dtype={'chrom': str})
        
        # Validate required columns
        required_cols = ['hg38']
        missing_cols = [col for col in required_cols if col not in snp_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in SNP list: {missing_cols}")
        
        # Process hg38 column
        snp_df['hg38'] = snp_df['hg38'].astype(str).str.strip()
        
        # Parse hg38 to create additional columns if not present
        if 'chrom' not in snp_df.columns or 'pos' not in snp_df.columns:
            parsed = snp_df['hg38'].str.split(':', expand=True)
            if len(parsed.columns) >= 4:
                snp_df['chrom'] = parsed[0]
                snp_df['pos'] = parsed[1]
                snp_df['ref'] = parsed[2].str.upper()
                snp_df['alt'] = parsed[3].str.upper()
        
        # Create variant ID
        snp_df['variant_id'] = snp_df['hg38'].str.upper()
        
        # Cache for reuse
        self._reference_cache[snplist_path] = snp_df
        
        return snp_df
    
    async def filter_by_locus(self, snp_df: pd.DataFrame, locus: str) -> pd.DataFrame:
        """Filter SNP list by locus."""
        if 'locus' not in snp_df.columns:
            logger.warning("No locus column in SNP list, returning all variants")
            return snp_df
        
        return snp_df[snp_df['locus'] == locus].copy()
    
    async def filter_by_variants(self, snp_df: pd.DataFrame, 
                               variant_ids: List[str]) -> pd.DataFrame:
        """Filter SNP list by specific variant IDs."""
        variant_set = set(v.upper() for v in variant_ids)
        return snp_df[snp_df['variant_id'].isin(variant_set)].copy()
    
    def get_variant_to_locus_mapping(self, snp_df: pd.DataFrame) -> Dict[str, str]:
        """
        Create mapping from variant ID to locus.
        
        Returns:
            Dictionary mapping variant_id to locus name
        """
        if 'locus' not in snp_df.columns:
            return {}
        
        return dict(zip(snp_df['variant_id'], snp_df['locus']))
    
    def get_locus_list(self, snp_df: pd.DataFrame) -> List[str]:
        """Get unique locus names from SNP list."""
        if 'locus' not in snp_df.columns:
            return []
        
        return snp_df['locus'].dropna().unique().tolist()
    
    async def validate_reference_alleles(self, snp_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate reference allele format and return any issues.
        
        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'invalid_format': [],
            'invalid_alleles': [],
            'duplicate_variants': []
        }
        
        # Check for proper hg38 format
        for idx, row in snp_df.iterrows():
            hg38 = str(row['hg38'])
            parts = hg38.split(':')
            
            if len(parts) != 4:
                issues['invalid_format'].append(f"Row {idx}: {hg38}")
                continue
            
            # Check alleles are valid nucleotides
            ref, alt = parts[2].upper(), parts[3].upper()
            valid_bases = {'A', 'T', 'G', 'C'}
            
            if not all(b in valid_bases for b in ref):
                issues['invalid_alleles'].append(f"Row {idx}: Invalid ref allele {ref}")
            
            if not all(b in valid_bases for b in alt):
                issues['invalid_alleles'].append(f"Row {idx}: Invalid alt allele {alt}")
        
        # Check for duplicates
        dup_mask = snp_df['variant_id'].duplicated(keep=False)
        if dup_mask.any():
            dup_variants = snp_df[dup_mask]['variant_id'].unique()
            issues['duplicate_variants'] = dup_variants.tolist()
        
        return issues
    
    def create_reference_subset(self, snp_df: pd.DataFrame, 
                              chromosome: Optional[str] = None) -> pd.DataFrame:
        """Create subset of reference for specific chromosome."""
        if chromosome is None:
            return snp_df.copy()
        
        # Normalize chromosome format
        chrom_norm = str(chromosome)
        if chrom_norm.lower().startswith('chr'):
            chrom_norm = chrom_norm[3:]
        
        # Filter by chromosome
        return snp_df[snp_df['chrom'] == chrom_norm].copy()
