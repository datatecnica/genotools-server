"""
Allele matching logic for harmonization.
"""

import pandas as pd
from typing import Tuple, List, Dict, Set, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AlleleMatcher:
    """
    Handles matching of alleles between genotype files and reference variants.
    
    Supports:
    - Exact matches
    - Strand flips (complement matching)
    - Allele swaps (ref/alt swap)
    - Combination of flip and swap
    """
    
    # Complement mapping for strand flips
    COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    
    def __init__(self, chunk_size: int = 500000):
        self.chunk_size = chunk_size
    
    async def find_matches(self, pfile: str, reference: str, 
                         out_prefix: str) -> Tuple[List[str], pd.DataFrame]:
        """
        Find matching variants between genotype file and reference.
        
        Args:
            pfile: Path to PLINK file prefix
            reference: Path to reference variant file
            out_prefix: Output prefix for match information
            
        Returns:
            Tuple of (list of matching variant IDs, match information DataFrame)
        """
        pvar_path = f"{pfile}.pvar"
        
        # Read reference variants
        ref_df = pd.read_csv(reference, dtype={'chrom': str})
        ref_variants = self._prepare_reference_variants(ref_df)
        
        # Process genotype file in chunks
        matches = []
        
        for chunk in pd.read_csv(pvar_path, sep='\t', comment='#', 
                                chunksize=self.chunk_size,
                                dtype={'#CHROM': str, 'POS': int}):
            chunk_matches = self._find_chunk_matches(chunk, ref_variants)
            matches.extend(chunk_matches)
        
        # Convert matches to DataFrame
        if not matches:
            return [], pd.DataFrame()
        
        match_df = pd.DataFrame(matches)
        
        # Save match information
        match_info_path = f"{out_prefix}_match_info.tsv"
        match_df.to_csv(match_info_path, sep='\t', index=False)
        
        # Return matching IDs
        matching_ids = match_df['id_geno'].tolist()
        
        return matching_ids, match_df
    
    def _prepare_reference_variants(self, ref_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Prepare reference variants for efficient matching.
        
        Returns:
            Dictionary mapping position keys to variant information
        """
        ref_variants = {}
        
        # Parse hg38 format and create variant entries
        for _, row in ref_df.iterrows():
            hg38 = str(row['hg38']).strip()
            parts = hg38.split(':')
            
            if len(parts) != 4:
                continue
            
            chrom, pos, ref, alt = parts
            chrom = self._normalize_chrom(chrom)
            
            # Create all possible representations
            variants = self._create_variant_representations(ref.upper(), alt.upper())
            
            # Store under position key
            pos_key = f"{chrom}:{pos}"
            if pos_key not in ref_variants:
                ref_variants[pos_key] = []
            
            ref_variants[pos_key].append({
                'variant_id': hg38.upper(),
                'ref': ref.upper(),
                'alt': alt.upper(),
                'representations': variants,
                'original_row': row
            })
        
        return ref_variants
    
    def _find_chunk_matches(self, chunk: pd.DataFrame, 
                          ref_variants: Dict[str, List[Dict]]) -> List[Dict]:
        """Find matches for a chunk of genotype data."""
        matches = []
        
        for _, geno_row in chunk.iterrows():
            chrom = self._normalize_chrom(str(geno_row['#CHROM']))
            pos = str(geno_row['POS'])
            pos_key = f"{chrom}:{pos}"
            
            # Check if position exists in reference
            if pos_key not in ref_variants:
                continue
            
            # Get genotype alleles
            geno_ref = str(geno_row['REF']).upper()
            geno_alt = str(geno_row['ALT']).upper()
            geno_id = str(geno_row['ID'])
            
            # Check each reference variant at this position
            for ref_var in ref_variants[pos_key]:
                match_type = self._check_match(
                    geno_ref, geno_alt, 
                    ref_var['ref'], ref_var['alt'],
                    ref_var['representations']
                )
                
                if match_type:
                    matches.append({
                        'id_geno': geno_id,
                        'variant_id_geno': f"{chrom}:{pos}:{geno_ref}:{geno_alt}",
                        'variant_id_ref': ref_var['variant_id'],
                        'match_type': match_type,
                        'swap_alleles': 'swap' in match_type,
                        'flip_strands': 'flip' in match_type,
                        'ref_allele': ref_var['ref'],
                        'alt_allele': ref_var['alt']
                    })
                    break  # Only take first match
        
        return matches
    
    def _create_variant_representations(self, ref: str, alt: str) -> Set[Tuple[str, str]]:
        """Create all possible representations of a variant."""
        representations = set()
        
        # Original
        representations.add((ref, alt))
        
        # Swapped
        representations.add((alt, ref))
        
        # Flipped
        ref_flip = self._flip_allele(ref)
        alt_flip = self._flip_allele(alt)
        representations.add((ref_flip, alt_flip))
        
        # Flipped and swapped
        representations.add((alt_flip, ref_flip))
        
        return representations
    
    def _check_match(self, geno_ref: str, geno_alt: str,
                    ref_ref: str, ref_alt: str,
                    representations: Set[Tuple[str, str]]) -> Optional[str]:
        """
        Check if genotype alleles match reference alleles.
        
        Returns:
            Match type string or None if no match
        """
        geno_pair = (geno_ref, geno_alt)
        
        # Check exact match
        if geno_pair == (ref_ref, ref_alt):
            return "exact"
        
        # Check swapped
        if geno_pair == (ref_alt, ref_ref):
            return "swap"
        
        # Check flipped
        ref_flip = self._flip_allele(ref_ref)
        alt_flip = self._flip_allele(ref_alt)
        if geno_pair == (ref_flip, alt_flip):
            return "flip"
        
        # Check flipped and swapped
        if geno_pair == (alt_flip, ref_flip):
            return "flip_swap"
        
        return None
    
    def _flip_allele(self, allele: str) -> str:
        """Get complement of an allele (strand flip)."""
        return ''.join(self.COMPLEMENT.get(base, base) for base in allele)
    
    def _normalize_chrom(self, chrom: str) -> str:
        """Normalize chromosome format."""
        # Remove 'chr' prefix if present
        if chrom.lower().startswith('chr'):
            chrom = chrom[3:]
        
        # Handle special cases
        if chrom == 'MT':
            chrom = 'M'
        
        return chrom
