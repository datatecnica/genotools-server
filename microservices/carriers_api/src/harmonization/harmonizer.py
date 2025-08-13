"""
Harmonization service with caching support.
"""

import tempfile
import shutil
from typing import Optional, List, Tuple, Dict
import pandas as pd
from pathlib import Path
import logging

from .cache import HarmonizationCache
from .matcher import AlleleMatcher
from ..storage.repository import StorageRepository
from ..models.carrier import HarmonizationMapping
from ..core.plink_operations import (
    ExtractSnpsCommand, UpdateAllelesCommand, SwapAllelesCommand,
    ExportCommand, FrequencyCommand
)

logger = logging.getLogger(__name__)


class HarmonizationService:
    """
    Service for harmonizing genotype data with reference alleles.
    
    Includes caching support to speed up repeated harmonization operations.
    """
    
    def __init__(self, storage: StorageRepository, cache: HarmonizationCache):
        self.storage = storage
        self.cache = cache
        self.matcher = AlleleMatcher()
    
    async def get_cached_mapping(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get cached harmonization mapping."""
        return await self.cache.get_mapping(cache_key)
    
    async def cache_mapping(self, cache_key: str, subset_snp_path: str):
        """Cache harmonization mapping from subset SNP file."""
        # This would extract the mapping information from the subset file
        # and store it in the cache
        pass
    
    async def harmonize_and_extract(self, geno_path: str, reference_path: str, 
                                  plink_out: str, use_cache: bool = True) -> Optional[str]:
        """
        Harmonize alleles and extract variants.
        
        Args:
            geno_path: Path to genotype PLINK files
            reference_path: Path to reference SNP list
            plink_out: Output path for PLINK files
            use_cache: Whether to use cached mappings
            
        Returns:
            Path to subset SNP file or None if no variants found
        """
        tmpdir = tempfile.mkdtemp()
        
        try:
            # Step 1: Find common SNPs
            common_snps_path, match_info_path = await self._find_common_snps(
                geno_path, reference_path, tmpdir
            )
            
            if not common_snps_path:
                return None
            
            # Step 2: Extract common SNPs
            extracted_prefix = await self._extract_snps(
                geno_path, common_snps_path, tmpdir
            )
            
            if not extracted_prefix:
                return None
            
            # Step 3: Harmonize alleles
            harmonized_prefix = await self._harmonize_alleles(
                extracted_prefix, reference_path, tmpdir, match_info_path
            )
            
            # Step 4: Export final data
            await self._export_plink(harmonized_prefix, plink_out)
            
            # Step 5: Create subset SNP file
            subset_snp_path = await self._create_subset_snp_file(
                plink_out, match_info_path, reference_path
            )
            
            return subset_snp_path
            
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    async def extract_with_cache(self, geno_path: str, reference_path: str,
                               plink_out: str, cached_mapping: pd.DataFrame) -> Optional[str]:
        """
        Extract variants using cached harmonization mapping.
        
        This is faster than full harmonization as it skips the matching step.
        """
        tmpdir = tempfile.mkdtemp()
        
        try:
            # Use cached mapping to create extraction list
            variant_ids = cached_mapping['variant_id_geno'].unique()
            extract_list_path = Path(tmpdir) / "cached_extract.txt"
            
            # Write variant IDs to file
            with open(extract_list_path, 'w') as f:
                for vid in variant_ids:
                    f.write(f"{vid}\n")
            
            # Extract variants
            extracted_prefix = await self._extract_snps(
                geno_path, str(extract_list_path), tmpdir
            )
            
            if not extracted_prefix:
                return None
            
            # Apply harmonization operations from cache
            harmonized_prefix = await self._apply_cached_harmonization(
                extracted_prefix, cached_mapping, tmpdir
            )
            
            # Export final data
            await self._export_plink(harmonized_prefix, plink_out)
            
            # Create subset SNP file from cache
            subset_snp_path = await self._create_subset_from_cache(
                plink_out, cached_mapping, reference_path
            )
            
            return subset_snp_path
            
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    async def _find_common_snps(self, pfile: str, reference: str, 
                              tmpdir: str) -> Tuple[Optional[str], Optional[str]]:
        """Find SNPs common between genotype and reference files."""
        common_prefix = Path(tmpdir) / "common_snps"
        
        # Use the matcher to find common variants
        common_ids, match_info = await self.matcher.find_matches(
            pfile, reference, str(common_prefix)
        )
        
        if not common_ids:
            logger.warning(f"No common SNPs found between {pfile} and {reference}")
            return None, None
        
        # Write common IDs to file
        common_ids_path = f"{common_prefix}_ids.txt"
        with open(common_ids_path, 'w') as f:
            for vid in common_ids:
                f.write(f"{vid}\n")
        
        # Write match info
        match_info_path = f"{common_prefix}_match_info.tsv"
        match_info.to_csv(match_info_path, sep='\t', index=False)
        
        return common_ids_path, match_info_path
    
    async def _extract_snps(self, geno_path: str, snp_list: str, 
                          tmpdir: str) -> Optional[str]:
        """Extract SNPs from genotype file."""
        extracted_prefix = str(Path(tmpdir) / "extracted")
        
        extract_cmd = ExtractSnpsCommand(
            geno_path, snp_list, extracted_prefix, output_chr='M'
        )
        
        try:
            extract_cmd.execute()
            return extracted_prefix
        except ValueError as e:
            if "No variants found after extraction" in str(e):
                logger.warning(f"No variants extracted from {geno_path}")
                return None
            raise
    
    async def _harmonize_alleles(self, pfile: str, reference: str,
                                tmpdir: str, match_info_path: str) -> str:
        """Harmonize alleles based on match information."""
        # Read match info
        match_info = pd.read_csv(match_info_path, sep='\t')
        
        harmonized_prefix = str(Path(tmpdir) / "harmonized")
        current_prefix = pfile
        
        # Apply necessary operations
        if 'swap_alleles' in match_info.columns:
            swap_needed = match_info[match_info['swap_alleles']]
            if not swap_needed.empty:
                # Create swap file
                swap_file = str(Path(tmpdir) / "swap_alleles.txt")
                swap_needed[['id_geno']].to_csv(swap_file, index=False, header=False)
                
                # Apply swap
                swap_cmd = SwapAllelesCommand(current_prefix, swap_file, harmonized_prefix)
                swap_cmd.execute()
                current_prefix = harmonized_prefix
        
        # Copy final result if needed
        if current_prefix != harmonized_prefix:
            import shutil
            for ext in ['.pgen', '.pvar', '.psam']:
                shutil.copy2(f"{current_prefix}{ext}", f"{harmonized_prefix}{ext}")
        
        return harmonized_prefix
    
    async def _export_plink(self, pfile: str, out: str):
        """Export PLINK files with additional statistics."""
        # Export main files
        export_cmd = ExportCommand(pfile=pfile, out=out)
        export_cmd.execute()
        
        # Calculate frequencies
        freq_cmd = FrequencyCommand(pfile=pfile, out=out)
        freq_cmd.execute()
    
    async def _create_subset_snp_file(self, plink_out: str, match_info_path: str,
                                    reference_path: str) -> str:
        """Create subset SNP file with harmonization results."""
        # Read match info
        match_info = pd.read_csv(match_info_path, sep='\t')
        
        # Read reference data
        ref_df = await self.storage.read_csv(reference_path, dtype={'chrom': str})
        
        # Process reference data
        ref_df['variant_id'] = ref_df['hg38'].str.upper()
        
        # Merge with match info
        subset_snps = match_info.merge(
            ref_df, left_on='variant_id_ref', right_on='variant_id', how='inner'
        )
        
        # Rename columns for consistency
        subset_snps = subset_snps.rename(columns={'id_geno': 'id'})
        
        # Save subset SNP file
        subset_snp_path = f"{plink_out}_subset_snps.csv"
        await self.storage.write_csv(subset_snps, subset_snp_path)
        
        return subset_snp_path
    
    async def _apply_cached_harmonization(self, pfile: str, cached_mapping: pd.DataFrame,
                                        tmpdir: str) -> str:
        """Apply harmonization operations from cached mapping."""
        harmonized_prefix = str(Path(tmpdir) / "harmonized")
        current_prefix = pfile
        
        # Apply swap operations if needed
        swap_needed = cached_mapping[cached_mapping['swap_alleles'] == True]
        if not swap_needed.empty:
            swap_file = str(Path(tmpdir) / "swap_alleles.txt")
            swap_needed[['variant_id_geno']].to_csv(swap_file, index=False, header=False)
            
            swap_cmd = SwapAllelesCommand(current_prefix, swap_file, harmonized_prefix)
            swap_cmd.execute()
            current_prefix = harmonized_prefix
        
        # Copy final result if needed
        if current_prefix != harmonized_prefix:
            import shutil
            for ext in ['.pgen', '.pvar', '.psam']:
                shutil.copy2(f"{current_prefix}{ext}", f"{harmonized_prefix}{ext}")
        
        return harmonized_prefix
    
    async def _create_subset_from_cache(self, plink_out: str, cached_mapping: pd.DataFrame,
                                      reference_path: str) -> str:
        """Create subset SNP file from cached mapping."""
        # Read reference data
        ref_df = await self.storage.read_csv(reference_path, dtype={'chrom': str})
        ref_df['variant_id'] = ref_df['hg38'].str.upper()
        
        # Merge with cached mapping
        subset_snps = cached_mapping.merge(
            ref_df, left_on='variant_id_ref', right_on='variant_id', how='inner'
        )
        
        # Rename for consistency
        subset_snps = subset_snps.rename(columns={'variant_id_geno': 'id'})
        
        # Save subset SNP file
        subset_snp_path = f"{plink_out}_subset_snps.csv"
        await self.storage.write_csv(subset_snps, subset_snp_path)
        
        return subset_snp_path
