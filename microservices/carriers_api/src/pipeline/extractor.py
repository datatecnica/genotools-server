"""
Variant extraction stage of the pipeline.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
from pathlib import Path

from .base import PipelineStage, PipelineContext
from ..harmonization.harmonizer import HarmonizationService
from ..storage.repository import StorageRepository


class VariantExtractionStage(PipelineStage[Dict[str, str], Tuple[pd.DataFrame, str]]):
    """
    Stage 1: Extract variants from genotype files using SNP list.
    
    This stage handles the initial extraction of variants from PLINK files,
    using harmonization to ensure correct allele matching.
    """
    
    def __init__(self, harmonizer: HarmonizationService, storage: StorageRepository):
        super().__init__("VariantExtraction")
        self.harmonizer = harmonizer
        self.storage = storage
    
    async def process(self, inputs: Dict[str, str], context: PipelineContext) -> Tuple[pd.DataFrame, str]:
        """
        Extract variants from genotype files.
        
        Args:
            inputs: Dictionary with 'geno_path' and 'snplist_path'
            context: Pipeline context
            
        Returns:
            Tuple of (variant statistics DataFrame, subset SNP file path)
        """
        geno_path = inputs['geno_path']
        snplist_path = inputs['snplist_path']
        
        # Generate output prefix for PLINK files
        output_dir = Path(context.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        plink_out = str(output_dir / f"{Path(context.output_path).stem}_snps")
        
        # Check if we have cached harmonization data
        cache_key = f"{context.dataset_type}_{context.release}_{Path(geno_path).stem}"
        cached_mapping = await self.harmonizer.get_cached_mapping(cache_key)
        
        if cached_mapping is not None:
            self.logger.info(f"Using cached harmonization mapping for {cache_key}")
            # Use cached mapping to speed up extraction
            subset_snp_path = await self.harmonizer.extract_with_cache(
                geno_path, snplist_path, plink_out, cached_mapping
            )
        else:
            self.logger.info(f"No cached mapping found, performing full harmonization")
            # Perform full harmonization and extraction
            subset_snp_path = await self.harmonizer.harmonize_and_extract(
                geno_path, snplist_path, plink_out
            )
            
            # Cache the mapping for future use
            if subset_snp_path:
                await self.harmonizer.cache_mapping(cache_key, subset_snp_path)
        
        # Handle case where no variants were found
        if subset_snp_path is None:
            self.logger.warning(f"No target variants found in {geno_path}")
            return pd.DataFrame(), None
        
        # Read variant statistics
        var_stats = await self._read_variant_stats(plink_out)
        
        # Store in context for later stages
        context.metadata['plink_out'] = plink_out
        context.metadata['subset_snp_path'] = subset_snp_path
        
        return var_stats, subset_snp_path
    
    async def _read_variant_stats(self, plink_out: str) -> pd.DataFrame:
        """Read frequency and missingness data from PLINK output."""
        var_stats = pd.DataFrame()
        
        try:
            # Read frequency data
            freq = await self.storage.read_csv(f"{plink_out}.afreq", sep='\t')
            freq.rename(columns={'ID': 'SNP'}, inplace=True)
            var_stats = freq
            
            # Read variant missingness data
            try:
                vmiss = await self.storage.read_csv(f"{plink_out}.vmiss", sep='\t')
                vmiss.rename(columns={'ID': 'SNP'}, inplace=True)
                
                # Merge frequency and missingness data
                var_stats = freq.merge(vmiss[['SNP', 'F_MISS']], on='SNP', how='left')
            except Exception:
                self.logger.warning(f"Missing data file {plink_out}.vmiss not found")
                
        except Exception:
            self.logger.error(f"Error reading variant statistics from {plink_out}")
            # Return empty DataFrame with expected column
            var_stats = pd.DataFrame(columns=['SNP'])
        
        return var_stats
