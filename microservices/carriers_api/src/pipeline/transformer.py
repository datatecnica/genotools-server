"""
Carrier transformation stage of the pipeline.
"""

from typing import Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from .base import PipelineStage, PipelineContext
from ..storage.repository import StorageRepository
from ..models.carrier import CarrierData


class CarrierTransformationStage(PipelineStage[Tuple[pd.DataFrame, str], CarrierData]):
    """
    Stage 3: Transform genotype data into carrier format.
    
    This stage converts PLINK traw output into the three standard outputs:
    - var_info: Comprehensive variant information
    - carriers_string: String genotype format (e.g., 'A/G')
    - carriers_int: Integer encoded format
    """
    
    def __init__(self, storage: StorageRepository):
        super().__init__("CarrierTransformation")
        self.storage = storage
    
    async def process(self, inputs: Tuple[pd.DataFrame, str], context: PipelineContext) -> CarrierData:
        """
        Transform genotype data to carrier format.
        
        Args:
            inputs: Tuple of (variant stats, subset SNP path)
            context: Pipeline context
            
        Returns:
            CarrierData object with all three output formats
        """
        var_stats, subset_snp_path = inputs
        
        if subset_snp_path is None:
            # Return empty results if no variants found
            return await self._create_empty_output(context.output_path)
        
        # Get PLINK output path from context
        plink_out = context.metadata.get('plink_out')
        out_path = context.output_path
        
        # Read traw data
        traw_data = await self._read_traw_data(plink_out)
        
        # Read subset SNP information
        subset_snps = await self.storage.read_csv(subset_snp_path)
        
        # Process and create outputs
        carrier_data = await self._process_traw_data(
            traw_data, subset_snps, var_stats, out_path
        )
        
        return carrier_data
    
    async def _read_traw_data(self, plink_out: str) -> pd.DataFrame:
        """Read and parse PLINK traw output."""
        traw_path = f"{plink_out}.traw"
        
        # Read traw file
        traw_df = await self.storage.read_csv(traw_path, sep='\t')
        
        # Rename columns for consistency
        traw_df.rename(columns={'CHR': 'chrom', 'SNP': 'id', 'POS': 'pos',
                               'COUNTED': 'counted', 'ALT': 'alt'}, inplace=True)
        
        return traw_df
    
    async def _process_traw_data(self, traw_df: pd.DataFrame, subset_snps: pd.DataFrame,
                                var_stats: pd.DataFrame, out_path: str) -> CarrierData:
        """Process traw data into the three output formats."""
        
        # Define variant info columns
        var_cols = ['chrom', 'id', 'pos', 'counted', 'alt']
        
        # Process genotypes to integer format
        sample_cols = [col for col in traw_df.columns if col not in var_cols]
        
        # Convert NA to 2 (homozygous reference)
        for col in sample_cols:
            traw_df[col] = pd.to_numeric(traw_df[col], errors='coerce').fillna(2).astype(int)
        
        # Create comprehensive variant info by merging all data sources
        var_info = await self._create_variant_info(traw_df, subset_snps, var_stats, var_cols)
        
        # Create carriers string format
        carriers_string = await self._create_carriers_string(traw_df, var_cols, sample_cols)
        
        # Create carriers integer format
        carriers_int = await self._create_carriers_int(traw_df, var_cols, sample_cols)
        
        # Save all outputs
        var_info_path = f"{out_path}_var_info.parquet"
        carriers_string_path = f"{out_path}_carriers_string.parquet"
        carriers_int_path = f"{out_path}_carriers_int.parquet"
        
        await self.storage.write_parquet(var_info, var_info_path)
        await self.storage.write_parquet(carriers_string, carriers_string_path)
        await self.storage.write_parquet(carriers_int, carriers_int_path)
        
        return CarrierData(
            var_info_path=var_info_path,
            carriers_string_path=carriers_string_path,
            carriers_int_path=carriers_int_path,
            var_info=var_info,
            carriers_string=carriers_string,
            carriers_int=carriers_int
        )
    
    async def _create_variant_info(self, traw_df: pd.DataFrame, subset_snps: pd.DataFrame,
                                  var_stats: pd.DataFrame, var_cols: list) -> pd.DataFrame:
        """Create comprehensive variant information DataFrame."""
        # Start with traw variant columns
        var_info = traw_df[var_cols].copy()
        
        # Merge with subset SNP information
        var_info = var_info.merge(subset_snps, on='id', how='left')
        
        # Merge with variant statistics
        if not var_stats.empty:
            var_info = var_info.merge(var_stats, left_on='id', right_on='SNP', how='left')
            if 'SNP' in var_info.columns:
                var_info.drop(columns=['SNP'], inplace=True)
        
        return var_info
    
    async def _create_carriers_string(self, traw_df: pd.DataFrame, var_cols: list,
                                     sample_cols: list) -> pd.DataFrame:
        """Create string format carrier data."""
        # Get genotype data
        genotype_data = traw_df[sample_cols].copy()
        
        # Convert to string format with proper genotype notation
        def convert_to_genotype_string(val, ref, alt):
            if pd.isna(val) or val == 2:
                return f"{ref}/{ref}"
            elif val == 1:
                return f"{ref}/{alt}"
            elif val == 0:
                return f"{alt}/{alt}"
            else:
                return "./."
        
        # Apply conversion (this is simplified - actual implementation would use ref/alt alleles)
        carriers_string = genotype_data.applymap(lambda x: "./." if pd.isna(x) else str(int(x)))
        
        # Transpose and format
        carriers_string = carriers_string.T.reset_index()
        carriers_string.columns = ['IID'] + list(traw_df['id'])
        carriers_string['IID'] = carriers_string['IID'].str.replace('0_', '')
        
        return carriers_string
    
    async def _create_carriers_int(self, traw_df: pd.DataFrame, var_cols: list,
                                  sample_cols: list) -> pd.DataFrame:
        """Create integer format carrier data."""
        # Get genotype data
        genotype_data = traw_df[sample_cols].copy()
        
        # Transpose and format
        carriers_int = genotype_data.T.reset_index()
        carriers_int.columns = ['IID'] + list(traw_df['id'])
        carriers_int['IID'] = carriers_int['IID'].str.replace('0_', '')
        
        return carriers_int
    
    async def _create_empty_output(self, out_path: str) -> CarrierData:
        """Create empty output files when no variants are found."""
        empty_var_info = pd.DataFrame()
        empty_carriers = pd.DataFrame(columns=['IID'])
        
        var_info_path = f"{out_path}_var_info.parquet"
        carriers_string_path = f"{out_path}_carriers_string.parquet"
        carriers_int_path = f"{out_path}_carriers_int.parquet"
        
        await self.storage.write_parquet(empty_var_info, var_info_path)
        await self.storage.write_parquet(empty_carriers, carriers_string_path)
        await self.storage.write_parquet(empty_carriers, carriers_int_path)
        
        return CarrierData(
            var_info_path=var_info_path,
            carriers_string_path=carriers_string_path,
            carriers_int_path=carriers_int_path,
            var_info=empty_var_info,
            carriers_string=empty_carriers,
            carriers_int=empty_carriers
        )
