"""
Result combination stage of the pipeline.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from .base import PipelineStage, PipelineContext
from ..storage.repository import StorageRepository
from ..models.carrier import CarrierData


class CombinerStage(PipelineStage[List[CarrierData], CarrierData]):
    """
    Combines carrier data across multiple sources (chromosomes or ancestries).
    
    This stage handles:
    - Chromosome combination for imputed data
    - Ancestry combination for population-level analysis
    """
    
    def __init__(self, storage: StorageRepository, key_file_path: Optional[str] = None):
        super().__init__("Combiner")
        self.storage = storage
        self.key_file_path = key_file_path
    
    async def process(self, carrier_data_list: List[CarrierData], context: PipelineContext) -> CarrierData:
        """
        Combine multiple carrier data objects into one.
        
        Args:
            carrier_data_list: List of CarrierData objects to combine
            context: Pipeline context
            
        Returns:
            Combined CarrierData object
        """
        if not carrier_data_list:
            raise ValueError("No carrier data to combine")
        
        if len(carrier_data_list) == 1:
            # Nothing to combine
            return carrier_data_list[0]
        
        combine_type = context.metadata.get('combine_type', 'chromosome')
        
        if combine_type == 'chromosome':
            return await self._combine_chromosomes(carrier_data_list, context)
        elif combine_type == 'ancestry':
            return await self._combine_ancestries(carrier_data_list, context)
        else:
            raise ValueError(f"Unknown combine type: {combine_type}")
    
    async def _combine_chromosomes(self, carrier_data_list: List[CarrierData], 
                                  context: PipelineContext) -> CarrierData:
        """Combine carrier data across chromosomes."""
        self.logger.info(f"Combining {len(carrier_data_list)} chromosomes")
        
        # Combine var_info
        var_info_dfs = []
        for cd in carrier_data_list:
            if cd.var_info is not None and not cd.var_info.empty:
                var_info_dfs.append(cd.var_info)
        
        combined_var_info = pd.concat(var_info_dfs, ignore_index=True) if var_info_dfs else pd.DataFrame()
        
        # Combine carriers_string
        carriers_string_dfs = []
        for cd in carrier_data_list:
            if cd.carriers_string is not None and not cd.carriers_string.empty:
                carriers_string_dfs.append(cd.carriers_string)
        
        combined_carriers_string = await self._merge_carrier_dfs(carriers_string_dfs, 'IID')
        
        # Combine carriers_int
        carriers_int_dfs = []
        for cd in carrier_data_list:
            if cd.carriers_int is not None and not cd.carriers_int.empty:
                carriers_int_dfs.append(cd.carriers_int)
        
        combined_carriers_int = await self._merge_carrier_dfs(carriers_int_dfs, 'IID')
        
        # Save combined files
        out_path = context.output_path
        var_info_path = f"{out_path}_var_info.parquet"
        carriers_string_path = f"{out_path}_carriers_string.parquet"
        carriers_int_path = f"{out_path}_carriers_int.parquet"
        
        await self.storage.write_parquet(combined_var_info, var_info_path)
        await self.storage.write_parquet(combined_carriers_string, carriers_string_path)
        await self.storage.write_parquet(combined_carriers_int, carriers_int_path)
        
        return CarrierData(
            var_info_path=var_info_path,
            carriers_string_path=carriers_string_path,
            carriers_int_path=carriers_int_path,
            var_info=combined_var_info,
            carriers_string=combined_carriers_string,
            carriers_int=combined_carriers_int
        )
    
    async def _combine_ancestries(self, carrier_data_list: List[CarrierData], 
                                 context: PipelineContext) -> CarrierData:
        """Combine carrier data across ancestries."""
        self.logger.info(f"Combining {len(carrier_data_list)} ancestries")
        
        # Get ancestry labels from context
        ancestry_labels = context.metadata.get('ancestry_labels', [])
        
        # Load master key if provided
        key_df = None
        if self.key_file_path:
            key_df = await self.storage.read_csv(self.key_file_path)
        
        # Combine var_info with ancestry-specific columns
        combined_var_info = await self._combine_ancestry_var_info(carrier_data_list, ancestry_labels)
        
        # Combine carriers_string
        carriers_string_dfs = []
        for cd in carrier_data_list:
            if cd.carriers_string is not None and not cd.carriers_string.empty:
                carriers_string_dfs.append(cd.carriers_string)
        
        combined_carriers_string = await self._merge_carrier_dfs(carriers_string_dfs, 'IID')
        
        # Combine carriers_int
        carriers_int_dfs = []
        for cd in carrier_data_list:
            if cd.carriers_int is not None and not cd.carriers_int.empty:
                carriers_int_dfs.append(cd.carriers_int)
        
        combined_carriers_int = await self._merge_carrier_dfs(carriers_int_dfs, 'IID')
        
        # Add study information if key file provided
        if key_df is not None:
            combined_carriers_string = await self._add_study_info(combined_carriers_string, key_df)
            combined_carriers_int = await self._add_study_info(combined_carriers_int, key_df)
        
        # Save combined files
        out_path = context.output_path
        var_info_path = f"{out_path}_var_info.parquet"
        carriers_string_path = f"{out_path}_carriers_string.parquet"
        carriers_int_path = f"{out_path}_carriers_int.parquet"
        
        await self.storage.write_parquet(combined_var_info, var_info_path)
        await self.storage.write_parquet(combined_carriers_string, carriers_string_path)
        await self.storage.write_parquet(combined_carriers_int, carriers_int_path)
        
        return CarrierData(
            var_info_path=var_info_path,
            carriers_string_path=carriers_string_path,
            carriers_int_path=carriers_int_path,
            var_info=combined_var_info,
            carriers_string=combined_carriers_string,
            carriers_int=combined_carriers_int
        )
    
    async def _merge_carrier_dfs(self, dfs: List[pd.DataFrame], merge_col: str) -> pd.DataFrame:
        """Merge multiple carrier DataFrames on a common column."""
        if not dfs:
            return pd.DataFrame()
        
        result = dfs[0]
        for df in dfs[1:]:
            # Get variant columns (excluding merge column)
            variant_cols = [col for col in df.columns if col != merge_col]
            # Only keep non-overlapping columns
            new_cols = [col for col in variant_cols if col not in result.columns]
            if new_cols:
                result = result.merge(df[[merge_col] + new_cols], on=merge_col, how='outer')
        
        # Fill missing values
        variant_cols = [col for col in result.columns if col != merge_col]
        for col in variant_cols:
            if result[col].dtype == 'object':
                result[col] = result[col].fillna('./.')
            else:
                result[col] = result[col].fillna(2)  # Homozygous reference
        
        return result
    
    async def _combine_ancestry_var_info(self, carrier_data_list: List[CarrierData],
                                        ancestry_labels: List[str]) -> pd.DataFrame:
        """Combine variant info with ancestry-specific frequency columns."""
        # Start with first non-empty var_info
        base_var_info = None
        for cd in carrier_data_list:
            if cd.var_info is not None and not cd.var_info.empty:
                base_var_info = cd.var_info.copy()
                break
        
        if base_var_info is None:
            return pd.DataFrame()
        
        # Define columns to make ancestry-specific
        freq_cols = ['ALT_FREQS', 'OBS_CT', 'A1', 'A2', 'A1_FREQ', 'F_MISS']
        
        # Combine frequency data from all ancestries
        for i, (cd, label) in enumerate(zip(carrier_data_list, ancestry_labels)):
            if cd.var_info is None or cd.var_info.empty:
                continue
                
            # Add ancestry-specific columns
            for col in freq_cols:
                if col in cd.var_info.columns:
                    new_col = f"{col}_{label}"
                    base_var_info[new_col] = cd.var_info[col]
        
        # Clean up redundant columns
        columns_to_remove = ['chrom.1', 'CHR', '#CHROM', 'REF', 'ALT', 'COUNTED', 'PROVISIONAL_REF?']
        base_var_info.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        
        return base_var_info
    
    async def _add_study_info(self, carriers_df: pd.DataFrame, key_df: pd.DataFrame) -> pd.DataFrame:
        """Add study information to carrier DataFrame."""
        # Merge with key file to add study column
        result = carriers_df.merge(key_df[['IID', 'study']], on='IID', how='left')
        
        # Reorder columns to put study after IID
        variant_cols = [col for col in result.columns if col not in ['IID', 'study']]
        result = result[['IID', 'study'] + variant_cols]
        
        return result
