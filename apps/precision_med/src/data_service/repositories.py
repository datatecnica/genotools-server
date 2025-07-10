"""
Repository layer for NBA and WGS carriers data access.
Optimized repository pattern with Parquet support and native filtering for performance.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

from src.core.config import DataConfig
from src.core.exceptions import DataAccessError, FileNotFoundError as CustomFileNotFoundError
from src.data_service.models import (
    NBADataset, NBAInfoRecord, NBAIntRecord, NBAStringRecord,
    WGSDataset, WGSInfoRecord, WGSIntRecord, WGSStringRecord
)


class BaseRepository(ABC):
    """Abstract base repository for data access operations."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._data_cache: Dict[str, Any] = {}
    
    @abstractmethod
    def load_data(self) -> Any:
        """Load data from source."""
        pass
    
    def clear_cache(self) -> None:
        """Clear cached data."""
        self._data_cache.clear()
    
    def _clean_dataframe_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame to avoid PyArrow serialization issues."""
        df_clean = df.copy()
        
        # Convert object columns with mixed types to string
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    # Try to keep numeric if possible, otherwise convert to string
                    pd.to_numeric(df_clean[col], errors='raise')
                except (ValueError, TypeError):
                    # Has mixed types, convert to string
                    df_clean[col] = df_clean[col].astype(str)
        
        return df_clean
    
    def _load_parquet_with_filtering(self, file_path: Path, 
                                    sample_ids: Optional[List[str]] = None,
                                    snp_names: Optional[List[str]] = None,
                                    data_type: str = 'info') -> pd.DataFrame:
        """Load Parquet file with native PyArrow filtering for optimal performance.
        
        Args:
            file_path: Path to Parquet file
            sample_ids: List of sample IDs to include (for int/string data)
            snp_names: List of SNP names to include
            data_type: Type of data ('info', 'int', 'string')
            
        Returns:
            Filtered DataFrame with optimal memory usage
        """
        try:
            # Determine column filtering
            columns = None
            if data_type != 'info' and snp_names:
                # For genotype data, get available columns and filter
                schema = pq.read_schema(file_path)
                all_cols = schema.names
                # Keep IID column and requested SNP columns
                snp_cols = [col for col in all_cols if col in snp_names]
                columns = ['IID'] + snp_cols
            elif data_type == 'info' and snp_names:
                # For info data, we'll filter after loading since we need snp_name column
                columns = None
            
            # Determine row filtering (only works for genotype data with IID column)
            filters = None
            if data_type != 'info' and sample_ids:
                filters = [('IID', 'in', sample_ids)]
            
            # Load with PyArrow native filtering
            table = pq.read_table(file_path, columns=columns, filters=filters)
            df = table.to_pandas()
            
            # Apply additional filtering for info data (PyArrow filters don't work on snp_name)
            if data_type == 'info' and snp_names:
                df = df[df['snp_name'].isin(snp_names)]
            
            return self._clean_dataframe_for_display(df)
            
        except Exception as e:
            raise DataAccessError(f"Failed to load Parquet file {file_path}: {e}")
    
    def _load_parquet_basic(self, file_path: Path) -> pd.DataFrame:
        """Load Parquet file without filtering for basic operations."""
        try:
            df = pd.read_parquet(file_path, engine='pyarrow')
            return self._clean_dataframe_for_display(df)
        except Exception as e:
            raise DataAccessError(f"Failed to load Parquet file {file_path}: {e}")

    def _filter_dataframe(self, df: pd.DataFrame, 
                         sample_ids: Optional[List[str]] = None,
                         snp_names: Optional[List[str]] = None,
                         data_type: str = 'info') -> pd.DataFrame:
        """Apply filtering to dataframes based on sample IDs and SNP names.
        
        NOTE: This method is now primarily for fallback cases.
        Use _load_parquet_with_filtering for optimal performance.
        
        Args:
            df: DataFrame to filter
            sample_ids: List of sample IDs to include (for int/string data)
            snp_names: List of SNP names to include
            data_type: Type of data ('info', 'int', 'string')
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        if data_type == 'info':
            # For info data, filter by SNP names in the 'snp_name' column
            if snp_names:
                filtered_df = filtered_df[filtered_df['snp_name'].isin(snp_names)]
        else:
            # For int/string data, filter by sample IDs and SNP name columns
            if sample_ids:
                filtered_df = filtered_df[filtered_df['IID'].isin(sample_ids)]
            
            if snp_names:
                # Keep IID column and only the specified SNP name columns
                snp_cols = [col for col in filtered_df.columns if col in snp_names]
                cols_to_keep = ['IID'] + snp_cols
                filtered_df = filtered_df[cols_to_keep]
        
        return filtered_df


class NBARepository(BaseRepository):
    """Repository for NBA carriers data access."""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self._info_df: Optional[pd.DataFrame] = None
        self._int_df: Optional[pd.DataFrame] = None
        self._string_df: Optional[pd.DataFrame] = None
        self._dataset: Optional[NBADataset] = None
    
    def load_info_df(self) -> pd.DataFrame:
        """Load NBA info DataFrame with lazy loading from Parquet."""
        if self._info_df is None:
            try:
                path = self.config.get_nba_file_path('info')
                self._info_df = self._load_parquet_basic(path)
            except Exception as e:
                raise DataAccessError(f"Failed to load NBA info data: {e}")
        return self._info_df
    
    def load_int_df(self) -> pd.DataFrame:
        """Load NBA integer genotype DataFrame with lazy loading from Parquet."""
        if self._int_df is None:
            try:
                path = self.config.get_nba_file_path('int')
                self._int_df = self._load_parquet_basic(path)
                # Map variant IDs to SNP names if needed
                self._int_df = self._map_variant_columns_to_snp_names(self._int_df)
            except Exception as e:
                raise DataAccessError(f"Failed to load NBA int data: {e}")
        return self._int_df
    
    def load_string_df(self) -> pd.DataFrame:
        """Load NBA string genotype DataFrame with lazy loading from Parquet."""
        if self._string_df is None:
            try:
                path = self.config.get_nba_file_path('string')
                self._string_df = self._load_parquet_basic(path)
                # Map variant IDs to SNP names if needed
                self._string_df = self._map_variant_columns_to_snp_names(self._string_df)
            except Exception as e:
                raise DataAccessError(f"Failed to load NBA string data: {e}")
        return self._string_df
    
    def load_all_dfs(self) -> Dict[str, pd.DataFrame]:
        """Load all NBA DataFrames and return as dictionary."""
        return {
            'info': self.load_info_df(),
            'int': self.load_int_df(),
            'string': self.load_string_df()
        }
    
    def load_data(self) -> NBADataset:
        """Load complete NBA dataset with data models."""
        if self._dataset is None:
            try:
                # Load DataFrames
                info_df = self.load_info_df()
                int_df = self.load_int_df()
                string_df = self.load_string_df()
                
                # Convert to data models (for future use)
                info_records = [NBAInfoRecord.from_pandas_row(row) for _, row in info_df.iterrows()]
                int_records = [NBAIntRecord.from_pandas_row(row) for _, row in int_df.iterrows()]
                string_records = [NBAStringRecord.from_pandas_row(row) for _, row in string_df.iterrows()]
                
                self._dataset = NBADataset(
                    info_data=info_records,
                    int_data=int_records,
                    string_data=string_records,
                    release=self.config.release
                )
            except Exception as e:
                raise DataAccessError(f"Failed to load NBA dataset: {e}")
        
        return self._dataset
    
    def load_filtered_info_df(self, snp_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered variant info data using native PyArrow filtering."""
        if not snp_names:
            return self.load_info_df()
        
        path = self.config.get_nba_file_path('info')
        return self._load_parquet_with_filtering(path, snp_names=snp_names, data_type='info')
    
    def load_filtered_int_df(self, sample_ids: Optional[List[str]] = None, 
                            snp_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered integer genotype data using native PyArrow filtering."""
        if not sample_ids and not snp_names:
            return self.load_int_df()
        
        path = self.config.get_nba_file_path('int')
        df = self._load_parquet_with_filtering(path, sample_ids=sample_ids, 
                                              snp_names=snp_names, data_type='int')
        return self._map_variant_columns_to_snp_names(df)
    
    def load_filtered_string_df(self, sample_ids: Optional[List[str]] = None, 
                               snp_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered string genotype data using native PyArrow filtering."""
        if not sample_ids and not snp_names:
            return self.load_string_df()
        
        path = self.config.get_nba_file_path('string')
        df = self._load_parquet_with_filtering(path, sample_ids=sample_ids, 
                                              snp_names=snp_names, data_type='string')
        return self._map_variant_columns_to_snp_names(df)
    
    def get_available_samples(self) -> List[str]:
        """Get list of available sample IDs."""
        int_df = self.load_int_df()
        return sorted(int_df['IID'].unique().tolist())
    
    def get_available_snp_names(self) -> List[str]:
        """Get list of available SNP names from info data."""
        info_df = self.load_info_df()
        return sorted(info_df['snp_name'].unique().tolist())
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the NBA data."""
        info_df = self.load_info_df()
        int_df = self.load_int_df()
        
        return {
            'num_variants': len(info_df),
            'num_samples': len(int_df),
            'loci': sorted(info_df['locus'].unique().tolist()),
            'release': self.config.release,
            'data_type': 'NBA'
        }
    
    def _map_variant_columns_to_snp_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map variant ID column names to SNP names using info DataFrame."""
        try:
            # Load info DataFrame directly to avoid circular calls
            if self._info_df is None:
                path = self.config.get_nba_file_path('info')
                info_df = self._load_parquet_basic(path)
            else:
                info_df = self._info_df
            
            # Create mapping from variant ID to SNP name
            id_to_snp_mapping = dict(zip(info_df['id'], info_df['snp_name']))
            
            # Get current column names
            current_columns = df.columns.tolist()
            
            # Create new column names, mapping variant IDs to SNP names
            new_columns = []
            for col in current_columns:
                if col == 'IID':
                    # Keep sample ID column as-is
                    new_columns.append(col)
                elif col in id_to_snp_mapping:
                    # Map variant ID to SNP name
                    snp_name = id_to_snp_mapping[col]
                    # Use SNP name, but add suffix if duplicate
                    base_name = snp_name
                    counter = 1
                    while base_name in new_columns:
                        base_name = f"{snp_name}_{counter}"
                        counter += 1
                    new_columns.append(base_name)
                else:
                    # Keep original name if no mapping found
                    new_columns.append(col)
            
            # Rename columns
            df_renamed = df.copy()
            df_renamed.columns = new_columns
            
            return df_renamed
            
        except Exception as e:
            # If mapping fails, return original DataFrame
            print(f"Warning: Could not map variant columns to SNP names: {e}")
            return df
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        super().clear_cache()
        self._info_df = None
        self._int_df = None
        self._string_df = None
        self._dataset = None


class WGSRepository(BaseRepository):
    """Repository for WGS carriers data access."""
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self._info_df: Optional[pd.DataFrame] = None
        self._int_df: Optional[pd.DataFrame] = None
        self._string_df: Optional[pd.DataFrame] = None
        self._dataset: Optional[WGSDataset] = None
    
    def load_info_df(self) -> pd.DataFrame:
        """Load WGS info DataFrame with lazy loading from Parquet."""
        if self._info_df is None:
            try:
                path = self.config.get_wgs_file_path('info')
                self._info_df = self._load_parquet_basic(path)
            except Exception as e:
                raise DataAccessError(f"Failed to load WGS info data: {e}")
        return self._info_df
    
    def load_int_df(self) -> pd.DataFrame:
        """Load WGS integer genotype DataFrame with lazy loading from Parquet."""
        if self._int_df is None:
            try:
                path = self.config.get_wgs_file_path('int')
                self._int_df = self._load_parquet_basic(path)
                # Map variant IDs to SNP names if needed
                self._int_df = self._map_variant_columns_to_snp_names(self._int_df)
            except Exception as e:
                raise DataAccessError(f"Failed to load WGS int data: {e}")
        return self._int_df
    
    def load_string_df(self) -> pd.DataFrame:
        """Load WGS string genotype DataFrame with lazy loading from Parquet."""
        if self._string_df is None:
            try:
                path = self.config.get_wgs_file_path('string')
                self._string_df = self._load_parquet_basic(path)
                # Map variant IDs to SNP names if needed
                self._string_df = self._map_variant_columns_to_snp_names(self._string_df)
            except Exception as e:
                raise DataAccessError(f"Failed to load WGS string data: {e}")
        return self._string_df
    
    def _map_variant_columns_to_snp_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map variant ID column names to SNP names using info DataFrame."""
        try:
            # Load info DataFrame directly to avoid circular calls
            if self._info_df is None:
                path = self.config.get_wgs_file_path('info')
                info_df = self._load_parquet_basic(path)
            else:
                info_df = self._info_df
            
            # Create mapping from variant ID to SNP name
            id_to_snp_mapping = dict(zip(info_df['id'], info_df['snp_name']))
            
            # Get current column names
            current_columns = df.columns.tolist()
            
            # Create new column names, mapping variant IDs to SNP names
            new_columns = []
            for col in current_columns:
                if col == 'IID':
                    # Keep sample ID column as-is
                    new_columns.append(col)
                elif col in id_to_snp_mapping:
                    # Map variant ID to SNP name
                    snp_name = id_to_snp_mapping[col]
                    # Use SNP name, but add suffix if duplicate
                    base_name = snp_name
                    counter = 1
                    while base_name in new_columns:
                        base_name = f"{snp_name}_{counter}"
                        counter += 1
                    new_columns.append(base_name)
                else:
                    # Keep original name if no mapping found
                    new_columns.append(col)
            
            # Rename columns
            df_renamed = df.copy()
            df_renamed.columns = new_columns
            
            return df_renamed
            
        except Exception as e:
            # If mapping fails, return original DataFrame
            print(f"Warning: Could not map WGS variant columns to SNP names: {e}")
            return df
    
    def load_all_dfs(self) -> Dict[str, pd.DataFrame]:
        """Load all WGS DataFrames and return as dictionary."""
        return {
            'info': self.load_info_df(),
            'int': self.load_int_df(),
            'string': self.load_string_df()
        }
    
    def load_data(self) -> WGSDataset:
        """Load complete WGS dataset with data models."""
        if self._dataset is None:
            try:
                # Load DataFrames
                info_df = self.load_info_df()
                int_df = self.load_int_df()
                string_df = self.load_string_df()
                
                # Convert to data models (for future use)
                info_records = [WGSInfoRecord.from_pandas_row(row) for _, row in info_df.iterrows()]
                int_records = [WGSIntRecord.from_pandas_row(row) for _, row in int_df.iterrows()]
                string_records = [WGSStringRecord.from_pandas_row(row) for _, row in string_df.iterrows()]
                
                self._dataset = WGSDataset(
                    info_data=info_records,
                    int_data=int_records,
                    string_data=string_records,
                    release=self.config.release,
                    data_type="WGS"
                )
            except Exception as e:
                raise DataAccessError(f"Failed to load WGS dataset: {e}")
        
        return self._dataset
    
    def load_filtered_info_df(self, snp_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered variant info data using native PyArrow filtering."""
        if not snp_names:
            return self.load_info_df()
        
        path = self.config.get_wgs_file_path('info')
        return self._load_parquet_with_filtering(path, snp_names=snp_names, data_type='info')
    
    def load_filtered_int_df(self, sample_ids: Optional[List[str]] = None, 
                            snp_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered integer genotype data using native PyArrow filtering."""
        if not sample_ids and not snp_names:
            return self.load_int_df()
        
        path = self.config.get_wgs_file_path('int')
        df = self._load_parquet_with_filtering(path, sample_ids=sample_ids, 
                                              snp_names=snp_names, data_type='int')
        return self._map_variant_columns_to_snp_names(df)
    
    def load_filtered_string_df(self, sample_ids: Optional[List[str]] = None, 
                               snp_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Load filtered string genotype data using native PyArrow filtering."""
        if not sample_ids and not snp_names:
            return self.load_string_df()
        
        path = self.config.get_wgs_file_path('string')
        df = self._load_parquet_with_filtering(path, sample_ids=sample_ids, 
                                              snp_names=snp_names, data_type='string')
        return self._map_variant_columns_to_snp_names(df)
    
    def get_available_samples(self) -> List[str]:
        """Get list of available sample IDs."""
        int_df = self.load_int_df()
        return sorted(int_df['IID'].unique().tolist())
    
    def get_available_snp_names(self) -> List[str]:
        """Get list of available SNP names from info data."""
        info_df = self.load_info_df()
        return sorted(info_df['snp_name'].unique().tolist())
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the WGS data."""
        info_df = self.load_info_df()
        int_df = self.load_int_df()
        
        return {
            'num_variants': len(info_df),
            'num_samples': len(int_df),
            'loci': sorted(info_df['locus'].unique().tolist()),
            'release': self.config.release,
            'data_type': 'WGS'
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        super().clear_cache()
        self._info_df = None
        self._int_df = None
        self._string_df = None
        self._dataset = None


def create_nba_repository(config: Optional[DataConfig] = None) -> NBARepository:
    """Factory function to create NBA repository."""
    if config is None:
        from src.core.config import config as default_config
        config = default_config
    
    return NBARepository(config)


def create_wgs_repository(config: Optional[DataConfig] = None) -> WGSRepository:
    """Factory function to create WGS repository."""
    if config is None:
        from src.core.config import config as default_config
        config = default_config
    
    return WGSRepository(config) 