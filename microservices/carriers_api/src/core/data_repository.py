import os
import pandas as pd


class DataRepository:
    @staticmethod
    def read_csv(file_path: str, **kwargs) -> pd.DataFrame:
        """Read data from CSV file"""
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def read_parquet(file_path: str, **kwargs) -> pd.DataFrame:
        """Read data from Parquet file"""
        return pd.read_parquet(file_path, **kwargs)
    
    @staticmethod
    def write_csv(data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Write data to CSV file"""
        data.to_csv(file_path, **kwargs)
    
    @staticmethod
    def write_parquet(data: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Write data to Parquet file using PyArrow engine"""
        # Set PyArrow as default engine if not specified
        if 'engine' not in kwargs:
            kwargs['engine'] = 'pyarrow'
        data.to_parquet(file_path, **kwargs)
    
    @staticmethod
    def remove_file(file_path: str) -> None:
        """Remove a file"""
        if os.path.exists(file_path):
            os.remove(file_path)
