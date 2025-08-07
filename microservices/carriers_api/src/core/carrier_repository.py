"""
Repository for carrier data operations.

This module provides repository implementations for genetic carrier data access
following the DataRepository pattern established in .cursorrules.
"""

import os
import pandas as pd
from typing import Dict, Optional
from src.core.recruitment_config import RecruitmentAnalysisConfig


class RecruitmentCarrierRepository:
    """
    Repository for GP2 carrier data following DataRepository pattern for recruitment analysis.
    
    Handles loading and validation of pre-processed carrier data files including
    variant information and carrier genotype data for recruitment analysis.
    """
    
    def __init__(self, config: RecruitmentAnalysisConfig):
        """
        Initialize the repository with configuration.
        
        Args:
            config: Analysis configuration object
        """
        self.config = config
        self._setup_paths()
    
    def _setup_paths(self) -> None:
        """Set up file paths for carrier data."""
        wgs_path = os.path.join(
            self.config.carriers_path,
            "wgs",
            f"release{self.config.release}"
        )
        
        self.paths = {
            'wgs_var_info': os.path.join(
                wgs_path,
                f"release{self.config.release}_var_info.parquet"
            ),
            'wgs_int': os.path.join(
                wgs_path,
                f"release{self.config.release}_carriers_int.parquet"
            ),
            'wgs_string': os.path.join(
                wgs_path,
                f"release{self.config.release}_carriers_string.parquet"
            )
        }
    
    def load_variant_info(self) -> pd.DataFrame:
        """
        Load variant information with validation.
        
        Returns:
            DataFrame containing variant information
            
        Raises:
            FileNotFoundError: If variant info file not found
            ValueError: If data validation fails
        """
        if not os.path.exists(self.paths['wgs_var_info']):
            raise FileNotFoundError(f"Variant info file not found: {self.paths['wgs_var_info']}")
        
        df = pd.read_parquet(self.paths['wgs_var_info'])
        self._validate_variant_info(df)
        return df
    
    def load_carriers_int(self) -> pd.DataFrame:
        """
        Load integer-encoded carrier data.
        
        Returns:
            DataFrame containing integer-encoded carrier genotypes
            
        Raises:
            FileNotFoundError: If carriers int file not found
        """
        if not os.path.exists(self.paths['wgs_int']):
            raise FileNotFoundError(f"Carriers int file not found: {self.paths['wgs_int']}")
        
        df = pd.read_parquet(self.paths['wgs_int'])
        self._validate_carrier_data(df)
        return df
    
    def load_carriers_string(self) -> pd.DataFrame:
        """
        Load string-encoded carrier data.
        
        Returns:
            DataFrame containing string-encoded carrier genotypes
            
        Raises:
            FileNotFoundError: If carriers string file not found
        """
        if not os.path.exists(self.paths['wgs_string']):
            raise FileNotFoundError(f"Carriers string file not found: {self.paths['wgs_string']}")
        
        df = pd.read_parquet(self.paths['wgs_string'])
        self._validate_carrier_data(df)
        return df
    
    def _validate_variant_info(self, df: pd.DataFrame) -> None:
        """
        Validate variant information data integrity.
        
        Args:
            df: Variant info DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Variant info data is empty")
        
        required_cols = ['id', 'locus']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Variant info missing required columns: {missing_cols}")
        
        # Check for duplicate variant IDs
        if df['id'].duplicated().any():
            n_dups = df['id'].duplicated().sum()
            print(f"Warning: {n_dups} duplicate variant IDs found")
    
    def _validate_carrier_data(self, df: pd.DataFrame) -> None:
        """
        Validate carrier data integrity.
        
        Args:
            df: Carrier DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Carrier data is empty")
        
        if 'IID' not in df.columns:
            raise ValueError("Carrier data missing required column: IID")
        
        # Check that we have at least one variant column
        non_id_cols = [col for col in df.columns if col != 'IID']
        if not non_id_cols:
            raise ValueError("Carrier data has no variant columns")