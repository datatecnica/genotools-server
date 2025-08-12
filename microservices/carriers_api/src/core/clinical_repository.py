"""
Repository for clinical data operations.

This module provides repository implementations for clinical data access
following the DataRepository pattern established in .cursorrules.
"""

import os
import pandas as pd
from typing import Dict, Optional, List
from src.core.recruitment_config import RecruitmentAnalysisConfig


class RecruitmentClinicalRepository:
    """
    Repository for GP2 clinical data following DataRepository pattern for recruitment analysis.
    
    Handles loading and validation of clinical data files including
    master key, extended clinical data, and data dictionary for recruitment analysis.
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
        """Set up file paths for clinical data."""
        self.paths = {
            'key_path': os.path.join(
                self.config.clinical_path,
                f"master_key_release{self.config.release}_final_vwb.csv"
            ),
            'dict_path': os.path.join(
                self.config.clinical_path,
                f"master_key_release{self.config.release}_data_dictionary.csv"
            ),
            'ext_clin_path': os.path.join(
                self.config.clinical_path,
                f"r{self.config.release}_extended_clinical_data_vwb.csv"
            )
        }
    
    def load_master_key(self) -> pd.DataFrame:
        """
        Load master key data with validation.
        
        Returns:
            DataFrame containing master key data
            
        Raises:
            FileNotFoundError: If master key file not found
            ValueError: If data validation fails
        """
        if not os.path.exists(self.paths['key_path']):
            raise FileNotFoundError(f"Master key file not found: {self.paths['key_path']}")
        
        df = pd.read_csv(self.paths['key_path'], low_memory=False)
        self._validate_master_key(df)
        return df
    
    def load_extended_clinical(self, first_visit_only: bool = True) -> pd.DataFrame:
        """
        Load extended clinical data with required columns.
        
        Args:
            first_visit_only: Whether to filter to first visit only (visit_month == 0)
            
        Returns:
            DataFrame containing extended clinical data
            
        Raises:
            FileNotFoundError: If extended clinical file not found
        """
        if not os.path.exists(self.paths['ext_clin_path']):
            raise FileNotFoundError(f"Extended clinical file not found: {self.paths['ext_clin_path']}")
        
        # Define required columns for extended clinical data
        required_cols = [
            "GP2ID", "Phenotype", "visit_month", "date_baseline_unix", 
            "date_visit_unix", "date_birth_unix", "age_at_onset", 
            "age_at_baseline", "primary_diagnosis", "last_diagnosis",
            "moca_total_score", "hoehn_and_yahr_stage", 
            "dat_sbr_caudate_mean", "date_enrollment", "study"
        ]
        
        # Check which columns exist in the file
        temp_df = pd.read_csv(self.paths['ext_clin_path'], nrows=5)
        available_cols = [col for col in required_cols if col in temp_df.columns]
        
        if len(available_cols) < len(required_cols):
            missing_cols = set(required_cols) - set(available_cols)
            print(f"Warning: Missing columns in extended clinical data: {missing_cols}")
        
        # Load data with available columns
        df = pd.read_csv(self.paths['ext_clin_path'], usecols=available_cols, low_memory=False)
        
        if first_visit_only and 'visit_month' in df.columns:
            df = df[df['visit_month'] == 0.0].copy()
        
        return df
    
    def load_data_dictionary(self) -> pd.DataFrame:
        """
        Load data dictionary.
        
        Returns:
            DataFrame containing data dictionary
            
        Raises:
            FileNotFoundError: If data dictionary file not found
        """
        if not os.path.exists(self.paths['dict_path']):
            raise FileNotFoundError(f"Data dictionary file not found: {self.paths['dict_path']}")
        
        return pd.read_csv(self.paths['dict_path'], low_memory=False)
    
    def _validate_master_key(self, df: pd.DataFrame) -> None:
        """
        Validate master key data integrity.
        
        Args:
            df: Master key DataFrame to validate
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("Master key data is empty")
        
        if 'GP2ID' not in df.columns:
            raise ValueError("Master key missing required column: GP2ID")
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=['GP2ID'], keep=False)]
        if not duplicates.empty:
            print(f"Warning: {len(duplicates)} duplicate GP2ID entries found in master key")