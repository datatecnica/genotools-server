"""
Data processing components for recruitment analysis.

This module provides processing components for carrier and clinical data
following the processor patterns established in .cursorrules.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from src.core.clinical_repository import RecruitmentClinicalRepository
from src.core.recruitment_config import RecruitmentAnalysisConfig


class RecruitmentCarrierProcessor:
    """
    Process carrier data into locus-specific DataFrames for recruitment analysis.
    
    Follows the processor pattern with single responsibility for
    transforming pre-processed carrier data into recruitment analysis-ready formats.
    """
    
    def __init__(self, config: RecruitmentAnalysisConfig):
        """
        Initialize the processor with configuration.
        
        Args:
            config: Analysis configuration object
        """
        self.config = config
    
    def process_carriers_by_locus(self, 
                                carriers_df: pd.DataFrame, 
                                variant_info_df: pd.DataFrame,
                                master_key_df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Process carrier data into locus-specific DataFrames.
        
        Args:
            carriers_df: Raw carrier data DataFrame
            variant_info_df: Variant information DataFrame
            master_key_df: Optional master key for ancestry information
            
        Returns:
            Dictionary mapping locus names to processed carrier DataFrames
        """
        # Create variant to locus mapping
        variant_to_locus = dict(zip(variant_info_df['id'], variant_info_df['locus']))
        
        # Get unique loci in data
        loci_in_data = self._get_loci_in_data(carriers_df, variant_to_locus)
        print(f"Found loci in data: {sorted(loci_in_data)}")
        
        # Process each locus
        locus_data = {}
        for locus in loci_in_data:
            locus_df = self._create_locus_dataframe(carriers_df, variant_to_locus, locus)
            if locus_df is not None and len(locus_df) > 0:
                # Add ancestry information if master key provided
                if master_key_df is not None:
                    locus_df = self._add_ancestry_info(locus_df, master_key_df)
                else:
                    locus_df['ancestry'] = 'Unknown'
                
                locus_data[locus] = locus_df
                print(f"{locus} carriers loaded: {len(locus_df)}")
        
        return locus_data
    
    def _get_loci_in_data(self, carriers_df: pd.DataFrame, 
                         variant_to_locus: Dict[str, str]) -> set:
        """Get unique loci present in carrier data."""
        loci = set()
        for col in carriers_df.columns:
            if col in variant_to_locus:
                loci.add(variant_to_locus[col])
        return loci
    
    def _create_locus_dataframe(self, carriers_df: pd.DataFrame, 
                               variant_to_locus: Dict[str, str], 
                               locus: str) -> Optional[pd.DataFrame]:
        """Create locus-specific DataFrame with carrier information."""
        # Get variant columns for this locus
        locus_variants = [col for col in carriers_df.columns 
                         if col in variant_to_locus and variant_to_locus[col] == locus]
        
        if not locus_variants:
            return None
        
        # Create locus DataFrame with IID and variant columns
        locus_df = carriers_df[['IID'] + locus_variants].copy()
        
        # Add locus information
        locus_df['locus'] = locus
        
        # Determine carrier status (0 or 1 = carrier, 2 = not carrier)
        # Using vectorized operation for better performance
        variant_data = locus_df[locus_variants]
        locus_df['is_carrier'] = (variant_data < 2).any(axis=1)
        
        # Filter to carriers only
        carriers_only = locus_df[locus_df['is_carrier']].copy()
        
        return carriers_only
    
    def _add_ancestry_info(self, carriers_df: pd.DataFrame, 
                          master_key_df: pd.DataFrame) -> pd.DataFrame:
        """Add ancestry information from master key."""
        # Prepare ancestry data
        ancestry_data = master_key_df[['GP2ID', 'nba_label']].copy()
        ancestry_data.rename(columns={'GP2ID': 'IID', 'nba_label': 'ancestry'}, inplace=True)
        
        # Merge with carriers
        merged = pd.merge(carriers_df, ancestry_data, on='IID', how='left')
        
        # Fill missing ancestry
        merged['ancestry'].fillna('Unknown', inplace=True)
        
        return merged


class RecruitmentClinicalProcessor:
    """
    Process clinical data for recruitment analysis.
    
    Handles transformation and preparation of clinical data
    for recruitment statistics generation.
    """
    
    def prepare_clinical_subset(self, 
                               extended_clinical: pd.DataFrame, 
                               master_key: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare clinical subset with relevant variables for recruitment analysis.
        
        Args:
            extended_clinical: Extended clinical data DataFrame
            master_key: Master key DataFrame
            
        Returns:
            Processed clinical DataFrame with calculated variables
        """
        # Select relevant columns that exist in the data
        available_cols = extended_clinical.columns.tolist()
        clinical_cols = [
            "GP2ID", "Phenotype", "visit_month", "date_baseline_unix", 
            "date_visit_unix", "date_birth_unix", "age_at_onset", 
            "age_at_baseline", "primary_diagnosis", "last_diagnosis",
            "moca_total_score", "hoehn_and_yahr_stage", 
            "dat_sbr_caudate_mean", "date_enrollment"
        ]
        
        # Filter to available columns
        clinical_cols = [col for col in clinical_cols if col in available_cols]
        clinical_subset = extended_clinical[clinical_cols].copy()
        
        # Merge with master key
        merged_clin = pd.merge(clinical_subset, master_key, on="GP2ID", how='inner')
        
        # Calculate derived variables
        merged_clin = self._calculate_derived_variables(merged_clin)
        
        # Standardize column names
        merged_clin = self._standardize_column_names(merged_clin)
        
        return merged_clin
    
    def _calculate_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived clinical variables."""
        # Calculate age at visit if baseline age and visit month available
        if 'age_at_baseline' in df.columns and 'visit_month' in df.columns:
            df['age_at_visit'] = df['age_at_baseline'] + df['visit_month'] / 12
        
        # Calculate disease duration if we have the necessary fields
        if 'age_at_visit' in df.columns and 'age_at_diagnosis' in df.columns:
            df['disease_duration'] = df['age_at_visit'] - df['age_at_diagnosis']
        elif 'age_at_baseline' in df.columns and 'age_at_onset' in df.columns:
            # Alternative calculation using age at onset
            df['disease_duration'] = df['age_at_baseline'] - df['age_at_onset']
        
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistency and de-duplicate conflicting sources.

        - Creates a single AAO column by coalescing age_at_onset and age_of_onset
        - Renames known fields to standardized names
        - Removes duplicate columns that can break pandas groupby aggregations
        """
        working = df.copy()

        # Coalesce age at onset columns safely into a single AAO
        has_age_at_onset = 'age_at_onset' in working.columns
        has_age_of_onset = 'age_of_onset' in working.columns
        if has_age_at_onset and has_age_of_onset:
            working['AAO'] = working['age_at_onset'].combine_first(working['age_of_onset'])
            working.drop(columns=['age_at_onset', 'age_of_onset'], inplace=True)
        elif has_age_at_onset:
            working.rename(columns={'age_at_onset': 'AAO'}, inplace=True)
        elif has_age_of_onset:
            working.rename(columns={'age_of_onset': 'AAO'}, inplace=True)

        # Standard renames
        rename_map = {
            'GP2ID': 'IID',
            'Phenotype': 'PHENO',
            'age_at_baseline': 'AGE',
            'nba_label': 'ANCESTRY',
        }
        existing_map = {k: v for k, v in rename_map.items() if k in working.columns}
        if existing_map:
            working.rename(columns=existing_map, inplace=True)

        # Drop any duplicated columns that may still exist after renaming
        # Keep the first occurrence to maintain deterministic behavior
        working = working.loc[:, ~working.columns.duplicated()]

        return working