"""
Analysis components for recruitment statistics.

This module provides analysis components for generating recruitment statistics
following the analyzer patterns established in .cursorrules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any


class RecruitmentAnalyzer:
    """
    Implementation of recruitment analysis following .cursorrules patterns.
    
    Generates comprehensive recruitment statistics for genetic carriers
    including demographic, clinical, and study distribution analyses.
    """
    
    def analyze_cohort_distribution(self, clinical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Analyze distribution of samples across study cohorts.
        
        Args:
            clinical_data: Dictionary containing clinical DataFrames
            
        Returns:
            Dictionary with cohort distribution statistics
        """
        results = {}
        
        # Analyze basic clinical data
        if 'key' in clinical_data:
            key_clean = clinical_data['key'].copy()
            if 'study' in key_clean.columns:
                key_clean['study'] = key_clean['study'].fillna('Unknown')
                results['basic_cohorts'] = (key_clean.groupby('study')
                                          .size()
                                          .reset_index(name='count')
                                          .sort_values('count', ascending=False))
        
        # Analyze extended clinical data
        if 'key_extended_first' in clinical_data:
            key_ext_clean = clinical_data['key_extended_first'].copy()
            if 'study' in key_ext_clean.columns:
                key_ext_clean['study'] = key_ext_clean['study'].fillna('Unknown')
                results['extended_cohorts'] = (key_ext_clean.groupby('study')
                                             .size()
                                             .reset_index(name='count')
                                             .sort_values('count', ascending=False))
        
        return results
    
    def analyze_carrier_distribution(self, carrier_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Analyze carrier distributions by ancestry for all loci.
        
        Args:
            carrier_data: Dictionary of carrier DataFrames by locus
            
        Returns:
            Dictionary with carrier distribution statistics
        """
        results = {}
        
        for locus, carriers_df in carrier_data.items():
            if 'ancestry' in carriers_df.columns:
                tmp = carriers_df.copy()
                tmp['ancestry'] = tmp['ancestry'].fillna('Unknown')
                results[f'{locus}_ancestry'] = (tmp.groupby('ancestry')
                                              .size()
                                              .reset_index(name='count')
                                              .sort_values('count', ascending=False))
        
        return results
    
    def generate_recruitment_stats(self, 
                                 locus: str, 
                                 carriers: pd.DataFrame, 
                                 clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive recruitment statistics for a specific locus.
        
        Args:
            locus: Locus name (e.g., 'LRRK2', 'GBA1')
            carriers: Carrier data DataFrame
            clinical: Clinical data DataFrame
            
        Returns:
            DataFrame with recruitment statistics by ancestry
        """
        # Merge carriers with clinical data
        carriers_clin = pd.merge(clinical, carriers, on="IID", how='right')
        
        # Remove duplicates keeping first occurrence
        carriers_clin = carriers_clin.drop_duplicates(subset=['IID'], keep='first')
        
        # Convert Hoehn & Yahr to numeric if present
        if 'hoehn_and_yahr_stage' in carriers_clin.columns:
            carriers_clin['hoehn_and_yahr_stage'] = pd.to_numeric(
                carriers_clin['hoehn_and_yahr_stage'], errors='coerce'
            )
        
        # Ensure ancestry column exists
        if 'ancestry' not in carriers_clin.columns:
            carriers_clin['ancestry'] = 'Unknown'
        else:
            carriers_clin['ancestry'] = carriers_clin['ancestry'].fillna('Unknown')
        
        # Generate statistics by ancestry
        grouped_data = self._aggregate_recruitment_stats(carriers_clin, locus)
        
        return grouped_data
    
    def _aggregate_recruitment_stats(self, df: pd.DataFrame, locus: str) -> pd.DataFrame:
        """
        Aggregate recruitment statistics by ancestry.
        
        Args:
            df: Merged carrier and clinical data
            locus: Locus name for column naming
            
        Returns:
            DataFrame with aggregated statistics
        """
        # Define aggregation functions based on available columns
        agg_dict = {'IID': 'size'}  # Total carriers
        
        # Add phenotype statistics if available
        if 'PHENO' in df.columns:
            agg_dict['PHENO'] = lambda x: (x == 'PD').sum()  # PD cases
        
        # Add age statistics if available
        if 'AGE' in df.columns:
            agg_dict['AGE'] = ['mean', 'std']
        
        # Add age at onset statistics if available
        if 'AAO' in df.columns:
            agg_dict['AAO'] = ['mean', 'std']
        
        # Add Hoehn & Yahr statistics if available
        if 'hoehn_and_yahr_stage' in df.columns:
            agg_dict['hoehn_and_yahr_stage'] = [
                lambda x: (~x.isna()).sum(),  # HY available
                lambda x: (x <= 2).sum(),     # HY ≤ 2
                lambda x: (x <= 3).sum(),     # HY ≤ 3
                lambda x: x.isna().sum()      # HY missing
            ]
        
        # Add MoCA statistics if available
        if 'moca_total_score' in df.columns:
            agg_dict['moca_total_score'] = [
                'mean',                       # Mean MoCA
                lambda x: (~x.isna()).sum(),  # MoCA available
                lambda x: (x >= 20).sum(),    # MoCA ≥ 20
                lambda x: (x >= 24).sum()     # MoCA ≥ 24
            ]
        
        # Add DAT scan statistics if available
        if 'dat_sbr_caudate_mean' in df.columns:
            agg_dict['dat_sbr_caudate_mean'] = [
                'mean',                       # Mean DAT
                lambda x: (~x.isna()).sum(),  # DAT available
                lambda x: x.isna().sum()      # DAT missing
            ]
        
        # Add disease duration statistics if available
        if 'disease_duration' in df.columns:
            agg_dict['disease_duration'] = [
                lambda x: (x <= 5).sum(),     # Duration ≤ 5 years
                lambda x: (x <= 7).sum(),     # Duration ≤ 7 years
                lambda x: x.isna().sum()      # Duration missing
            ]
        
        # Perform aggregation
        grouped_data = df.groupby('ancestry').agg(agg_dict)
        
        # Flatten multi-level column names
        new_columns = []
        for col in grouped_data.columns:
            if isinstance(col, tuple):
                if col[1] == 'size':
                    new_columns.append(f'{locus}_total')
                elif col[1] == '<lambda>':
                    # Handle lambda functions
                    if col[0] == 'PHENO':
                        new_columns.append(f'{locus}_PD')
                    elif col[0] == 'hoehn_and_yahr_stage':
                        lambda_idx = list(grouped_data.columns).index(col)
                        hy_names = ['HY_available', 'HY_stage2_or_less', 
                                   'HY_stage3_or_less', 'HY_missing']
                        # Determine which HY stat this is
                        hy_lambda_cols = [c for c in grouped_data.columns 
                                         if c[0] == 'hoehn_and_yahr_stage' and c[1] == '<lambda>']
                        hy_idx = hy_lambda_cols.index(col)
                        new_columns.append(hy_names[hy_idx])
                    elif col[0] == 'moca_total_score':
                        moca_lambda_cols = [c for c in grouped_data.columns 
                                           if c[0] == 'moca_total_score' and c[1] == '<lambda>']
                        moca_idx = moca_lambda_cols.index(col)
                        moca_names = ['MoCA_available', 'MoCA_20_plus', 'MoCA_24_plus']
                        new_columns.append(moca_names[moca_idx])
                    elif col[0] == 'dat_sbr_caudate_mean':
                        dat_lambda_cols = [c for c in grouped_data.columns 
                                          if c[0] == 'dat_sbr_caudate_mean' and c[1] == '<lambda>']
                        dat_idx = dat_lambda_cols.index(col)
                        dat_names = ['DAT_available', 'DAT_missing']
                        new_columns.append(dat_names[dat_idx])
                    elif col[0] == 'disease_duration':
                        dur_lambda_cols = [c for c in grouped_data.columns 
                                          if c[0] == 'disease_duration' and c[1] == '<lambda>']
                        dur_idx = dur_lambda_cols.index(col)
                        dur_names = ['Duration_5y_or_less', 'Duration_7y_or_less', 'Duration_missing']
                        new_columns.append(dur_names[dur_idx])
                else:
                    # Handle named aggregations
                    if col[1] == 'mean':
                        new_columns.append(f'{col[0]}_mean')
                    elif col[1] == 'std':
                        new_columns.append(f'{col[0]}_std')
                    else:
                        new_columns.append(f'{col[0]}_{col[1]}')
            else:
                new_columns.append(col)
        
        grouped_data.columns = new_columns
        
        return grouped_data
    
    def analyze_study_distributions(self, 
                                  locus: str, 
                                  carriers: pd.DataFrame,
                                  clinical: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze carrier distribution across study cohorts.
        
        Args:
            locus: Locus name
            carriers: Carrier data DataFrame
            clinical: Clinical data DataFrame
            
        Returns:
            DataFrame with carrier counts by study
        """
        # Merge carriers with clinical data
        carriers_clin = pd.merge(clinical, carriers, on="IID", how='right')
        
        # Ensure study column exists
        if 'study' not in carriers_clin.columns:
            return pd.DataFrame({'study': ['Unknown'], 'count': [len(carriers)]})
        
        # Fill missing study values
        carriers_clin['study'] = carriers_clin['study'].fillna('Unknown')
        
        # Count by study
        study_dist = (carriers_clin.groupby('study')
                     .size()
                     .reset_index(name='count')
                     .sort_values('count', ascending=False))
        
        return study_dist