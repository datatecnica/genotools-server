import pathlib
import os
import sys
import pandas as pd
from functools import reduce
import seaborn as sns  
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import urllib.parse
from google.cloud import bigquery
import subprocess
from plotnine import *
from typing import Dict, Tuple, Optional


class PrecisionMedRecruitmentAnalyzer:
    """
    Precision Medicine Recruitment Analysis Pipeline for GP2 Release 9 data.
    
    Analyzes clinical trial recruitment potential by evaluating genetic carriers
    (LRRK2 and GBA1) and generating recruitment statistics by ancestry and study cohort.
    """
    
    def __init__(self,
                 clinical_data_path: str,
                 clinical_dict_path: str,
                 key_path: str,
                 key_extended_path: str,
                 dict_extended_path: str,
                 release_version: int,
                 lrrk2_nba_path: str,
                 lrrk2_wgs_path: str,
                 gba1_nba_path: str,
                 gba1_wgs_path: str):
        """
        Initialize the analyzer with data paths.
        
        Args:
            rel9_base_path: Base path to GP2 Release 9 data
            work_dir: Working directory for output files
            lrrk2_nba_path: Path to LRRK2 NBA carrier data file
            lrrk2_wgs_path: Path to LRRK2 WGS carrier data file
            gba1_nba_path: Path to GBA1 NBA carrier data file
            gba1_wgs_path: Path to GBA1 WGS carrier data file
        """
        # Set paths (all required)
        self.release_version = release_version
        # self.rel_path = pathlib.Path(rel9_base_path)
        # self.work_dir = work_dir
        
        # Individual carrier file paths
        self.lrrk2_nba_path = lrrk2_nba_path
        self.lrrk2_wgs_path = lrrk2_wgs_path
        self.gba1_nba_path = gba1_nba_path
        self.gba1_wgs_path = gba1_wgs_path
        
        # Initialize data paths
        self._setup_paths()
        
        # Data containers
        self.clinical_data = {}
        self.carrier_data = {}
        self.analysis_results = {}
        
    def _setup_paths(self):
        """Set up all file paths for data loading."""
        self.paths = {
            'clinical_data': self.clinical_data_path,
            'dict_path': self.clinical_dict_path,
            'key_path': self.key_path,
            'key_extended_path': self.key_extended_path,
            'dict_extended_path': self.dict_extended_path,
            'lrrk2_nba': self.lrrk2_nba_path,
            'lrrk2_wgs': self.lrrk2_wgs_path,
            'gba1_nba': self.gba1_nba_path,
            'gba1_wgs': self.gba1_wgs_path
        }
        
        # Reference: Original hard-coded paths from notebook
        # rel9_base_path: pathlib.Path.home() / 'workspace/gp2_tier2_eu_release9_18122024'
        # work_dir: '/home/jupyter/clinical_trial/'
        # lrrk2_nba_path: '/home/jupyter/clinical_trial/LRRK2_carriers_all_NBA_raw_cleaned.csv'
        # lrrk2_wgs_path: '/home/jupyter/clinical_trial/LRRK2_carriers_all_WGS_updated.csv'
        # gba1_nba_path: '/home/jupyter/clinical_trial/GBA1_carriers_all_NBA_raw_cleaned.csv'
        # gba1_wgs_path: '/home/jupyter/clinical_trial/GBA1_carriers_all_WGS_updated.csv'
    
    def load_clinical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all clinical data files from GP2 Release 9.
        
        Returns:
            Dictionary containing all loaded clinical DataFrames
        """
        print("Loading clinical data...")
        
        # Load data dictionary
        self.clinical_data['dict_df'] = pd.read_csv(self.paths['dict_path'], low_memory=False)
        print(f"Loaded data dictionary: {self.clinical_data['dict_df'].shape}")
        
        # Load master key
        self.clinical_data['key'] = pd.read_csv(self.paths['key_path'], low_memory=False)
        print(f"Loaded master key: {self.clinical_data['key'].shape}")
        self._check_duplicates(self.clinical_data['key'], 'GP2ID', 'master key')
        
        # Load extended clinical data
        self.clinical_data['key_extended'] = pd.read_csv(self.paths['key_extended_path'], low_memory=False)
        print(f"Loaded extended data: {self.clinical_data['key_extended'].shape}")
        self._check_duplicates(self.clinical_data['key_extended'], 'GP2ID', 'extended data')
        
        # Filter to first visit only
        self.clinical_data['key_extended_first'] = self.clinical_data['key_extended'].loc[
            self.clinical_data['key_extended']['visit_month'] == 0.0
        ]
        print(f"First visit only: {self.clinical_data['key_extended_first'].shape}")
        
        # Load extended dictionary
        self.clinical_data['dict_extended'] = pd.read_csv(self.paths['dict_path'], low_memory=False)
        
        return self.clinical_data
    
    def _check_duplicates(self, df: pd.DataFrame, column: str, data_name: str):
        """Check for duplicates in specified column."""
        duplicates = df[df.duplicated(subset=column, keep=False)]
        if not duplicates.empty:
            print(f"⚠️  Duplicate entries found in {data_name}: {len(duplicates)}")
        else:
            print(f"✅ No duplicates found in {data_name}")
    
    def load_carrier_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and process genetic carrier data for LRRK2 and GBA1.
        
        Returns:
            Dictionary containing processed carrier DataFrames
        """
        print("\nLoading genetic carrier data...")
        
        # Load LRRK2 carriers
        lrrk2_nba = pd.read_csv(self.paths['lrrk2_nba'], delimiter="\t")
        lrrk2_wgs = pd.read_csv(self.paths['lrrk2_wgs'], sep='\t')
        
        self.carrier_data['LRRK2'] = pd.concat([lrrk2_nba, lrrk2_wgs], axis=0, ignore_index=True)
        self.carrier_data['LRRK2'] = self.carrier_data['LRRK2'].drop_duplicates(subset=['IID'])
        print(f"LRRK2 carriers loaded: {len(self.carrier_data['LRRK2'])}")
        
        # Load GBA1 carriers
        gba1_nba = pd.read_csv(self.paths['gba1_nba'], sep='\t')
        gba1_wgs = pd.read_csv(self.paths['gba1_wgs'], sep='\t')
        
        self.carrier_data['GBA1'] = pd.concat([gba1_nba, gba1_wgs], axis=0, ignore_index=True)
        self.carrier_data['GBA1'] = self.carrier_data['GBA1'].drop_duplicates(subset=['IID'])
        print(f"GBA1 carriers loaded: {len(self.carrier_data['GBA1'])}")
        
        return self.carrier_data
    
    def analyze_cohort_distribution(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze distribution of samples across different study cohorts.
        
        Returns:
            Dictionary with cohort distribution statistics
        """
        print("\nAnalyzing cohort distributions...")
        
        results = {}
        
        # Fill NaN values and analyze basic clinical data
        key_clean = self.clinical_data['key'].copy()
        key_clean['study'].fillna('Unknown', inplace=True)
        results['basic_cohorts'] = key_clean.groupby('study').size().reset_index(name='count')
        
        # Analyze extended clinical data
        key_ext_clean = self.clinical_data['key_extended_first'].copy()
        key_ext_clean['study'].fillna('Unknown', inplace=True)
        results['extended_cohorts'] = key_ext_clean.groupby('study').size().reset_index(name='count')
        
        # Analyze carrier distributions
        for gene in ['LRRK2', 'GBA1']:
            if gene in self.carrier_data:
                results[f'{gene}_ancestry'] = (self.carrier_data[gene]
                                             .groupby('ancestry')
                                             .size()
                                             .reset_index(name='count'))
        
        self.analysis_results['cohort_distributions'] = results
        return results
    
    def prepare_clinical_subset(self) -> pd.DataFrame:
        """
        Prepare clinical subset with relevant variables for recruitment analysis.
        
        Returns:
            Processed clinical DataFrame with calculated variables
        """
        print("\nPreparing clinical subset...")
        
        # Select relevant columns
        clinical_cols = [
            "GP2ID", "Phenotype", "visit_month", "date_baseline_unix", "date_visit_unix", 
            "date_birth_unix", "age_at_onset", "age_at_baseline", "primary_diagnosis", 
            "last_diagnosis", "moca_total_score", "hoehn_and_yahr_stage", 
            "dat_sbr_caudate_mean", "date_enrollment"
        ]
        
        clinical_subset = self.clinical_data['key_extended_first'][clinical_cols].copy()
        
        # Merge with master key
        merged_clin = pd.merge(clinical_subset, self.clinical_data['key'], on="GP2ID")
        
        # Calculate derived variables
        merged_clin['age_at_visit'] = merged_clin['age_at_baseline'] + merged_clin['visit_month']/12
        merged_clin['disease_duration'] = merged_clin['age_at_visit'] - merged_clin['age_at_diagnosis']
        
        # Standardize column names
        merged_clin.rename(columns={
            'GP2ID': 'IID', 
            'Phenotype': 'PHENO', 
            'age_at_baseline': 'AGE', 
            'age_of_onset': 'AAO', 
            'label': 'ANCESTRY'
        }, inplace=True)
        
        self.clinical_data['merged_clinical'] = merged_clin
        return merged_clin
    
    def generate_recruitment_analysis(self, gene: str) -> pd.DataFrame:
        """
        Generate comprehensive recruitment analysis for a specific gene.
        
        Args:
            gene: Gene name ('LRRK2' or 'GBA1')
            
        Returns:
            DataFrame with recruitment statistics by ancestry
        """
        print(f"\nGenerating recruitment analysis for {gene}...")
        
        if gene not in self.carrier_data:
            raise ValueError(f"Carrier data for {gene} not loaded")
        
        # Merge carriers with clinical data
        carriers_clin = pd.merge(
            self.clinical_data['merged_clinical'], 
            self.carrier_data[gene], 
            on="IID", 
            how='right'
        )
        
        # Remove duplicates
        carriers_clin = carriers_clin.drop_duplicates(subset=['IID'])
        
        # Convert Hoehn & Yahr to numeric
        carriers_clin['hoehn_and_yahr_stage'] = pd.to_numeric(
            carriers_clin['hoehn_and_yahr_stage'], errors='coerce'
        )
        
        # Generate comprehensive statistics by ancestry
        grouped_data = carriers_clin.groupby('ancestry').agg({
            'IID': 'size',  # Total carriers
            'PHENO': lambda x: (x == 'PD').sum(),  # PD cases
            'AGE': 'mean',  # Mean age
            'AAO': 'mean',  # Mean age at onset
            'hoehn_and_yahr_stage': [
                lambda x: (~x.isna()).sum(),  # HY available
                lambda x: (x <= 2).sum(),     # HY ≤ 2
                lambda x: (x <= 3).sum(),     # HY ≤ 3
                lambda x: x.isna().sum()      # HY missing
            ],
            'moca_total_score': [
                'mean',                       # Mean MoCA
                lambda x: (~x.isna()).sum(),  # MoCA available
                lambda x: (x >= 20).sum(),    # MoCA ≥ 20
                lambda x: (x >= 24).sum()     # MoCA ≥ 24
            ],
            'dat_sbr_caudate_mean': [
                'mean',                       # Mean DAT
                lambda x: (~x.isna()).sum(),  # DAT available
                lambda x: x.isna().sum()      # DAT missing
            ],
            'disease_duration': [
                lambda x: (x <= 5).sum(),     # Duration ≤ 5 years
                lambda x: (x <= 7).sum(),     # Duration ≤ 7 years
                lambda x: x.isna().sum()      # Duration missing
            ]
        })
        
        # Flatten column names
        grouped_data.columns = [
            f'{gene}_total', f'{gene}_PD', 'Age_mean', 'AAO_mean',
            'HY_available', 'HY_stage2_or_less', 'HY_stage3_or_less', 'HY_missing',
            'MoCA_mean', 'MoCA_available', 'MoCA_20_plus', 'MoCA_24_plus',
            'DAT_mean', 'DAT_available', 'DAT_missing',
            'Duration_5y_or_less', 'Duration_7y_or_less', 'Duration_missing'
        ]
        
        self.analysis_results[f'{gene}_recruitment'] = grouped_data
        return grouped_data
    
    def analyze_study_distributions(self, gene: str) -> pd.DataFrame:
        """
        Analyze carrier distribution across study cohorts.
        
        Args:
            gene: Gene name ('LRRK2' or 'GBA1')
            
        Returns:
            DataFrame with carrier counts by study
        """
        carriers_clin = pd.merge(
            self.clinical_data['merged_clinical'], 
            self.carrier_data[gene], 
            on="IID", 
            how='right'
        )
        
        carriers_clin['study'].fillna('Unknown', inplace=True)
        study_dist = carriers_clin.groupby('study').size().reset_index(name='count')
        
        return study_dist
    
    def export_results(self, output_dir: Optional[str] = None):
        """
        Export all analysis results to CSV files.
        
        Args:
            output_dir: Directory to save results (defaults to work_dir)
        """
        output_dir = output_dir or self.work_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting results to {output_dir}...")
        
        # Export recruitment analyses
        for gene in ['LRRK2', 'GBA1']:
            if f'{gene}_recruitment' in self.analysis_results:
                filename = f'{gene}_recruitment_analysis_release9.csv'
                filepath = os.path.join(output_dir, filename)
                self.analysis_results[f'{gene}_recruitment'].to_csv(filepath, index=True)
                print(f"Exported {filename}")
        
        # Export carrier lists
        for gene in ['LRRK2', 'GBA1']:
            if gene in self.carrier_data:
                filename = f'{gene}_carriers_list_release9.csv'
                filepath = os.path.join(output_dir, filename)
                self.carrier_data[gene].to_csv(filepath, index=False)
                print(f"Exported {filename}")
    
    def run_full_analysis(self) -> Dict:
        """
        Run the complete recruitment analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🚀 Starting Precision Medicine Recruitment Analysis")
        print("=" * 60)
        
        # Load all data
        self.load_clinical_data()
        self.load_carrier_data()
        
        # Analyze distributions
        self.analyze_cohort_distribution()
        
        # Prepare clinical data
        self.prepare_clinical_subset()
        
        # Generate recruitment analyses
        for gene in ['LRRK2', 'GBA1']:
            self.generate_recruitment_analysis(gene)
            study_dist = self.analyze_study_distributions(gene)
            self.analysis_results[f'{gene}_study_distribution'] = study_dist
        
        # Export results
        self.export_results()
        
        print("\n✅ Analysis complete!")
        return self.analysis_results
    
    def print_summary(self):
        """Print a summary of the analysis results."""
        print("\n📊 ANALYSIS SUMMARY")
        print("=" * 40)
        
        if 'cohort_distributions' in self.analysis_results:
            cohorts = self.analysis_results['cohort_distributions']
            print(f"Clinical cohorts analyzed: {len(cohorts.get('basic_cohorts', []))}")
            print(f"Extended data cohorts: {len(cohorts.get('extended_cohorts', []))}")
        
        for gene in ['LRRK2', 'GBA1']:
            if gene in self.carrier_data:
                total_carriers = len(self.carrier_data[gene])
                print(f"{gene} carriers: {total_carriers}")
                
                if f'{gene}_recruitment' in self.analysis_results:
                    recruitment_data = self.analysis_results[f'{gene}_recruitment']
                    ancestries = len(recruitment_data)
                    print(f"  - Analyzed across {ancestries} ancestry groups")


def main():
    """Main function to run the analysis."""
    # Initialize analyzer with original paths from notebook
    analyzer = PrecisionMedRecruitmentAnalyzer(
        rel9_base_path=str(pathlib.Path.home() / 'workspace/gp2_tier2_eu_release9_18122024'),
        work_dir='/home/jupyter/clinical_trial/',
        lrrk2_nba_path='/home/jupyter/clinical_trial/LRRK2_carriers_all_NBA_raw_cleaned.csv',
        lrrk2_wgs_path='/home/jupyter/clinical_trial/LRRK2_carriers_all_WGS_updated.csv',
        gba1_nba_path='/home/jupyter/clinical_trial/GBA1_carriers_all_NBA_raw_cleaned.csv',
        gba1_wgs_path='/home/jupyter/clinical_trial/GBA1_carriers_all_WGS_updated.csv'
    )
    
    # Example with custom paths:
    # analyzer = PrecisionMedRecruitmentAnalyzer(
    #     rel9_base_path='/path/to/custom/gp2_release9_data',
    #     work_dir='/path/to/custom/output',
    #     lrrk2_nba_path='/custom/path/LRRK2_NBA.csv',
    #     lrrk2_wgs_path='/custom/path/LRRK2_WGS.csv',
    #     gba1_nba_path='/custom/path/GBA1_NBA.csv',
    #     gba1_wgs_path='/custom/path/GBA1_WGS.csv'
    # )
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    # Display key results
    print("\n📈 KEY FINDINGS")
    print("-" * 30)
    
    for gene in ['LRRK2', 'GBA1']:
        if f'{gene}_recruitment' in results:
            print(f"\n{gene} Recruitment Analysis:")
            print(results[f'{gene}_recruitment'])


if __name__ == "__main__":
    main()
