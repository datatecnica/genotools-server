import os
import pandas as pd
import numpy as np
from typing import Dict, Optional


class PrecisionMedRecruitmentAnalyzer:
    """
    Precision Medicine Recruitment Analysis Pipeline for GP2 Release 9 data.
    
    Analyzes clinical trial recruitment potential by evaluating genetic carriers
    (LRRK2 and GBA1) and generating recruitment statistics by ancestry and study cohort.
    """
    
    def __init__(self,
                 release: str,
                 mnt_path: str = "~/gcs_mounts",
                 output_dir: Optional[str] = None):
        """
        Initialize the analyzer with data paths based on release structure.
        
        Args:
            release: GP2 release version (e.g., "10")
            mnt_path: Base mount path for data files
            output_dir: Output directory for results files (optional)
        """
        # Set release version
        self.release = release
        self.release_version = int(release)
        
        # Set up paths based on release structure
        self.mnt_path = mnt_path
        self.carriers_path = f"{mnt_path}/genotools_server/carriers"
        self.release_path = f"{mnt_path}/gp2tier2_vwb/release{release}"
        self.clinical_path = f"{self.release_path}/clinical_data"
        
        # Clinical data paths
        self.key_path = f"{self.clinical_path}/master_key_release{release}_final_vwb.csv"
        self.key_dict_path = f"{self.clinical_path}/master_key_release{release}_data_dictionary.csv"
        self.ext_clin_path = f"{self.clinical_path}/r{release}_extended_clinical_data_vwb.csv"
        
        # Carrier data paths (parquet format)
        self.wgs_var_info_path = f"{self.carriers_path}/wgs/release{release}/release{release}_var_info.parquet"
        self.wgs_int_path = f"{self.carriers_path}/wgs/release{release}/release{release}_carriers_int.parquet"
        self.wgs_string_path = f"{self.carriers_path}/wgs/release{release}/release{release}_carriers_string.parquet"
        
        # Set output directory
        self.output_dir = output_dir
        
        # Initialize data paths
        self._setup_paths()
        
        # Data containers
        self.clinical_data = {}
        self.carrier_data = {}
        self.analysis_results = {}
        
    def _setup_paths(self):
        """Set up all file paths for data loading."""
        self.paths = {
            'key_path': self.key_path,
            'dict_path': self.key_dict_path,
            'key_extended_path': self.ext_clin_path,
            'wgs_var_info': self.wgs_var_info_path,
            'wgs_int': self.wgs_int_path,
            'wgs_string': self.wgs_string_path
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
        Load all clinical data files from GP2 Release.
        
        Returns:
            Dictionary containing all loaded clinical DataFrames
        """
        print("Loading clinical data...")
        
        # Define necessary columns for extended clinical data
        extended_clinical_cols = [
            "GP2ID",
            "Phenotype", 
            "visit_month",
            "date_baseline_unix",
            "date_visit_unix", 
            "date_birth_unix",
            "age_at_onset",
            "age_at_baseline",
            "primary_diagnosis",
            "last_diagnosis",
            "moca_total_score",
            "hoehn_and_yahr_stage",
            "dat_sbr_caudate_mean",
            "date_enrollment",
            "study"  # Also needed for cohort analysis
        ]
        
        # Load data dictionary
        self.clinical_data['dict_df'] = pd.read_csv(self.paths['dict_path'], low_memory=False)
        print(f"Loaded data dictionary: {self.clinical_data['dict_df'].shape}")
        
        # Load master key
        self.clinical_data['key'] = pd.read_csv(self.paths['key_path'], low_memory=False)
        print(f"Loaded master key: {self.clinical_data['key'].shape}")
        self._check_duplicates(self.clinical_data['key'], 'GP2ID', 'master key')
        
        # Load extended clinical data with only necessary columns
        self.clinical_data['key_extended'] = pd.read_csv(
            self.paths['key_extended_path'], 
            usecols=extended_clinical_cols, 
            low_memory=False
        )
        print(f"Loaded extended data: {self.clinical_data['key_extended'].shape}")
        self._check_duplicates(self.clinical_data['key_extended'], 'GP2ID', 'extended data')
        
        # Filter to first visit only
        self.clinical_data['key_extended_first'] = self.clinical_data['key_extended'].loc[
            self.clinical_data['key_extended']['visit_month'] == 0.0
        ]
        print(f"First visit only: {self.clinical_data['key_extended_first'].shape}")
        
        # Load extended dictionary (using same as regular dict for now)
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
        Load and process genetic carrier data from parquet files.
        Creates locus-specific DataFrames for each gene present in the data.
        
        Returns:
            Dictionary containing processed carrier DataFrames by locus
        """
        print("\nLoading genetic carrier data...")
        
        # Load variant info to map variants to loci
        var_info_df = pd.read_parquet(self.paths['wgs_var_info'])
        print(f"Loaded variant info: {var_info_df.shape}")
        
        # Load carrier data (parquet format)
        carriers_int_df = pd.read_parquet(self.paths['wgs_int'])
        carriers_string_df = pd.read_parquet(self.paths['wgs_string'])
        print(f"Loaded carriers int data: {carriers_int_df.shape}")
        print(f"Loaded carriers string data: {carriers_string_df.shape}")
        
        # Combine int and string carriers (assuming they have the same structure)
        # For now, we'll use the int carriers as the main data
        carriers_df = carriers_int_df.copy()
        
        # Create a mapping from variant ID to locus
        variant_to_locus = dict(zip(var_info_df['id'], var_info_df['locus']))
        
        # Get all unique loci present in the data
        loci_in_data = set()
        for col in carriers_df.columns:
            if col in variant_to_locus:
                loci_in_data.add(variant_to_locus[col])
        
        print(f"Found loci in data: {sorted(loci_in_data)}")
        
        # Create locus-specific DataFrames
        for locus in loci_in_data:
            # Get all variant columns for this locus
            locus_variants = [col for col in carriers_df.columns 
                            if col in variant_to_locus and variant_to_locus[col] == locus]
            
            if not locus_variants:
                continue
                
            # Create locus-specific DataFrame
            locus_df = carriers_df[['IID'] + locus_variants].copy()
            
            # Add locus information
            locus_df['locus'] = locus
            
            # Create carrier status (0 or 1 = carrier, 2 = not carrier)
            locus_df['is_carrier'] = locus_df[locus_variants].lt(2).any(axis=1)
            
            # Filter to only carriers
            locus_carriers = locus_df[locus_df['is_carrier']].copy()
            
            # Merge with master key to get ancestry information from nba_label
            if hasattr(self, 'clinical_data') and 'key' in self.clinical_data:
                # Use master key to get ancestry information
                master_key = self.clinical_data['key'][['GP2ID', 'nba_label']].copy()
                master_key.rename(columns={'GP2ID': 'IID', 'nba_label': 'ancestry'}, inplace=True)
                
                # Merge carriers with ancestry information
                locus_carriers = pd.merge(locus_carriers, master_key, on='IID', how='left')
                
                # Fill missing ancestry with 'Unknown'
                locus_carriers['ancestry'].fillna('Unknown', inplace=True)
                
                print(f"Added ancestry information from master key for {locus}")
            else:
                # Fallback if master key not loaded yet
                locus_carriers['ancestry'] = 'Unknown'
                print(f"Warning: Master key not loaded, using 'Unknown' ancestry for {locus}")
            
            # Store in carrier_data dictionary
            self.carrier_data[locus] = locus_carriers
            
            print(f"{locus} carriers loaded: {len(locus_carriers)}")
        
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
        
        # Analyze carrier distributions for all loci
        for locus in self.carrier_data.keys():
            results[f'{locus}_ancestry'] = (self.carrier_data[locus]
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
            'nba_label': 'ANCESTRY'
        }, inplace=True)
        
        self.clinical_data['merged_clinical'] = merged_clin
        return merged_clin
    
    def generate_recruitment_analysis(self, locus: str) -> pd.DataFrame:
        """
        Generate comprehensive recruitment analysis for a specific locus.
        
        Args:
            locus: Locus name (e.g., 'LRRK2', 'GBA1', 'PARK7', etc.)
            
        Returns:
            DataFrame with recruitment statistics by ancestry
        """
        print(f"\nGenerating recruitment analysis for {locus}...")
        
        if locus not in self.carrier_data:
            raise ValueError(f"Carrier data for {locus} not loaded")
        
        # Merge carriers with clinical data
        carriers_clin = pd.merge(
            self.clinical_data['merged_clinical'], 
            self.carrier_data[locus], 
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
            f'{locus}_total', f'{locus}_PD', 'Age_mean', 'AAO_mean',
            'HY_available', 'HY_stage2_or_less', 'HY_stage3_or_less', 'HY_missing',
            'MoCA_mean', 'MoCA_available', 'MoCA_20_plus', 'MoCA_24_plus',
            'DAT_mean', 'DAT_available', 'DAT_missing',
            'Duration_5y_or_less', 'Duration_7y_or_less', 'Duration_missing'
        ]
        
        self.analysis_results[f'{locus}_recruitment'] = grouped_data
        return grouped_data
    
    def analyze_study_distributions(self, locus: str) -> pd.DataFrame:
        """
        Analyze carrier distribution across study cohorts.
        
        Args:
            locus: Locus name (e.g., 'LRRK2', 'GBA1', 'PARK7', etc.)
            
        Returns:
            DataFrame with carrier counts by study
        """
        carriers_clin = pd.merge(
            self.clinical_data['merged_clinical'], 
            self.carrier_data[locus], 
            on="IID", 
            how='right'
        )
        
        carriers_clin['study'].fillna('Unknown', inplace=True)
        study_dist = carriers_clin.groupby('study').size().reset_index(name='count')
        
        return study_dist
    
    def export_results(self, output_dir: Optional[str] = None):
        """
        Export all analysis results to CSV files and locus-specific DataFrames to JSON.
        
        Args:
            output_dir: Directory to save results (defaults to output_dir)
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting results to {output_dir}...")
        
        # Export recruitment analyses for all loci
        for locus in self.carrier_data.keys():
            if f'{locus}_recruitment' in self.analysis_results:
                filename = f'{locus}_recruitment_analysis_release{self.release_version}.csv'
                filepath = os.path.join(output_dir, filename)
                self.analysis_results[f'{locus}_recruitment'].to_csv(filepath, index=True)
                print(f"Exported {filename}")
        
        # Export carrier lists for all loci
        for locus in self.carrier_data.keys():
            filename = f'{locus}_carriers_list_release{self.release_version}.csv'
            filepath = os.path.join(output_dir, filename)
            self.carrier_data[locus].to_csv(filepath, index=False)
            print(f"Exported {filename}")
        
        # Export locus-specific DataFrames as JSON
        self.export_locus_dataframes_as_json(output_dir)
        
        # Export recruitment analysis results as JSON
        self.export_recruitment_analysis_as_json(output_dir)
    
    def export_locus_dataframes_as_json(self, output_dir: Optional[str] = None):
        """
        Export locus-specific DataFrames as JSON format with locus names as keys.
        
        Args:
            output_dir: Directory to save results (defaults to output_dir)
            
        Returns:
            Dictionary with locus names as keys and carrier data as values
            
        Example JSON structure:
        {
          "LRRK2": [
            {
              "IID": "BBDP_000002",
              "chr12:40340404:T:C": 2.0,
              "chr12:40363526:G:A": 0.0,
              "locus": "LRRK2",
              "is_carrier": true,
              "ancestry": "EUR"
            }
          ],
          "GBA1": [
            {
              "IID": "BBDP_000005", 
              "chr1:155204131:A:G": 2.0,
              "locus": "GBA1",
              "is_carrier": true,
              "ancestry": "AJ"
            }
          ]
        }
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting locus-specific DataFrames as JSON...")
        
        # Create a dictionary with locus names as keys and DataFrames as values
        locus_dataframes = {}
        
        for locus in self.carrier_data.keys():
            # Convert DataFrame to JSON-serializable format
            locus_df = self.carrier_data[locus].copy()
            
            # Convert any non-serializable types to strings
            for col in locus_df.columns:
                if locus_df[col].dtype == 'object':
                    locus_df[col] = locus_df[col].astype(str)
            
            # Convert to records format (list of dictionaries)
            locus_dataframes[locus] = locus_df.to_dict('records')
        
        # Export to JSON file
        filename = f'locus_specific_dataframes_release{self.release_version}.json'
        filepath = os.path.join(output_dir, filename)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(locus_dataframes, f, indent=2, default=str)
        
        print(f"Exported {filename}")
        print(f"Contains data for {len(locus_dataframes)} loci: {list(locus_dataframes.keys())}")
        
        return locus_dataframes
    
    def export_recruitment_analysis_as_json(self, output_dir: Optional[str] = None):
        """
        Export recruitment analysis results as JSON format.
        
        Args:
            output_dir: Directory to save results (defaults to output_dir)
        """
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nExporting recruitment analysis results as JSON...")
        
        # Create a dictionary with recruitment analysis results
        recruitment_results = {}
        
        for locus in self.carrier_data.keys():
            if f'{locus}_recruitment' in self.analysis_results:
                # Convert DataFrame to JSON-serializable format
                recruitment_df = self.analysis_results[f'{locus}_recruitment'].copy()
                
                # Convert to records format (list of dictionaries)
                recruitment_results[locus] = recruitment_df.to_dict('records')
        
        # Export to JSON file
        filename = f'recruitment_analysis_results_release{self.release_version}.json'
        filepath = os.path.join(output_dir, filename)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(recruitment_results, f, indent=2, default=str)
        
        print(f"Exported {filename}")
        print(f"Contains recruitment analysis for {len(recruitment_results)} loci: {list(recruitment_results.keys())}")
        
        return recruitment_results
    
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
        
        # Generate recruitment analyses for all loci
        for locus in self.carrier_data.keys():
            self.generate_recruitment_analysis(locus)
            study_dist = self.analyze_study_distributions(locus)
            self.analysis_results[f'{locus}_study_distribution'] = study_dist
        
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
        
        for locus in self.carrier_data.keys():
            total_carriers = len(self.carrier_data[locus])
            print(f"{locus} carriers: {total_carriers}")
            
            if f'{locus}_recruitment' in self.analysis_results:
                recruitment_data = self.analysis_results[f'{locus}_recruitment']
                ancestries = len(recruitment_data)
                print(f"  - Analyzed across {ancestries} ancestry groups")


def main():
    """Main function to run the analysis."""
    # Initialize analyzer with new path structure
    analyzer = PrecisionMedRecruitmentAnalyzer(
        release="10",
        mnt_path="~/gcs_mounts",
        output_dir="~/gcs_mounts/clinical_trial_output/release10"
    )
    
    # Example with custom paths:
    # analyzer = PrecisionMedRecruitmentAnalyzer(
    #     release="9",
    #     mnt_path="/custom/mount/path",
    #     output_dir="/custom/output/path"
    # )
    
    # Example for different releases:
    # analyzer = PrecisionMedRecruitmentAnalyzer(
    #     release="10",  # or "9", "11", etc.
    #     mnt_path="~/gcs_mounts",
    #     output_dir="~/gcs_mounts/clinical_trial_output/release10"
    # )
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Print summary
    analyzer.print_summary()
    
    # Display key results
    print("\n📈 KEY FINDINGS")
    print("-" * 30)
    
    for locus in results.keys():
        if locus.endswith('_recruitment'):
            print(f"\n{locus.replace('_recruitment', '')} Recruitment Analysis:")
            print(results[locus])


if __name__ == "__main__":
    main()
