"""
Export components for recruitment analysis results.

This module provides export functionality for analysis results
following the file operation patterns established in .cursorrules.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
from src.core.recruitment_config import RecruitmentAnalysisConfig
from src.core.data_repository import DataRepository


class RecruitmentResultsExporter:
    """
    Export analysis results following .cursorrules file operation patterns.
    
    Handles exporting of recruitment analysis results to various formats
    including CSV and JSON, with proper file naming conventions.
    """
    
    def __init__(self, config: RecruitmentAnalysisConfig):
        """
        Initialize the exporter with configuration.
        
        Args:
            config: Analysis configuration object
        """
        self.config = config
        self.data_repo = DataRepository()
    
    def export_results(self, 
                      analysis_results: Dict[str, Any], 
                      carrier_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """
        Export all analysis results to files.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            carrier_data: Dictionary containing carrier data by locus
            
        Returns:
            Dictionary mapping result types to file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        exported_files = {}
        
        # Export recruitment analyses
        recruitment_files = self._export_recruitment_analyses(analysis_results)
        exported_files.update(recruitment_files)
        
        # Export carrier lists
        carrier_files = self._export_carrier_lists(carrier_data)
        exported_files.update(carrier_files)
        
        # Export cohort distributions
        if 'cohort_distributions' in analysis_results:
            cohort_files = self._export_cohort_distributions(
                analysis_results['cohort_distributions']
            )
            exported_files.update(cohort_files)
        
        # Export JSON formats
        json_files = self._export_json_formats(analysis_results, carrier_data)
        exported_files.update(json_files)
        
        print(f"\n✅ Exported {len(exported_files)} files to {self.config.output_dir}")
        
        return exported_files
    
    def _export_recruitment_analyses(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Export recruitment analyses to CSV files."""
        exported = {}
        
        for key, data in analysis_results.items():
            if key.endswith('_recruitment') and isinstance(data, pd.DataFrame):
                locus = key.replace('_recruitment', '')
                filename = f'{locus}_recruitment_analysis_release{self.config.release}.csv'
                filepath = os.path.join(self.config.output_dir, filename)
                
                # Use DataRepository for consistent file operations
                self.data_repo.write_csv(data, filepath, index=True)
                exported[f'{locus}_recruitment'] = filepath
                print(f"Exported {filename}")
        
        return exported
    
    def _export_carrier_lists(self, carrier_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Export carrier lists to CSV files."""
        exported = {}
        
        for locus, data in carrier_data.items():
            filename = f'{locus}_carriers_list_release{self.config.release}.csv'
            filepath = os.path.join(self.config.output_dir, filename)
            
            # Use DataRepository for consistent file operations
            self.data_repo.write_csv(data, filepath, index=False)
            exported[f'{locus}_carriers'] = filepath
            print(f"Exported {filename}")
        
        return exported
    
    def _export_cohort_distributions(self, cohort_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Export cohort distribution data to CSV files."""
        exported = {}
        
        for cohort_type, data in cohort_data.items():
            if isinstance(data, pd.DataFrame):
                filename = f'{cohort_type}_distribution_release{self.config.release}.csv'
                filepath = os.path.join(self.config.output_dir, filename)
                
                self.data_repo.write_csv(data, filepath, index=False)
                exported[cohort_type] = filepath
                print(f"Exported {filename}")
        
        return exported
    
    def _export_json_formats(self, 
                           analysis_results: Dict[str, Any],
                           carrier_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Export data in JSON formats."""
        exported = {}
        
        # Export locus-specific DataFrames as JSON
        locus_json_path = self._export_locus_dataframes_as_json(carrier_data)
        if locus_json_path:
            exported['locus_dataframes_json'] = locus_json_path
        
        # Export recruitment analysis results as JSON
        recruitment_json_path = self._export_recruitment_analysis_as_json(analysis_results)
        if recruitment_json_path:
            exported['recruitment_analysis_json'] = recruitment_json_path
        
        # Export summary statistics as JSON
        summary_json_path = self._export_summary_statistics_as_json(
            analysis_results, carrier_data
        )
        if summary_json_path:
            exported['summary_statistics_json'] = summary_json_path
        
        return exported
    
    def _export_locus_dataframes_as_json(self, 
                                       carrier_data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Export locus-specific DataFrames as JSON.
        
        Args:
            carrier_data: Dictionary of carrier DataFrames by locus
            
        Returns:
            Path to exported JSON file
        """
        locus_dataframes = {}
        
        for locus, df in carrier_data.items():
            # Convert DataFrame to JSON-serializable format
            df_copy = df.copy()
            
            # Convert any non-serializable types
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].astype(str)
                elif pd.api.types.is_numeric_dtype(df_copy[col]):
                    # Replace NaN with None for JSON serialization
                    df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
            
            # Convert to records format (list of dictionaries)
            locus_dataframes[locus] = df_copy.to_dict('records')
        
        # Export to JSON file
        filename = f'locus_specific_dataframes_release{self.config.release}.json'
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(locus_dataframes, f, indent=2, default=str)
        
        print(f"Exported {filename}")
        print(f"Contains data for {len(locus_dataframes)} loci: {list(locus_dataframes.keys())}")
        
        return filepath
    
    def _export_recruitment_analysis_as_json(self, 
                                           analysis_results: Dict[str, Any]) -> Optional[str]:
        """
        Export recruitment analysis results as JSON.
        
        Args:
            analysis_results: Dictionary containing analysis results
            
        Returns:
            Path to exported JSON file
        """
        recruitment_results = {}
        
        for key, data in analysis_results.items():
            if key.endswith('_recruitment') and isinstance(data, pd.DataFrame):
                locus = key.replace('_recruitment', '')
                
                # Convert DataFrame to JSON-serializable format
                df_copy = data.copy()
                
                # Convert index (ancestry) to a column
                df_copy.reset_index(inplace=True)
                
                # Handle any non-serializable types
                for col in df_copy.columns:
                    if pd.api.types.is_numeric_dtype(df_copy[col]):
                        df_copy[col] = df_copy[col].where(pd.notna(df_copy[col]), None)
                
                recruitment_results[locus] = df_copy.to_dict('records')
        
        if not recruitment_results:
            return None
        
        # Export to JSON file
        filename = f'recruitment_analysis_results_release{self.config.release}.json'
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(recruitment_results, f, indent=2, default=str)
        
        print(f"Exported {filename}")
        print(f"Contains recruitment analysis for {len(recruitment_results)} loci")
        
        return filepath
    
    def _export_summary_statistics_as_json(self,
                                         analysis_results: Dict[str, Any],
                                         carrier_data: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Export summary statistics as JSON.
        
        Args:
            analysis_results: Dictionary containing analysis results
            carrier_data: Dictionary containing carrier data by locus
            
        Returns:
            Path to exported JSON file
        """
        summary = {
            'release': self.config.release,
            'loci_analyzed': list(carrier_data.keys()),
            'total_loci': len(carrier_data),
            'locus_carrier_counts': {},
            'cohort_summary': {}
        }
        
        # Add carrier counts per locus
        for locus, df in carrier_data.items():
            summary['locus_carrier_counts'][locus] = len(df)
        
        # Add cohort distribution summary if available
        if 'cohort_distributions' in analysis_results:
            cohorts = analysis_results['cohort_distributions']
            if 'basic_cohorts' in cohorts and isinstance(cohorts['basic_cohorts'], pd.DataFrame):
                summary['cohort_summary']['total_cohorts'] = len(cohorts['basic_cohorts'])
                summary['cohort_summary']['largest_cohort'] = (
                    cohorts['basic_cohorts'].iloc[0]['study'] 
                    if len(cohorts['basic_cohorts']) > 0 else None
                )
        
        # Export to JSON file
        filename = f'analysis_summary_release{self.config.release}.json'
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Exported {filename}")
        
        return filepath