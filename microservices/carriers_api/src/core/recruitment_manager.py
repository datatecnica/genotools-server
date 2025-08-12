"""
Main orchestrator for precision medicine recruitment analysis.

This module provides the main manager class that orchestrates all components
following the CarrierAnalysisManager pattern from .cursorrules.
"""

import os
from typing import Dict, Any, Optional
from src.core.recruitment_config import RecruitmentAnalysisConfig
from src.core.clinical_repository import RecruitmentClinicalRepository
from src.core.carrier_repository import RecruitmentCarrierRepository
from src.core.recruitment_processor import RecruitmentCarrierProcessor, RecruitmentClinicalProcessor
from src.core.recruitment_analyzer import RecruitmentAnalyzer
from src.core.recruitment_exporter import RecruitmentResultsExporter


class PrecisionMedRecruitmentManager:
    """
    Main orchestrator for precision medicine recruitment analysis.
    
    Follows the CarrierAnalysisManager pattern from .cursorrules with proper
    dependency injection and separation of concerns. Coordinates all components
    to perform comprehensive recruitment analysis for genetic carriers.
    """
    
    def __init__(self, config: RecruitmentAnalysisConfig):
        """
        Initialize the manager with configuration and all dependencies.
        
        Args:
            config: Analysis configuration object
        """
        self.config = config
        
        # Initialize repositories
        self.clinical_repo = RecruitmentClinicalRepository(config)
        self.carrier_repo = RecruitmentCarrierRepository(config)
        
        # Initialize processors
        self.carrier_processor = RecruitmentCarrierProcessor(config)
        self.clinical_processor = RecruitmentClinicalProcessor()
        
        # Initialize analyzer
        self.recruitment_analyzer = RecruitmentAnalyzer()
        
        # Initialize exporter
        self.results_exporter = RecruitmentResultsExporter(config)
        
        # Data containers
        self.clinical_data = {}
        self.carrier_data = {}
        self.analysis_results = {}
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run the complete recruitment analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
            
        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data validation fails
        """
        print("🚀 Starting Precision Medicine Recruitment Analysis")
        print(f"   Release: {self.config.release}")
        print(f"   Output: {self.config.output_dir}")
        print("=" * 60)
        
        try:
            # Load all data
            self._load_clinical_data()
            self._load_carrier_data()
            
            # Analyze distributions
            self._analyze_cohort_distributions()
            self._analyze_carrier_distributions()
            
            # Prepare clinical data
            self._prepare_clinical_data()
            
            # Generate recruitment analyses for all loci
            self._generate_recruitment_analyses()
            
            # Export results
            self._export_results()
            
            print("\n✅ Analysis complete!")
            return self.analysis_results
            
        except FileNotFoundError as e:
            print(f"\n❌ File not found: {str(e)}")
            raise
        except ValueError as e:
            print(f"\n❌ Data validation error: {str(e)}")
            raise
        except Exception as e:
            print(f"\n❌ Analysis failed: {str(e)}")
            raise
    
    def _load_clinical_data(self) -> None:
        """Load all clinical data files."""
        print("\nLoading clinical data...")
        
        # Load data dictionary
        self.clinical_data['dict_df'] = self.clinical_repo.load_data_dictionary()
        print(f"  ✓ Loaded data dictionary: {self.clinical_data['dict_df'].shape}")
        
        # Load master key
        self.clinical_data['key'] = self.clinical_repo.load_master_key()
        print(f"  ✓ Loaded master key: {self.clinical_data['key'].shape}")
        
        # Load extended clinical data (first visit only)
        self.clinical_data['key_extended_first'] = self.clinical_repo.load_extended_clinical(
            first_visit_only=True
        )
        print(f"  ✓ Loaded extended clinical data: {self.clinical_data['key_extended_first'].shape}")
        
        # Also keep full extended data for reference
        self.clinical_data['key_extended'] = self.clinical_repo.load_extended_clinical(
            first_visit_only=False
        )
    
    def _load_carrier_data(self) -> None:
        """Load and process genetic carrier data."""
        print("\nLoading genetic carrier data...")
        
        # Load variant information
        variant_info_df = self.carrier_repo.load_variant_info()
        print(f"  ✓ Loaded variant info: {variant_info_df.shape}")
        
        # Load carrier data (using int format as primary)
        carriers_int_df = self.carrier_repo.load_carriers_int()
        print(f"  ✓ Loaded carriers data: {carriers_int_df.shape}")
        
        # Process carriers by locus
        self.carrier_data = self.carrier_processor.process_carriers_by_locus(
            carriers_int_df, 
            variant_info_df,
            self.clinical_data['key']  # Pass master key for ancestry info
        )
        
        print(f"\n  Summary: Loaded carrier data for {len(self.carrier_data)} loci")
        for locus, carriers in self.carrier_data.items():
            print(f"    - {locus}: {len(carriers)} carriers")
    
    def _analyze_cohort_distributions(self) -> None:
        """Analyze cohort distributions."""
        print("\nAnalyzing cohort distributions...")
        
        cohort_results = self.recruitment_analyzer.analyze_cohort_distribution(
            self.clinical_data
        )
        self.analysis_results['cohort_distributions'] = cohort_results
        
        # Print summary
        if 'basic_cohorts' in cohort_results:
            print(f"  ✓ Found {len(cohort_results['basic_cohorts'])} cohorts in master key")
        if 'extended_cohorts' in cohort_results:
            print(f"  ✓ Found {len(cohort_results['extended_cohorts'])} cohorts in extended data")
    
    def _analyze_carrier_distributions(self) -> None:
        """Analyze carrier distributions by ancestry."""
        print("\nAnalyzing carrier distributions by ancestry...")
        
        carrier_dist_results = self.recruitment_analyzer.analyze_carrier_distribution(
            self.carrier_data
        )
        self.analysis_results.update(carrier_dist_results)
        
        # Print summary
        for key, dist in carrier_dist_results.items():
            if key.endswith('_ancestry'):
                locus = key.replace('_ancestry', '')
                print(f"  ✓ {locus}: {len(dist)} ancestry groups")
    
    def _prepare_clinical_data(self) -> None:
        """Prepare clinical data for analysis."""
        print("\nPreparing clinical data subset...")
        
        self.clinical_data['merged_clinical'] = self.clinical_processor.prepare_clinical_subset(
            self.clinical_data['key_extended_first'],
            self.clinical_data['key']
        )
        
        print(f"  ✓ Prepared clinical subset: {self.clinical_data['merged_clinical'].shape}")
    
    def _generate_recruitment_analyses(self) -> None:
        """Generate recruitment analyses for all loci."""
        print("\nGenerating recruitment analyses...")
        
        for locus in self.carrier_data.keys():
            print(f"  Processing {locus}...")
            
            # Generate recruitment statistics
            recruitment_stats = self.recruitment_analyzer.generate_recruitment_stats(
                locus, 
                self.carrier_data[locus], 
                self.clinical_data['merged_clinical']
            )
            self.analysis_results[f'{locus}_recruitment'] = recruitment_stats
            
            # Analyze study distributions
            study_dist = self.recruitment_analyzer.analyze_study_distributions(
                locus,
                self.carrier_data[locus],
                self.clinical_data['merged_clinical']
            )
            self.analysis_results[f'{locus}_study_distribution'] = study_dist
            
            print(f"    ✓ Generated statistics for {len(recruitment_stats)} ancestry groups")
    
    def _export_results(self) -> None:
        """Export all results."""
        print("\nExporting results...")
        
        exported_files = self.results_exporter.export_results(
            self.analysis_results,
            self.carrier_data
        )
        
        self.analysis_results['exported_files'] = exported_files
    
    def print_summary(self) -> None:
        """Print a summary of the analysis results."""
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Release information
        print(f"\nRelease: {self.config.release}")
        print(f"Output directory: {self.config.output_dir}")
        
        # Cohort summary
        if 'cohort_distributions' in self.analysis_results:
            cohorts = self.analysis_results['cohort_distributions']
            if 'basic_cohorts' in cohorts:
                print(f"\nClinical cohorts analyzed: {len(cohorts['basic_cohorts'])}")
                # Show top 5 cohorts
                top_cohorts = cohorts['basic_cohorts'].head(5)
                print("\nTop 5 cohorts by sample count:")
                for _, row in top_cohorts.iterrows():
                    print(f"  - {row['study']}: {row['count']} samples")
        
        # Carrier summary
        print(f"\nGenetic loci analyzed: {len(self.carrier_data)}")
        for locus in sorted(self.carrier_data.keys()):
            total_carriers = len(self.carrier_data[locus])
            print(f"  - {locus}: {total_carriers} carriers")
            
            # Show ancestry distribution if available
            if f'{locus}_ancestry' in self.analysis_results:
                ancestry_dist = self.analysis_results[f'{locus}_ancestry']
                n_ancestries = len(ancestry_dist)
                print(f"    Distributed across {n_ancestries} ancestry groups")
        
        # Export summary
        if 'exported_files' in self.analysis_results:
            n_files = len(self.analysis_results['exported_files'])
            print(f"\nExported {n_files} result files")


def create_recruitment_analyzer(release: str, 
                               mnt_path: str = "~/gcs_mounts",
                               output_dir: Optional[str] = None) -> PrecisionMedRecruitmentManager:
    """
    Factory function to create a configured recruitment analyzer instance.
    
    Args:
        release: GP2 release version (e.g., "10")
        mnt_path: Base mount path for data files
        output_dir: Output directory for results (optional)
        
    Returns:
        Configured PrecisionMedRecruitmentManager instance
    """
    config = RecruitmentAnalysisConfig(
        release=release,
        mnt_path=mnt_path,
        output_dir=output_dir
    )
    
    return PrecisionMedRecruitmentManager(config)