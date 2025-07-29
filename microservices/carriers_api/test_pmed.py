#!/usr/bin/env python3
"""
Test script for Precision Medicine Recruitment Analysis Pipeline.

Usage:
   python test_pmed.py --release 10 \
    --key-path ~/gcs_mounts/gp2tier2_vwb/release10/clinical_data/master_key_release10_final_vwb.csv \
    --key-dict-path ~/gcs_mounts/gp2tier2_vwb/release10/clinical_data/master_key_release10_data_dictionary.csv \
    --ext-clin-path ~/gcs_mounts/gp2tier2_vwb/release10/clinical_data/r10_extended_clinical_data_vwb.csv \
    --wgs-var-info-path ~/gcs_mounts/genotools_server/carriers/wgs/release10/release10_var_info.parquet \
    --wgs-int-path ~/gcs_mounts/genotools_server/carriers/wgs/release10/release10_carriers_int.parquet \
    --wgs-string-path ~/gcs_mounts/genotools_server/carriers/wgs/release10/release10_carriers_string.parquet \
    --output-dir ~/gcs_mounts/genotools_server/carriers/wgs/release10/
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path to import the module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from precision_med_recruitment import PrecisionMedRecruitmentAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Precision Medicine Recruitment Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_pmed.py --release 10 --key-path /path/to/master_key.csv --ext-clin-path /path/to/extended_clinical.csv
    python test_pmed.py --release 10 --mnt-path ~/gcs_mounts --output-dir ~/output
    python test_pmed.py --help
        """
    )
    
    # Release and general settings
    parser.add_argument(
        '--release',
        type=str,
        required=True,
        help='GP2 release version (e.g., "10", "9")'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results files (auto-generated if not provided)'
    )
    
    # Clinical data paths
    parser.add_argument(
        '--key-path',
        type=str,
        help='Path to master key file (e.g., master_key_release10_final_vwb.csv)'
    )
    
    parser.add_argument(
        '--key-dict-path',
        type=str,
        help='Path to master key dictionary file (e.g., master_key_release10_data_dictionary.csv)'
    )
    
    parser.add_argument(
        '--ext-clin-path',
        type=str,
        help='Path to extended clinical data file (e.g., r10_extended_clinical_data_vwb.csv)'
    )
    
    # Carrier data paths
    parser.add_argument(
        '--wgs-var-info-path',
        type=str,
        help='Path to WGS variant info parquet file (e.g., release10_var_info.parquet)'
    )
    
    parser.add_argument(
        '--wgs-int-path',
        type=str,
        help='Path to WGS carriers int parquet file (e.g., release10_carriers_int.parquet)'
    )
    
    parser.add_argument(
        '--wgs-string-path',
        type=str,
        help='Path to WGS carriers string parquet file (e.g., release10_carriers_string.parquet)'
    )
    
    # Legacy mount path option (for backward compatibility)
    parser.add_argument(
        '--mnt-path',
        type=str,
        help='Base mount path for data files (used only if individual paths not provided)'
    )
    
    # Other options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check paths and data availability without running full analysis'
    )
    
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Only export results (assumes data is already loaded)'
    )
    
    return parser.parse_args()


def check_paths(analyzer, verbose=False):
    """Check if all required data paths exist."""
    print("🔍 Checking data paths...")
    
    # Get paths based on analyzer type
    if hasattr(analyzer, 'key_path'):
        # Custom analyzer with individual paths
        paths_to_check = [
            ('Master Key', analyzer.key_path),
            ('Key Dictionary', analyzer.key_dict_path),
            ('Extended Clinical', analyzer.ext_clin_path),
            ('WGS Variant Info', analyzer.wgs_var_info_path),
            ('WGS Carriers Int', analyzer.wgs_int_path),
            ('WGS Carriers String', analyzer.wgs_string_path)
        ]
    else:
        # Standard analyzer with mount path
        paths_to_check = [
            ('Master Key', analyzer.key_path),
            ('Key Dictionary', analyzer.key_dict_path),
            ('Extended Clinical', analyzer.ext_clin_path),
            ('WGS Variant Info', analyzer.wgs_var_info_path),
            ('WGS Carriers Int', analyzer.wgs_int_path),
            ('WGS Carriers String', analyzer.wgs_string_path)
        ]
    
    missing_paths = []
    
    for name, path in paths_to_check:
        expanded_path = os.path.expanduser(path)
        exists = os.path.exists(expanded_path)
        
        if verbose:
            status = "✅" if exists else "❌"
            print(f"  {status} {name}: {expanded_path}")
        else:
            print(f"  {name}: {'✅' if exists else '❌'}")
        
        if not exists:
            missing_paths.append((name, expanded_path))
    
    if missing_paths:
        print(f"\n❌ Missing {len(missing_paths)} required files:")
        for name, path in missing_paths:
            print(f"  - {name}: {path}")
        return False
    
    print("✅ All required data paths found!")
    return True


def run_analysis(args):
    """Run the precision medicine recruitment analysis."""
    print("🚀 Precision Medicine Recruitment Analysis Test")
    print("=" * 60)
    
    # Check if individual paths are provided or if we should use mount path
    use_individual_paths = any([
        args.key_path, args.key_dict_path, args.ext_clin_path,
        args.wgs_var_info_path, args.wgs_int_path, args.wgs_string_path
    ])
    
    if use_individual_paths:
        # Use individual paths
        print(f"📁 Using individual file paths for release {args.release}...")
        
        # Check that all required paths are provided
        required_paths = {
            '--key-path': args.key_path,
            '--key-dict-path': args.key_dict_path,
            '--ext-clin-path': args.ext_clin_path,
            '--wgs-var-info-path': args.wgs_var_info_path,
            '--wgs-int-path': args.wgs_int_path,
            '--wgs-string-path': args.wgs_string_path,
            '--output-dir': args.output_dir
        }
        
        missing_paths = [arg for arg, path in required_paths.items() if not path]
        if missing_paths:
            print(f"❌ Missing required arguments: {', '.join(missing_paths)}")
            print("Please provide all individual paths including --output-dir or use --mnt-path for automatic path generation.")
            return False
        
        # Create a custom analyzer with individual paths
        analyzer = create_custom_analyzer(args)
        
    else:
        # Use mount path for automatic path generation
        if not args.mnt_path:
            print("❌ Either provide all individual paths or specify --mnt-path for automatic path generation.")
            return False
            
        print(f"📁 Using mount path for release {args.release}...")
        analyzer = PrecisionMedRecruitmentAnalyzer(
            release=args.release,
            mnt_path=args.mnt_path,
            output_dir=args.output_dir
        )
    
    # Check paths
    if not check_paths(analyzer, args.verbose):
        print("\n❌ Cannot proceed - missing required data files.")
        return False
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"  Release: {args.release}")
    print(f"  Output Directory: {analyzer.output_dir}")
    print(f"  Verbose: {args.verbose}")
    if use_individual_paths:
        print(f"  Mode: Individual paths")
    else:
        print(f"  Mode: Mount path ({args.mnt_path})")
    
    if args.dry_run:
        print("\n🔍 Dry run completed - paths checked successfully!")
        return True
    
    try:
        if args.export_only:
            print("\n📤 Running export only...")
            # This would require data to be already loaded
            print("❌ Export-only mode not implemented yet - run full analysis first")
            return False
        else:
            print("\n🔬 Running full analysis...")
            results = analyzer.run_full_analysis()
            
            print("\n📊 Analysis completed successfully!")
            print(f"Results stored in: {analyzer.output_dir}")
            
            # Print summary
            analyzer.print_summary()
            
            return True
            
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def create_custom_analyzer(args):
    """Create a custom analyzer with individual paths."""
    # Create a custom class that inherits from PrecisionMedRecruitmentAnalyzer
    class CustomPrecisionMedRecruitmentAnalyzer(PrecisionMedRecruitmentAnalyzer):
        def __init__(self, release, key_path, key_dict_path, ext_clin_path,
                     wgs_var_info_path, wgs_int_path, wgs_string_path, output_dir=None):
            # Set release version
            self.release = release
            self.release_version = int(release)
            
            # Set individual paths
            self.key_path = key_path
            self.key_dict_path = key_dict_path
            self.ext_clin_path = ext_clin_path
            self.wgs_var_info_path = wgs_var_info_path
            self.wgs_int_path = wgs_int_path
            self.wgs_string_path = wgs_string_path
            
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
        
        def export_results(self, output_dir=None):
            """Override to use output_dir instead of work_dir."""
            output_dir = output_dir or self.output_dir
            return super().export_results(output_dir)
        
        def export_locus_dataframes_as_json(self, output_dir=None):
            """Override to use output_dir instead of work_dir."""
            output_dir = output_dir or self.output_dir
            return super().export_locus_dataframes_as_json(output_dir)
        
        def export_recruitment_analysis_as_json(self, output_dir=None):
            """Override to use output_dir instead of work_dir."""
            output_dir = output_dir or self.output_dir
            return super().export_recruitment_analysis_as_json(output_dir)
    
    return CustomPrecisionMedRecruitmentAnalyzer(
        release=args.release,
        key_path=args.key_path,
        key_dict_path=args.key_dict_path,
        ext_clin_path=args.ext_clin_path,
        wgs_var_info_path=args.wgs_var_info_path,
        wgs_int_path=args.wgs_int_path,
        wgs_string_path=args.wgs_string_path,
        output_dir=args.output_dir
    )


def main():
    """Main function."""
    args = parse_arguments()
    
    # Expand user paths for all path arguments
    if args.mnt_path:
        args.mnt_path = os.path.expanduser(args.mnt_path)
    if args.output_dir:
        args.output_dir = os.path.expanduser(args.output_dir)
    if args.key_path:
        args.key_path = os.path.expanduser(args.key_path)
    if args.key_dict_path:
        args.key_dict_path = os.path.expanduser(args.key_dict_path)
    if args.ext_clin_path:
        args.ext_clin_path = os.path.expanduser(args.ext_clin_path)
    if args.wgs_var_info_path:
        args.wgs_var_info_path = os.path.expanduser(args.wgs_var_info_path)
    if args.wgs_int_path:
        args.wgs_int_path = os.path.expanduser(args.wgs_int_path)
    if args.wgs_string_path:
        args.wgs_string_path = os.path.expanduser(args.wgs_string_path)
    
    # Run analysis
    success = run_analysis(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
