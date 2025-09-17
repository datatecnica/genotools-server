#!/usr/bin/env python3
"""
Run carriers pipeline for harmonization/extraction across ALL datatypes (NBA, WGS, IMPUTED).
Executes the full genomic carrier screening pipeline with separate data type outputs.
"""

import sys
import os
import logging
import argparse
import time
from pathlib import Path
from typing import List

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import Settings
from app.processing.coordinator import ExtractionCoordinator
from app.processing.extractor import VariantExtractor
from app.processing.transformer import GenotypeTransformer
from app.models.analysis import DataType

def parse_args():
    """Parse command line arguments."""
    # Get default ancestries from settings
    default_settings = Settings()
    all_ancestries = default_settings.ANCESTRIES
    
    parser = argparse.ArgumentParser(description='Run carriers pipeline across all data types')
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,  # Will use config-based path if not specified
        help='Output path prefix (default: config-based results directory)'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default='carriers_analysis',
        help='Job name for output files (default: carriers_analysis)'
    )
    parser.add_argument(
        '--ancestries',
        type=str,
        nargs='+',
        default=all_ancestries,  # Use all ancestries from config
        help=f'Ancestries to process (default: all {len(all_ancestries)} ancestries from config)'
    )
    parser.add_argument(
        '--data-types',
        type=str,
        nargs='+',
        choices=['NBA', 'WGS', 'IMPUTED'],
        default=['NBA', 'WGS', 'IMPUTED'],
        help='Data types to process (default: all)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,  # Use auto-detected settings
        help='Maximum workers (default: auto-detect)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        default=True,
        help='Use performance optimizations (default: True)'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        default=False,
        help='Skip extraction phase if results already exist (default: False)'
    )
    parser.add_argument(
        '--enable-probe-selection',
        action='store_true',
        default=True,
        help='Enable probe quality analysis and selection (default: True)'
    )
    parser.add_argument(
        '--no-probe-selection',
        action='store_true',
        default=False,
        help='Disable probe selection analysis'
    )
    return parser.parse_args()


def print_system_info():
    """Print system information for performance context."""
    import psutil
    logger = logging.getLogger(__name__)
    
    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    logger.info("=== System Information ===")
    logger.info(f"CPU cores: {cpu_count}")
    logger.info(f"Total RAM: {memory_gb:.1f} GB")
    logger.info(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

def check_extraction_results_exist(output_dir: str, job_name: str, data_types: List[str]) -> bool:
    """Check if extraction results already exist for all requested data types."""
    required_files = []
    for data_type in data_types:
        parquet_file = os.path.join(output_dir, f"{job_name}_{data_type}.parquet")
        required_files.append(parquet_file)

    # Check if all files exist and have non-zero size
    all_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in required_files)
    return all_exist

def analyze_results(results: dict, logger):
    """Analyze and report pipeline results."""
    logger.info("\n=== Pipeline Results Analysis ===")
    
    if results['success']:
        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"📋 Job ID: {results['job_id']}")
        logger.info(f"⏱️  Execution time: {results['execution_time_seconds']:.1f}s")
        
        # Analyze output files by data type
        output_files = results['output_files']
        data_type_files = {}
        
        for file_key, file_path in output_files.items():
            if '_' in file_key:
                data_type = file_key.split('_')[0]
                if data_type not in data_type_files:
                    data_type_files[data_type] = []
                data_type_files[data_type].append(file_key)
        
        logger.info(f"📁 Generated files for {len(data_type_files)} data types:")
        for data_type, files in data_type_files.items():
            logger.info(f"   {data_type}: {len(files)} files ({', '.join(files)})")
        
        # Summary information
        if 'summary' in results and results['summary']:
            summary = results['summary']
            logger.info(f"🧬 Total variants: {summary.get('total_variants', 0)}")
            logger.info(f"👥 Total samples: {summary.get('total_samples', 0)}")
            
            # Data type breakdown if available
            if 'by_data_type' in summary:
                logger.info("📊 Variants by data type:")
                for data_type, count in summary['by_data_type'].items():
                    logger.info(f"   {data_type}: {count} variants")
            
            # Ancestry breakdown if available  
            if 'by_ancestry' in summary:
                logger.info("🌍 Variants by ancestry:")
                for ancestry, count in summary['by_ancestry'].items():
                    logger.info(f"   {ancestry}: {count} variants")
        
        return True
        
    else:
        logger.error("❌ Pipeline failed!")
        for error in results['errors']:
            logger.error(f"   Error: {error}")
        return False

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        print_system_info()
        
        # Initialize settings with optimization
        if args.optimize:
            logger.info("🚀 Using auto-optimized performance settings")
            settings = Settings.create_optimized()
        else:
            logger.info("📊 Using default settings")
            settings = Settings()
        
        logger.info(f"⚙️  Performance settings: {settings.max_workers} workers, {settings.chunk_size} chunk_size, {settings.process_cap} process_cap")
        
        # Initialize components
        extractor = VariantExtractor(settings)
        transformer = GenotypeTransformer()
        coordinator = ExtractionCoordinator(extractor, transformer, settings)
        
        # Handle output path - use config-based results directory if not specified
        if args.output is None:
            # Use config-based results directory
            output_dir = settings.results_path
            custom_name = args.job_name
            full_output_path = os.path.join(output_dir, custom_name)
        else:
            # Use user-specified path
            output_dir = os.path.dirname(args.output)
            custom_name = os.path.basename(args.output)
            full_output_path = args.output
            if not output_dir:
                output_dir = "."
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Use default SNP list
        logger.info("📋 Using default precision medicine SNP list")
        snp_list_path = settings.snp_list_path
        
        # Convert data type strings to enum
        data_type_enums = [DataType[dt] for dt in args.data_types]

        # Handle probe selection flag logic
        enable_probe_selection = args.enable_probe_selection and not args.no_probe_selection

        logger.info("=== Pipeline Configuration ===")
        logger.info(f"📊 Data types: {args.data_types}")
        logger.info(f"🌍 Ancestries ({len(args.ancestries)}): {args.ancestries}")
        logger.info(f"📋 SNP list: {snp_list_path}")
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📝 Job name: {custom_name}")
        logger.info(f"🎯 Full output path: {full_output_path}")
        logger.info(f"⚡ Parallel: {args.parallel}")
        logger.info(f"👥 Max workers: {args.max_workers or 'auto-detect'}")
        logger.info(f"🔧 Using {'config-based' if args.output is None else 'custom'} output location")
        logger.info(f"📋 Skip extraction: {args.skip_extraction}")
        logger.info(f"🔬 Probe selection: {enable_probe_selection}")

        # Check if we should skip extraction
        if args.skip_extraction:
            if check_extraction_results_exist(output_dir, custom_name, args.data_types):
                logger.info("✅ Extraction results found. Skipping extraction phase...")
                logger.info("📁 Existing files will be used for any postprocessing")

                # Create a minimal results structure for existing files
                output_files = {}
                for data_type in args.data_types:
                    parquet_file = os.path.join(output_dir, f"{custom_name}_{data_type}.parquet")
                    output_files[f"{data_type}_parquet"] = parquet_file

                # Run probe selection on existing results if enabled
                if enable_probe_selection:
                    logger.info("🔬 Running probe selection analysis on existing results...")
                    probe_selection_results = coordinator.run_probe_selection_postprocessing(
                        output_dir=output_dir,
                        output_name=custom_name,
                        data_types=data_type_enums
                    )
                    if probe_selection_results:
                        output_files.update(probe_selection_results)

                results = {
                    'success': True,
                    'job_id': custom_name,
                    'execution_time_seconds': 0.0,
                    'output_files': output_files,
                    'summary': {'note': 'Skipped extraction - used existing results'},
                    'skipped_extraction': True
                }
            else:
                logger.warning("⚠️ Skip extraction requested but no valid results found.")
                logger.info("🚀 Running full extraction pipeline...")
                start_time = time.time()
                results = coordinator.run_full_extraction_pipeline(
                    snp_list_path=snp_list_path,
                    data_types=data_type_enums,
                    output_dir=output_dir,
                    ancestries=args.ancestries,
                    parallel=args.parallel,
                    max_workers=args.max_workers,  # Use auto-detect if None
                    output_name=custom_name,
                    enable_probe_selection=enable_probe_selection
                )
        else:
            # Normal pipeline execution (will overwrite existing results)
            logger.info("\n🚀 Starting carriers pipeline extraction...")
            start_time = time.time()
            results = coordinator.run_full_extraction_pipeline(
                snp_list_path=snp_list_path,
                data_types=data_type_enums,
                output_dir=output_dir,
                ancestries=args.ancestries,
                parallel=args.parallel,
                max_workers=args.max_workers,  # Use auto-detect if None
                output_name=custom_name,
                enable_probe_selection=enable_probe_selection
            )
        
        # Calculate timing only if extraction was run
        if not results.get('skipped_extraction', False):
            end_time = time.time()
            logger.info(f"⏱️  Total pipeline time: {end_time - start_time:.1f}s")
        else:
            logger.info("⏱️  Extraction skipped - no timing measured")
        
        # Analyze and report results
        success = analyze_results(results, logger)
        
        if success:
            if results.get('skipped_extraction', False):
                logger.info("\n🎯 Carriers Pipeline Complete (Extraction Skipped)!")
                logger.info("   📁 Used existing NBA/WGS/IMPUTED datasets")
                logger.info("   🧬 Genotype data ready for carrier analysis")
                logger.info("   💡 Use without --skip-extraction to regenerate extraction data")
            else:
                logger.info("\n🎯 Carriers Pipeline Complete!")
                logger.info("   📁 Generated separate NBA/WGS/IMPUTED datasets")
                logger.info("   🧬 Genotype data ready for carrier analysis")
                logger.info("   📊 Quality reports and harmonization summaries available")
        
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())