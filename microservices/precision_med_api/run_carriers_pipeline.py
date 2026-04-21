#!/usr/bin/env python3
"""
Run carriers pipeline for harmonization/extraction across ALL datatypes (NBA, WGS, IMPUTED, EXOMES).
Executes the full genomic carrier screening pipeline with separate data type outputs.

NOTE: Keep CLI arguments in sync with run_carriers_pipeline_api.py
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
from app.core.logging_config import setup_logging, get_progress_logger, get_log_file_path
from app.core.progress import PipelineProgress
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
        choices=['NBA', 'WGS', 'IMPUTED', 'EXOMES'],
        default=['NBA', 'WGS', 'IMPUTED', 'EXOMES'],
        help='Data types to process (default: all four - NBA, WGS, IMPUTED, EXOMES)'
    )
    parser.add_argument(
        '--release',
        type=str,
        required=True,
        help='GP2 release version (required, e.g., 11). EXOMES requires release 8+'
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
        '--skip-probe-selection',
        action='store_true',
        default=False,
        help='Skip probe selection phase if results already exist (default: False)'
    )
    parser.add_argument(
        '--skip-locus-reports',
        action='store_true',
        default=False,
        help='Skip locus report generation (default: False)'
    )
    parser.add_argument(
        '--skip-coverage-profiling',
        action='store_true',
        default=False,
        help='Skip coverage profiling phase (default: False)'
    )
    parser.add_argument(
        '--retry-failed',
        type=str,
        default=None,
        metavar='FAILED_FILES_JSON',
        help='Path to a failed_files.json from a previous run. Retries only failed files and merges with existing results.'
    )
    parser.add_argument(
        '--dosage-het-min',
        type=float,
        default=0.5,
        help='Minimum dosage to call heterozygous (default: 0.5 for soft calls, use 0.9 for hard calls)'
    )
    parser.add_argument(
        '--dosage-het-max',
        type=float,
        default=1.5,
        help='Maximum dosage to call heterozygous (default: 1.5 for soft calls, use 1.1 for hard calls)'
    )
    parser.add_argument(
        '--dosage-hom-min',
        type=float,
        default=1.5,
        help='Minimum dosage to call homozygous (default: 1.5 for soft calls, use 1.9 for hard calls)'
    )
    return parser.parse_args()


def print_system_info():
    """Print system information for performance context (to file log only)."""
    import psutil
    logger = logging.getLogger(__name__)

    cpu_count = os.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    logger.debug("=== System Information ===")
    logger.debug(f"CPU cores: {cpu_count}")
    logger.debug(f"Total RAM: {memory_gb:.1f} GB")
    logger.debug(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

def check_extraction_results_exist(output_dir: str, job_name: str, data_types: List[str]) -> bool:
    """Check if extraction results already exist for all requested data types."""
    required_files = []
    for data_type in data_types:
        parquet_file = os.path.join(output_dir, f"{job_name}_{data_type}.parquet")
        required_files.append(parquet_file)

    # Check if all files exist and have non-zero size
    all_exist = all(os.path.exists(f) and os.path.getsize(f) > 0 for f in required_files)
    return all_exist

def analyze_results(results: dict, logger, progress_logger):
    """Analyze and report pipeline results."""
    logger.debug("\n=== Pipeline Results Analysis ===")

    if results['success']:
        # Log detailed info to file
        logger.debug(f"Job ID: {results['job_id']}")
        logger.debug(f"Execution time: {results['execution_time_seconds']:.1f}s")

        # Analyze output files by data type
        output_files = results['output_files']
        data_type_files = {}

        for file_key, file_path in output_files.items():
            if '_' in file_key:
                data_type = file_key.split('_')[0]
                if data_type not in data_type_files:
                    data_type_files[data_type] = []
                data_type_files[data_type].append(file_key)

        logger.debug(f"Generated files for {len(data_type_files)} data types:")
        for data_type, files in data_type_files.items():
            logger.debug(f"   {data_type}: {len(files)} files ({', '.join(files)})")

        # Summary information
        if 'summary' in results and results['summary']:
            summary = results['summary']
            total_variants = summary.get('total_variants', 0)
            total_samples = summary.get('total_samples', 0)

            # Show summary to console
            progress_logger.info(f"  Variants: {total_variants} | Samples: {total_samples}")

            # Data type breakdown to file log
            if 'by_data_type' in summary:
                logger.debug("Variants by data type:")
                for data_type, count in summary['by_data_type'].items():
                    logger.debug(f"   {data_type}: {count} variants")

        return True

    else:
        progress_logger.error("Pipeline failed!")
        for error in results['errors']:
            progress_logger.error(f"  Error: {error}")
        return False

def main():
    # Parse command line arguments
    args = parse_args()

    # Determine output directory early for log file placement
    temp_settings = Settings(release=args.release)
    if args.output is None:
        output_dir = temp_settings.results_path
    else:
        output_dir = os.path.dirname(args.output) or "."

    # Setup centralized logging (console quiet, file detailed)
    log_file = setup_logging(log_dir=output_dir, job_name=args.job_name)
    logger = logging.getLogger(__name__)
    progress = get_progress_logger()

    try:
        print_system_info()

        # Initialize settings with optimization and release
        dosage_overrides = {
            'dosage_het_min': args.dosage_het_min,
            'dosage_het_max': args.dosage_het_max,
            'dosage_hom_min': args.dosage_hom_min,
        }
        if args.optimize:
            logger.debug("Using auto-optimized performance settings")
            settings = Settings.create_optimized(release=args.release, **dosage_overrides)
        else:
            logger.debug("Using default settings")
            settings = Settings(release=args.release, **dosage_overrides)
        
        logger.debug(f"Performance settings: {settings.max_workers} workers, {settings.chunk_size} chunk_size, {settings.process_cap} process_cap")
        logger.debug(f"Dosage thresholds: het=[{settings.dosage_het_min}, {settings.dosage_het_max}), hom>={settings.dosage_hom_min}")

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
        snp_list_path = settings.snp_list_path

        # Convert data type strings to enum
        data_type_enums = [DataType[dt] for dt in args.data_types]

        # Handle probe selection logic - enabled by default unless skipped
        enable_probe_selection = not args.skip_probe_selection

        # Handle locus reports logic - enabled by default unless skipped
        enable_locus_reports = not args.skip_locus_reports

        # Handle coverage profiling logic - enabled by default unless skipped
        enable_coverage_profiling = not args.skip_coverage_profiling

        # Console: Show minimal config summary
        progress.info(f"Release {args.release} | {', '.join(args.data_types)} | {len(args.ancestries)} ancestries")
        progress.info(f"Output: {output_dir}")

        # File log: Show detailed config
        logger.debug("=== Pipeline Configuration ===")
        logger.debug(f"Release: {args.release}")
        logger.debug(f"Data types: {args.data_types}")
        logger.debug(f"Ancestries ({len(args.ancestries)}): {args.ancestries}")
        logger.debug(f"SNP list: {snp_list_path}")
        logger.debug(f"Output directory: {output_dir}")
        logger.debug(f"Job name: {custom_name}")
        logger.debug(f"Full output path: {full_output_path}")
        logger.debug(f"Parallel: {args.parallel}")
        logger.debug(f"Max workers: {args.max_workers or 'auto-detect'}")
        logger.debug(f"Skip extraction: {args.skip_extraction}")
        logger.debug(f"Probe selection: {enable_probe_selection}")
        logger.debug(f"Locus reports: {enable_locus_reports}")
        logger.debug(f"Coverage profiling: {enable_coverage_profiling}")
        logger.debug(f"Retry failed: {args.retry_failed or 'No'}")

        # Check if we should retry failed extractions
        if args.retry_failed:
            if not os.path.exists(args.retry_failed):
                progress.error(f"Failed files JSON not found: {args.retry_failed}")
                return 1

            progress.info(f"\nRetrying failed extractions from {args.retry_failed}")
            start_time = time.time()
            results = coordinator.retry_failed_extraction(
                failed_files_json_path=args.retry_failed,
                snp_list_path=snp_list_path,
                output_dir=output_dir,
                max_workers=args.max_workers,
                enable_probe_selection=enable_probe_selection,
                enable_locus_reports=enable_locus_reports
            )

        # Check if we should skip extraction
        elif args.skip_extraction:
            if check_extraction_results_exist(output_dir, custom_name, args.data_types):
                progress.info("Extraction results found. Skipping extraction phase...")
                logger.debug("Existing files will be used for any postprocessing")

                # Create a minimal results structure for existing files
                output_files = {}
                for data_type in args.data_types:
                    parquet_file = os.path.join(output_dir, f"{custom_name}_{data_type}.parquet")
                    output_files[f"{data_type}_parquet"] = parquet_file

                # Run probe selection on existing results if enabled
                if enable_probe_selection:
                    progress.info("  Running probe selection...")
                    probe_selection_results = coordinator.run_probe_selection_postprocessing(
                        output_dir=output_dir,
                        output_name=custom_name,
                        data_types=data_type_enums
                    )
                    if probe_selection_results:
                        output_files.update(probe_selection_results)

                # Run locus reports on existing results if enabled
                if enable_locus_reports:
                    progress.info("  Running locus reports...")
                    locus_report_results = coordinator.run_locus_report_postprocessing(
                        output_dir=output_dir,
                        output_name=custom_name,
                        data_types=data_type_enums
                    )
                    if locus_report_results:
                        output_files.update(locus_report_results)

                # Run coverage profiling on existing results if enabled
                if enable_coverage_profiling:
                    progress.info("  Running coverage profiling...")
                    coverage_results = coordinator.run_coverage_profiling_postprocessing(
                        output_dir=output_dir,
                        output_name=custom_name,
                        data_types=data_type_enums,
                        max_workers=args.max_workers
                    )
                    if coverage_results:
                        output_files.update(coverage_results)

                results = {
                    'success': True,
                    'job_id': custom_name,
                    'execution_time_seconds': 0.0,
                    'output_files': output_files,
                    'summary': {'note': 'Skipped extraction - used existing results'},
                    'skipped_extraction': True
                }
            else:
                progress.warning("Skip extraction requested but no valid results found.")
                progress.info("Running full extraction pipeline...")
                start_time = time.time()
                results = coordinator.run_full_extraction_pipeline(
                    snp_list_path=snp_list_path,
                    data_types=data_type_enums,
                    output_dir=output_dir,
                    ancestries=args.ancestries,
                    parallel=args.parallel,
                    max_workers=args.max_workers,
                    output_name=custom_name,
                    enable_probe_selection=enable_probe_selection,
                    enable_locus_reports=enable_locus_reports,
                    enable_coverage_profiling=enable_coverage_profiling
                )
        else:
            # Normal pipeline execution (will overwrite existing results)
            progress.info("\nStarting extraction pipeline...")
            start_time = time.time()
            results = coordinator.run_full_extraction_pipeline(
                snp_list_path=snp_list_path,
                data_types=data_type_enums,
                output_dir=output_dir,
                ancestries=args.ancestries,
                parallel=args.parallel,
                max_workers=args.max_workers,
                output_name=custom_name,
                enable_probe_selection=enable_probe_selection,
                enable_locus_reports=enable_locus_reports,
                enable_coverage_profiling=enable_coverage_profiling
            )

        # Calculate timing only if extraction was run
        if not results.get('skipped_extraction', False):
            end_time = time.time()
            elapsed = end_time - start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            if minutes > 0:
                progress.info(f"\nPipeline completed in {minutes}m {seconds}s")
            else:
                progress.info(f"\nPipeline completed in {seconds}s")
        else:
            progress.info("\nPostprocessing completed")

        # Analyze and report results
        success = analyze_results(results, logger, progress)

        # Show log file location
        log_path = get_log_file_path()
        if log_path:
            progress.info(f"Detailed logs: {log_path}")

        return 0 if success else 1

    except Exception as e:
        progress.error(f"Pipeline failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())