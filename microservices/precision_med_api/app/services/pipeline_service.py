"""
Pipeline execution service - business logic extracted from CLI script.

Coordinates full carriers pipeline execution including extraction,
probe selection, and locus report generation.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..core.config import Settings
from ..processing.coordinator import ExtractionCoordinator
from ..processing.extractor import VariantExtractor
from ..processing.transformer import GenotypeTransformer
from ..models.analysis import DataType
from ..models.api import PipelineRequest, PipelineResponse, JobStatus

logger = logging.getLogger(__name__)


def print_system_info():
    """Print system information for performance context."""
    import psutil

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


def analyze_results(results: dict, logger_instance) -> bool:
    """Analyze and report pipeline results."""
    logger_instance.info("\n=== Pipeline Results Analysis ===")

    if results['success']:
        logger_instance.info("✅ Pipeline completed successfully!")
        logger_instance.info(f"📋 Job ID: {results['job_id']}")
        logger_instance.info(f"⏱️  Execution time: {results['execution_time_seconds']:.1f}s")

        # Analyze output files by data type
        output_files = results['output_files']
        data_type_files = {}

        for file_key, file_path in output_files.items():
            if '_' in file_key:
                data_type = file_key.split('_')[0]
                if data_type not in data_type_files:
                    data_type_files[data_type] = []
                data_type_files[data_type].append(file_key)

        logger_instance.info(f"📁 Generated files for {len(data_type_files)} data types:")
        for data_type, files in data_type_files.items():
            logger_instance.info(f"   {data_type}: {len(files)} files ({', '.join(files)})")

        # Summary information
        if 'summary' in results and results['summary']:
            summary = results['summary']
            logger_instance.info(f"🧬 Total variants: {summary.get('total_variants', 0)}")
            logger_instance.info(f"👥 Total samples: {summary.get('total_samples', 0)}")

            # Data type breakdown if available
            if 'by_data_type' in summary:
                logger_instance.info("📊 Variants by data type:")
                for data_type, count in summary['by_data_type'].items():
                    logger_instance.info(f"   {data_type}: {count} variants")

            # Ancestry breakdown if available
            if 'by_ancestry' in summary:
                logger_instance.info("🌍 Variants by ancestry:")
                for ancestry, count in summary['by_ancestry'].items():
                    logger_instance.info(f"   {ancestry}: {count} variants")

        return True

    else:
        logger_instance.error("❌ Pipeline failed!")
        for error in results['errors']:
            logger_instance.error(f"   Error: {error}")
        return False


class PipelineService:
    """
    Service for executing carriers pipeline with full configuration support.

    Extracted from CLI script to enable API access while maintaining
    exact same functionality.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def execute_pipeline(self, request: PipelineRequest) -> Dict[str, Any]:
        """
        Execute carriers pipeline with given configuration.

        Args:
            request: Pipeline execution parameters

        Returns:
            Dictionary with execution results (matches CLI script format)
        """
        try:
            print_system_info()

            # Initialize settings with optimization
            if request.optimize:
                self.logger.info("🚀 Using auto-optimized performance settings")
                settings = Settings.create_optimized()
            else:
                self.logger.info("📊 Using default settings")
                settings = Settings()

            self.logger.info(
                f"⚙️  Performance settings: {settings.max_workers} workers, "
                f"{settings.chunk_size} chunk_size, {settings.process_cap} process_cap"
            )

            # Initialize components
            extractor = VariantExtractor(settings)
            transformer = GenotypeTransformer()
            coordinator = ExtractionCoordinator(extractor, transformer, settings)

            # Handle ancestries - use all from settings if not specified
            ancestries = request.ancestries if request.ancestries else settings.ANCESTRIES

            # Handle output path - use config-based results directory if not specified
            if request.output_dir is None:
                output_dir = settings.results_path
                custom_name = request.job_name
                full_output_path = os.path.join(output_dir, custom_name)
            else:
                output_dir = os.path.dirname(request.output_dir)
                custom_name = os.path.basename(request.output_dir)
                full_output_path = request.output_dir
                if not output_dir:
                    output_dir = "."

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Use default SNP list
            self.logger.info("📋 Using default precision medicine SNP list")
            snp_list_path = settings.snp_list_path

            # Convert data type strings to enum
            data_type_enums = [DataType[dt] for dt in request.data_types]

            # Handle probe selection logic - enabled by default unless skipped
            enable_probe_selection = not request.skip_probe_selection

            # Handle locus reports logic - enabled by default unless skipped
            enable_locus_reports = not request.skip_locus_reports

            self.logger.info("=== Pipeline Configuration ===")
            self.logger.info(f"📊 Data types: {request.data_types}")
            self.logger.info(f"🌍 Ancestries ({len(ancestries)}): {ancestries}")
            self.logger.info(f"📋 SNP list: {snp_list_path}")
            self.logger.info(f"📁 Output directory: {output_dir}")
            self.logger.info(f"📝 Job name: {custom_name}")
            self.logger.info(f"🎯 Full output path: {full_output_path}")
            self.logger.info(f"⚡ Parallel: {request.parallel}")
            self.logger.info(f"👥 Max workers: {request.max_workers or 'auto-detect'}")
            self.logger.info(f"🔧 Using {'config-based' if request.output_dir is None else 'custom'} output location")
            self.logger.info(f"📋 Skip extraction: {request.skip_extraction}")
            self.logger.info(f"🔬 Probe selection: {enable_probe_selection}")
            self.logger.info(f"📊 Locus reports: {enable_locus_reports}")

            # Check if we should skip extraction
            if request.skip_extraction:
                if check_extraction_results_exist(output_dir, custom_name, request.data_types):
                    self.logger.info("✅ Extraction results found. Skipping extraction phase...")
                    self.logger.info("📁 Existing files will be used for any postprocessing")

                    # Create a minimal results structure for existing files
                    output_files = {}
                    for data_type in request.data_types:
                        parquet_file = os.path.join(output_dir, f"{custom_name}_{data_type}.parquet")
                        output_files[f"{data_type}_parquet"] = parquet_file

                    # Run probe selection on existing results if enabled
                    if enable_probe_selection:
                        self.logger.info("🔬 Running probe selection analysis on existing results...")
                        probe_selection_results = coordinator.run_probe_selection_postprocessing(
                            output_dir=output_dir,
                            output_name=custom_name,
                            data_types=data_type_enums
                        )
                        if probe_selection_results:
                            output_files.update(probe_selection_results)

                    # Run locus reports on existing results if enabled
                    if enable_locus_reports:
                        self.logger.info("📊 Running locus report generation on existing results...")
                        locus_report_results = coordinator.run_locus_report_postprocessing(
                            output_dir=output_dir,
                            output_name=custom_name,
                            data_types=data_type_enums
                        )
                        if locus_report_results:
                            output_files.update(locus_report_results)

                    results = {
                        'success': True,
                        'job_id': custom_name,
                        'execution_time_seconds': 0.0,
                        'output_files': output_files,
                        'summary': {'note': 'Skipped extraction - used existing results'},
                        'skipped_extraction': True
                    }
                else:
                    self.logger.warning("⚠️ Skip extraction requested but no valid results found.")
                    self.logger.info("🚀 Running full extraction pipeline...")
                    start_time = time.time()
                    results = coordinator.run_full_extraction_pipeline(
                        snp_list_path=snp_list_path,
                        data_types=data_type_enums,
                        output_dir=output_dir,
                        ancestries=ancestries,
                        parallel=request.parallel,
                        max_workers=request.max_workers,
                        output_name=custom_name,
                        enable_probe_selection=enable_probe_selection,
                        enable_locus_reports=enable_locus_reports
                    )
                    results['execution_time_seconds'] = time.time() - start_time
            else:
                # Normal pipeline execution (will overwrite existing results)
                self.logger.info("\n🚀 Starting carriers pipeline extraction...")
                start_time = time.time()
                results = coordinator.run_full_extraction_pipeline(
                    snp_list_path=snp_list_path,
                    data_types=data_type_enums,
                    output_dir=output_dir,
                    ancestries=ancestries,
                    parallel=request.parallel,
                    max_workers=request.max_workers,
                    output_name=custom_name,
                    enable_probe_selection=enable_probe_selection,
                    enable_locus_reports=enable_locus_reports
                )
                results['execution_time_seconds'] = time.time() - start_time

            # Calculate timing only if extraction was run
            if not results.get('skipped_extraction', False):
                self.logger.info(f"⏱️  Total pipeline time: {results['execution_time_seconds']:.1f}s")
            else:
                self.logger.info("⏱️  Extraction skipped - no timing measured")

            # Analyze and report results
            success = analyze_results(results, self.logger)

            if success:
                if results.get('skipped_extraction', False):
                    self.logger.info("\n🎯 Carriers Pipeline Complete (Extraction Skipped)!")
                    self.logger.info("   📁 Used existing NBA/WGS/IMPUTED datasets")
                    self.logger.info("   🧬 Genotype data ready for carrier analysis")
                    self.logger.info("   💡 Use without --skip-extraction to regenerate extraction data")
                else:
                    self.logger.info("\n🎯 Carriers Pipeline Complete!")
                    self.logger.info("   📁 Generated separate NBA/WGS/IMPUTED datasets")
                    self.logger.info("   🧬 Genotype data ready for carrier analysis")
                    self.logger.info("   📊 Quality reports and harmonization summaries available")

            return results

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())

            return {
                'success': False,
                'job_id': request.job_name,
                'execution_time_seconds': 0.0,
                'output_files': {},
                'summary': {},
                'errors': [str(e), traceback.format_exc()]
            }
