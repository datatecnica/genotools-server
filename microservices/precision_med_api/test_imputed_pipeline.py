#!/usr/bin/env python3
"""
Test script for IMPUTED harmonization/extraction pipeline with AAC and AFR ancestries.
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import Settings
from app.processing.coordinator import ExtractionCoordinator
from app.processing.extractor import VariantExtractor
from app.processing.transformer import GenotypeTransformer
from app.models.analysis import DataType

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test IMPUTED harmonization/extraction pipeline')
    parser.add_argument(
        '--output', 
        type=str, 
        default='/tmp/imputed_processpool_test/imputed_aac_afr_test',
        help='Output path prefix (e.g., /tmp/results/my_analysis) - creates files like my_analysis.traw, my_analysis_harmonization_report.json'
    )
    parser.add_argument(
        '--ancestries',
        type=str,
        nargs='+',
        default=['AAC', 'AFR'],
        help='Ancestries to process (default: AAC AFR)'
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
   
        settings = Settings()
        extractor = VariantExtractor(settings)
        transformer = GenotypeTransformer()
        coordinator = ExtractionCoordinator(extractor, transformer, settings)
        
        # Test parameters from command line
        snp_list_path = settings.snp_list_path  # Uses default precision med SNP list
        
        # Parse output path to extract directory and basename
        output_dir = os.path.dirname(args.output)
        custom_name = os.path.basename(args.output)
        ancestries = args.ancestries
        
        # Ensure output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            # If no directory specified, use current directory
            output_dir = "."
        
        logger.info(f"Testing IMPUTED pipeline with ancestries: {ancestries}")
        logger.info(f"SNP list: {snp_list_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output basename: {custom_name}")
        logger.info(f"Full output path: {args.output}")
        logger.info("ProcessPool parallelization ENABLED for testing")
        
        # Run pipeline with ProcessPool enabled (AAC ancestry only for testing)
        results = coordinator.run_full_extraction_pipeline(
            snp_list_path=snp_list_path,
            data_types=[DataType.IMPUTED],
            output_dir=output_dir,
            ancestries=['AAC'],  # Test with just AAC for speed
            output_formats=['traw', 'parquet'],
            parallel=True,  # Enable ProcessPool parallelization
            max_workers=3,  # Use 3 workers for testing
            output_name=custom_name  # Use custom name instead of auto-generated
        )
        
        # Print results
        if results['success']:
            logger.info("Pipeline completed successfully!")
            logger.info(f"Job ID: {results['job_id']}")
            logger.info(f"Execution time: {results['execution_time_seconds']:.1f}s")
            logger.info(f"Output files: {list(results['output_files'].keys())}")
            
            if 'summary' in results:
                summary = results['summary']
                logger.info(f"Total variants: {summary.get('total_variants', 0)}")
                logger.info(f"Total samples (actual): {summary.get('total_samples', 0)}")
        else:
            logger.error("Pipeline failed!")
            for error in results['errors']:
                logger.error(f"Error: {error}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()