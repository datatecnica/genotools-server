#!/usr/bin/env python3
"""
Test script for NBA harmonization/extraction pipeline with AAC ancestry.
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
    parser = argparse.ArgumentParser(description='Test NBA harmonization/extraction pipeline')
    parser.add_argument(
        '--output', 
        type=str, 
        default='/tmp/nba_aac_test/nba_aac_test',
        help='Output path prefix (e.g., /tmp/results/my_analysis) - creates files like my_analysis.traw, my_analysis_harmonization_report.json'
    )
    parser.add_argument(
        '--ancestry',
        type=str,
        default='AAC',
        help='Ancestry to process (default: AAC)'
    )
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components for cache-free mode
        settings = Settings()
        extractor = VariantExtractor(settings)  # Cache-free mode - no cache_dir parameter needed
        transformer = GenotypeTransformer()
        coordinator = ExtractionCoordinator(extractor, transformer, settings)
        
        # Test parameters from command line
        snp_list_path = settings.snp_list_path  # Uses default precision med SNP list
        
        # Parse output path to extract directory and basename
        output_dir = os.path.dirname(args.output)
        custom_name = os.path.basename(args.output)
        ancestry = args.ancestry
        
        # Ensure output directory exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            # If no directory specified, use current directory
            output_dir = "."
        
        logger.info(f"Testing NBA pipeline with {ancestry} ancestry")
        logger.info(f"SNP list: {snp_list_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Output basename: {custom_name}")
        logger.info(f"Full output path: {args.output}")
        
        # Run pipeline
        results = coordinator.run_full_extraction_pipeline(
            snp_list_path=snp_list_path,
            data_types=[DataType.NBA],
            output_dir=output_dir,
            ancestries=[ancestry],
            output_formats=['traw', 'parquet'],
            parallel=False,  # Disable for debugging
            max_workers=1,
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