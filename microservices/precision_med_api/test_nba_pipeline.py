#!/usr/bin/env python3
"""
Test script for NBA harmonization/extraction pipeline with AAC ancestry.
"""

import sys
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import Settings
from app.processing.coordinator import ExtractionCoordinator
from app.processing.extractor import VariantExtractor
from app.processing.transformer import GenotypeTransformer
from app.models.analysis import DataType

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        settings = Settings()
        cache_dir = settings.get_cache_path()
        extractor = VariantExtractor(cache_dir, settings)
        transformer = GenotypeTransformer()
        coordinator = ExtractionCoordinator(extractor, transformer, settings)
        
        # Test parameters
        snp_list_path = settings.snp_list_path  # Uses default precision med SNP list
        output_dir = "/tmp/nba_aac_test"
        ancestry = "AAC"
        
        logger.info(f"Testing NBA pipeline with {ancestry} ancestry")
        logger.info(f"SNP list: {snp_list_path}")
        logger.info(f"Output: {output_dir}")
        
        # Run pipeline
        results = coordinator.run_full_extraction_pipeline(
            snp_list_path=snp_list_path,
            data_types=[DataType.NBA],
            output_dir=output_dir,
            ancestries=[ancestry],
            output_formats=['traw', 'parquet'],
            parallel=False,  # Disable for debugging
            max_workers=1
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
                logger.info(f"Total samples: {summary.get('total_samples', 0)}")
        else:
            logger.error("Pipeline failed!")
            for error in results['errors']:
                logger.error(f"Error: {error}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    main()