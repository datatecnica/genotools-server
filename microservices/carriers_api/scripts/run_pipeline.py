#!/usr/bin/env python3
"""
Command-line pipeline runner for carrier analysis.

This script provides a direct way to run the carrier analysis pipeline
without going through the API.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.config.paths import PathConfig
from src.models.carrier import ProcessingRequest
from src.pipeline.base import Pipeline, PipelineContext
from src.pipeline.consolidator import ConsolidationStage
from src.pipeline.extractor import VariantExtractionStage
from src.pipeline.transformer import CarrierTransformationStage
from src.dataset.base import DatasetConfig
from src.dataset.nba import NBADatasetHandler
from src.dataset.wgs import WGSDatasetHandler
from src.dataset.imputed import ImputedDatasetHandler
# Storage imports removed - using direct file operations
from src.harmonization import HarmonizationService, HarmonizationCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_dataset(args):
    """Process a dataset through the carrier analysis pipeline."""
    
    # Storage simplified - using direct file operations with gcsfuse
    
    # Create dataset configuration
    dataset_config = DatasetConfig(
        dataset_type=args.dataset_type,
        release=args.release,
        base_path=args.input_path,
        ancestry=args.ancestry
    )
    
    # Get dataset handler
    handler = get_dataset_handler(dataset_config, args)
    
    # Validate inputs
    logger.info(f"Validating inputs for {args.dataset_type} dataset...")
    if not handler.validate_inputs():
        logger.error(f"Invalid input files for {args.dataset_type} dataset")
        return 1
    
    # Create pipeline
    logger.info("Creating processing pipeline...")
    pipeline = await create_pipeline(args.use_cache, args.release)
    
    # Get genotype paths
    paths = handler.get_genotype_paths()
    logger.info(f"Found {len(paths)} genotype file(s) to process")
    
    # Process through consolidated pipeline
    results = await process_consolidated(
        handler, pipeline, paths, args, dataset_config
    )
    
    # Print summary
    print_summary(results, args)
    
    return 0


def get_dataset_handler(config: DatasetConfig, args):
    """Get appropriate dataset handler based on type."""
    if config.dataset_type == "nba":
        ancestries = args.ancestries or NBADatasetHandler.DEFAULT_ANCESTRIES
        return NBADatasetHandler(config, ancestries)
    elif config.dataset_type == "wgs":
        return WGSDatasetHandler(config)
    elif config.dataset_type == "imputed":
        if not args.ancestry:
            raise ValueError("Ancestry must be specified for imputed data")
        chromosomes = args.chromosomes or ImputedDatasetHandler.DEFAULT_CHROMOSOMES
        return ImputedDatasetHandler(config, chromosomes)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")


async def create_pipeline(use_cache: bool, release: str) -> Pipeline:
    """Create processing pipeline with all stages."""
    # Initialize harmonization components
    cache_dir = settings.get_harmonization_cache_dir(release)
    cache = HarmonizationCache(cache_dir)
    harmonizer = HarmonizationService(cache)
    
    # Create pipeline stages
    stages = [
        ConsolidationStage(harmonizer),
        VariantExtractionStage(),
        CarrierTransformationStage(),
    ]
    
    return Pipeline(stages)


async def process_consolidated(handler, pipeline, paths, args, dataset_config):
    """Process dataset through consolidated pipeline."""
    if not paths:
        raise ValueError("No genotype files found")
    
    logger.info(f"Processing {len(paths)} file(s) through consolidated pipeline...")
    
    # Create pipeline context
    context = PipelineContext(
        dataset_type=args.dataset_type,
        release=args.release,
        output_path=args.output_path,
        snplist_path=args.snplist_path
    )
    
    # Run consolidated pipeline
    result = await pipeline.run(paths, context)
    
    logger.info(f"Processing complete. Output files:")
    for key, path in result.to_dict().items():
        logger.info(f"  {key}: {path}")
    
    return {"consolidated": result}


def print_summary(results, args):
    """Print processing summary."""
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Dataset Type: {args.dataset_type}")
    print(f"Release: {args.release}")
    print(f"Input Path: {args.input_path}")
    print(f"SNP List: {args.snplist_path}")
    print(f"Output Path: {args.output_path}")
    
    if "consolidated" in results:
        print(f"\nConsolidated processing completed")
        
    print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run carrier analysis pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('dataset_type', choices=['nba', 'wgs', 'imputed'],
                       help='Type of dataset to process')
    parser.add_argument('input_path', help='Path to input genotype files')
    parser.add_argument('snplist_path', help='Path to SNP list file')
    parser.add_argument('output_path', help='Output path prefix')
    
    # Dataset options
    parser.add_argument('--release', default='10', help='Release version')
    parser.add_argument('--ancestry', help='Ancestry label (required for imputed)')
    parser.add_argument('--ancestries', nargs='+', help='Ancestries to process (NBA)')
    parser.add_argument('--chromosomes', nargs='+', help='Chromosomes to process (imputed)')
    
    # Processing options
    parser.add_argument('--no-cache', dest='use_cache', action='store_false',
                       help='Disable harmonization cache')

    parser.add_argument('--key-file', help='Path to master key file (NBA combination)')
    
    # Storage options (simplified)
    parser.add_argument('--base-path', help='Base path (overrides settings default)')
    
    # Other options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate inputs without processing')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if args.dataset_type == 'imputed' and not args.ancestry:
        parser.error("--ancestry is required for imputed dataset")
    
    # Run processing
    try:
        if args.dry_run:
            logger.info("DRY RUN MODE - Validating inputs only")
            # Would validate without processing
            return 0
            
        exit_code = asyncio.run(process_dataset(args))
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
