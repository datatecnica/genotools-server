"""
Simplified Carriers API with unified pipeline architecture.
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import asyncio
from pathlib import Path

from src.config.settings import settings
from src.config.paths import PathConfig
from src.models.carrier import ProcessingRequest, HarmonizationMapping
from src.pipeline.base import Pipeline, PipelineContext
from src.pipeline.extractor import VariantExtractionStage
from src.pipeline.transformer import CarrierTransformationStage
from src.pipeline.combiner import CombinerStage
from src.dataset.base import DatasetConfig
from src.dataset.nba import NBADatasetHandler
from src.dataset.wgs import WGSDatasetHandler
from src.dataset.imputed import ImputedDatasetHandler
from src.storage.file_storage import FileStorageRepository
from src.storage.cloud_storage import CloudStorageRepository
from src.harmonization import HarmonizationService, HarmonizationCache
from src.core.security import get_api_key

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Carriers API",
    description="Unified API for processing genetic carrier information",
    version="2.0.0"
)

# Initialize storage based on settings
if settings.storage_type == "gcs":
    storage = CloudStorageRepository(settings.gcs_mount_path)
else:
    storage = FileStorageRepository(settings.local_storage_path)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info(f"Starting Carriers API with storage type: {settings.storage_type}")
    logger.info(f"Settings: {settings.to_dict()}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "storage_type": settings.storage_type
    }


class ProcessRequest(BaseModel):
    """Unified request model for all dataset types."""
    dataset_type: str  # 'nba', 'wgs', or 'imputed'
    input_path: str
    snplist_path: str
    output_path: str
    release: str = "10"
    ancestry: Optional[str] = None  # For imputed data
    ancestries: Optional[List[str]] = None  # For NBA multi-ancestry
    chromosomes: Optional[List[str]] = None  # For imputed chromosomes
    use_cache: bool = True


class HarmonizationRequest(BaseModel):
    """Request for precomputing harmonization mappings."""
    dataset_type: str
    release: str
    snplist_path: str
    force_update: bool = False  # Force recomputation even if cache exists


@app.post("/process")
async def process_carriers(
    request: ProcessRequest,
    api_key: str = Depends(get_api_key) if settings.require_api_key else None
):
    """
    Unified endpoint for processing all dataset types.
    
    This endpoint automatically detects the dataset structure and applies
    the appropriate processing pipeline.
    """
    try:
        # Create dataset configuration
        dataset_config = DatasetConfig(
            dataset_type=request.dataset_type,
            release=request.release,
            base_path=request.input_path,
            ancestry=request.ancestry
        )
        
        # Get appropriate dataset handler
        handler = _get_dataset_handler(dataset_config, request)
        
        # Validate inputs
        if not handler.validate_inputs():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid input files for {request.dataset_type} dataset"
            )
        
        # Create pipeline
        pipeline = await _create_pipeline(storage, request.use_cache)
        
        # Process based on dataset type
        if handler.should_combine_results():
            # Process with combination (NBA or Imputed)
            results = await _process_with_combination(
                handler, pipeline, request, dataset_config
            )
        else:
            # Process single file (WGS)
            results = await _process_single(
                handler, pipeline, request, dataset_config
            )
        
        return {
            "status": "success",
            "dataset_type": request.dataset_type,
            "release": request.release,
            "outputs": results
        }
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}"
        )


@app.post("/harmonization/precompute")
async def precompute_harmonization(
    request: HarmonizationRequest,
    api_key: str = Depends(get_api_key) if settings.require_api_key else None
):
    """
    Precompute harmonization mappings for faster processing.
    
    This endpoint builds a cache of allele harmonization mappings that can
    be reused across multiple processing runs.
    """
    try:
        # Get path configuration
        path_config = PathConfig.from_settings(request.release)
        
        # Initialize harmonization cache
        cache = HarmonizationCache(
            storage,
            path_config.harmonization_cache
        )
        
        # Check existing cache
        if not request.force_update:
            existing = await cache.get_full_mapping(
                request.dataset_type,
                request.release
            )
            if not existing.empty:
                return {
                    "status": "exists",
                    "message": "Harmonization cache already exists",
                    "cache_stats": await cache.get_cache_stats()
                }
        
        # Create dataset config
        dataset_config = DatasetConfig(
            dataset_type=request.dataset_type,
            release=request.release,
            base_path=path_config.get_dataset_path(request.dataset_type)
        )
        
        # Get handler to enumerate files
        handler = _get_dataset_handler(dataset_config, None)
        genotype_paths = handler.get_genotype_paths()
        
        # Build harmonization mappings
        mappings = []
        harmonizer = HarmonizationService(storage, cache)
        
        for geno_path, metadata in genotype_paths:
            logger.info(f"Processing harmonization for {geno_path}")
            
            # Find common variants and build mapping
            # This is a simplified version - actual implementation would
            # extract the mapping details from harmonization process
            new_mappings = await _build_harmonization_mappings(
                harmonizer, geno_path, request.snplist_path, metadata
            )
            mappings.extend(new_mappings)
        
        # Update cache
        await cache.update_mapping(
            request.dataset_type,
            request.release,
            mappings
        )
        
        return {
            "status": "success",
            "message": f"Built harmonization cache for {len(genotype_paths)} files",
            "mappings_count": len(mappings),
            "cache_stats": await cache.get_cache_stats()
        }
        
    except Exception as e:
        logger.error(f"Harmonization precompute failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Harmonization precompute failed: {str(e)}"
        )


def _get_dataset_handler(config: DatasetConfig, request: Optional[ProcessRequest]):
    """Get appropriate dataset handler based on type."""
    if config.dataset_type == "nba":
        ancestries = request.ancestries if request else None
        return NBADatasetHandler(config, ancestries)
    elif config.dataset_type == "wgs":
        return WGSDatasetHandler(config)
    elif config.dataset_type == "imputed":
        chromosomes = request.chromosomes if request else None
        return ImputedDatasetHandler(config, chromosomes)
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")


async def _create_pipeline(storage, use_cache: bool) -> Pipeline:
    """Create processing pipeline with all stages."""
    # Initialize harmonization components
    cache_dir = settings.get_harmonization_cache_dir()
    cache = HarmonizationCache(storage, cache_dir)
    harmonizer = HarmonizationService(storage, cache)
    
    # Create pipeline stages
    stages = [
        VariantExtractionStage(harmonizer, storage),
        CarrierTransformationStage(storage),
    ]
    
    return Pipeline(stages)


async def _process_single(handler, pipeline, request, dataset_config):
    """Process a single genotype file (WGS)."""
    paths = handler.get_genotype_paths()
    
    if not paths:
        raise ValueError("No genotype files found")
    
    geno_path, metadata = paths[0]
    
    # Create pipeline context
    context = PipelineContext(
        dataset_type=request.dataset_type,
        release=request.release,
        output_path=request.output_path,
        **metadata
    )
    
    # Run pipeline
    inputs = {
        'geno_path': geno_path,
        'snplist_path': request.snplist_path
    }
    
    result = await pipeline.run(inputs, context)
    
    return result.to_dict()


async def _process_with_combination(handler, pipeline, request, dataset_config):
    """Process multiple files with combination (NBA or Imputed)."""
    paths = handler.get_genotype_paths()
    
    if not paths:
        raise ValueError("No genotype files found")
    
    # Process each file
    results = []
    for geno_path, metadata in paths:
        # Determine output path for individual result
        if request.dataset_type == "nba":
            output_path = handler.get_output_path(
                request.output_path,
                ancestry=metadata.get('ancestry')
            )
        else:  # imputed
            output_path = handler.get_output_path(
                request.output_path,
                chromosome=metadata.get('chromosome')
            )
        
        # Create context
        context = PipelineContext(
            dataset_type=request.dataset_type,
            release=request.release,
            output_path=output_path,
            **metadata
        )
        
        # Run pipeline
        inputs = {
            'geno_path': geno_path,
            'snplist_path': request.snplist_path
        }
        
        result = await pipeline.run(inputs, context)
        results.append(result)
    
    # Combine results if needed
    if len(results) > 1:
        # Create combiner stage
        key_file = None  # Would be loaded from config for NBA
        combiner = CombinerStage(storage, key_file)
        
        # Create combination context
        combine_context = PipelineContext(
            dataset_type=request.dataset_type,
            release=request.release,
            output_path=handler.get_output_path(request.output_path, combined=True),
            combine_type=handler.get_combine_strategy(),
            ancestry_labels=[p[1].get('ancestry') for p in paths] if request.dataset_type == "nba" else None
        )
        
        # Run combination
        combined_result = await combiner.process(results, combine_context)
        
        return {
            "individual": [r.to_dict() for r in results],
            "combined": combined_result.to_dict()
        }
    
    return results[0].to_dict() if results else {}


async def _build_harmonization_mappings(harmonizer, geno_path, snplist_path, metadata):
    """Build harmonization mappings for caching."""
    # This is a placeholder - actual implementation would extract
    # mapping details from the harmonization process
    mappings = []
    
    # Would run harmonization and extract the mapping information
    # For now, return empty list
    
    return mappings


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_refactored:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers
    )
