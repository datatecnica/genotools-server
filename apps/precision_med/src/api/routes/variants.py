"""
Variant-related API endpoints.
"""
import sys
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ...models.variant import VariantInfo, FilterCriteria
from ...repositories import VariantRepository
from ..dependencies import get_variant_repository

router = APIRouter(prefix="/variants", tags=["variants"])


@router.get("/", response_model=List[VariantInfo])
async def list_variants(
    limit: Optional[int] = Query(default=10, ge=1, le=100, description="Number of variants to return"),
    locus: Optional[str] = Query(default=None, description="Filter by gene/locus (e.g., 'GBA1')"),
    variant_repo: VariantRepository = Depends(get_variant_repository)
) -> List[VariantInfo]:
    """
    Get a list of variants with optional filtering.
    
    - **limit**: Number of variants to return (1-100)
    - **locus**: Optional gene/locus filter (e.g., 'GBA1', 'LRRK2')
    """
    try:
        # Ensure repository is loaded
        if not variant_repo.is_loaded:
            variant_repo.load()
        
        # Create filter criteria
        filters = FilterCriteria(
            loci=[locus] if locus else None,
            limit=limit
        )
        
        # Get filtered variants
        variants = variant_repo.filter(filters)
        
        return variants
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving variants: {str(e)}"
        )


@router.get("/loci")
async def get_available_loci(
    variant_repo: VariantRepository = Depends(get_variant_repository)
) -> List[str]:
    """
    Get a list of all available gene/loci in the dataset.
    """
    try:
        # Ensure repository is loaded
        if not variant_repo.is_loaded:
            variant_repo.load()
        
        loci = variant_repo.get_loci()
        return sorted(loci)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving loci: {str(e)}"
        ) 