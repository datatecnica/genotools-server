"""
FastAPI dependencies for repository injection.
"""
import sys
from pathlib import Path
from functools import lru_cache

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ..repositories import VariantRepository, ClinicalRepository, SampleRepository


@lru_cache()
def get_variant_repository() -> VariantRepository:
    """Get a cached instance of the variant repository."""
    return VariantRepository(data_source="WGS")  # Default to WGS for now


@lru_cache()
def get_clinical_repository() -> ClinicalRepository:
    """Get a cached instance of the clinical repository."""
    return ClinicalRepository()


@lru_cache()
def get_sample_repository() -> SampleRepository:
    """Get a cached instance of the sample repository."""
    return SampleRepository() 