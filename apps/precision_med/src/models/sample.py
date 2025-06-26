"""
Sample data models for GP2 Precision Medicine Data Browser.
"""
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from .clinical import ClinicalMetadata
from .variant import VariantCarrierData


class SampleCarrier(BaseModel):
    """Model for sample-level carrier data combined with clinical metadata."""
    
    # Core identifier
    sample_id: str = Field(..., description="Sample ID (IID from carrier files)")
    
    # Clinical metadata (optional - may not be available for all samples)
    clinical: Optional[ClinicalMetadata] = Field(None, description="Clinical metadata")
    
    # Carrier data for variants
    carriers: Dict[str, Optional[float]] = Field(default_factory=dict, description="Carrier status by variant ID")
    
    # Data source information
    data_source: str = Field(..., description="Primary data source (NBA or WGS)")
    
    @property
    def has_clinical_data(self) -> bool:
        """Check if sample has associated clinical data."""
        return self.clinical is not None
    
    @property
    def ancestry_label(self) -> Optional[str]:
        """Get ancestry label from clinical data."""
        if not self.clinical:
            return None
        return self.clinical.primary_ancestry_label
    
    @property
    def study(self) -> Optional[str]:
        """Get study name from clinical data."""
        if not self.clinical:
            return None
        return self.clinical.study
    
    @property
    def diagnosis(self) -> Optional[str]:
        """Get diagnosis from clinical data."""
        if not self.clinical:
            return None
        return self.clinical.diagnosis
    
    def get_carrier_status(self, variant_id: str) -> Optional[float]:
        """Get carrier status for a specific variant."""
        return self.carriers.get(variant_id)
    
    def is_carrier(self, variant_id: str) -> bool:
        """Check if sample is a carrier for a specific variant."""
        status = self.get_carrier_status(variant_id)
        return status is not None and status > 0.0
    
    def is_homozygous(self, variant_id: str) -> bool:
        """Check if sample is homozygous for a specific variant."""
        return self.get_carrier_status(variant_id) == 2.0
    
    def is_heterozygous(self, variant_id: str) -> bool:
        """Check if sample is heterozygous for a specific variant."""
        return self.get_carrier_status(variant_id) == 1.0
    
    def get_carried_variants(self) -> List[str]:
        """Get list of variant IDs where sample is a carrier."""
        return [
            variant_id for variant_id, status in self.carriers.items()
            if status is not None and status > 0.0
        ]
    
    def get_carrier_count(self) -> int:
        """Get total number of variants carried by this sample."""
        return len(self.get_carried_variants())


class SampleFilterCriteria(BaseModel):
    """Model for sample filtering criteria."""
    
    # Data source filters
    data_sources: Optional[List[str]] = Field(None, description="Filter by data source (NBA, WGS)")
    
    # Clinical filters (delegated to clinical filter criteria)
    studies: Optional[List[str]] = Field(None, description="Filter by study names")
    ancestry_labels: Optional[List[str]] = Field(None, description="Filter by ancestry labels")
    diagnoses: Optional[List[str]] = Field(None, description="Filter by diagnosis")
    biological_sex: Optional[List[str]] = Field(None, description="Filter by biological sex")
    
    # Carrier status filters
    carried_variants: Optional[List[str]] = Field(None, description="Filter by carried variant IDs")
    min_carrier_count: Optional[int] = Field(None, description="Minimum number of variants carried")
    max_carrier_count: Optional[int] = Field(None, description="Maximum number of variants carried")
    
    # Clinical data availability
    has_clinical_data: Optional[bool] = Field(None, description="Filter by clinical data availability")
    gdpr_compliant_only: Optional[bool] = Field(None, description="Include only GDPR compliant samples")
    
    # Pagination
    limit: Optional[int] = Field(None, description="Maximum number of results")
    offset: Optional[int] = Field(0, description="Number of results to skip")


class SampleCarrierSummary(BaseModel):
    """Summary statistics for sample carrier data."""
    
    total_samples: int = Field(..., description="Total number of samples")
    samples_with_clinical: int = Field(..., description="Samples with clinical data")
    samples_by_source: Dict[str, int] = Field(..., description="Sample counts by data source")
    samples_by_ancestry: Dict[str, int] = Field(..., description="Sample counts by ancestry")
    samples_by_study: Dict[str, int] = Field(..., description="Sample counts by study")
    avg_variants_per_sample: float = Field(..., description="Average variants carried per sample")
    total_carrier_observations: int = Field(..., description="Total carrier observations across all samples") 