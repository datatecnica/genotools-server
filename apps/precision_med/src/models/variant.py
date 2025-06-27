"""
Variant data models for GP2 Precision Medicine Data Browser.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class VariantInfo(BaseModel):
    """Model for variant information data."""
    
    # Core variant identifiers
    id: str = Field(..., description="Variant ID (chr:pos:ref:alt format)")
    snp_name: str = Field(..., description="SNP name (protein change)")
    snp_name_alt: Optional[str] = Field(None, description="Alternative SNP name")
    locus: str = Field(..., description="Gene locus/symbol")
    rsid: Optional[str] = Field(None, description="dbSNP RS ID")
    
    # Genomic coordinates
    hg38: str = Field(..., description="hg38 coordinates")
    hg19: str = Field(..., description="hg19 coordinates")
    chrom: str = Field(..., description="Chromosome")
    pos: int = Field(..., description="Genomic position")
    
    # Alleles
    a1: str = Field(..., description="Reference allele")
    a2: str = Field(..., description="Alternative allele")
    
    # Metadata
    ancestry: Optional[str] = Field(None, description="Ancestry information")
    submitter_email: Optional[str] = Field(None, description="Submitter email")
    precision_medicine: Optional[str] = Field(None, description="Precision medicine flag")
    pipeline: Optional[str] = Field(None, description="Analysis pipeline")
    
    # Quality metrics
    alt_freqs: Optional[float] = Field(None, description="Alternative allele frequency")
    obs_ct: Optional[int] = Field(None, description="Observation count")
    f_miss: Optional[float] = Field(None, description="Missing data frequency")
    
    model_config = {
        # Allow field name case mapping (Pydantic v2 syntax)
        "validate_assignment": True,
        "extra": "ignore"
    }


class FilterCriteria(BaseModel):
    """Model for variant filtering criteria."""
    
    # Gene/locus filters
    loci: Optional[List[str]] = Field(None, description="Filter by gene loci")
    
    # Quality filters
    min_obs_ct: Optional[int] = Field(None, description="Minimum observation count")
    max_f_miss: Optional[float] = Field(None, description="Maximum missing data frequency")
    
    # Frequency filters
    min_alt_freq: Optional[float] = Field(None, description="Minimum alternative allele frequency")
    max_alt_freq: Optional[float] = Field(None, description="Maximum alternative allele frequency")
    
    # Ancestry filters
    ancestry: Optional[List[str]] = Field(None, description="Filter by ancestry")
    
    # Precision medicine filter
    precision_medicine_only: Optional[bool] = Field(False, description="Include only precision medicine variants")
    
    # Pagination
    limit: Optional[int] = Field(None, description="Maximum number of results")
    offset: Optional[int] = Field(0, description="Number of results to skip")


class VariantCarrierData(BaseModel):
    """Model for variant carrier status data."""
    
    variant_id: str = Field(..., description="Variant ID")
    sample_id: str = Field(..., description="Sample ID (IID)")
    carrier_status: Optional[float] = Field(None, description="Carrier status (0.0, 1.0, 2.0, or None for missing)")
    
    @property
    def is_carrier(self) -> bool:
        """Check if sample is a carrier (heterozygous or homozygous)."""
        return self.carrier_status is not None and self.carrier_status > 0.0
    
    @property
    def is_homozygous(self) -> bool:
        """Check if sample is homozygous for the variant."""
        return self.carrier_status == 2.0
    
    @property
    def is_heterozygous(self) -> bool:
        """Check if sample is heterozygous for the variant."""
        return self.carrier_status == 1.0
    
    @property
    def is_missing(self) -> bool:
        """Check if carrier status is missing."""
        return self.carrier_status is None 