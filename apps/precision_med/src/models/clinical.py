"""
Clinical data models for GP2 Precision Medicine Data Browser.
"""
from typing import Optional, List
from pydantic import BaseModel, Field


class ClinicalMetadata(BaseModel):
    """Model for clinical metadata from master key."""
    
    # Core identifiers
    gp2_id: str = Field(..., description="GP2 sample ID", alias="GP2ID")
    study: str = Field(..., description="Study name")
    
    # Data availability flags
    nba: int = Field(..., description="NBA data available (0/1)")
    wgs: int = Field(..., description="WGS data available (0/1)")
    clinical_exome: int = Field(..., description="Clinical exome data available (0/1)")
    extended_clinical_data: int = Field(..., description="Extended clinical data available (0/1)")
    gdpr: int = Field(..., description="GDPR compliance flag (0/1)", alias="GDPR")
    
    # Data quality info
    nba_prune_reason: Optional[str] = Field(None, description="NBA pruning reason")
    nba_related: Optional[int] = Field(None, description="NBA related flag")
    nba_label: Optional[str] = Field(None, description="NBA ancestry label")
    wgs_prune_reason: Optional[str] = Field(None, description="WGS pruning reason")
    wgs_label: Optional[str] = Field(None, description="WGS ancestry label")
    
    # Study characteristics
    study_arm: Optional[str] = Field(None, description="Study arm")
    study_type: Optional[str] = Field(None, description="Study type")
    diagnosis: Optional[str] = Field(None, description="Primary diagnosis")
    baseline_gp2_phenotype_for_qc: Optional[str] = Field(None, description="Baseline phenotype for QC")
    baseline_gp2_phenotype: Optional[str] = Field(None, description="Baseline phenotype")
    
    # Demographics
    biological_sex_for_qc: Optional[str] = Field(None, description="Biological sex for QC")
    age_at_sample_collection: Optional[float] = Field(None, description="Age at sample collection")
    age_of_onset: Optional[float] = Field(None, description="Age of onset")
    
    @property
    def has_nba_data(self) -> bool:
        """Check if sample has NBA data available."""
        return self.nba == 1
    
    @property
    def has_wgs_data(self) -> bool:
        """Check if sample has WGS data available."""
        return self.wgs == 1
    
    @property
    def primary_ancestry_label(self) -> Optional[str]:
        """Get the primary ancestry label (prioritize WGS over NBA)."""
        if self.wgs_label:
            return self.wgs_label
        return self.nba_label
    
    @property
    def data_source(self) -> str:
        """Get the primary data source for this sample."""
        if self.has_wgs_data:
            return "WGS"
        elif self.has_nba_data:
            return "NBA"
        else:
            return "None"
    
    class Config:
        allow_population_by_field_name = True


class ClinicalFilterCriteria(BaseModel):
    """Model for clinical data filtering criteria."""
    
    # Study filters
    studies: Optional[List[str]] = Field(None, description="Filter by study names")
    
    # Data availability filters
    has_nba: Optional[bool] = Field(None, description="Filter by NBA data availability")
    has_wgs: Optional[bool] = Field(None, description="Filter by WGS data availability")
    has_clinical_exome: Optional[bool] = Field(None, description="Filter by clinical exome availability")
    
    # Ancestry filters
    nba_labels: Optional[List[str]] = Field(None, description="Filter by NBA ancestry labels")
    wgs_labels: Optional[List[str]] = Field(None, description="Filter by WGS ancestry labels")
    ancestry_labels: Optional[List[str]] = Field(None, description="Filter by any ancestry label")
    
    # Clinical filters
    diagnoses: Optional[List[str]] = Field(None, description="Filter by diagnosis")
    study_arms: Optional[List[str]] = Field(None, description="Filter by study arm")
    study_types: Optional[List[str]] = Field(None, description="Filter by study type")
    
    # Demographics filters
    biological_sex: Optional[List[str]] = Field(None, description="Filter by biological sex")
    min_age_at_collection: Optional[float] = Field(None, description="Minimum age at sample collection")
    max_age_at_collection: Optional[float] = Field(None, description="Maximum age at sample collection")
    min_age_of_onset: Optional[float] = Field(None, description="Minimum age of onset")
    max_age_of_onset: Optional[float] = Field(None, description="Maximum age of onset")
    
    # GDPR compliance
    gdpr_compliant_only: Optional[bool] = Field(None, description="Include only GDPR compliant samples")
    
    # Pagination
    limit: Optional[int] = Field(None, description="Maximum number of results")
    offset: Optional[int] = Field(0, description="Number of results to skip") 