from typing import Optional
from pydantic import BaseModel, Field


class KeyRecord(BaseModel):
    GP2ID: str = Field(description="GP2 identifier")
    study: Optional[str] = Field(default=None, description="Study name")
    nba: Optional[bool] = Field(default=None, description="NBA availability")
    wgs: Optional[bool] = Field(default=None, description="WGS availability")
    clinical_exome: Optional[bool] = Field(default=None, description="Clinical exome availability")
    extended_clinical_data: Optional[bool] = Field(default=None, description="Extended clinical data availability")
    GDPR: Optional[bool] = Field(default=None, description="GDPR status")
    nba_prune_reason: Optional[str] = Field(default=None, description="NBA prune reason")
    nba_related: Optional[bool] = Field(default=None, description="NBA related status")
    nba_label: Optional[str] = Field(default=None, description="NBA label")
    wgs_prune_reason: Optional[str] = Field(default=None, description="WGS prune reason")
    wgs_label: Optional[str] = Field(default=None, description="WGS label")
    study_arm: Optional[str] = Field(default=None, description="Study arm")
    study_type: Optional[str] = Field(default=None, description="Study type")
    diagnosis: Optional[str] = Field(default=None, description="Diagnosis")
    baseline_GP2_phenotype_for_qc: Optional[str] = Field(default=None, description="Baseline GP2 phenotype for QC")
    baseline_GP2_phenotype: Optional[str] = Field(default=None, description="Baseline GP2 phenotype")
    biological_sex_for_qc: Optional[str] = Field(default=None, description="Biological sex for QC")
    age_at_sample_collection: Optional[float] = Field(default=None, description="Age at sample collection")
    age_of_onset: Optional[float] = Field(default=None, description="Age of onset")
    age_at_diagnosis: Optional[float] = Field(default=None, description="Age at diagnosis")
    age_at_death: Optional[float] = Field(default=None, description="Age at death")
    age_at_last_follow_up: Optional[float] = Field(default=None, description="Age at last follow-up")
    race_for_qc: Optional[str] = Field(default=None, description="Race for QC")
    family_history_for_qc: Optional[str] = Field(default=None, description="Family history for QC")
    region_for_qc: Optional[str] = Field(default=None, description="Region for QC")
    manifest_id: Optional[str] = Field(default=None, description="Manifest ID")
    Genotyping_site: Optional[str] = Field(default=None, description="Genotyping site")
    sample_type: Optional[str] = Field(default=None, description="Sample type")
    amppd_wgs: Optional[bool] = Field(default=None, description="AMP-PD WGS availability")

    class Config:
        populate_by_name = True