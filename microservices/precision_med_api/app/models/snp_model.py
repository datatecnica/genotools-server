from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SNPRecord(BaseModel):
    snp_name: str = Field(description="SNP name")
    snp_name_alt: Optional[str] = Field(default=None, description="Alternative SNP name")
    locus: str = Field(description="Gene locus")
    rsid: Optional[str] = Field(default=None, description="Reference SNP ID")
    hg38: str = Field(description="HG38 genome coordinates")
    hg19: str = Field(description="HG19 genome coordinates")
    ancestry: Optional[str] = Field(default=None, description="Ancestry information")
    submitter_email: Optional[str] = Field(default=None, description="Submitter email")
    precision_medicine: Optional[str] = Field(default=None, description="Precision medicine flag")
    pipeline: Optional[str] = Field(default=None, description="Pipeline information")

    model_config = ConfigDict(populate_by_name=True)