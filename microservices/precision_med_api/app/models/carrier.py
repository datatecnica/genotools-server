from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class GenotypeValue(str, Enum):
    HOMOZYGOUS_REF = "0/0"
    HETEROZYGOUS = "0/1"
    HOMOZYGOUS_ALT = "1/1"
    MISSING = "./."
    
    @property
    def is_carrier(self) -> bool:
        return self in (GenotypeValue.HETEROZYGOUS, GenotypeValue.HOMOZYGOUS_ALT)
    
    @property
    def allele_count(self) -> int:
        if self == GenotypeValue.HOMOZYGOUS_REF:
            return 0
        elif self == GenotypeValue.HETEROZYGOUS:
            return 1
        elif self == GenotypeValue.HOMOZYGOUS_ALT:
            return 2
        else:
            return 0


class Genotype(BaseModel):
    sample_id: str = Field(..., min_length=1, description="Sample identifier")
    variant_id: str = Field(..., description="Variant identifier (chr:pos:ref:alt)")
    gt: GenotypeValue = Field(..., description="Genotype value")
    ancestry: Optional[str] = Field(None, description="Sample ancestry")
    
    @field_validator('variant_id')
    @classmethod
    def validate_variant_id(cls, v: str) -> str:
        parts = v.split(':')
        if len(parts) != 4:
            raise ValueError(f"Invalid variant_id format: {v}. Expected chr:pos:ref:alt")
        return v
    
    @property
    def is_carrier(self) -> bool:
        return self.gt.is_carrier
    
    @property
    def allele_count(self) -> int:
        return self.gt.allele_count
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_id": "GP2_001234",
                "variant_id": "1:762273:G:A",
                "gt": "0/1",
                "ancestry": "EUR"
            }
        }
    )


class Carrier(Genotype):
    sex: Optional[str] = Field(None, description="Biological sex (M/F)")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age at sampling")
    study_arm: Optional[str] = Field(None, description="Study arm or cohort")
    clinical_status: Optional[str] = Field(None, description="Clinical status")
    family_history: Optional[bool] = Field(None, description="Family history of condition")
    phenotype: Optional[str] = Field(None, description="Associated phenotype")
    additional_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional clinical metadata"
    )
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v: Optional[str]) -> Optional[str]:
        if v and v.upper() not in ['M', 'F', 'MALE', 'FEMALE']:
            raise ValueError(f"Invalid sex value: {v}. Must be M/F or Male/Female")
        return v.upper()[0] if v else None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sample_id": "GP2_001234",
                "variant_id": "1:762273:G:A",
                "gt": "0/1",
                "ancestry": "EUR",
                "sex": "F",
                "age": 45,
                "study_arm": "PD_GWAS",
                "clinical_status": "affected",
                "family_history": True,
                "phenotype": "Parkinson's Disease",
                "additional_metadata": {
                    "onset_age": 42,
                    "severity": "moderate"
                }
            }
        }
    )


class CarrierStatistics(BaseModel):
    total_samples: int = Field(..., ge=0, description="Total number of samples analyzed")
    total_carriers: int = Field(..., ge=0, description="Total number of carriers identified")
    carrier_frequency: float = Field(..., ge=0, le=1, description="Overall carrier frequency")
    heterozygous_count: int = Field(..., ge=0, description="Number of heterozygous carriers")
    homozygous_count: int = Field(..., ge=0, description="Number of homozygous carriers")
    missing_count: int = Field(..., ge=0, description="Number of samples with missing genotypes")
    
    @property
    def genotyped_samples(self) -> int:
        return self.total_samples - self.missing_count
    
    @property
    def allele_frequency(self) -> float:
        if self.genotyped_samples == 0:
            return 0.0
        total_alleles = self.genotyped_samples * 2
        alt_alleles = self.heterozygous_count + (self.homozygous_count * 2)
        return alt_alleles / total_alleles if total_alleles > 0 else 0.0
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_samples": 10000,
                "total_carriers": 150,
                "carrier_frequency": 0.015,
                "heterozygous_count": 145,
                "homozygous_count": 5,
                "missing_count": 25
            }
        }
    )


class AncestryCarrierStats(CarrierStatistics):
    ancestry: str = Field(..., description="Ancestry group")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ancestry": "EUR",
                "total_samples": 5000,
                "total_carriers": 75,
                "carrier_frequency": 0.015,
                "heterozygous_count": 72,
                "homozygous_count": 3,
                "missing_count": 10
            }
        }
    )


class CarrierReport(BaseModel):
    variant_id: str = Field(..., description="Variant identifier")
    gene: str = Field(..., description="Gene symbol")
    inheritance_pattern: str = Field(..., description="Inheritance pattern")
    overall_statistics: CarrierStatistics = Field(..., description="Overall carrier statistics")
    ancestry_statistics: List[AncestryCarrierStats] = Field(
        default_factory=list,
        description="Carrier statistics by ancestry"
    )
    carriers: List[Carrier] = Field(
        default_factory=list,
        description="List of identified carriers"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional report metadata"
    )
    
    @property
    def total_carrier_count(self) -> int:
        return len([c for c in self.carriers if c.is_carrier])
    
    @property
    def carriers_by_ancestry(self) -> Dict[str, List[Carrier]]:
        result = {}
        for carrier in self.carriers:
            if carrier.ancestry:
                if carrier.ancestry not in result:
                    result[carrier.ancestry] = []
                result[carrier.ancestry].append(carrier)
        return result
    
    @property
    def carriers_by_genotype(self) -> Dict[str, List[Carrier]]:
        result = {
            GenotypeValue.HETEROZYGOUS.value: [],
            GenotypeValue.HOMOZYGOUS_ALT.value: []
        }
        for carrier in self.carriers:
            if carrier.gt == GenotypeValue.HETEROZYGOUS:
                result[GenotypeValue.HETEROZYGOUS.value].append(carrier)
            elif carrier.gt == GenotypeValue.HOMOZYGOUS_ALT:
                result[GenotypeValue.HOMOZYGOUS_ALT.value].append(carrier)
        return result
    
    def get_ancestry_stats(self, ancestry: str) -> Optional[AncestryCarrierStats]:
        for stats in self.ancestry_statistics:
            if stats.ancestry == ancestry:
                return stats
        return None
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "variant_id": "1:762273:G:A",
                "gene": "BRCA2",
                "inheritance_pattern": "AD",
                "overall_statistics": {
                    "total_samples": 10000,
                    "total_carriers": 150,
                    "carrier_frequency": 0.015,
                    "heterozygous_count": 145,
                    "homozygous_count": 5,
                    "missing_count": 25
                },
                "ancestry_statistics": [
                    {
                        "ancestry": "EUR",
                        "total_samples": 5000,
                        "total_carriers": 75,
                        "carrier_frequency": 0.015,
                        "heterozygous_count": 72,
                        "homozygous_count": 3,
                        "missing_count": 10
                    }
                ],
                "carriers": [],
                "metadata": {
                    "analysis_date": "2024-01-01",
                    "reference_genome": "GRCh38"
                }
            }
        }
    )