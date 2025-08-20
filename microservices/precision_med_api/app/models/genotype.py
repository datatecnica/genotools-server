from typing import Optional, Dict, List
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict


class GenotypeRecord(BaseModel):
    """Model for individual genotype data from PLINK traw files (numeric format)"""
    
    # Sample identifier
    IID: str = Field(..., description="Individual/Sample ID")
    
    # Genotype data - numeric values from PLINK traw (0/1/2 or NaN for missing)
    genotypes: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Dictionary of variant_id -> numeric genotype (0=hom_ref, 1=het, 2=hom_alt, None=missing)"
    )
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "IID": "BBDP_000002",
                "genotypes": {
                    "chr12:40340400:G:A": 0.0,
                    "chr12:40340404:T:C": 1.0,
                    "chr1:155235205:C:T": 2.0,
                    "chr1:155235805:AG:A": None  # missing genotype
                }
            }
        }
    )


class GenotypeCallFormat(str, Enum):
    """Enumeration of genotype call formats"""
    NUMERIC = "numeric"  # 0/1/2 format (0=hom_ref, 1=het, 2=hom_alt) from PLINK traw
    STRING = "string"    # WT/WT, WT/MUT format for display purposes


class GenotypeData(BaseModel):
    """Container for genotype data with metadata"""
    
    samples: List[GenotypeRecord] = Field(..., description="List of sample genotypes")
    variants: List[str] = Field(..., description="List of variant IDs in the dataset")
    format: GenotypeCallFormat = Field(..., description="Format of genotype calls")
    source: Optional[str] = Field(None, description="Data source (WGS, NBA, Imputed)")
    release: Optional[str] = Field(None, description="Release version")
    ancestry: Optional[str] = Field(None, description="Ancestry group if applicable")
    chromosome: Optional[str] = Field(None, description="Chromosome if subset")
    
    @property
    def num_samples(self) -> int:
        return len(self.samples)
    
    @property
    def num_variants(self) -> int:
        return len(self.variants)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "samples": [
                    {
                        "IID": "SAMPLE_001",
                        "genotypes": {
                            "chr12:40340400:G:A": 0.0,
                            "chr12:40340404:T:C": 1.0
                        }
                    }
                ],
                "variants": ["chr12:40340400:G:A", "chr12:40340404:T:C"],
                "format": "numeric",
                "source": "WGS",
                "release": "10",
                "ancestry": None
            }
        }
    )


class CarrierStatus(BaseModel):
    """Model for carrier status results"""
    
    IID: str = Field(..., description="Individual/Sample ID")
    variant_id: str = Field(..., description="Variant identifier")
    genotype: Optional[float] = Field(..., description="Numeric genotype value (0/1/2 or None)")
    is_carrier: bool = Field(..., description="Whether sample is a carrier")
    carrier_type: Optional[str] = Field(None, description="Type of carrier (heterozygous/homozygous)")
    gene: Optional[str] = Field(None, description="Gene symbol")
    snp_name: Optional[str] = Field(None, description="SNP name")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "IID": "SAMPLE_001",
                "variant_id": "chr12:40340400:G:A",
                "genotype": 1.0,
                "is_carrier": True,
                "carrier_type": "heterozygous",
                "gene": "LRRK2",
                "snp_name": "G2019S"
            }
        }
    )