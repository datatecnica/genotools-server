from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class InheritancePattern(str, Enum):
    AD = "AD"  # Autosomal Dominant
    AR = "AR"  # Autosomal Recessive
    XL = "XL"  # X-Linked
    MT = "MT"  # Mitochondrial


class Variant(BaseModel):
    chromosome: str = Field(..., description="Chromosome (1-22, X, Y, MT)")
    position: int = Field(..., ge=0, description="Genomic position (0-based)")
    ref: str = Field(..., min_length=1, description="Reference allele")
    alt: str = Field(..., min_length=1, description="Alternative allele")
    gene: str = Field(..., min_length=1, description="Gene symbol")
    rsid: Optional[str] = Field(None, description="dbSNP RS ID")
    inheritance_pattern: InheritancePattern = Field(..., description="Inheritance pattern")
    
    @field_validator('chromosome')
    @classmethod
    def validate_chromosome(cls, v: str) -> str:
        valid_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        if v not in valid_chroms:
            raise ValueError(f"Invalid chromosome: {v}. Must be one of {valid_chroms}")
        return v
    
    @field_validator('ref', 'alt')
    @classmethod
    def validate_allele(cls, v: str) -> str:
        valid_bases = set('ACGTMRWSYKVHDBN-')
        if not all(c.upper() in valid_bases for c in v):
            raise ValueError(f"Invalid allele: {v}. Must contain only valid nucleotide codes")
        return v.upper()
    
    @field_validator('rsid')
    @classmethod
    def validate_rsid(cls, v: Optional[str]) -> Optional[str]:
        if v and not v.startswith('rs'):
            raise ValueError(f"Invalid rsid: {v}. Must start with 'rs'")
        return v
    
    @property
    def variant_id(self) -> str:
        return f"{self.chromosome}:{self.position}:{self.ref}:{self.alt}"
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chromosome": "1",
                "position": 762273,
                "ref": "G",
                "alt": "A",
                "gene": "BRCA2",
                "rsid": "rs121913023",
                "inheritance_pattern": "AD"
            }
        }
    )


class VariantList(BaseModel):
    variants: List[Variant] = Field(..., description="List of variants")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional metadata about the variant list"
    )
    name: Optional[str] = Field(None, description="Name or identifier for this variant list")
    description: Optional[str] = Field(None, description="Description of the variant list")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    version: Optional[str] = Field(None, description="Version of the variant list")
    
    @property
    def total_variants(self) -> int:
        return len(self.variants)
    
    @property
    def variants_by_chromosome(self) -> Dict[str, List[Variant]]:
        result = {}
        for variant in self.variants:
            if variant.chromosome not in result:
                result[variant.chromosome] = []
            result[variant.chromosome].append(variant)
        return result
    
    @property
    def variants_by_gene(self) -> Dict[str, List[Variant]]:
        result = {}
        for variant in self.variants:
            if variant.gene not in result:
                result[variant.gene] = []
            result[variant.gene].append(variant)
        return result
    
    @property
    def inheritance_patterns(self) -> Dict[str, int]:
        counts = {}
        for variant in self.variants:
            pattern = variant.inheritance_pattern.value
            counts[pattern] = counts.get(pattern, 0) + 1
        return counts
    
    def get_variants_for_chromosome(self, chromosome: str) -> List[Variant]:
        return [v for v in self.variants if v.chromosome == chromosome]
    
    def get_variants_for_gene(self, gene: str) -> List[Variant]:
        return [v for v in self.variants if v.gene == gene]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "Pathogenic Carrier Screening Panel",
                "description": "Curated list of pathogenic variants for carrier screening",
                "version": "1.0",
                "variants": [
                    {
                        "chromosome": "1",
                        "position": 762273,
                        "ref": "G",
                        "alt": "A",
                        "gene": "BRCA2",
                        "rsid": "rs121913023",
                        "inheritance_pattern": "AD"
                    }
                ],
                "metadata": {
                    "source": "ClinVar",
                    "curation_date": "2024-01-01"
                }
            }
        }
    )