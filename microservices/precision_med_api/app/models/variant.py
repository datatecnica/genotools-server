from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class InheritancePattern(str, Enum):
    AD = "AD"  # Autosomal Dominant
    AR = "AR"  # Autosomal Recessive
    XL = "XL"  # X-Linked
    MT = "MT"  # Mitochondrial


class Variant(BaseModel):
    """Model for input genomic variants from SNP lists (pre-extraction)"""
    
    # Variant annotation (from SNP list)
    snp_name: Optional[str] = Field(None, description="SNP name (e.g., G2019S)")
    snp_name_alt: Optional[str] = Field(None, description="Alternative SNP name (e.g., p.Gly2019Ser)")
    locus: str = Field(..., description="Gene locus/symbol")
    rsid: Optional[str] = Field(None, description="dbSNP RS ID")
    
    # Genome coordinates
    hg38: str = Field(..., description="HG38 genome coordinates (chr:pos:ref:alt)")
    hg19: str = Field(..., description="HG19 genome coordinates (chr:pos:ref:alt)")
    
    # Clinical/submission metadata
    ancestry: Optional[str] = Field(None, description="Ancestry information")
    precision_medicine: Optional[str] = Field(None, description="Precision medicine flag")
    pipeline: Optional[str] = Field(None, description="Pipeline information")
    submitter_email: Optional[str] = Field(None, description="Submitter email")
    
    # Computed properties from coordinates
    @property
    def chromosome(self) -> str:
        """Extract chromosome from HG38 coordinates"""
        return self.hg38.split(':')[0].replace('chr', '')
    
    @property
    def position(self) -> int:
        """Extract position from HG38 coordinates"""
        return int(self.hg38.split(':')[1])
    
    @property
    def ref(self) -> str:
        """Extract reference allele from HG38 coordinates"""
        return self.hg38.split(':')[2]
    
    @property
    def alt(self) -> str:
        """Extract alternative allele from HG38 coordinates"""
        return self.hg38.split(':')[3]
    
    @field_validator('rsid')
    @classmethod
    def validate_rsid(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == '.' or v == 'NaN':
            return None
        if not v.startswith('rs'):
            raise ValueError(f"Invalid rsid: {v}. Must start with 'rs'")
        return v
    
    @field_validator('precision_medicine')
    @classmethod
    def normalize_precision_medicine(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == 'NaN':
            return None
        return v
    
    @property
    def variant_id(self) -> str:
        """Generate variant ID from HG38 coordinates"""
        return self.hg38
    
    @property
    def gene_symbol(self) -> str:
        """Return gene symbol"""
        return self.locus
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "snp_name": "G2019S",
                "snp_name_alt": "p.Gly2019Ser",
                "locus": "LRRK2",
                "rsid": "rs34637584",
                "hg38": "12:40340400:G:A",
                "hg19": "12:40734202:G:A",
                "ancestry": "multi",
                "precision_medicine": "yes",
                "submitter_email": "lara.lange@nih.gov"
            }
        }
    )


class ProcessedVariant(BaseModel):
    """Model for variants after PLINK processing with population statistics"""
    
    # Variant identification
    id: str = Field(..., description="Variant ID in format chr:pos:ref:alt")
    snp_name: Optional[str] = Field(None, description="SNP name (e.g., G2019S)")
    snp_name_alt: Optional[str] = Field(None, description="Alternative SNP name (e.g., p.Gly2019Ser)")
    locus: str = Field(..., description="Gene locus/symbol")
    rsid: Optional[str] = Field(None, description="dbSNP RS ID")
    
    # Genomic coordinates
    hg38: str = Field(..., description="HG38 genome coordinates")
    hg19: str = Field(..., description="HG19 genome coordinates")
    chrom: int = Field(..., description="Chromosome number")
    pos: int = Field(..., description="Position")
    a1: str = Field(..., description="Allele 1 (reference)")
    a2: str = Field(..., description="Allele 2 (alternate)")
    
    # Clinical metadata
    ancestry: Optional[str] = Field(None, description="Ancestry information")
    submitter_email: Optional[str] = Field(None, description="Submitter email")
    precision_medicine: Optional[str] = Field(None, description="Precision medicine flag")
    pipeline: Optional[str] = Field(None, description="Pipeline information")
    
    # Population statistics (from PLINK output)
    ALT_FREQS: float = Field(..., description="Alternative allele frequency")
    OBS_CT: int = Field(..., description="Observation count (number of samples)")
    F_MISS: float = Field(..., description="Missing data frequency")
    
    @property
    def chromosome(self) -> str:
        """Get chromosome as string"""
        return str(self.chrom)
    
    @property
    def position(self) -> int:
        """Get position"""
        return self.pos
    
    @property
    def ref(self) -> str:
        """Get reference allele"""
        return self.a1
    
    @property
    def alt(self) -> str:
        """Get alternative allele"""
        return self.a2
    
    @property
    def variant_id(self) -> str:
        """Get variant ID"""
        return self.id
    
    @property
    def gene_symbol(self) -> str:
        """Get gene symbol"""
        return self.locus
    
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "id": "chr12:40340400:G:A",
                "snp_name": "G2019S",
                "snp_name_alt": "p.Gly2019Ser",
                "locus": "LRRK2",
                "rsid": "rs34637584",
                "hg38": "12:40340400:G:A",
                "hg19": "12:40734202:G:A",
                "chrom": 12,
                "pos": 40340400,
                "a1": "G",
                "a2": "A",
                "ancestry": "multi",
                "precision_medicine": "yes",
                "submitter_email": "lara.lange@nih.gov",
                "ALT_FREQS": 0.023191,
                "OBS_CT": 41956,
                "F_MISS": 0.002757
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