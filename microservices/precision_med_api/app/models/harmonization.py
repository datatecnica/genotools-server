"""
Models for variant harmonization and allele transformation.

Defines data structures for tracking variant harmonization between SNP lists
and PLINK files, including transformation metadata and statistics.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict


class HarmonizationAction(str, Enum):
    """Actions needed to harmonize PLINK variants to SNP list format."""
    EXACT = "EXACT"           # Perfect match, no transformation needed
    SWAP = "SWAP"             # Alleles swapped (A1<->A2), transform genotypes
    FLIP = "FLIP"             # Strand flipped (A<->T, C<->G), same genotypes  
    FLIP_SWAP = "FLIP_SWAP"   # Both strand flip and allele swap
    INVALID = "INVALID"       # Cannot be harmonized (e.g., different SNPs)
    AMBIGUOUS = "AMBIGUOUS"   # Strand ambiguous (A/T or C/G), needs manual review
    
    @property
    def requires_genotype_transform(self) -> bool:
        """Whether this action requires genotype transformation."""
        return self in (HarmonizationAction.SWAP, HarmonizationAction.FLIP_SWAP)
    
    @property
    def is_valid(self) -> bool:
        """Whether harmonization is valid."""
        return self not in (HarmonizationAction.INVALID, HarmonizationAction.AMBIGUOUS)


class HarmonizationRecord(BaseModel):
    """Record of variant harmonization between SNP list and PLINK file."""
    
    # Identifiers
    snp_list_id: str = Field(..., description="Variant ID from SNP list")
    pgen_variant_id: str = Field(..., description="Variant ID from PVAR file")
    
    # Genomic coordinates
    chromosome: str = Field(..., description="Chromosome")
    position: int = Field(..., description="Genomic position")
    
    # SNP list alleles (reference orientation)
    snp_list_a1: str = Field(..., description="Reference allele from SNP list")
    snp_list_a2: str = Field(..., description="Alternate allele from SNP list")
    
    # PLINK file alleles (may be flipped/swapped)
    pgen_a1: str = Field(..., description="A1 allele from PGEN file")
    pgen_a2: str = Field(..., description="A2 allele from PGEN file")
    
    # Harmonization metadata
    harmonization_action: HarmonizationAction = Field(..., description="Action needed for harmonization")
    genotype_transform: Optional[str] = Field(None, description="Formula for genotype transformation (e.g., '2-x')")
    
    # File metadata
    file_path: str = Field(..., description="Path to PLINK file")
    data_type: str = Field(..., description="Data type (NBA, WGS, IMPUTED)")
    ancestry: Optional[str] = Field(None, description="Ancestry group")
    
    @field_validator('chromosome')
    @classmethod
    def normalize_chromosome(cls, v: str) -> str:
        """Normalize chromosome format."""
        return v.replace('chr', '').upper()
    
    @field_validator('snp_list_a1', 'snp_list_a2', 'pgen_a1', 'pgen_a2')
    @classmethod
    def normalize_allele(cls, v: str) -> str:
        """Normalize allele format."""
        return v.upper().strip()
    
    @property
    def variant_key(self) -> str:
        """Generate unique variant key."""
        return f"{self.chromosome}:{self.position}:{self.snp_list_a1}:{self.snp_list_a2}"
    
    @property
    def requires_transformation(self) -> bool:
        """Whether genotype transformation is required."""
        return self.harmonization_action.requires_genotype_transform
    
    @property
    def is_strand_ambiguous(self) -> bool:
        """Whether variant is strand ambiguous (A/T or C/G)."""
        alleles = {self.snp_list_a1, self.snp_list_a2}
        return alleles == {"A", "T"} or alleles == {"C", "G"}
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "snp_list_id": "chr1:123456:A:G",
                "pgen_variant_id": "1:123456:G:A",
                "chromosome": "1",
                "position": 123456,
                "snp_list_a1": "A",
                "snp_list_a2": "G", 
                "pgen_a1": "G",
                "pgen_a2": "A",
                "harmonization_action": "SWAP",
                "genotype_transform": "2-x",
                "file_path": "/path/to/file.pgen",
                "data_type": "NBA",
                "ancestry": "EUR"
            }
        }
    )


class HarmonizationStats(BaseModel):
    """Statistics for harmonization results."""
    
    total_variants: int = Field(..., description="Total variants processed")
    exact_matches: int = Field(0, description="Variants with exact matches")
    swapped_alleles: int = Field(0, description="Variants with swapped alleles")
    flipped_strand: int = Field(0, description="Variants with flipped strand")
    flip_and_swap: int = Field(0, description="Variants with flip and swap")
    invalid_variants: int = Field(0, description="Variants that cannot be harmonized")
    ambiguous_variants: int = Field(0, description="Strand-ambiguous variants")
    
    # Processing metadata
    processing_time_seconds: Optional[float] = Field(None, description="Processing time")
    cache_file_path: Optional[str] = Field(None, description="Path to cache file")
    harmonized_files_generated: bool = Field(False, description="Whether harmonized PLINK files were generated")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    @property
    def harmonized_variants(self) -> int:
        """Number of successfully harmonized variants."""
        return self.exact_matches + self.swapped_alleles + self.flipped_strand + self.flip_and_swap
    
    @property
    def harmonization_rate(self) -> float:
        """Proportion of variants successfully harmonized."""
        return self.harmonized_variants / self.total_variants if self.total_variants > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        """Proportion of variants that failed harmonization."""
        return (self.invalid_variants + self.ambiguous_variants) / self.total_variants if self.total_variants > 0 else 0.0
    
    @property
    def summary_dict(self) -> Dict[str, Any]:
        """Summary dictionary for reporting."""
        return {
            "total_variants": self.total_variants,
            "harmonized": self.harmonized_variants,
            "harmonization_rate": round(self.harmonization_rate, 4),
            "exact_matches": self.exact_matches,
            "swapped_alleles": self.swapped_alleles,
            "flipped_strand": self.flipped_strand,
            "flip_and_swap": self.flip_and_swap,
            "invalid": self.invalid_variants,
            "ambiguous": self.ambiguous_variants,
            "failure_rate": round(self.failure_rate, 4)
        }
    
    def update_from_records(self, records: List[HarmonizationRecord]) -> None:
        """Update stats from harmonization records."""
        self.total_variants = len(records)
        
        action_counts = {}
        for record in records:
            action = record.harmonization_action
            action_counts[action] = action_counts.get(action, 0) + 1
        
        self.exact_matches = action_counts.get(HarmonizationAction.EXACT, 0)
        self.swapped_alleles = action_counts.get(HarmonizationAction.SWAP, 0)
        self.flipped_strand = action_counts.get(HarmonizationAction.FLIP, 0)
        self.flip_and_swap = action_counts.get(HarmonizationAction.FLIP_SWAP, 0)
        self.invalid_variants = action_counts.get(HarmonizationAction.INVALID, 0)
        self.ambiguous_variants = action_counts.get(HarmonizationAction.AMBIGUOUS, 0)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_variants": 400,
                "exact_matches": 200,
                "swapped_alleles": 150,
                "flipped_strand": 30,
                "flip_and_swap": 15,
                "invalid_variants": 3,
                "ambiguous_variants": 2,
                "processing_time_seconds": 45.2,
                "cache_file_path": "/path/to/cache.parquet"
            }
        }
    )


class ExtractionPlan(BaseModel):
    """Plan for multi-source variant extraction."""
    
    snp_list_ids: List[str] = Field(..., description="Variant IDs to extract")
    data_sources: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Data sources and their file paths"
    )
    expected_total_variants: int = Field(..., description="Expected number of variants")
    expected_total_samples: int = Field(0, description="Expected number of samples")
    
    # Execution metadata
    created_at: datetime = Field(default_factory=datetime.now)
    estimated_duration_minutes: Optional[float] = Field(None, description="Estimated execution time")
    
    @property
    def num_files(self) -> int:
        """Total number of files to process."""
        return sum(len(source_info.get('files', [])) for source_info in self.data_sources.values())
    
    @property
    def data_types(self) -> List[str]:
        """List of data types in the plan."""
        return list(self.data_sources.keys())
    
    def add_data_source(self, data_type: str, files: List[str], ancestries: Optional[List[str]] = None) -> None:
        """Add a data source to the extraction plan."""
        self.data_sources[data_type] = {
            "files": files,
            "ancestries": ancestries or [],
            "num_files": len(files)
        }
    
    def get_files_for_data_type(self, data_type: str) -> List[str]:
        """Get file list for a specific data type."""
        return self.data_sources.get(data_type, {}).get("files", [])
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "snp_list_ids": ["chr1:123456:A:G", "chr2:789012:C:T"],
                "data_sources": {
                    "NBA": {
                        "files": ["/path/EUR.pgen", "/path/EAS.pgen"],
                        "ancestries": ["EUR", "EAS"],
                        "num_files": 2
                    },
                    "IMPUTED": {
                        "files": ["/path/chr1_EUR.pgen", "/path/chr2_EUR.pgen"],
                        "ancestries": ["EUR"],
                        "num_files": 2
                    }
                },
                "expected_total_variants": 2,
                "expected_total_samples": 100000
            }
        }
    )