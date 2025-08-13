"""
Carrier data models.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd
from pathlib import Path


@dataclass
class CarrierData:
    """
    Represents the three standard outputs from carrier analysis.
    
    Attributes:
        var_info_path: Path to variant information parquet file
        carriers_string_path: Path to string format carriers parquet file
        carriers_int_path: Path to integer format carriers parquet file
        var_info: Optional DataFrame with variant information
        carriers_string: Optional DataFrame with string genotypes
        carriers_int: Optional DataFrame with integer genotypes
    """
    var_info_path: str
    carriers_string_path: str
    carriers_int_path: str
    var_info: Optional[pd.DataFrame] = None
    carriers_string: Optional[pd.DataFrame] = None
    carriers_int: Optional[pd.DataFrame] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary of file paths."""
        return {
            'var_info': self.var_info_path,
            'carriers_string': self.carriers_string_path,
            'carriers_int': self.carriers_int_path
        }
    
    def validate_paths(self) -> bool:
        """Validate that all output files exist."""
        paths = [self.var_info_path, self.carriers_string_path, self.carriers_int_path]
        return all(Path(p).exists() for p in paths)
    
    def get_variant_count(self) -> int:
        """Get number of variants."""
        if self.var_info is not None:
            return len(self.var_info)
        return 0
    
    def get_sample_count(self) -> int:
        """Get number of samples."""
        if self.carriers_int is not None:
            return len(self.carriers_int)
        return 0


@dataclass
class ProcessingRequest:
    """Request for processing carrier data."""
    dataset_type: str  # 'nba', 'wgs', or 'imputed'
    input_path: str  # Path to genotype files
    snplist_path: str  # Path to SNP list
    output_path: str  # Output path prefix
    release: str = "10"  # Release version
    ancestry: Optional[str] = None  # For dataset-specific processing
    ancestries: Optional[list] = None  # For multi-ancestry processing
    chromosomes: Optional[list] = None  # For chromosome-specific processing
    use_cache: bool = True  # Use harmonization cache
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dataset_type': self.dataset_type,
            'input_path': self.input_path,
            'snplist_path': self.snplist_path,
            'output_path': self.output_path,
            'release': self.release,
            'ancestry': self.ancestry,
            'ancestries': self.ancestries,
            'chromosomes': self.chromosomes,
            'use_cache': self.use_cache
        }


@dataclass
class HarmonizationMapping:
    """Represents a harmonization mapping between genotype and reference variants."""
    data_type: str
    release: str
    ancestry: Optional[str] = None
    chromosome: Optional[str] = None
    geno_prefix: Optional[str] = None
    variant_id_ref: str = None  # Reference variant ID
    variant_id_geno: str = None  # Genotype variant ID
    ref_allele: str = None
    alt_allele: str = None
    swap_alleles: bool = False
    flip_strands: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'data_type': self.data_type,
            'release': self.release,
            'ancestry': self.ancestry,
            'chromosome': self.chromosome,
            'geno_prefix': self.geno_prefix,
            'variant_id_ref': self.variant_id_ref,
            'variant_id_geno': self.variant_id_geno,
            'ref_allele': self.ref_allele,
            'alt_allele': self.alt_allele,
            'swap_alleles': self.swap_alleles,
            'flip_strands': self.flip_strands
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HarmonizationMapping':
        """Create from dictionary."""
        return cls(**data)
