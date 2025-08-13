"""
Dataset configuration models.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class DatasetPaths:
    """Paths for different dataset types."""
    nba_base: str = ""
    wgs_base: str = ""
    imputed_base: str = ""
    output_base: str = ""
    harmonization_cache: str = ""
    
    def get_path(self, dataset_type: str) -> str:
        """Get base path for a dataset type."""
        mapping = {
            'nba': self.nba_base,
            'wgs': self.wgs_base,
            'imputed': self.imputed_base
        }
        return mapping.get(dataset_type, "")


@dataclass
class AncestryConfig:
    """Configuration for ancestry-specific processing."""
    label: str  # Ancestry label (e.g., 'AAC', 'EUR')
    display_name: str  # Display name
    enabled: bool = True  # Whether to process this ancestry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'label': self.label,
            'display_name': self.display_name,
            'enabled': self.enabled
        }


@dataclass 
class DatasetMetadata:
    """Metadata about a dataset."""
    dataset_type: str
    release: str
    creation_date: Optional[str] = None
    source: Optional[str] = None
    description: Optional[str] = None
    sample_count: Optional[int] = None
    variant_count: Optional[int] = None
    ancestries: List[str] = field(default_factory=list)
    chromosomes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'dataset_type': self.dataset_type,
            'release': self.release,
            'creation_date': self.creation_date,
            'source': self.source,
            'description': self.description,
            'sample_count': self.sample_count,
            'variant_count': self.variant_count,
            'ancestries': self.ancestries,
            'chromosomes': self.chromosomes
        }


# Default ancestry configurations
DEFAULT_ANCESTRIES = [
    AncestryConfig('AAC', 'African American Caribbean'),
    AncestryConfig('AFR', 'African'),
    AncestryConfig('AJ', 'Ashkenazi Jewish'),
    AncestryConfig('AMR', 'Admixed American'),
    AncestryConfig('CAH', 'Central Asian Hispanic'),
    AncestryConfig('CAS', 'Central Asian'),
    AncestryConfig('EAS', 'East Asian'),
    AncestryConfig('EUR', 'European'),
    AncestryConfig('FIN', 'Finnish'),
    AncestryConfig('MDE', 'Middle Eastern'),
    AncestryConfig('SAS', 'South Asian')
]


def get_ancestry_labels() -> List[str]:
    """Get list of ancestry labels."""
    return [anc.label for anc in DEFAULT_ANCESTRIES if anc.enabled]


def get_ancestry_config(label: str) -> Optional[AncestryConfig]:
    """Get ancestry configuration by label."""
    for anc in DEFAULT_ANCESTRIES:
        if anc.label == label:
            return anc
    return None
