"""
Path configurations and utilities.
"""

import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

from .settings import settings


@dataclass
class PathConfig:
    """Centralized path configuration."""
    
    # Base paths
    mount_path: Path
    output_base: Path
    harmonization_cache: Path
    clinical_data: Path
    
    # Dataset-specific paths
    nba_path: Path
    wgs_path: Path
    imputed_path: Path
    
    @classmethod
    def from_settings(cls, release: str) -> 'PathConfig':
        """Create PathConfig from application settings."""
        mount = Path(settings.base_path).expanduser()
        
        return cls(
            mount_path=mount,
            output_base=Path(settings.get_output_base(release)),
            harmonization_cache=Path(settings.get_harmonization_cache_dir(release)),
            clinical_data=Path(settings.get_clinical_data_path(release)),
            nba_path=Path(settings.get_dataset_path("nba", release)),
            wgs_path=Path(settings.get_dataset_path("wgs", release)),
            imputed_path=Path(settings.get_dataset_path("imputed", release))
        )
    
    def get_dataset_path(self, dataset_type: str) -> Path:
        """Get path for a specific dataset type."""
        mapping = {
            "nba": self.nba_path,
            "wgs": self.wgs_path,
            "imputed": self.imputed_path
        }
        return mapping.get(dataset_type, self.mount_path)
    
    def get_output_path(self, dataset_type: str, ancestry: Optional[str] = None, 
                       chromosome: Optional[str] = None, combined: bool = False) -> Path:
        """
        Get output path for results.
        
        Args:
            dataset_type: Type of dataset ('nba', 'wgs', 'imputed')
            ancestry: Ancestry label (optional)
            chromosome: Chromosome number (optional)
            combined: Whether this is for combined output
            
        Returns:
            Output path
        """
        base = self.output_base / dataset_type
        
        if dataset_type == "wgs":
            # WGS has simple structure
            return base / f"release{self.mount_path.name}"
        
        elif dataset_type == "nba":
            if combined:
                return base / "combined" / f"nba_release{self.mount_path.name}_combined"
            elif ancestry:
                return base / ancestry / f"{ancestry}_release{self.mount_path.name}"
            else:
                return base
        
        elif dataset_type == "imputed":
            if ancestry:
                ancestry_base = base / ancestry
                if chromosome:
                    return ancestry_base / f"{ancestry}_release{self.mount_path.name}_chr{chromosome}"
                else:
                    return ancestry_base / f"{ancestry}_release{self.mount_path.name}"
            else:
                return base
        
        return base
    
    def ensure_output_dirs(self):
        """Create output directories if they don't exist."""
        dirs = [
            self.output_base,
            self.harmonization_cache,
            self.output_base / "nba",
            self.output_base / "wgs",
            self.output_base / "imputed"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def validate_input_paths(self) -> Dict[str, bool]:
        """Validate that input paths exist."""
        return {
            "mount": self.mount_path.exists(),
            "nba": self.nba_path.exists(),
            "wgs": self.wgs_path.exists(),
            "imputed": self.imputed_path.exists(),
            "clinical": self.clinical_data.exists()
        }


# Standard paths used throughout the application
STANDARD_PATHS = {
    "snplist_filename": "carriers_snp_list.csv",
    "master_key_pattern": "master_key_release{release}_final_vwb.csv",
    "data_dict_pattern": "data_dictionary_release{release}_vwb.csv",
    "extended_clinical_pattern": "extended_clinical_release{release}_vwb.parquet"
}


def get_snplist_path(base_dir: Path) -> Path:
    """Get standard SNP list path."""
    return base_dir / "summary_data" / STANDARD_PATHS["snplist_filename"]


def get_master_key_path(clinical_dir: Path, release: str) -> Path:
    """Get master key file path."""
    filename = STANDARD_PATHS["master_key_pattern"].format(release=release)
    return clinical_dir / filename


def get_data_dict_path(clinical_dir: Path, release: str) -> Path:
    """Get data dictionary path."""
    filename = STANDARD_PATHS["data_dict_pattern"].format(release=release)
    return clinical_dir / filename


def get_extended_clinical_path(clinical_dir: Path, release: str) -> Path:
    """Get extended clinical data path."""
    filename = STANDARD_PATHS["extended_clinical_pattern"].format(release=release)
    return clinical_dir / filename
