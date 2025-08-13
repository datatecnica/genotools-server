"""
WGS (Whole Genome Sequencing) dataset handler.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path

from .base import DatasetHandler, DatasetConfig


class WGSDatasetHandler(DatasetHandler):
    """
    Handler for WGS dataset type.
    
    WGS data has a single consolidated PLINK2 file across all ancestries:
    - Format: R{version}_wgs_carrier_vars.{pgen|pvar|psam}
    - Example: R10_wgs_carrier_vars.pgen
    """
    
    def get_file_patterns(self) -> List[str]:
        """Get file patterns for WGS data."""
        return [f"R{self.config.release}_wgs_carrier_vars"]
    
    def get_genotype_paths(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get list of WGS genotype files to process."""
        paths = []
        
        # WGS has a single file
        prefix = Path(self.config.base_path) / f"R{self.config.release}_wgs_carrier_vars"
        
        # Check if files exist
        if (Path(f"{prefix}.pgen").exists() and 
            Path(f"{prefix}.pvar").exists() and 
            Path(f"{prefix}.psam").exists()):
            
            metadata = {
                "dataset_type": "wgs",
                "release": self.config.release,
                "is_consolidated": True
            }
            paths.append((str(prefix), metadata))
        else:
            print(f"Warning: WGS files not found at {prefix}")
        
        return paths
    
    def should_combine_results(self) -> bool:
        """WGS data is already consolidated, no combination needed."""
        return False
    
    def get_combine_strategy(self) -> str:
        """WGS doesn't need combination."""
        return None
    
    def get_output_path(self, base_output: str, **kwargs) -> str:
        """
        Get output path for WGS data.
        
        Args:
            base_output: Base output directory
            
        Returns:
            Output path
        """
        base_path = Path(base_output) / "wgs" / f"release{self.config.release}"
        return str(base_path / f"release{self.config.release}")
    
    def get_snplist_path(self) -> str:
        """Get SNP list path for WGS data."""
        return self.config.snplist_path
