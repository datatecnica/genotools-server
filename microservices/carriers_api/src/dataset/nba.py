"""
NBA (NeuroBooster Array) dataset handler.
"""

from typing import List, Dict, Any, Tuple
from pathlib import Path

from .base import DatasetHandler, DatasetConfig


class NBADatasetHandler(DatasetHandler):
    """
    Handler for NBA dataset type.
    
    NBA data has single consolidated PLINK2 files per ancestry:
    - Format: {ancestry}_release{version}_vwb.{pgen|pvar|psam}
    - Example: AAC_release10_vwb.pgen
    """
    
    # Default ancestry labels if not specified
    DEFAULT_ANCESTRIES = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
    
    def __init__(self, config: DatasetConfig, ancestries: List[str] = None):
        super().__init__(config)
        self.ancestries = ancestries or self.DEFAULT_ANCESTRIES
    
    def get_file_patterns(self) -> List[str]:
        """Get file patterns for NBA data."""
        return [f"{ancestry}_release{self.config.release}_vwb" for ancestry in self.ancestries]
    
    def get_genotype_paths(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get list of NBA genotype files to process."""
        paths = []
        
        base_dir = Path(self.config.base_path)
        
        # Check each ancestry
        for ancestry in self.ancestries:
            ancestry_dir = base_dir / ancestry
            prefix = ancestry_dir / f"{ancestry}_release{self.config.release}_vwb"
            
            # Check if files exist
            if (Path(f"{prefix}.pgen").exists() and 
                Path(f"{prefix}.pvar").exists() and 
                Path(f"{prefix}.psam").exists()):
                
                metadata = {
                    "ancestry": ancestry,
                    "dataset_type": "nba",
                    "release": self.config.release
                }
                paths.append((str(prefix), metadata))
            else:
                # Log warning about missing files
                print(f"Warning: NBA files not found for ancestry {ancestry} at {prefix}")
        
        return paths
    
    def should_combine_results(self) -> bool:
        """NBA results should be combined across ancestries."""
        return len(self.ancestries) > 1
    
    def get_combine_strategy(self) -> str:
        """NBA uses ancestry combination strategy."""
        return "ancestry"
    
    def get_output_path(self, base_output: str, ancestry: str = None, **kwargs) -> str:
        """
        Get output path for NBA data.
        
        Args:
            base_output: Base output directory
            ancestry: Specific ancestry (optional)
            
        Returns:
            Output path
        """
        base_path = Path(base_output) / "nba" / f"release{self.config.release}"
        
        if ancestry:
            # Individual ancestry output
            return str(base_path / ancestry / f"{ancestry}_release{self.config.release}")
        else:
            # Combined output
            return str(base_path / "combined" / f"nba_release{self.config.release}_combined")
    
    def get_snplist_path(self, ancestry: str = None) -> str:
        """Get SNP list path for NBA data."""
        # NBA uses the same SNP list for all ancestries
        return self.config.snplist_path
