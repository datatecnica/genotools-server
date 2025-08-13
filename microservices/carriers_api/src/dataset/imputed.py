"""
Imputed dataset handler.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .base import DatasetHandler, DatasetConfig


class ImputedDatasetHandler(DatasetHandler):
    """
    Handler for Imputed dataset type.
    
    Imputed data has chromosome-split PLINK2 files per ancestry:
    - Format: chr{chrom}_{ancestry}_release{version}_vwb.{pgen|pvar|psam}
    - Example: chr1_AAC_release10_vwb.pgen
    """
    
    # Default chromosomes if not specified
    DEFAULT_CHROMOSOMES = [str(i) for i in range(1, 23)] + ['X']
    
    def __init__(self, config: DatasetConfig, chromosomes: List[str] = None):
        super().__init__(config)
        self.chromosomes = chromosomes or self.DEFAULT_CHROMOSOMES
        
        # For imputed data, ancestry must be specified
        if not config.ancestry:
            raise ValueError("Ancestry must be specified for imputed data")
    
    def get_file_patterns(self) -> List[str]:
        """Get file patterns for imputed data."""
        patterns = []
        for chrom in self.chromosomes:
            patterns.append(f"chr{chrom}_{self.config.ancestry}_release{self.config.release}_vwb")
        return patterns
    
    def get_genotype_paths(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get list of imputed genotype files to process."""
        paths = []
        
        base_dir = Path(self.config.base_path) / self.config.ancestry
        
        # Check each chromosome
        for chrom in self.chromosomes:
            prefix = base_dir / f"chr{chrom}_{self.config.ancestry}_release{self.config.release}_vwb"
            
            # Check if files exist
            if (Path(f"{prefix}.pgen").exists() and 
                Path(f"{prefix}.pvar").exists() and 
                Path(f"{prefix}.psam").exists()):
                
                metadata = {
                    "ancestry": self.config.ancestry,
                    "chromosome": chrom,
                    "dataset_type": "imputed",
                    "release": self.config.release
                }
                paths.append((str(prefix), metadata))
        
        return paths
    
    def should_combine_results(self) -> bool:
        """Imputed results need to be combined across chromosomes."""
        return True
    
    def get_combine_strategy(self) -> str:
        """Imputed uses chromosome combination strategy."""
        return "chromosome"
    
    def get_output_path(self, base_output: str, chromosome: str = None, **kwargs) -> str:
        """
        Get output path for imputed data.
        
        Args:
            base_output: Base output directory
            chromosome: Specific chromosome (optional)
            
        Returns:
            Output path
        """
        base_path = Path(base_output) / "imputed" / f"release{self.config.release}" / self.config.ancestry
        
        if chromosome:
            # Individual chromosome output
            return str(base_path / f"{self.config.ancestry}_release{self.config.release}_chr{chromosome}")
        else:
            # Combined output
            return str(base_path / f"{self.config.ancestry}_release{self.config.release}")
    
    def get_snplist_path(self) -> str:
        """Get SNP list path for imputed data."""
        return self.config.snplist_path
    
    @staticmethod
    def process_all_ancestries(base_config: DatasetConfig, ancestries: List[str], 
                             chromosomes: Optional[List[str]] = None) -> List['ImputedDatasetHandler']:
        """
        Create handlers for multiple ancestries.
        
        Args:
            base_config: Base configuration
            ancestries: List of ancestry labels
            chromosomes: List of chromosomes (optional)
            
        Returns:
            List of ImputedDatasetHandler instances
        """
        handlers = []
        for ancestry in ancestries:
            config = DatasetConfig(
                dataset_type="imputed",
                release=base_config.release,
                base_path=base_config.base_path,
                ancestry=ancestry
            )
            handler = ImputedDatasetHandler(config, chromosomes)
            handlers.append(handler)
        return handlers
