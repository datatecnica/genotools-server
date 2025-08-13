"""
Base classes for dataset handlers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    dataset_type: str
    release: str
    base_path: str
    ancestry: Optional[str] = None
    chromosome: Optional[str] = None
    
    def get_identifier(self) -> str:
        """Get unique identifier for this dataset configuration."""
        parts = [self.dataset_type, self.release]
        if self.ancestry:
            parts.append(self.ancestry)
        if self.chromosome:
            parts.append(f"chr{self.chromosome}")
        return "_".join(parts)


class DatasetHandler(ABC):
    """Abstract handler for different dataset types."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    @abstractmethod
    def get_file_patterns(self) -> List[str]:
        """
        Get file patterns for this dataset type.
        
        Returns:
            List of file patterns (glob-style)
        """
        pass
    
    @abstractmethod
    def get_genotype_paths(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get list of genotype file paths to process.
        
        Returns:
            List of tuples: (path, metadata) where metadata contains
            information like ancestry, chromosome, etc.
        """
        pass
    
    @abstractmethod
    def should_combine_results(self) -> bool:
        """Whether results need combining (e.g., across chromosomes)."""
        pass
    
    @abstractmethod
    def get_combine_strategy(self) -> str:
        """
        Get strategy for combining results.
        
        Returns:
            'chromosome' for combining chromosomes
            'ancestry' for combining ancestries
            None if no combination needed
        """
        pass
    
    @abstractmethod
    def get_output_path(self, base_output: str, **kwargs) -> str:
        """
        Get output path for this dataset.
        
        Args:
            base_output: Base output directory
            **kwargs: Additional parameters (ancestry, chromosome, etc.)
            
        Returns:
            Full output path
        """
        pass
    
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        paths = self.get_genotype_paths()
        if not paths:
            return False
            
        for path, _ in paths:
            # Check for PLINK2 files
            pgen_path = f"{path}.pgen"
            if not Path(pgen_path).exists():
                return False
                
        return True
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about this dataset."""
        return {
            "type": self.config.dataset_type,
            "release": self.config.release,
            "base_path": self.config.base_path,
            "ancestry": self.config.ancestry,
            "chromosome": self.config.chromosome,
            "file_patterns": self.get_file_patterns(),
            "requires_combination": self.should_combine_results(),
            "combine_strategy": self.get_combine_strategy() if self.should_combine_results() else None
        }
