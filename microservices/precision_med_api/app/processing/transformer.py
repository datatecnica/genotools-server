"""
Genotype transformation logic for allele harmonization.

Handles genotype transformations needed when harmonizing variants between
SNP lists and PLINK files, including allele swaps and strand flips.
"""

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GenotypeTransformer:
    """Handles genotype transformations for allele harmonization."""
    
    def __init__(self):
        # Mapping from transformation string to function
        self.transform_functions = {
            "2-x": self._transform_swap,
            "x": self._transform_identity,
            None: self._transform_identity
        }
    
    @staticmethod
    def _transform_identity(genotypes: np.ndarray) -> np.ndarray:
        """Identity transformation - no change."""
        return genotypes.copy()
    
    @staticmethod
    def _transform_swap(genotypes: np.ndarray) -> np.ndarray:
        """
        Swap transformation for allele orientation.
        0 (AA) -> 2 (BB), 1 (AB) -> 1 (AB), 2 (BB) -> 0 (AA)
        """
        return 2 - genotypes
    
    def apply_transformation_by_formula(
        self, 
        genotypes: np.ndarray, 
        formula: Optional[str]
    ) -> np.ndarray:
        """
        Apply transformation using formula string.
        
        Args:
            genotypes: Array of genotypes
            formula: Transformation formula (e.g., "2-x", "x", None)
            
        Returns:
            Transformed genotypes array
        """
        if formula in self.transform_functions:
            return self.transform_functions[formula](genotypes)
        else:
            logger.warning(f"Unknown transformation formula: {formula}, applying identity")
            return self._transform_identity(genotypes)