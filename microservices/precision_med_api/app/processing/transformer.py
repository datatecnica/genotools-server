"""
Genotype transformation logic for allele harmonization.

Handles genotype transformations needed when harmonizing variants between
SNP lists and PLINK files, including allele swaps and strand flips.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any
import logging

from ..models.harmonization import HarmonizationAction

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
    
    def transform_for_swap(self, genotypes: np.ndarray) -> np.ndarray:
        """Transform genotypes for allele swap (A1<->A2)."""
        return self._transform_swap(genotypes)
    
    def transform_for_flip(self, genotypes: np.ndarray) -> np.ndarray:
        """
        Transform genotypes for strand flip.
        Strand flip doesn't change genotype counts, only allele labels.
        """
        return self._transform_identity(genotypes)
    
    def transform_for_flip_swap(self, genotypes: np.ndarray) -> np.ndarray:
        """Transform genotypes for both strand flip and allele swap."""
        # Strand flip doesn't affect genotypes, so only apply swap
        return self._transform_swap(genotypes)
    
    def apply_transformation(
        self, 
        genotypes: np.ndarray, 
        action: Union[str, HarmonizationAction]
    ) -> np.ndarray:
        """
        Apply transformation based on harmonization action.
        
        Args:
            genotypes: Array of genotypes (0, 1, 2, or NaN for missing)
            action: Harmonization action or transformation string
            
        Returns:
            Transformed genotypes array
        """
        if isinstance(action, HarmonizationAction):
            action_str = action.value
        else:
            action_str = str(action)
        
        if action_str == "EXACT" or action_str == "FLIP":
            return self.transform_for_flip(genotypes)
        elif action_str == "SWAP":
            return self.transform_for_swap(genotypes)
        elif action_str == "FLIP_SWAP":
            return self.transform_for_flip_swap(genotypes)
        else:
            logger.warning(f"Unknown harmonization action: {action_str}, applying identity")
            return self._transform_identity(genotypes)
    
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
    
    def transform_matrix(
        self, 
        genotype_matrix: np.ndarray, 
        actions: List[Union[str, HarmonizationAction]]
    ) -> np.ndarray:
        """
        Transform genotype matrix with different actions per variant.
        
        Args:
            genotype_matrix: 2D array (variants x samples)
            actions: List of harmonization actions for each variant
            
        Returns:
            Transformed genotype matrix
        """
        if len(actions) != genotype_matrix.shape[0]:
            raise ValueError(f"Number of actions ({len(actions)}) must match number of variants ({genotype_matrix.shape[0]})")
        
        transformed_matrix = np.zeros_like(genotype_matrix)
        
        for i, action in enumerate(actions):
            transformed_matrix[i, :] = self.apply_transformation(genotype_matrix[i, :], action)
        
        return transformed_matrix
    
    def batch_transform_by_formula(
        self, 
        genotype_matrix: np.ndarray, 
        formulas: List[Optional[str]]
    ) -> np.ndarray:
        """
        Batch transform genotypes using formula strings.
        
        Args:
            genotype_matrix: 2D array (variants x samples)
            formulas: List of transformation formulas for each variant
            
        Returns:
            Transformed genotype matrix
        """
        if len(formulas) != genotype_matrix.shape[0]:
            raise ValueError(f"Number of formulas ({len(formulas)}) must match number of variants ({genotype_matrix.shape[0]})")
        
        transformed_matrix = np.zeros_like(genotype_matrix)
        
        for i, formula in enumerate(formulas):
            transformed_matrix[i, :] = self.apply_transformation_by_formula(genotype_matrix[i, :], formula)
        
        return transformed_matrix
    
    def validate_transformation(
        self, 
        original: np.ndarray, 
        transformed: np.ndarray, 
        action: Union[str, HarmonizationAction]
    ) -> bool:
        """
        Validate that transformation was applied correctly.
        
        Args:
            original: Original genotypes
            transformed: Transformed genotypes
            action: Harmonization action applied
            
        Returns:
            True if transformation is valid
        """
        try:
            if isinstance(action, HarmonizationAction):
                action_str = action.value
            else:
                action_str = str(action)
            
            # Check array shapes match
            if original.shape != transformed.shape:
                return False
            
            # For identity transformations (EXACT, FLIP), arrays should be identical
            if action_str in ("EXACT", "FLIP"):
                return np.array_equal(original, transformed, equal_nan=True)
            
            # For swap transformations, check that 0<->2 swap was applied correctly
            elif action_str in ("SWAP", "FLIP_SWAP"):
                # Create expected result
                expected = 2 - original
                return np.array_equal(expected, transformed, equal_nan=True)
            
            else:
                logger.warning(f"Cannot validate unknown action: {action_str}")
                return True  # Assume valid for unknown actions
                
        except Exception as e:
            logger.error(f"Error validating transformation: {e}")
            return False
    
    def get_allele_counts(
        self, 
        genotypes: np.ndarray, 
        action: Union[str, HarmonizationAction]
    ) -> Dict[str, int]:
        """
        Get allele counts after transformation.
        
        Args:
            genotypes: Genotype array (0, 1, 2)
            action: Harmonization action
            
        Returns:
            Dictionary with allele counts
        """
        # Apply transformation
        transformed = self.apply_transformation(genotypes, action)
        
        # Remove missing values
        valid_gts = transformed[~np.isnan(transformed)]
        
        if len(valid_gts) == 0:
            return {"n_samples": 0, "n_allele1": 0, "n_allele2": 0, "frequency_allele2": 0.0}
        
        # Count genotypes
        n_00 = np.sum(valid_gts == 0)  # Homozygous reference
        n_01 = np.sum(valid_gts == 1)  # Heterozygous
        n_11 = np.sum(valid_gts == 2)  # Homozygous alternate
        
        # Count alleles
        n_allele1 = 2 * n_00 + n_01  # Reference allele count
        n_allele2 = 2 * n_11 + n_01  # Alternate allele count
        total_alleles = n_allele1 + n_allele2
        
        return {
            "n_samples": len(valid_gts),
            "n_00": n_00,
            "n_01": n_01, 
            "n_11": n_11,
            "n_allele1": n_allele1,
            "n_allele2": n_allele2,
            "frequency_allele2": n_allele2 / total_alleles if total_alleles > 0 else 0.0
        }
    
    def compare_allele_frequencies(
        self, 
        genotypes_before: np.ndarray,
        genotypes_after: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare allele frequencies before and after transformation.
        
        Args:
            genotypes_before: Original genotypes
            genotypes_after: Transformed genotypes
            
        Returns:
            Dictionary with frequency comparison
        """
        # Calculate frequencies for both
        before_counts = self.get_allele_counts(genotypes_before, "EXACT")
        after_counts = self.get_allele_counts(genotypes_after, "EXACT")
        
        return {
            "frequency_before": before_counts["frequency_allele2"],
            "frequency_after": after_counts["frequency_allele2"],
            "frequency_diff": abs(before_counts["frequency_allele2"] - after_counts["frequency_allele2"]),
            "samples_before": before_counts["n_samples"],
            "samples_after": after_counts["n_samples"]
        }
    
    def transform_dataframe(
        self, 
        df: pd.DataFrame,
        genotype_columns: List[str],
        action_column: str = "harmonization_action",
        formula_column: str = "genotype_transform"
    ) -> pd.DataFrame:
        """
        Transform genotypes in a DataFrame.
        
        Args:
            df: DataFrame with genotype data
            genotype_columns: List of columns containing genotypes
            action_column: Column with harmonization actions
            formula_column: Column with transformation formulas
            
        Returns:
            DataFrame with transformed genotypes
        """
        df_transformed = df.copy()
        
        for idx, row in df.iterrows():
            # Get transformation info
            if formula_column in row and pd.notna(row[formula_column]):
                formula = row[formula_column]
            elif action_column in row:
                action = row[action_column]
                formula = "2-x" if action in ("SWAP", "FLIP_SWAP") else "x"
            else:
                formula = "x"  # Identity
            
            # Transform each genotype column
            for col in genotype_columns:
                if col in df_transformed.columns and pd.notna(row[col]):
                    original_gts = np.array([row[col]])
                    transformed_gts = self.apply_transformation_by_formula(original_gts, formula)
                    df_transformed.at[idx, col] = transformed_gts[0]
        
        return df_transformed
    
    def get_transformation_summary(
        self, 
        harmonization_records: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get summary of transformations needed.
        
        Args:
            harmonization_records: DataFrame with harmonization records
            
        Returns:
            Summary dictionary
        """
        if harmonization_records.empty:
            return {"total_variants": 0}
        
        summary = {"total_variants": len(harmonization_records)}
        
        # Count by action
        action_counts = harmonization_records["harmonization_action"].value_counts().to_dict()
        for action in HarmonizationAction:
            summary[f"action_{action.value.lower()}"] = action_counts.get(action.value, 0)
        
        # Count requiring transformation
        requires_transform = harmonization_records["genotype_transform"].notna().sum()
        summary["requires_transformation"] = requires_transform
        summary["transformation_rate"] = requires_transform / len(harmonization_records)
        
        # Most common transformations
        transform_counts = harmonization_records["genotype_transform"].value_counts().to_dict()
        summary["transformation_formulas"] = transform_counts
        
        return summary