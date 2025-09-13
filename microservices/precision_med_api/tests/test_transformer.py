"""
Tests for genotype transformation functionality.
"""

import pytest
import numpy as np

from app.processing.transformer import GenotypeTransformer


class TestGenotypeTransformer:
    """Test the GenotypeTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.transformer = GenotypeTransformer()
        
        # Test genotype arrays
        self.test_genotypes = np.array([0, 1, 2, 0, 1, 2, np.nan])
        self.simple_genotypes = np.array([0, 1, 2])
    
    def test_transform_identity(self):
        """Test identity transformation."""
        result = self.transformer._transform_identity(self.test_genotypes)
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
    
    def test_transform_swap(self):
        """Test swap transformation (2-x)."""
        result = self.transformer._transform_swap(self.simple_genotypes)
        expected = np.array([2, 1, 0])  # 0->2, 1->1, 2->0
        np.testing.assert_array_equal(result, expected)
        
        # Test with missing values
        genotypes_with_nan = np.array([0, 1, 2, np.nan])
        result_with_nan = self.transformer._transform_swap(genotypes_with_nan)
        expected_with_nan = np.array([2, 1, 0, np.nan])
        assert np.array_equal(result_with_nan, expected_with_nan, equal_nan=True)
    
    def test_apply_transformation_by_formula(self):
        """Test applying transformation by formula string."""
        # Test "2-x" formula
        result = self.transformer.apply_transformation_by_formula(self.simple_genotypes, "2-x")
        expected = np.array([2, 1, 0])
        np.testing.assert_array_equal(result, expected)
        
        # Test "x" formula (identity)
        result = self.transformer.apply_transformation_by_formula(self.test_genotypes, "x")
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
        
        # Test None formula (identity)
        result = self.transformer.apply_transformation_by_formula(self.test_genotypes, None)
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
        
        # Test unknown formula (should default to identity with warning)
        result = self.transformer.apply_transformation_by_formula(self.test_genotypes, "unknown")
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)