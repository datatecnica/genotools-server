"""
Tests for genotype transformation functionality.
"""

import pytest
import numpy as np
import pandas as pd
from typing import List

from app.processing.transformer import GenotypeTransformer
from app.models.harmonization import HarmonizationAction


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
    
    def test_transform_for_swap(self):
        """Test transform_for_swap method."""
        result = self.transformer.transform_for_swap(self.simple_genotypes)
        expected = np.array([2, 1, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_transform_for_flip(self):
        """Test transform_for_flip method (should be identity)."""
        result = self.transformer.transform_for_flip(self.test_genotypes)
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
    
    def test_transform_for_flip_swap(self):
        """Test transform_for_flip_swap method (should be same as swap)."""
        result = self.transformer.transform_for_flip_swap(self.simple_genotypes)
        expected = np.array([2, 1, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_apply_transformation_by_action(self):
        """Test applying transformation by harmonization action."""
        # Test EXACT
        result = self.transformer.apply_transformation(self.test_genotypes, HarmonizationAction.EXACT)
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
        
        # Test FLIP
        result = self.transformer.apply_transformation(self.test_genotypes, HarmonizationAction.FLIP)
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
        
        # Test SWAP
        result = self.transformer.apply_transformation(self.simple_genotypes, HarmonizationAction.SWAP)
        expected = np.array([2, 1, 0])
        np.testing.assert_array_equal(result, expected)
        
        # Test FLIP_SWAP
        result = self.transformer.apply_transformation(self.simple_genotypes, HarmonizationAction.FLIP_SWAP)
        expected = np.array([2, 1, 0])
        np.testing.assert_array_equal(result, expected)
    
    def test_apply_transformation_by_string(self):
        """Test applying transformation by string action."""
        # Test string actions
        result = self.transformer.apply_transformation(self.simple_genotypes, "SWAP")
        expected = np.array([2, 1, 0])
        np.testing.assert_array_equal(result, expected)
        
        result = self.transformer.apply_transformation(self.test_genotypes, "EXACT")
        assert np.array_equal(result, self.test_genotypes, equal_nan=True)
    
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
    
    def test_transform_matrix(self):
        """Test transforming a matrix of genotypes."""
        # Create test matrix (3 variants x 5 samples)
        genotype_matrix = np.array([
            [0, 1, 2, 0, 1],  # Variant 1
            [2, 1, 0, 2, 1],  # Variant 2
            [1, 0, 1, 2, 0]   # Variant 3
        ])
        
        actions = [HarmonizationAction.EXACT, HarmonizationAction.SWAP, HarmonizationAction.FLIP]
        
        result = self.transformer.transform_matrix(genotype_matrix, actions)
        
        # Expected results
        expected = np.array([
            [0, 1, 2, 0, 1],  # Variant 1 (EXACT): no change
            [0, 1, 2, 0, 1],  # Variant 2 (SWAP): 2->0, 1->1, 0->2
            [1, 0, 1, 2, 0]   # Variant 3 (FLIP): no change
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_batch_transform_by_formula(self):
        """Test batch transformation by formula."""
        genotype_matrix = np.array([
            [0, 1, 2],
            [2, 1, 0],
            [1, 0, 1]
        ])
        
        formulas = ["x", "2-x", None]
        
        result = self.transformer.batch_transform_by_formula(genotype_matrix, formulas)
        
        expected = np.array([
            [0, 1, 2],  # "x": identity
            [0, 1, 2],  # "2-x": swap
            [1, 0, 1]   # None: identity
        ])
        
        np.testing.assert_array_equal(result, expected)
    
    def test_validate_transformation(self):
        """Test transformation validation."""
        original = np.array([0, 1, 2, 0, 1])
        
        # Test valid identity transformation
        transformed_identity = np.array([0, 1, 2, 0, 1])
        assert self.transformer.validate_transformation(
            original, transformed_identity, HarmonizationAction.EXACT
        ) == True
        
        # Test valid swap transformation
        transformed_swap = np.array([2, 1, 0, 2, 1])
        assert self.transformer.validate_transformation(
            original, transformed_swap, HarmonizationAction.SWAP
        ) == True
        
        # Test invalid transformation
        invalid_transformed = np.array([1, 0, 1, 1, 0])
        assert self.transformer.validate_transformation(
            original, invalid_transformed, HarmonizationAction.SWAP
        ) == False
    
    def test_get_allele_counts(self):
        """Test allele count calculation."""
        genotypes = np.array([0, 0, 1, 1, 2, 2, np.nan])  # 2 x 0/0, 2 x 0/1, 2 x 1/1, 1 missing
        
        counts = self.transformer.get_allele_counts(genotypes, HarmonizationAction.EXACT)
        
        expected_counts = {
            'n_samples': 6,
            'n_00': 2,
            'n_01': 2,
            'n_11': 2,
            'n_allele1': 6,  # 2*2 + 2 = 6 reference alleles
            'n_allele2': 6,  # 2*2 + 2 = 6 alternate alleles
            'frequency_allele2': 0.5
        }
        
        for key, expected_value in expected_counts.items():
            assert counts[key] == expected_value
    
    def test_get_allele_counts_with_swap(self):
        """Test allele counts with swap transformation."""
        genotypes = np.array([0, 1, 2])  # Before swap: 0, 1, 2
        
        counts = self.transformer.get_allele_counts(genotypes, HarmonizationAction.SWAP)
        
        # After swap: 2, 1, 0
        expected_counts = {
            'n_samples': 3,
            'n_00': 1,  # Was 2, now 0
            'n_01': 1,  # Stays 1
            'n_11': 1,  # Was 0, now 2
            'n_allele1': 3,  # 1*2 + 1 = 3
            'n_allele2': 3,  # 1*2 + 1 = 3
            'frequency_allele2': 0.5
        }
        
        for key, expected_value in expected_counts.items():
            assert counts[key] == expected_value
    
    def test_compare_allele_frequencies(self):
        """Test allele frequency comparison."""
        before = np.array([0, 1, 2, 0])  # Frequency = 3/8 = 0.375
        after = np.array([2, 1, 0, 2])   # Frequency = 5/8 = 0.625
        
        comparison = self.transformer.compare_allele_frequencies(before, after)
        
        assert comparison['frequency_before'] == 0.375
        assert comparison['frequency_after'] == 0.625
        assert comparison['frequency_diff'] == 0.25
        assert comparison['samples_before'] == 4
        assert comparison['samples_after'] == 4
    
    def test_transform_dataframe(self):
        """Test transforming genotypes in a DataFrame."""
        df = pd.DataFrame({
            'variant_id': ['var1', 'var2', 'var3'],
            'harmonization_action': ['EXACT', 'SWAP', 'FLIP'],
            'genotype_transform': [None, '2-x', None],
            'sample1': [0, 2, 1],
            'sample2': [1, 1, 0],
            'sample3': [2, 0, 1]
        })
        
        genotype_cols = ['sample1', 'sample2', 'sample3']
        result = self.transformer.transform_dataframe(df, genotype_cols)
        
        # Check transformations
        assert result.loc[0, 'sample1'] == 0  # EXACT: no change
        assert result.loc[1, 'sample1'] == 0  # SWAP: 2 -> 0
        assert result.loc[1, 'sample3'] == 2  # SWAP: 0 -> 2
        assert result.loc[2, 'sample1'] == 1  # FLIP: no change
    
    def test_get_transformation_summary(self):
        """Test getting transformation summary."""
        df = pd.DataFrame({
            'harmonization_action': ['EXACT', 'SWAP', 'FLIP', 'SWAP', 'FLIP_SWAP'],
            'genotype_transform': [None, '2-x', None, '2-x', '2-x']
        })
        
        summary = self.transformer.get_transformation_summary(df)
        
        assert summary['total_variants'] == 5
        assert summary['action_exact'] == 1
        assert summary['action_swap'] == 2
        assert summary['action_flip'] == 1
        assert summary['action_flip_swap'] == 1
        assert summary['requires_transformation'] == 3  # SWAP and FLIP_SWAP
        assert summary['transformation_rate'] == 0.6
        assert summary['transformation_formulas']['2-x'] == 3