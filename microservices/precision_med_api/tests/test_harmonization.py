"""
Tests for variant harmonization functionality.
"""

import pytest
import pandas as pd
import numpy as np
from typing import List, Tuple

from app.models.harmonization import HarmonizationAction, HarmonizationRecord, HarmonizationStats
from app.processing.cache import AlleleHarmonizer


class TestAlleleHarmonizer:
    """Test the AlleleHarmonizer class."""
    
    def test_complement_allele(self):
        """Test allele complement function."""
        harmonizer = AlleleHarmonizer()
        
        assert harmonizer.complement_allele('A') == 'T'
        assert harmonizer.complement_allele('T') == 'A'
        assert harmonizer.complement_allele('C') == 'G'
        assert harmonizer.complement_allele('G') == 'C'
        assert harmonizer.complement_allele('N') == 'N'
        assert harmonizer.complement_allele('.') == '.'
        
        # Test case insensitive
        assert harmonizer.complement_allele('a') == 'T'
        assert harmonizer.complement_allele('t') == 'A'
    
    def test_check_strand_ambiguous(self):
        """Test strand ambiguous detection."""
        harmonizer = AlleleHarmonizer()
        
        # Ambiguous pairs
        assert harmonizer.check_strand_ambiguous('A', 'T') == True
        assert harmonizer.check_strand_ambiguous('T', 'A') == True
        assert harmonizer.check_strand_ambiguous('C', 'G') == True
        assert harmonizer.check_strand_ambiguous('G', 'C') == True
        
        # Non-ambiguous pairs
        assert harmonizer.check_strand_ambiguous('A', 'C') == False
        assert harmonizer.check_strand_ambiguous('A', 'G') == False
        assert harmonizer.check_strand_ambiguous('T', 'C') == False
        assert harmonizer.check_strand_ambiguous('T', 'G') == False
    
    def test_get_all_representations(self):
        """Test getting all possible allele representations."""
        harmonizer = AlleleHarmonizer()
        
        representations = harmonizer.get_all_representations('A', 'C')
        expected = [
            ('A', 'C', 'EXACT'),
            ('C', 'A', 'SWAP'),
            ('T', 'G', 'FLIP'),
            ('G', 'T', 'FLIP_SWAP')
        ]
        
        assert len(representations) == 4
        assert representations == expected
    
    def test_determine_harmonization_exact_match(self):
        """Test harmonization determination for exact matches."""
        harmonizer = AlleleHarmonizer()
        
        action, transform = harmonizer.determine_harmonization('A', 'C', 'A', 'C')
        assert action == 'EXACT'
        assert transform is None
    
    def test_determine_harmonization_swap(self):
        """Test harmonization determination for allele swaps."""
        harmonizer = AlleleHarmonizer()
        
        action, transform = harmonizer.determine_harmonization('A', 'C', 'C', 'A')
        assert action == 'SWAP'
        assert transform == '2-x'
    
    def test_determine_harmonization_flip(self):
        """Test harmonization determination for strand flips."""
        harmonizer = AlleleHarmonizer()
        
        action, transform = harmonizer.determine_harmonization('A', 'C', 'T', 'G')
        assert action == 'FLIP'
        assert transform is None
    
    def test_determine_harmonization_flip_swap(self):
        """Test harmonization determination for flip and swap."""
        harmonizer = AlleleHarmonizer()
        
        action, transform = harmonizer.determine_harmonization('A', 'C', 'G', 'T')
        assert action == 'FLIP_SWAP'
        assert transform == '2-x'
    
    def test_determine_harmonization_ambiguous(self):
        """Test harmonization determination for ambiguous variants."""
        harmonizer = AlleleHarmonizer()
        
        action, transform = harmonizer.determine_harmonization('A', 'T', 'A', 'T')
        assert action == 'AMBIGUOUS'
        assert transform is None
    
    def test_determine_harmonization_invalid(self):
        """Test harmonization determination for invalid variants."""
        harmonizer = AlleleHarmonizer()
        
        action, transform = harmonizer.determine_harmonization('A', 'C', 'A', 'G')
        assert action == 'INVALID'
        assert transform is None


class TestHarmonizationRecord:
    """Test the HarmonizationRecord model."""
    
    def test_harmonization_record_creation(self):
        """Test creating a harmonization record."""
        record = HarmonizationRecord(
            snp_list_id="chr1:123456:A:C",
            pgen_variant_id="1:123456:C:A",
            chromosome="1",
            position=123456,
            snp_list_a1="A",
            snp_list_a2="C",
            pgen_a1="C",
            pgen_a2="A",
            harmonization_action=HarmonizationAction.SWAP,
            genotype_transform="2-x",
            file_path="/path/to/file.pgen",
            data_type="NBA",
            ancestry="EUR"
        )
        
        assert record.variant_key == "1:123456:A:C"
        assert record.requires_transformation == True
        assert record.is_strand_ambiguous == False
    
    def test_harmonization_record_normalization(self):
        """Test field normalization in harmonization record."""
        record = HarmonizationRecord(
            snp_list_id="chr1:123456:A:C",
            pgen_variant_id="1:123456:C:A",
            chromosome="chr1",  # Should be normalized
            position=123456,
            snp_list_a1=" a ",  # Should be normalized
            snp_list_a2=" c ",  # Should be normalized
            pgen_a1="C",
            pgen_a2="A",
            harmonization_action=HarmonizationAction.EXACT,
            file_path="/path/to/file.pgen",
            data_type="NBA"
        )
        
        assert record.chromosome == "1"
        assert record.snp_list_a1 == "A"
        assert record.snp_list_a2 == "C"
    
    def test_strand_ambiguous_detection(self):
        """Test strand ambiguous detection in harmonization record."""
        # A/T variant
        record1 = HarmonizationRecord(
            snp_list_id="chr1:123456:A:T",
            pgen_variant_id="1:123456:A:T",
            chromosome="1",
            position=123456,
            snp_list_a1="A",
            snp_list_a2="T",
            pgen_a1="A",
            pgen_a2="T",
            harmonization_action=HarmonizationAction.AMBIGUOUS,
            file_path="/path/to/file.pgen",
            data_type="NBA"
        )
        assert record1.is_strand_ambiguous == True
        
        # C/G variant
        record2 = HarmonizationRecord(
            snp_list_id="chr1:123456:C:G",
            pgen_variant_id="1:123456:C:G",
            chromosome="1",
            position=123456,
            snp_list_a1="C",
            snp_list_a2="G",
            pgen_a1="C",
            pgen_a2="G",
            harmonization_action=HarmonizationAction.AMBIGUOUS,
            file_path="/path/to/file.pgen",
            data_type="NBA"
        )
        assert record2.is_strand_ambiguous == True
        
        # Non-ambiguous variant
        record3 = HarmonizationRecord(
            snp_list_id="chr1:123456:A:C",
            pgen_variant_id="1:123456:A:C",
            chromosome="1",
            position=123456,
            snp_list_a1="A",
            snp_list_a2="C",
            pgen_a1="A",
            pgen_a2="C",
            harmonization_action=HarmonizationAction.EXACT,
            file_path="/path/to/file.pgen",
            data_type="NBA"
        )
        assert record3.is_strand_ambiguous == False


class TestHarmonizationStats:
    """Test the HarmonizationStats model."""
    
    def test_harmonization_stats_creation(self):
        """Test creating harmonization statistics."""
        stats = HarmonizationStats(
            total_variants=100,
            exact_matches=50,
            swapped_alleles=30,
            flipped_strand=15,
            flip_and_swap=3,
            invalid_variants=1,
            ambiguous_variants=1
        )
        
        assert stats.harmonized_variants == 98
        assert stats.harmonization_rate == 0.98
        assert stats.failure_rate == 0.02
    
    def test_harmonization_stats_update_from_records(self):
        """Test updating stats from harmonization records."""
        records = [
            HarmonizationRecord(
                snp_list_id=f"chr1:{i}:A:C",
                pgen_variant_id=f"1:{i}:A:C",
                chromosome="1",
                position=i,
                snp_list_a1="A",
                snp_list_a2="C",
                pgen_a1="A",
                pgen_a2="C",
                harmonization_action=HarmonizationAction.EXACT,
                file_path="/path/to/file.pgen",
                data_type="NBA"
            ) for i in range(100000, 100050)  # 50 exact matches
        ]
        
        # Add some swapped variants
        for i in range(100050, 100080):  # 30 swapped
            records.append(
                HarmonizationRecord(
                    snp_list_id=f"chr1:{i}:A:C",
                    pgen_variant_id=f"1:{i}:C:A",
                    chromosome="1",
                    position=i,
                    snp_list_a1="A",
                    snp_list_a2="C",
                    pgen_a1="C",
                    pgen_a2="A",
                    harmonization_action=HarmonizationAction.SWAP,
                    genotype_transform="2-x",
                    file_path="/path/to/file.pgen",
                    data_type="NBA"
                )
            )
        
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(records)
        
        assert stats.total_variants == 80
        assert stats.exact_matches == 50
        assert stats.swapped_alleles == 30
        assert stats.harmonization_rate == 1.0
    
    def test_harmonization_stats_summary_dict(self):
        """Test summary dictionary generation."""
        stats = HarmonizationStats(
            total_variants=100,
            exact_matches=50,
            swapped_alleles=30,
            flipped_strand=15,
            flip_and_swap=3,
            invalid_variants=1,
            ambiguous_variants=1
        )
        
        summary = stats.summary_dict
        
        expected_keys = [
            'total_variants', 'harmonized', 'harmonization_rate',
            'exact_matches', 'swapped_alleles', 'flipped_strand',
            'flip_and_swap', 'invalid', 'ambiguous', 'failure_rate'
        ]
        
        for key in expected_keys:
            assert key in summary
        
        assert summary['total_variants'] == 100
        assert summary['harmonized'] == 98
        assert summary['harmonization_rate'] == 0.98
        assert summary['failure_rate'] == 0.02