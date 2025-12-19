"""Tests for strand checking logic."""

import pytest

from imputation_harmonizer.checks.strand import check_strand_and_alleles
from imputation_harmonizer.models import AlleleAction, ExcludeReason, StrandAction


class TestStrandCheck:
    """Test the 6 possible outcomes from check_strand_and_alleles."""

    def test_case1_strand_ok_refalt_ok(self) -> None:
        """Case 1: Alleles match exactly - no action needed."""
        # ref_freq = 1 - alt_af = 1 - 0.3 = 0.7
        # bim_af should be close to ref_freq (0.7), not alt_af (0.3)
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.3,
            bim_af=0.68,  # Close to ref_freq of 0.7
        )
        assert result.exclude is False
        assert result.strand_action == StrandAction.NONE
        assert result.allele_action == AlleleAction.NONE
        assert result.check_code == 1

    def test_case1_strand_ok_refalt_ok_good_freq(self) -> None:
        """Case 1: Alleles match exactly with acceptable frequency."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.3,
            bim_af=0.68,  # Close to ref_freq of 0.7
        )
        assert result.exclude is False
        assert result.strand_action == StrandAction.NONE
        assert result.allele_action == AlleleAction.NONE
        assert result.check_code == 1
        assert result.ref_freq == pytest.approx(0.7)

    def test_case2_strand_ok_refalt_swapped(self) -> None:
        """Case 2: Alleles swapped - need to force reference allele."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("G", "A"),
            ref_alt_af=0.3,
            bim_af=0.28,  # Close to alt_af of 0.3
        )
        assert result.exclude is False
        assert result.strand_action == StrandAction.NONE
        assert result.allele_action == AlleleAction.FORCE_REF
        assert result.force_ref_allele == "A"
        assert result.check_code == 2
        assert result.ref_freq == pytest.approx(0.3)

    def test_case3_strand_flip_refalt_ok(self) -> None:
        """Case 3: Opposite strand - need to flip."""
        # BIM has T/C which is complement of A/G
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("T", "C"),
            ref_alt_af=0.3,
            bim_af=0.68,  # Close to ref_freq of 0.7
        )
        assert result.exclude is False
        assert result.strand_action == StrandAction.FLIP
        assert result.allele_action == AlleleAction.NONE
        assert result.check_code == 3
        assert result.ref_freq == pytest.approx(0.7)

    def test_case4_strand_flip_refalt_swapped(self) -> None:
        """Case 4: Strand flipped and alleles swapped."""
        # BIM has C/T which is complement and swapped of A/G
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("C", "T"),
            ref_alt_af=0.3,
            bim_af=0.28,  # Close to alt_af of 0.3
        )
        assert result.exclude is False
        assert result.strand_action == StrandAction.FLIP
        assert result.allele_action == AlleleAction.FORCE_REF
        assert result.force_ref_allele == "A"
        assert result.check_code == 4

    def test_case5_palindromic_high_maf_excluded(self) -> None:
        """Case 5: Palindromic SNP with MAF > 0.4 - exclude."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "T"),
            bim_alleles=("A", "T"),
            ref_alt_af=0.45,  # MAF = 0.45 > 0.4
            bim_af=0.44,
        )
        assert result.exclude is True
        assert result.exclude_reason == ExcludeReason.PALINDROMIC_HIGH_MAF
        assert result.check_code == 5

    def test_case5_palindromic_gc_high_maf(self) -> None:
        """Case 5: G/C palindromic SNP with high MAF - exclude."""
        result = check_strand_and_alleles(
            ref_alleles=("G", "C"),
            bim_alleles=("G", "C"),
            ref_alt_af=0.48,
            bim_af=0.47,
        )
        assert result.exclude is True
        assert result.exclude_reason == ExcludeReason.PALINDROMIC_HIGH_MAF
        assert result.check_code == 5

    def test_palindromic_low_maf_ok(self) -> None:
        """Palindromic SNP with MAF <= 0.4 should be OK."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "T"),
            bim_alleles=("A", "T"),
            ref_alt_af=0.2,  # MAF = 0.2 <= 0.4
            bim_af=0.78,  # Close to ref_freq of 0.8
        )
        assert result.exclude is False
        assert result.check_code == 1

    def test_case6_freq_diff_excluded(self) -> None:
        """Case 6: Large frequency difference - exclude."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.3,
            bim_af=0.05,  # ref_freq = 0.7, diff = 0.65 > 0.2
        )
        assert result.exclude is True
        assert result.exclude_reason == ExcludeReason.FREQ_DIFF_TOO_HIGH
        assert result.check_code == 6

    def test_case0_allele_mismatch(self) -> None:
        """Case 0: No allele match - exclude."""
        # A/G vs C/A - no match even with complement
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("C", "A"),
            ref_alt_af=0.3,
            bim_af=0.3,
        )
        assert result.exclude is True
        assert result.exclude_reason == ExcludeReason.ALLELE_MISMATCH
        assert result.check_code == 0

    def test_custom_thresholds(self) -> None:
        """Test with custom thresholds."""
        # Palindromic with MAF 0.35 - excluded with threshold 0.3
        result = check_strand_and_alleles(
            ref_alleles=("A", "T"),
            bim_alleles=("A", "T"),
            ref_alt_af=0.35,
            bim_af=0.34,
            palindrome_maf_threshold=0.3,
        )
        assert result.exclude is True
        assert result.exclude_reason == ExcludeReason.PALINDROMIC_HIGH_MAF

    def test_custom_freq_diff_threshold(self) -> None:
        """Test with custom frequency difference threshold."""
        # freq diff = 0.15 - excluded with threshold 0.1
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.3,
            bim_af=0.55,  # ref_freq = 0.7, diff = 0.15
            freq_diff_threshold=0.1,
        )
        assert result.exclude is True
        assert result.exclude_reason == ExcludeReason.FREQ_DIFF_TOO_HIGH


class TestFrequencyCalculation:
    """Tests for frequency calculation in different cases."""

    def test_case1_frequency(self) -> None:
        """Case 1: ref_freq = 1 - alt_af."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("A", "G"),
            ref_alt_af=0.25,
            bim_af=0.73,
        )
        assert result.ref_freq == pytest.approx(0.75)
        assert result.freq_diff == pytest.approx(0.02)

    def test_case2_frequency(self) -> None:
        """Case 2: ref_freq = alt_af (alleles swapped)."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("G", "A"),
            ref_alt_af=0.25,
            bim_af=0.23,
        )
        assert result.ref_freq == pytest.approx(0.25)
        assert result.freq_diff == pytest.approx(0.02)

    def test_case3_frequency(self) -> None:
        """Case 3: ref_freq = 1 - alt_af (strand flipped)."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("T", "C"),
            ref_alt_af=0.25,
            bim_af=0.73,
        )
        assert result.ref_freq == pytest.approx(0.75)
        assert result.freq_diff == pytest.approx(0.02)

    def test_case4_frequency(self) -> None:
        """Case 4: ref_freq = alt_af (flipped and swapped)."""
        result = check_strand_and_alleles(
            ref_alleles=("A", "G"),
            bim_alleles=("C", "T"),
            ref_alt_af=0.25,
            bim_af=0.23,
        )
        assert result.ref_freq == pytest.approx(0.25)
        assert result.freq_diff == pytest.approx(0.02)
