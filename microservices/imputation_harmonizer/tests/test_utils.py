"""Tests for utility functions."""

import pytest

from imputation_harmonizer.utils import (
    complement,
    complement_pair,
    extract_rsid,
    is_indel,
    is_palindromic,
    make_allele_key,
    make_chrpos_key,
    normalize_chromosome,
    sort_alleles,
)


class TestComplement:
    """Tests for complement function."""

    @pytest.mark.parametrize(
        "base,expected",
        [
            ("A", "T"),
            ("T", "A"),
            ("C", "G"),
            ("G", "C"),
            ("N", "N"),
        ],
    )
    def test_complement_bases(self, base: str, expected: str) -> None:
        """Test complement for each base."""
        assert complement(base) == expected

    def test_complement_unknown(self) -> None:
        """Test complement for unknown character returns same."""
        assert complement("X") == "X"


class TestComplementPair:
    """Tests for complement_pair function."""

    def test_complement_pair_ag(self) -> None:
        """Test complement of A/G pair."""
        assert complement_pair("A", "G") == ("T", "C")

    def test_complement_pair_ct(self) -> None:
        """Test complement of C/T pair."""
        assert complement_pair("C", "T") == ("G", "A")


class TestPalindromic:
    """Tests for is_palindromic function."""

    @pytest.mark.parametrize(
        "a1,a2,expected",
        [
            ("A", "T", True),
            ("T", "A", True),
            ("G", "C", True),
            ("C", "G", True),
            ("A", "G", False),
            ("A", "C", False),
            ("T", "G", False),
            ("T", "C", False),
        ],
    )
    def test_is_palindromic(self, a1: str, a2: str, expected: bool) -> None:
        """Test palindromic detection for various allele pairs."""
        assert is_palindromic(a1, a2) == expected


class TestIsIndel:
    """Tests for is_indel function."""

    @pytest.mark.parametrize(
        "a1,a2,expected",
        [
            ("A", "G", False),  # SNP
            ("C", "T", False),  # SNP
            ("-", "A", True),  # Deletion marker
            ("A", "-", True),  # Deletion marker
            ("I", "A", True),  # Insertion marker
            ("D", "A", True),  # Deletion marker
            ("AT", "A", True),  # Multi-base
            ("A", "ATG", True),  # Multi-base
        ],
    )
    def test_is_indel(self, a1: str, a2: str, expected: bool) -> None:
        """Test indel detection for various allele pairs."""
        assert is_indel(a1, a2) == expected


class TestSortAlleles:
    """Tests for sort_alleles function."""

    def test_sort_already_sorted(self) -> None:
        """Test sorting when already in order."""
        assert sort_alleles("A", "G") == ("A", "G")

    def test_sort_reversed(self) -> None:
        """Test sorting when in reverse order."""
        assert sort_alleles("G", "A") == ("A", "G")

    def test_sort_same(self) -> None:
        """Test sorting when alleles are same (shouldn't happen but handle)."""
        assert sort_alleles("A", "A") == ("A", "A")


class TestMakeChrposKey:
    """Tests for make_chrpos_key function."""

    def test_make_key(self) -> None:
        """Test key generation."""
        assert make_chrpos_key("1", 10000) == "1-10000"

    def test_make_key_string_chr(self) -> None:
        """Test key generation with X chromosome."""
        assert make_chrpos_key("X", 50000) == "X-50000"


class TestMakeAlleleKey:
    """Tests for make_allele_key function."""

    def test_make_key_sorted(self) -> None:
        """Test key with already sorted alleles."""
        assert make_allele_key("1", 10000, "A", "G") == "1-10000-A:G"

    def test_make_key_unsorted(self) -> None:
        """Test key with unsorted alleles (should be sorted)."""
        assert make_allele_key("1", 10000, "G", "A") == "1-10000-A:G"


class TestNormalizeChromosome:
    """Tests for normalize_chromosome function."""

    @pytest.mark.parametrize(
        "input_chr,expected",
        [
            ("1", "1"),
            ("01", "1"),
            ("22", "22"),
            ("chr1", "1"),
            ("Chr1", "1"),
            ("CHR1", "1"),
            ("chrX", "X"),
            ("X", "X"),
            ("MT", "MT"),
        ],
    )
    def test_normalize(self, input_chr: str, expected: str) -> None:
        """Test chromosome normalization for various formats."""
        assert normalize_chromosome(input_chr) == expected


class TestExtractRsid:
    """Tests for extract_rsid function."""

    def test_simple_rsid(self) -> None:
        """Test extraction from simple rsID."""
        assert extract_rsid("rs123") == "rs123"

    def test_compound_rsid(self) -> None:
        """Test extraction from compound ID."""
        assert extract_rsid("rs123:10177:A:C") == "rs123"

    def test_non_rsid(self) -> None:
        """Test extraction from non-rsID returns None."""
        assert extract_rsid("10:12345:A:G") is None

    def test_empty_string(self) -> None:
        """Test extraction from empty string."""
        assert extract_rsid("") is None
