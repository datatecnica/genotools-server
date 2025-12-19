"""Tests for file parsers."""

from pathlib import Path

import pytest

from imputation_harmonizer.parsers.bim import count_bim_variants, parse_bim
from imputation_harmonizer.parsers.frq import parse_frq


class TestBimParser:
    """Tests for BIM file parser."""

    def test_parse_valid_bim(self, tmp_path: Path) -> None:
        """Test parsing a valid BIM file."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text(
            "1\trs123\t0\t10000\tA\tG\n"
            "1\trs456\t0\t20000\tC\tT\n"
            "2\trs789\t0.5\t30000\tG\tA\n"
        )

        variants = list(parse_bim(bim_file))

        assert len(variants) == 3

        assert variants[0].chr == "1"
        assert variants[0].id == "rs123"
        assert variants[0].genetic_dist == 0.0
        assert variants[0].pos == 10000
        assert variants[0].allele1 == "A"
        assert variants[0].allele2 == "G"
        assert variants[0].freq is None

        assert variants[2].chr == "2"
        assert variants[2].genetic_dist == 0.5

    def test_parse_with_frequencies(self, tmp_path: Path) -> None:
        """Test parsing BIM with frequency lookup."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")

        frequencies = {"rs123": 0.35}
        variants = list(parse_bim(bim_file, frequencies))

        assert len(variants) == 1
        assert variants[0].freq == 0.35

    def test_parse_chr_prefix(self, tmp_path: Path) -> None:
        """Test parsing BIM with 'chr' prefix."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text("chr1\trs123\t0\t10000\tA\tG\n")

        variants = list(parse_bim(bim_file))

        assert len(variants) == 1
        assert variants[0].chr == "1"  # Should be normalized

    def test_parse_lowercase_alleles(self, tmp_path: Path) -> None:
        """Test that lowercase alleles are converted to uppercase."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text("1\trs123\t0\t10000\ta\tg\n")

        variants = list(parse_bim(bim_file))

        assert variants[0].allele1 == "A"
        assert variants[0].allele2 == "G"

    def test_parse_space_separated(self, tmp_path: Path) -> None:
        """Test parsing space-separated BIM file."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text("1 rs123 0 10000 A G\n")

        variants = list(parse_bim(bim_file))

        assert len(variants) == 1
        assert variants[0].id == "rs123"

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        missing_file = tmp_path / "missing.bim"

        with pytest.raises(FileNotFoundError):
            list(parse_bim(missing_file))

    def test_parse_invalid_format(self, tmp_path: Path) -> None:
        """Test error on invalid format."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text("1\trs123\t0\t10000\n")  # Only 4 columns

        with pytest.raises(ValueError, match="expected 6 columns"):
            list(parse_bim(bim_file))

    def test_count_variants(self, tmp_path: Path) -> None:
        """Test counting variants in BIM file."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text(
            "1\trs123\t0\t10000\tA\tG\n"
            "1\trs456\t0\t20000\tC\tT\n"
            "2\trs789\t0\t30000\tG\tA\n"
        )

        count = count_bim_variants(bim_file)
        assert count == 3


class TestFrqParser:
    """Tests for FRQ file parser."""

    def test_parse_valid_frq(self, tmp_path: Path) -> None:
        """Test parsing a valid FRQ file."""
        frq_file = tmp_path / "test.frq"
        frq_file.write_text(
            "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "1\trs456\tC\tT\t0.25\t1000\n"
        )

        frequencies = parse_frq(frq_file)

        assert len(frequencies) == 2
        assert frequencies["rs123"] == pytest.approx(0.30)
        assert frequencies["rs456"] == pytest.approx(0.25)

    def test_parse_frq_with_whitespace(self, tmp_path: Path) -> None:
        """Test parsing FRQ with leading whitespace (PLINK format)."""
        frq_file = tmp_path / "test.frq"
        frq_file.write_text(
            " CHR            SNP   A1   A2          MAF  NCHROBS\n"
            "   1      rs123456    A    G       0.1234     2000\n"
        )

        frequencies = parse_frq(frq_file)

        assert len(frequencies) == 1
        assert frequencies["rs123456"] == pytest.approx(0.1234)

    def test_parse_frq_missing_file(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        missing_file = tmp_path / "missing.frq"

        with pytest.raises(FileNotFoundError):
            parse_frq(missing_file)

    def test_parse_frq_skips_na(self, tmp_path: Path) -> None:
        """Test that NA values are skipped."""
        frq_file = tmp_path / "test.frq"
        frq_file.write_text(
            "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "1\trs456\tC\tT\tNA\t0\n"  # NA frequency
            "1\trs789\tG\tA\t0.15\t1000\n"
        )

        frequencies = parse_frq(frq_file)

        assert len(frequencies) == 2
        assert "rs456" not in frequencies
