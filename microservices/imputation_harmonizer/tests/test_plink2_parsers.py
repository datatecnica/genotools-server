"""Tests for PLINK2 format parsers (pvar and afreq)."""

import gzip
from pathlib import Path

import pytest

from imputation_harmonizer.parsers.pvar import count_pvar_variants, parse_pvar
from imputation_harmonizer.parsers.afreq import parse_afreq


class TestPvarParser:
    """Test PLINK2 .pvar file parsing."""

    def test_parse_valid_pvar(self, tmp_path: Path) -> None:
        """Parse a valid .pvar file."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t10000\trs123\tA\tG\n"
            "1\t20000\trs456\tC\tT\n"
            "2\t30000\trs789\tG\tA\n"
        )

        variants = list(parse_pvar(pvar_file))

        assert len(variants) == 3
        assert variants[0].id == "rs123"
        assert variants[0].chr == "1"
        assert variants[0].pos == 10000
        assert variants[0].allele1 == "A"
        assert variants[0].allele2 == "G"

    def test_parse_pvar_with_frequencies(self, tmp_path: Path) -> None:
        """Parse .pvar with frequency data."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t10000\trs123\tA\tG\n"
            "1\t20000\trs456\tC\tT\n"
        )

        frequencies = {"rs123": 0.25, "rs456": 0.40}
        variants = list(parse_pvar(pvar_file, frequencies))

        assert variants[0].freq == 0.25
        assert variants[1].freq == 0.40

    def test_parse_pvar_chr_prefix(self, tmp_path: Path) -> None:
        """Parse .pvar with 'chr' prefix in chromosome."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "chr1\t10000\trs123\tA\tG\n"
            "chr22\t20000\trs456\tC\tT\n"
        )

        variants = list(parse_pvar(pvar_file))

        # Chromosomes should be normalized
        assert variants[0].chr == "1"
        assert variants[1].chr == "22"

    def test_parse_gzipped_pvar(self, tmp_path: Path) -> None:
        """Parse gzipped .pvar file."""
        pvar_file = tmp_path / "test.pvar.gz"
        with gzip.open(pvar_file, "wt") as f:
            f.write(
                "#CHROM\tPOS\tID\tREF\tALT\n"
                "1\t10000\trs123\tA\tG\n"
                "1\t20000\trs456\tC\tT\n"
            )

        variants = list(parse_pvar(pvar_file))

        assert len(variants) == 2
        assert variants[0].id == "rs123"

    def test_count_pvar_variants(self, tmp_path: Path) -> None:
        """Count variants in .pvar file."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t10000\trs123\tA\tG\n"
            "1\t20000\trs456\tC\tT\n"
            "2\t30000\trs789\tG\tA\n"
        )

        count = count_pvar_variants(pvar_file)
        assert count == 3

    def test_parse_pvar_multiallelic_included(self, tmp_path: Path) -> None:
        """Multiallelic sites are included (alleles contain comma)."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t10000\trs123\tA\tG\n"
            "1\t20000\trs456\tC\tT,G\n"  # Multiallelic
            "1\t30000\trs789\tG\tA\n"
        )

        variants = list(parse_pvar(pvar_file))

        # All variants are included
        assert len(variants) == 3
        assert variants[0].id == "rs123"
        # Multiallelic ALT field is kept as-is
        assert variants[1].id == "rs456"
        assert variants[1].allele2 == "T,G"
        assert variants[2].id == "rs789"


class TestAfreqParser:
    """Test PLINK2 .afreq file parsing."""

    def test_parse_valid_afreq(self, tmp_path: Path) -> None:
        """Parse a valid .afreq file."""
        afreq_file = tmp_path / "test.afreq"
        afreq_file.write_text(
            "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "1\trs456\tC\tT\t0.25\t1000\n"
            "2\trs789\tG\tA\t0.15\t1000\n"
        )

        frequencies = parse_afreq(afreq_file)

        assert len(frequencies) == 3
        assert abs(frequencies["rs123"] - 0.30) < 0.01
        assert abs(frequencies["rs456"] - 0.25) < 0.01
        assert abs(frequencies["rs789"] - 0.15) < 0.01

    def test_parse_gzipped_afreq(self, tmp_path: Path) -> None:
        """Parse gzipped .afreq file."""
        afreq_file = tmp_path / "test.afreq.gz"
        with gzip.open(afreq_file, "wt") as f:
            f.write(
                "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
                "1\trs123\tA\tG\t0.30\t1000\n"
                "1\trs456\tC\tT\t0.25\t1000\n"
            )

        frequencies = parse_afreq(afreq_file)

        assert len(frequencies) == 2
        assert abs(frequencies["rs123"] - 0.30) < 0.01

    def test_parse_afreq_na_values(self, tmp_path: Path) -> None:
        """Handle NA values in .afreq file."""
        afreq_file = tmp_path / "test.afreq"
        afreq_file.write_text(
            "#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "1\trs456\tC\tT\tNA\t0\n"
            "1\trs789\tG\tA\t.\t0\n"
        )

        frequencies = parse_afreq(afreq_file)

        # NA and . values should be skipped
        assert "rs123" in frequencies
        assert "rs456" not in frequencies
        assert "rs789" not in frequencies

    def test_parse_afreq_missing_file(self, tmp_path: Path) -> None:
        """Handle missing .afreq file."""
        with pytest.raises(FileNotFoundError):
            parse_afreq(tmp_path / "nonexistent.afreq")


class TestPvarEdgeCases:
    """Test edge cases for .pvar parsing."""

    def test_pvar_with_qual_filter_info(self, tmp_path: Path) -> None:
        """Parse .pvar with QUAL, FILTER, INFO columns."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t10000\trs123\tA\tG\t.\tPASS\t.\n"
            "1\t20000\trs456\tC\tT\t30\tPASS\tAF=0.25\n"
        )

        variants = list(parse_pvar(pvar_file))

        assert len(variants) == 2
        assert variants[0].id == "rs123"
        assert variants[1].id == "rs456"

    def test_pvar_indels(self, tmp_path: Path) -> None:
        """Parse .pvar with indels."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t10000\trs123\tA\tG\n"
            "1\t20000\trs456\tATG\tA\n"  # Deletion
            "1\t30000\trs789\tA\tATG\n"  # Insertion
        )

        variants = list(parse_pvar(pvar_file))

        # All variants should be parsed
        assert len(variants) == 3
        assert variants[1].allele1 == "ATG"
        assert variants[1].allele2 == "A"

    def test_pvar_empty_file(self, tmp_path: Path) -> None:
        """Parse empty .pvar file."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text("#CHROM\tPOS\tID\tREF\tALT\n")

        variants = list(parse_pvar(pvar_file))
        assert len(variants) == 0

    def test_pvar_lowercase_alleles(self, tmp_path: Path) -> None:
        """Handle lowercase alleles."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t10000\trs123\ta\tg\n"
        )

        variants = list(parse_pvar(pvar_file))

        # Alleles should be uppercased
        assert variants[0].allele1 == "A"
        assert variants[0].allele2 == "G"
