"""Integration tests for the variant checking logic."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imputation_harmonizer.config import Config
from imputation_harmonizer.checks.comparator import check_variants
from imputation_harmonizer.models import Statistics
from imputation_harmonizer.parsers.bim import parse_bim
from imputation_harmonizer.parsers.frq import parse_frq
from imputation_harmonizer.reference.hrc import HRCPanel
from imputation_harmonizer.writers.plink_files import PlinkFileWriter


def run_variant_check(config: Config) -> None:
    """Run just the variant checking portion without PLINK2 pipeline.

    This is a simplified version of run_check() for testing purposes.
    It skips the PLINK2 pipeline which requires PLINK2 to be installed.
    """
    stats = Statistics()

    # Load reference panel
    if config.panel == "hrc":
        reference = HRCPanel()
    else:
        from imputation_harmonizer.reference.kg import KGPanel
        reference = KGPanel()

    reference.load(
        filepath=config.ref_file,
        population=config.population,
        verbose=config.verbose,
    )

    # Parse frequency file
    frequencies = parse_frq(config.frq_file)

    # Process variants and write output files
    assert config.output_dir is not None

    with PlinkFileWriter(
        output_dir=config.output_dir,
        file_stem=config.file_stem,
        panel_name=config.panel_name,
    ) as writer:
        bim_variants = parse_bim(config.bim_file, frequencies)

        for bim_variant, result in check_variants(bim_variants, reference, config, stats):
            writer.write_result(result)

    reference.clear()


class TestVariantChecking:
    """Integration tests for the variant checking logic."""

    def test_full_pipeline_creates_output_files(
        self, sample_files: dict[str, Path]
    ) -> None:
        """Test that all expected output files are created."""
        config = Config(
            bim_file=sample_files["bim"],
            frq_file=sample_files["frq"],
            ref_file=sample_files["ref"],
            panel="hrc",
            output_dir=sample_files["dir"],
            generate_report=False,  # Skip report for this test
        )

        run_variant_check(config)

        # Check all output files exist
        assert (sample_files["dir"] / "Exclude-test-HRC.txt").exists()
        assert (sample_files["dir"] / "Strand-Flip-test-HRC.txt").exists()
        assert (sample_files["dir"] / "Force-Allele1-test-HRC.txt").exists()
        assert (sample_files["dir"] / "Position-test-HRC.txt").exists()
        assert (sample_files["dir"] / "Chromosome-test-HRC.txt").exists()
        assert (sample_files["dir"] / "ID-test-HRC.txt").exists()
        assert (sample_files["dir"] / "FreqPlot-test-HRC.txt").exists()

    def test_palindromic_snp_excluded(self, sample_files: dict[str, Path]) -> None:
        """Test that high-MAF palindromic SNP is excluded."""
        config = Config(
            bim_file=sample_files["bim"],
            frq_file=sample_files["frq"],
            ref_file=sample_files["ref"],
            panel="hrc",
            output_dir=sample_files["dir"],
            generate_report=False,
        )

        run_variant_check(config)

        # rs789 is A/T with MAF 0.45 > 0.4, should be excluded
        excludes = (sample_files["dir"] / "Exclude-test-HRC.txt").read_text()
        assert "rs789" in excludes

    def test_strand_flip_detected(self, sample_files: dict[str, Path]) -> None:
        """Test that strand flip is detected."""
        config = Config(
            bim_file=sample_files["bim"],
            frq_file=sample_files["frq"],
            ref_file=sample_files["ref"],
            panel="hrc",
            output_dir=sample_files["dir"],
            generate_report=False,
        )

        run_variant_check(config)

        # rs101 has T/C in BIM, A/G in ref (complement), needs flip
        flips = (sample_files["dir"] / "Strand-Flip-test-HRC.txt").read_text()
        assert "rs101" in flips

    def test_force_allele_detected(self, sample_files: dict[str, Path]) -> None:
        """Test that ref/alt swap is detected."""
        config = Config(
            bim_file=sample_files["bim"],
            frq_file=sample_files["frq"],
            ref_file=sample_files["ref"],
            panel="hrc",
            output_dir=sample_files["dir"],
            generate_report=False,
        )

        run_variant_check(config)

        # rs102 has G/A in BIM, A/G in ref (swapped), needs force allele
        force = (sample_files["dir"] / "Force-Allele1-test-HRC.txt").read_text()
        assert "rs102" in force


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_missing_bim(self, tmp_path: Path) -> None:
        """Test validation error for missing BIM file."""
        config = Config(
            bim_file=tmp_path / "missing.bim",
            frq_file=tmp_path / "test.frq",
            ref_file=tmp_path / "ref.tab",
            panel="hrc",
            output_dir=tmp_path,
        )

        # Create the other files
        (tmp_path / "test.frq").write_text("CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n")
        (tmp_path / "ref.tab").write_text("#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n")

        errors = config.validate()
        assert any("BIM file not found" in e for e in errors)

    def test_validate_invalid_freq_threshold(self, tmp_path: Path) -> None:
        """Test validation error for invalid frequency threshold."""
        # Create dummy files
        (tmp_path / "test.bim").write_text("1\trs1\t0\t100\tA\tG\n")
        (tmp_path / "test.frq").write_text("CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n")
        (tmp_path / "ref.tab").write_text("#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n")

        config = Config(
            bim_file=tmp_path / "test.bim",
            frq_file=tmp_path / "test.frq",
            ref_file=tmp_path / "ref.tab",
            panel="hrc",
            output_dir=tmp_path,
            freq_diff_threshold=1.5,  # Invalid
        )

        errors = config.validate()
        assert any("freq_diff_threshold" in e for e in errors)

    def test_validate_invalid_population(self, tmp_path: Path) -> None:
        """Test validation error for invalid 1000G population."""
        # Create dummy files
        (tmp_path / "test.bim").write_text("1\trs1\t0\t100\tA\tG\n")
        (tmp_path / "test.frq").write_text("CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n")
        (tmp_path / "ref.tab").write_text("#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n")

        config = Config(
            bim_file=tmp_path / "test.bim",
            frq_file=tmp_path / "test.frq",
            ref_file=tmp_path / "ref.tab",
            panel="1000g",
            population="INVALID",
            output_dir=tmp_path,
        )

        errors = config.validate()
        assert any("Invalid population" in e for e in errors)


class TestNotInReference:
    """Tests for variants not in reference."""

    def test_variant_not_in_reference(self, tmp_path: Path) -> None:
        """Test that variants not in reference are excluded."""
        # BIM with variant not in reference
        bim_file = tmp_path / "test.bim"
        bim_file.write_text(
            "1\trs123\t0\t10000\tA\tG\n"
            "1\trs999\t0\t99999\tC\tT\n"  # Not in reference
        )

        frq_file = tmp_path / "test.frq"
        frq_file.write_text(
            "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "1\trs999\tC\tT\t0.25\t1000\n"
        )

        ref_file = tmp_path / "ref.tab"
        ref_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
            # rs999 at position 99999 not in reference
        )

        config = Config(
            bim_file=bim_file,
            frq_file=frq_file,
            ref_file=ref_file,
            panel="hrc",
            output_dir=tmp_path,
            generate_report=False,
        )

        run_variant_check(config)

        excludes = (tmp_path / "Exclude-test-HRC.txt").read_text()
        assert "rs999" in excludes


class TestIndels:
    """Tests for indel handling."""

    def test_indels_excluded(self, tmp_path: Path) -> None:
        """Test that indels are excluded."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text(
            "1\trs123\t0\t10000\tA\tG\n"
            "1\trs456\t0\t20000\t-\tA\n"  # Indel
            "1\trs789\t0\t30000\tAT\tA\n"  # Multi-base indel
        )

        frq_file = tmp_path / "test.frq"
        frq_file.write_text(
            "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "1\trs456\t-\tA\t0.10\t1000\n"
            "1\trs789\tAT\tA\t0.15\t1000\n"
        )

        ref_file = tmp_path / "ref.tab"
        ref_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
            "1\t20000\trs456\tA\tAT\t100\t1000\t0.10\n"
            "1\t30000\trs789\tA\tAT\t150\t1000\t0.15\n"
        )

        config = Config(
            bim_file=bim_file,
            frq_file=frq_file,
            ref_file=ref_file,
            panel="hrc",
            output_dir=tmp_path,
            generate_report=False,
        )

        run_variant_check(config)

        excludes = (tmp_path / "Exclude-test-HRC.txt").read_text()
        assert "rs456" in excludes  # Deletion marker
        assert "rs789" in excludes  # Multi-base


class TestAltChromosomes:
    """Tests for alternative chromosome handling."""

    def test_alt_chr_excluded_hrc(self, tmp_path: Path) -> None:
        """Test that X, Y, MT are excluded for HRC."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text(
            "1\trs123\t0\t10000\tA\tG\n"
            "X\trs456\t0\t20000\tC\tT\n"  # X chromosome
            "23\trs789\t0\t30000\tG\tA\n"  # Chr 23 (X in numeric)
        )

        frq_file = tmp_path / "test.frq"
        frq_file.write_text(
            "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
            "1\trs123\tA\tG\t0.30\t1000\n"
            "X\trs456\tC\tT\t0.25\t1000\n"
            "23\trs789\tG\tA\t0.40\t1000\n"
        )

        ref_file = tmp_path / "ref.tab"
        ref_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        )

        config = Config(
            bim_file=bim_file,
            frq_file=frq_file,
            ref_file=ref_file,
            panel="hrc",
            output_dir=tmp_path,
            generate_report=False,
        )

        run_variant_check(config)

        excludes = (tmp_path / "Exclude-test-HRC.txt").read_text()
        assert "rs456" in excludes  # X
        assert "rs789" in excludes  # 23
