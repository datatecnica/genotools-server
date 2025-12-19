"""Tests for format_detection module."""

from pathlib import Path

import pytest

from imputation_harmonizer.format_detection import (
    PlinkFormat,
    PlinkFileSet,
    detect_plink_format,
    find_variant_file,
    get_freq_file_extension,
    get_prefix,
    resolve_plink_fileset,
)


class TestPlinkFormat:
    """Test PlinkFormat enum."""

    def test_plink1_exists(self) -> None:
        """PLINK1 format exists."""
        assert PlinkFormat.PLINK1 is not None

    def test_plink2_exists(self) -> None:
        """PLINK2 format exists."""
        assert PlinkFormat.PLINK2 is not None


class TestDetectPlinkFormat:
    """Test PLINK format detection from variant file."""

    def test_detect_bim_format(self, tmp_path: Path) -> None:
        """Detect PLINK1 from .bim file."""
        bim_file = tmp_path / "test.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")

        assert detect_plink_format(bim_file) == PlinkFormat.PLINK1

    def test_detect_pvar_format(self, tmp_path: Path) -> None:
        """Detect PLINK2 from .pvar file."""
        pvar_file = tmp_path / "test.pvar"
        pvar_file.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t10000\trs123\tA\tG\n")

        assert detect_plink_format(pvar_file) == PlinkFormat.PLINK2

    def test_detect_bim_gz_format(self, tmp_path: Path) -> None:
        """Detect PLINK1 from .bim.gz file."""
        import gzip

        bim_file = tmp_path / "test.bim.gz"
        with gzip.open(bim_file, "wt") as f:
            f.write("1\trs123\t0\t10000\tA\tG\n")

        assert detect_plink_format(bim_file) == PlinkFormat.PLINK1

    def test_detect_pvar_gz_format(self, tmp_path: Path) -> None:
        """Detect PLINK2 from .pvar.gz file."""
        import gzip

        pvar_file = tmp_path / "test.pvar.gz"
        with gzip.open(pvar_file, "wt") as f:
            f.write("#CHROM\tPOS\tID\tREF\tALT\n")

        assert detect_plink_format(pvar_file) == PlinkFormat.PLINK2

    def test_detect_unknown_format(self, tmp_path: Path) -> None:
        """Unknown extension raises error."""
        unknown_file = tmp_path / "test.txt"
        unknown_file.write_text("data")

        with pytest.raises(ValueError, match="Unrecognized variant file extension"):
            detect_plink_format(unknown_file)


class TestGetPrefix:
    """Test file prefix extraction."""

    def test_get_prefix_bim(self, tmp_path: Path) -> None:
        """Get prefix from .bim file."""
        bim_file = tmp_path / "data.bim"
        assert get_prefix(bim_file) == tmp_path / "data"

    def test_get_prefix_pvar(self, tmp_path: Path) -> None:
        """Get prefix from .pvar file."""
        pvar_file = tmp_path / "data.pvar"
        assert get_prefix(pvar_file) == tmp_path / "data"

    def test_get_prefix_gzipped(self, tmp_path: Path) -> None:
        """Get prefix from gzipped file."""
        pvar_file = tmp_path / "data.pvar.gz"
        assert get_prefix(pvar_file) == tmp_path / "data"


class TestGetFreqFileExtension:
    """Test frequency file extension lookup."""

    def test_plink1_frq(self) -> None:
        """PLINK1 uses .frq extension."""
        assert get_freq_file_extension(PlinkFormat.PLINK1) == ".frq"

    def test_plink2_afreq(self) -> None:
        """PLINK2 uses .afreq extension."""
        assert get_freq_file_extension(PlinkFormat.PLINK2) == ".afreq"


class TestFindVariantFile:
    """Test finding variant files from prefix."""

    def test_find_pvar(self, tmp_path: Path) -> None:
        """Find .pvar file (PLINK2)."""
        pvar_file = tmp_path / "data.pvar"
        pvar_file.write_text("#CHROM\tPOS\tID\tREF\tALT\n")

        result = find_variant_file(tmp_path / "data")
        assert result == pvar_file

    def test_find_bim(self, tmp_path: Path) -> None:
        """Find .bim file (PLINK1)."""
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")

        result = find_variant_file(tmp_path / "data")
        assert result == bim_file

    def test_find_pvar_preferred(self, tmp_path: Path) -> None:
        """Prefer .pvar over .bim when both exist."""
        pvar_file = tmp_path / "data.pvar"
        pvar_file.write_text("#CHROM\tPOS\tID\tREF\tALT\n")
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")

        result = find_variant_file(tmp_path / "data")
        assert result == pvar_file

    def test_find_gzipped(self, tmp_path: Path) -> None:
        """Find gzipped variant file."""
        import gzip

        pvar_file = tmp_path / "data.pvar.gz"
        with gzip.open(pvar_file, "wt") as f:
            f.write("#CHROM\tPOS\tID\tREF\tALT\n")

        result = find_variant_file(tmp_path / "data")
        assert result == pvar_file

    def test_find_no_files(self, tmp_path: Path) -> None:
        """Return None when no files found."""
        result = find_variant_file(tmp_path / "nonexistent")
        assert result is None


class TestResolvePlinkFileset:
    """Test resolving complete PLINK filesets."""

    def test_resolve_plink1_fileset(self, tmp_path: Path) -> None:
        """Resolve complete PLINK1 fileset."""
        # Create PLINK1 files
        (tmp_path / "data.bed").write_bytes(b"")
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")
        (tmp_path / "data.fam").write_text("FAM001 IND001 0 0 1 -9\n")

        fileset = resolve_plink_fileset(bim_file)

        assert fileset.format == PlinkFormat.PLINK1
        assert fileset.variant_file == bim_file
        assert fileset.genotype_file == tmp_path / "data.bed"
        assert fileset.sample_file == tmp_path / "data.fam"

    def test_resolve_plink2_fileset(self, tmp_path: Path) -> None:
        """Resolve complete PLINK2 fileset."""
        # Create PLINK2 files
        (tmp_path / "data.pgen").write_bytes(b"")
        pvar_file = tmp_path / "data.pvar"
        pvar_file.write_text("#CHROM\tPOS\tID\tREF\tALT\n")
        (tmp_path / "data.psam").write_text("#FID\tIID\n")

        fileset = resolve_plink_fileset(pvar_file)

        assert fileset.format == PlinkFormat.PLINK2
        assert fileset.variant_file == pvar_file
        assert fileset.genotype_file == tmp_path / "data.pgen"
        assert fileset.sample_file == tmp_path / "data.psam"

    def test_resolve_with_freq_file(self, tmp_path: Path) -> None:
        """Resolve fileset with explicit frequency file."""
        # Create PLINK1 files
        (tmp_path / "data.bed").write_bytes(b"")
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")
        (tmp_path / "data.fam").write_text("FAM001 IND001 0 0 1 -9\n")
        frq_file = tmp_path / "data.frq"
        frq_file.write_text("CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n")

        fileset = resolve_plink_fileset(bim_file, freq_file=frq_file)

        assert fileset.freq_file == frq_file

    def test_resolve_auto_detect_freq(self, tmp_path: Path) -> None:
        """Auto-detect frequency file if present."""
        # Create PLINK1 files
        (tmp_path / "data.bed").write_bytes(b"")
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")
        (tmp_path / "data.fam").write_text("FAM001 IND001 0 0 1 -9\n")
        frq_file = tmp_path / "data.frq"
        frq_file.write_text("CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n")

        fileset = resolve_plink_fileset(bim_file)

        assert fileset.freq_file == frq_file

    def test_resolve_missing_genotype_file(self, tmp_path: Path) -> None:
        """Raise error when genotype file missing."""
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")
        (tmp_path / "data.fam").write_text("FAM001 IND001 0 0 1 -9\n")

        with pytest.raises(FileNotFoundError, match="Genotype file not found"):
            resolve_plink_fileset(bim_file)

    def test_resolve_missing_sample_file(self, tmp_path: Path) -> None:
        """Raise error when sample file missing."""
        (tmp_path / "data.bed").write_bytes(b"")
        bim_file = tmp_path / "data.bim"
        bim_file.write_text("1\trs123\t0\t10000\tA\tG\n")

        with pytest.raises(FileNotFoundError, match="Sample file not found"):
            resolve_plink_fileset(bim_file)
