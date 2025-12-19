"""Tests for TOPMed panel loader."""

import gzip
from pathlib import Path

import pytest

from imputation_harmonizer.reference.topmed import TOPMedPanel


@pytest.fixture
def mini_topmed(tmp_path: Path) -> Path:
    """Create a minimal TOPMed reference file."""
    topmed_file = tmp_path / "topmed_mini.tab"
    topmed_file.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
        "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        "1\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
        "2\t30000\trs789\tG\tA\t150\t1000\t0.15\n"
        "22\t40000\trs101\tT\tC\t400\t1000\t0.40\n"
    )
    return topmed_file


@pytest.fixture
def mini_topmed_gz(tmp_path: Path) -> Path:
    """Create a minimal gzipped TOPMed reference file."""
    topmed_file = tmp_path / "topmed_mini.tab.gz"
    with gzip.open(topmed_file, "wt") as f:
        f.write(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
            "1\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
            "2\t30000\trs789\tG\tA\t150\t1000\t0.15\n"
            "22\t40000\trs101\tT\tC\t400\t1000\t0.40\n"
        )
    return topmed_file


class TestTOPMedPanel:
    """Test TOPMed panel loading."""

    def test_load_valid_topmed(self, mini_topmed: Path) -> None:
        """Load a valid TOPMed reference file."""
        panel = TOPMedPanel()
        panel.load(mini_topmed)

        assert len(panel) == 4
        assert panel.has_position("1", 10000)
        assert panel.has_position("1", 20000)
        assert panel.has_position("2", 30000)
        assert panel.has_position("22", 40000)

    def test_load_gzipped_topmed(self, mini_topmed_gz: Path) -> None:
        """Load a gzipped TOPMed reference file."""
        panel = TOPMedPanel()
        panel.load(mini_topmed_gz)

        assert len(panel) == 4
        assert panel.has_position("1", 10000)

    def test_load_with_chromosome_filter(self, mini_topmed: Path) -> None:
        """Load only variants from specific chromosome."""
        panel = TOPMedPanel()
        panel.load(mini_topmed, chromosome="1")

        # Only chr1 variants should be loaded
        assert len(panel) == 2
        assert panel.has_position("1", 10000)
        assert panel.has_position("1", 20000)
        assert not panel.has_position("2", 30000)
        assert not panel.has_position("22", 40000)

    def test_load_chromosome_22(self, mini_topmed: Path) -> None:
        """Load only chromosome 22 variants."""
        panel = TOPMedPanel()
        panel.load(mini_topmed, chromosome="22")

        assert len(panel) == 1
        assert panel.has_position("22", 40000)

    def test_get_variant(self, mini_topmed: Path) -> None:
        """Get variant data by position."""
        panel = TOPMedPanel()
        panel.load(mini_topmed)

        variant = panel.get_by_position("1", 10000)
        assert variant is not None
        assert variant.id == "rs123"
        assert variant.ref == "A"
        assert variant.alt == "G"
        assert abs(variant.alt_af - 0.30) < 0.01

    def test_has_id(self, mini_topmed: Path) -> None:
        """Check variant existence by ID."""
        panel = TOPMedPanel()
        panel.load(mini_topmed)

        assert panel.has_id("rs123")
        assert panel.has_id("rs456")
        assert not panel.has_id("rs999")

    def test_get_chrpos_for_id(self, mini_topmed: Path) -> None:
        """Get chromosome-position key for an ID."""
        panel = TOPMedPanel()
        panel.load(mini_topmed)

        result = panel.get_chrpos_for_id("rs789")
        assert result is not None
        # Returns chr-pos key as string
        assert result == "2-30000"

    def test_clear(self, mini_topmed: Path) -> None:
        """Clear panel data."""
        panel = TOPMedPanel()
        panel.load(mini_topmed)

        assert len(panel) == 4
        panel.clear()
        assert len(panel) == 0

    def test_missing_file(self, tmp_path: Path) -> None:
        """Handle missing file gracefully."""
        panel = TOPMedPanel()
        with pytest.raises(FileNotFoundError):
            panel.load(tmp_path / "nonexistent.tab")


class TestTOPMedWithDotIds:
    """Test handling of variants with '.' as ID."""

    def test_dot_id_handling(self, tmp_path: Path) -> None:
        """Variants with '.' ID are loaded but not indexed by ID."""
        topmed_file = tmp_path / "topmed_dot.tab"
        topmed_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\t.\tA\tG\t300\t1000\t0.30\n"
            "1\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
        )

        panel = TOPMedPanel()
        panel.load(topmed_file)

        # Both variants should be loaded (by position)
        assert len(panel) == 2
        assert panel.has_position("1", 10000)
        assert panel.has_position("1", 20000)

        # The '.' ID is NOT indexed (not searchable by ID)
        assert not panel.has_id(".")

        # But named ID is indexed
        assert panel.has_id("rs456")

    def test_dot_id_variant_accessible_by_position(self, tmp_path: Path) -> None:
        """Variants with '.' ID can be retrieved by position."""
        topmed_file = tmp_path / "topmed_dot.tab"
        topmed_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\t.\tA\tG\t300\t1000\t0.30\n"
        )

        panel = TOPMedPanel()
        panel.load(topmed_file)

        variant = panel.get_by_position("1", 10000)
        assert variant is not None
        assert variant.id == "."  # ID is stored as-is
        assert variant.ref == "A"
        assert variant.alt == "G"


class TestTOPMedChromosomeNormalization:
    """Test chromosome naming normalization."""

    def test_chr_prefix_normalized(self, tmp_path: Path) -> None:
        """Chromosomes with 'chr' prefix are normalized."""
        topmed_file = tmp_path / "topmed_chr.tab"
        topmed_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "chr1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
            "chr22\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
        )

        panel = TOPMedPanel()
        panel.load(topmed_file)

        # Should be accessible with normalized chromosome
        assert panel.has_position("1", 10000)
        assert panel.has_position("22", 20000)

    def test_chromosome_filter_with_chr_prefix(self, tmp_path: Path) -> None:
        """Chromosome filter works with chr-prefixed data."""
        topmed_file = tmp_path / "topmed_chr.tab"
        topmed_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "chr1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
            "chr2\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
        )

        panel = TOPMedPanel()
        panel.load(topmed_file, chromosome="1")

        assert len(panel) == 1
        assert panel.has_position("1", 10000)
