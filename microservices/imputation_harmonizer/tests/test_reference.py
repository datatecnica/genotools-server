"""Tests for reference panel loaders."""

from pathlib import Path

import pytest

from imputation_harmonizer.reference.hrc import HRCPanel
from imputation_harmonizer.reference.kg import KGPanel


class TestHRCPanel:
    """Tests for HRC reference panel loader."""

    def test_load_valid_hrc(self, tmp_path: Path) -> None:
        """Test loading a valid HRC file."""
        hrc_file = tmp_path / "test_hrc.tab"
        hrc_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
            "1\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
            "2\t30000\trs789\tG\tA\t400\t1000\t0.40\n"
        )

        panel = HRCPanel()
        panel.load(hrc_file)

        assert len(panel) == 3

        # Test position lookup
        var = panel.get_by_position("1", 10000)
        assert var is not None
        assert var.id == "rs123"
        assert var.ref == "A"
        assert var.alt == "G"
        assert var.alt_af == pytest.approx(0.30)

        # Test ID lookup
        var = panel.get_by_id("rs456")
        assert var is not None
        assert var.pos == 20000

    def test_load_hrc_with_dot_id(self, tmp_path: Path) -> None:
        """Test HRC file with '.' ID (unnamed variant)."""
        hrc_file = tmp_path / "test_hrc.tab"
        hrc_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\t.\tA\tG\t300\t1000\t0.30\n"
        )

        panel = HRCPanel()
        panel.load(hrc_file)

        assert len(panel) == 1

        # Should be found by position
        var = panel.get_by_position("1", 10000)
        assert var is not None
        assert var.id == "."

        # Should NOT be indexed by ID
        assert panel.get_by_id(".") is None

    def test_load_hrc_missing_file(self, tmp_path: Path) -> None:
        """Test error on missing file."""
        missing_file = tmp_path / "missing.tab"

        panel = HRCPanel()
        with pytest.raises(FileNotFoundError):
            panel.load(missing_file)

    def test_has_position(self, tmp_path: Path) -> None:
        """Test has_position method."""
        hrc_file = tmp_path / "test_hrc.tab"
        hrc_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        )

        panel = HRCPanel()
        panel.load(hrc_file)

        assert panel.has_position("1", 10000) is True
        assert panel.has_position("1", 99999) is False

    def test_has_id(self, tmp_path: Path) -> None:
        """Test has_id method."""
        hrc_file = tmp_path / "test_hrc.tab"
        hrc_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        )

        panel = HRCPanel()
        panel.load(hrc_file)

        assert panel.has_id("rs123") is True
        assert panel.has_id("rs999") is False

    def test_get_chrpos_for_id(self, tmp_path: Path) -> None:
        """Test get_chrpos_for_id method."""
        hrc_file = tmp_path / "test_hrc.tab"
        hrc_file.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
            "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        )

        panel = HRCPanel()
        panel.load(hrc_file)

        assert panel.get_chrpos_for_id("rs123") == "1-10000"
        assert panel.get_chrpos_for_id("rs999") is None


class TestKGPanel:
    """Tests for 1000G reference panel loader."""

    def test_load_valid_1000g(self, tmp_path: Path) -> None:
        """Test loading a valid 1000G file."""
        kg_file = tmp_path / "test_1000g.legend"
        kg_file.write_text(
            "id\tchr\tposition\ta0\ta1\tTYPE\tAFR\tAMR\tEAS\tEUR\tSAS\tALL\n"
            "rs123:10000:A:G\t1\t10000\tA\tG\tBiallelic_SNP\t0.20\t0.25\t0.30\t0.35\t0.40\t0.30\n"
            "rs456:20000:C:T\t1\t20000\tC\tT\tBiallelic_SNP\t0.15\t0.20\t0.25\t0.30\t0.35\t0.25\n"
        )

        panel = KGPanel()
        panel.load(kg_file, population="ALL")

        assert len(panel) == 2

        # Test position lookup
        var = panel.get_by_position("1", 10000)
        assert var is not None
        assert var.ref == "A"
        assert var.alt == "G"
        assert var.alt_af == pytest.approx(0.30)

    def test_load_1000g_population_specific(self, tmp_path: Path) -> None:
        """Test loading 1000G with specific population."""
        kg_file = tmp_path / "test_1000g.legend"
        kg_file.write_text(
            "id\tchr\tposition\ta0\ta1\tTYPE\tAFR\tAMR\tEAS\tEUR\tSAS\tALL\n"
            "rs123:10000:A:G\t1\t10000\tA\tG\tBiallelic_SNP\t0.20\t0.25\t0.30\t0.35\t0.40\t0.30\n"
        )

        panel = KGPanel()
        panel.load(kg_file, population="EUR")

        var = panel.get_by_position("1", 10000)
        assert var is not None
        assert var.alt_af == pytest.approx(0.35)  # EUR column

    def test_load_1000g_invalid_population(self, tmp_path: Path) -> None:
        """Test error on invalid population."""
        kg_file = tmp_path / "test_1000g.legend"
        kg_file.write_text(
            "id\tchr\tposition\ta0\ta1\tTYPE\tAFR\tAMR\tEAS\tEUR\tSAS\tALL\n"
            "rs123:10000:A:G\t1\t10000\tA\tG\tBiallelic_SNP\t0.20\t0.25\t0.30\t0.35\t0.40\t0.30\n"
        )

        panel = KGPanel()
        with pytest.raises(ValueError, match="Population 'INVALID' not found"):
            panel.load(kg_file, population="INVALID")

    def test_load_1000g_rsid_extraction(self, tmp_path: Path) -> None:
        """Test rsID extraction from compound ID."""
        kg_file = tmp_path / "test_1000g.legend"
        kg_file.write_text(
            "id\tchr\tposition\ta0\ta1\tTYPE\tAFR\tAMR\tEAS\tEUR\tSAS\tALL\n"
            "rs123:10000:A:G\t1\t10000\tA\tG\tBiallelic_SNP\t0.20\t0.25\t0.30\t0.35\t0.40\t0.30\n"
        )

        panel = KGPanel()
        panel.load(kg_file)

        # Should be able to look up by just rsID
        var = panel.get_by_id("rs123")
        assert var is not None
        assert var.pos == 10000

    def test_load_1000g_multiallelic(self, tmp_path: Path) -> None:
        """Test multiallelic handling (alleles set to N:N)."""
        kg_file = tmp_path / "test_1000g.legend"
        kg_file.write_text(
            "id\tchr\tposition\ta0\ta1\tTYPE\tAFR\tAMR\tEAS\tEUR\tSAS\tALL\n"
            "rs123:10000:A:G\t1\t10000\tA\tG\tMultiallelic_SNP\t0.20\t0.25\t0.30\t0.35\t0.40\t0.30\n"
        )

        panel = KGPanel()
        panel.load(kg_file)

        var = panel.get_by_position("1", 10000)
        assert var is not None
        assert var.ref == "N"  # Set to N for multiallelic
        assert var.alt == "N"

    def test_load_1000g_non_rsid(self, tmp_path: Path) -> None:
        """Test handling of non-rsID variants."""
        kg_file = tmp_path / "test_1000g.legend"
        kg_file.write_text(
            "id\tchr\tposition\ta0\ta1\tTYPE\tAFR\tAMR\tEAS\tEUR\tSAS\tALL\n"
            "1:10000:A:G\t1\t10000\tA\tG\tBiallelic_SNP\t0.20\t0.25\t0.30\t0.35\t0.40\t0.30\n"
        )

        panel = KGPanel()
        panel.load(kg_file)

        # Should be found by position
        var = panel.get_by_position("1", 10000)
        assert var is not None

        # Should NOT be indexed by ID (not an rsID)
        assert panel.get_by_id("1:10000:A:G") is None
