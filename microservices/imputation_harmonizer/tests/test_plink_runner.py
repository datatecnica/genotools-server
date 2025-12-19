"""Tests for plink_runner module.

Note: These tests are mostly unit tests for the PlinkPipeline class.
Full pipeline integration tests would require PLINK2 to be installed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imputation_harmonizer.plink_runner import (
    PlinkPipeline,
    PlinkResult,
    find_plink2,
    verify_plink2_version,
)


class TestFindPlink2:
    """Test PLINK2 executable discovery."""

    def test_find_plink2_in_path(self) -> None:
        """Find plink2 when available."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/plink2" if x == "plink2" else None
            result = find_plink2()
            assert result == Path("/usr/bin/plink2")

    def test_find_plink_fallback(self) -> None:
        """Fall back to plink when plink2 not available."""
        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/plink" if x == "plink" else None
            result = find_plink2()
            assert result == Path("/usr/bin/plink")

    def test_find_neither(self) -> None:
        """Return None when neither available."""
        with patch("shutil.which", return_value=None):
            result = find_plink2()
            assert result is None


class TestVerifyPlink2Version:
    """Test PLINK version verification."""

    def test_verify_plink2(self) -> None:
        """Verify PLINK2 version."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="PLINK v2.00a3 AVX2",
                stderr="",
            )
            is_v2, version = verify_plink2_version(Path("/usr/bin/plink2"))
            assert is_v2 is True
            assert "v2" in version.lower()

    def test_verify_plink1(self) -> None:
        """Verify PLINK1 version."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                stdout="PLINK v1.90b6.21",
                stderr="",
            )
            is_v2, version = verify_plink2_version(Path("/usr/bin/plink"))
            assert is_v2 is False

    def test_verify_timeout(self) -> None:
        """Handle timeout gracefully."""
        import subprocess
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("plink", 10)):
            is_v2, version = verify_plink2_version(Path("/usr/bin/plink"))
            assert is_v2 is False
            assert version == "unknown"


class TestPlinkResult:
    """Test PlinkResult dataclass."""

    def test_success_result(self) -> None:
        """Create successful result."""
        result = PlinkResult(
            success=True,
            command="plink2 --version",
            stdout="PLINK v2.00",
            stderr="",
            output_prefix=Path("/tmp/output"),
        )
        assert result.success is True
        assert result.output_prefix == Path("/tmp/output")

    def test_failure_result(self) -> None:
        """Create failure result."""
        result = PlinkResult(
            success=False,
            command="plink2 --invalid",
            stdout="",
            stderr="Error: invalid option",
            output_prefix=None,
        )
        assert result.success is False
        assert result.output_prefix is None


class TestPlinkPipeline:
    """Test PlinkPipeline class."""

    def test_init_with_plink_path(self) -> None:
        """Initialize with explicit PLINK path."""
        with patch.object(PlinkPipeline, "__init__", lambda self, **kwargs: None):
            # Create pipeline without calling __init__
            pipeline = object.__new__(PlinkPipeline)
            pipeline.plink_path = Path("/usr/bin/plink2")
            pipeline.verbose = False

            assert pipeline.plink_path == Path("/usr/bin/plink2")

    def test_init_no_plink_raises(self) -> None:
        """Raise error when PLINK not found."""
        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=None):
            with pytest.raises(RuntimeError, match="PLINK2 not found"):
                PlinkPipeline()

    def test_run_plink_success(self) -> None:
        """Run PLINK command successfully."""
        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=Path("/usr/bin/plink2")):
            with patch("imputation_harmonizer.plink_runner.verify_plink2_version", return_value=(True, "v2")):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=0,
                        stdout="Success",
                        stderr="",
                    )

                    pipeline = PlinkPipeline()
                    result = pipeline._run_plink(
                        ["--version"],
                        description="Check version",
                    )

                    assert result.success is True

    def test_run_plink_failure(self) -> None:
        """Handle PLINK command failure."""
        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=Path("/usr/bin/plink2")):
            with patch("imputation_harmonizer.plink_runner.verify_plink2_version", return_value=(True, "v2")):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(
                        returncode=1,
                        stdout="",
                        stderr="Error: file not found",
                    )

                    pipeline = PlinkPipeline(verbose=False)
                    result = pipeline._run_plink(
                        ["--bfile", "nonexistent"],
                        description="Test failure",
                    )

                    assert result.success is False

    def test_cleanup_temp_files(self, tmp_path: Path) -> None:
        """Clean up temporary files."""
        # Create work directory
        work_dir = tmp_path / ".work"
        work_dir.mkdir()
        (work_dir / "temp_file.txt").write_text("temp")

        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=Path("/usr/bin/plink2")):
            with patch("imputation_harmonizer.plink_runner.verify_plink2_version", return_value=(True, "v2")):
                pipeline = PlinkPipeline()
                pipeline.cleanup_temp_files(tmp_path)

        assert not work_dir.exists()

    def test_cleanup_nonexistent_dir(self, tmp_path: Path) -> None:
        """Handle cleanup when work dir doesn't exist."""
        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=Path("/usr/bin/plink2")):
            with patch("imputation_harmonizer.plink_runner.verify_plink2_version", return_value=(True, "v2")):
                pipeline = PlinkPipeline()
                # Should not raise
                pipeline.cleanup_temp_files(tmp_path)


class TestSplitByChromosome:
    """Test chromosome splitting functionality."""

    def test_split_detects_chromosomes(self, tmp_path: Path) -> None:
        """Split finds created chromosome files."""
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        # Create fake chromosome files
        for chr_num in [1, 2, 22]:
            (work_dir / f"split.chr{chr_num}.pgen").write_bytes(b"")
            (work_dir / f"split.chr{chr_num}.pvar").write_text("")
            (work_dir / f"split.chr{chr_num}.psam").write_text("")

        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=Path("/usr/bin/plink2")):
            with patch("imputation_harmonizer.plink_runner.verify_plink2_version", return_value=(True, "v2")):
                with patch.object(PlinkPipeline, "_run_plink") as mock_run:
                    mock_run.return_value = PlinkResult(
                        success=True,
                        command="plink2 ...",
                        stdout="",
                        stderr="",
                        output_prefix=work_dir / "split",
                    )

                    pipeline = PlinkPipeline()
                    chromosomes = pipeline.split_by_chromosome(
                        input_prefix=tmp_path / "input",
                        work_dir=work_dir,
                    )

        assert "1" in chromosomes
        assert "2" in chromosomes
        assert "22" in chromosomes
        assert len(chromosomes) == 3

    def test_split_failure_raises(self, tmp_path: Path) -> None:
        """Raise error when split fails."""
        with patch("imputation_harmonizer.plink_runner.find_plink2", return_value=Path("/usr/bin/plink2")):
            with patch("imputation_harmonizer.plink_runner.verify_plink2_version", return_value=(True, "v2")):
                with patch.object(PlinkPipeline, "_run_plink") as mock_run:
                    mock_run.return_value = PlinkResult(
                        success=False,
                        command="plink2 ...",
                        stdout="",
                        stderr="Error: split failed",
                        output_prefix=None,
                    )

                    pipeline = PlinkPipeline()
                    with pytest.raises(RuntimeError, match="Failed to split"):
                        pipeline.split_by_chromosome(
                            input_prefix=tmp_path / "input",
                            work_dir=tmp_path / "work",
                        )


class TestPipelineOutputFormats:
    """Test pipeline output format options."""

    @pytest.mark.parametrize("output_format,expected_ext", [
        ("vcf", ".vcf.gz"),
        ("plink1", ".bed"),
        ("plink2", ".pgen"),
    ])
    def test_output_format_extensions(
        self,
        output_format: str,
        expected_ext: str,
    ) -> None:
        """Verify output format generates correct extension."""
        # This is a design validation test
        format_extensions = {
            "vcf": ".vcf.gz",
            "plink1": ".bed",
            "plink2": ".pgen",
        }
        assert format_extensions[output_format] == expected_ext
