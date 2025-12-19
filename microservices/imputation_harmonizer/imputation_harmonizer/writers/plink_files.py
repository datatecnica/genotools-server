"""PLINK update file writers.

[claude-assisted] Implements output file generation matching the original
Perl script's output format (lines 207-249).

Output files (must match Perl format exactly):
- Exclude-{stem}-{panel}.txt - SNP IDs to remove
- Strand-Flip-{stem}-{panel}.txt - SNP IDs needing strand flip
- Force-Allele1-{stem}-{panel}.txt - SNP ID + ref allele (tab-separated)
- Position-{stem}-{panel}.txt - SNP ID + new position
- Chromosome-{stem}-{panel}.txt - SNP ID + new chromosome
- ID-{stem}-{panel}.txt - Old ID + new ID
- FreqPlot-{stem}-{panel}.txt - Frequency comparison data
"""

from pathlib import Path
from types import TracebackType
from typing import TextIO

from imputation_harmonizer.models import AlleleAction, CheckResult, StrandAction


class PlinkFileWriter:
    """Manages all PLINK update file outputs.

    Opens all output files on initialization and provides methods to
    write check results. Implements context manager protocol for
    automatic cleanup.

    Usage:
        with PlinkFileWriter(output_dir, "data", "HRC") as writer:
            for result in check_results:
                writer.write_result(result)
    """

    def __init__(self, output_dir: Path, file_stem: str, panel_name: str) -> None:
        """Initialize writer and open all output files.

        Args:
            output_dir: Directory for output files
            file_stem: Base filename (from BIM file)
            panel_name: Reference panel name ("HRC" or "1000G")
        """
        self.output_dir = output_dir
        self.file_stem = file_stem
        self.panel_name = panel_name

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Open all output files
        self.exclude_file = self._open(f"Exclude-{file_stem}-{panel_name}.txt")
        self.strand_file = self._open(f"Strand-Flip-{file_stem}-{panel_name}.txt")
        self.force_file = self._open(f"Force-Allele1-{file_stem}-{panel_name}.txt")
        self.position_file = self._open(f"Position-{file_stem}-{panel_name}.txt")
        self.chromosome_file = self._open(f"Chromosome-{file_stem}-{panel_name}.txt")
        self.id_file = self._open(f"ID-{file_stem}-{panel_name}.txt")
        self.freq_plot_file = self._open(f"FreqPlot-{file_stem}-{panel_name}.txt")

        # Track counts for summary
        self.exclude_count = 0
        self.strand_flip_count = 0
        self.force_allele_count = 0
        self.position_update_count = 0
        self.chromosome_update_count = 0
        self.id_update_count = 0
        self.freq_plot_count = 0

    def _open(self, filename: str) -> TextIO:
        """Open a file in the output directory.

        Args:
            filename: Name of file to open

        Returns:
            Open file handle for writing
        """
        filepath = self.output_dir / filename
        return open(filepath, "w")

    def write_result(self, result: CheckResult) -> None:
        """Write a single check result to appropriate output files.

        Args:
            result: CheckResult from variant comparison
        """
        # If excluded, write to exclude file only
        if result.exclude:
            self.exclude_file.write(f"{result.snp_id}\n")
            self.exclude_count += 1
            return

        # Strand flip
        if result.strand_action == StrandAction.FLIP:
            self.strand_file.write(f"{result.snp_id}\n")
            self.strand_flip_count += 1

        # Force reference allele
        if result.allele_action == AlleleAction.FORCE_REF and result.force_ref_allele:
            self.force_file.write(f"{result.snp_id}\t{result.force_ref_allele}\n")
            self.force_allele_count += 1

        # Position update
        if result.update_position is not None:
            self.position_file.write(f"{result.snp_id}\t{result.update_position}\n")
            self.position_update_count += 1

        # Chromosome update
        if result.update_chromosome is not None:
            self.chromosome_file.write(f"{result.snp_id}\t{result.update_chromosome}\n")
            self.chromosome_update_count += 1

        # ID update
        if result.update_id is not None:
            self.id_file.write(f"{result.snp_id}\t{result.update_id}\n")
            self.id_update_count += 1

        # Frequency plot data
        # Format: snp_id  ref_freq  bim_freq  diff  check_code
        if result.ref_freq is not None:
            bim_freq = result.bim_freq if result.bim_freq is not None else 0.0
            diff = result.freq_diff if result.freq_diff is not None else 0.0
            check_code = result.check_code if result.check_code is not None else 0
            self.freq_plot_file.write(
                f"{result.snp_id}\t{result.ref_freq}\t{bim_freq}\t{diff}\t{check_code}\n"
            )
            self.freq_plot_count += 1

    def close(self) -> None:
        """Close all file handles."""
        for f in [
            self.exclude_file,
            self.strand_file,
            self.force_file,
            self.position_file,
            self.chromosome_file,
            self.id_file,
            self.freq_plot_file,
        ]:
            f.close()

    def __enter__(self) -> "PlinkFileWriter":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit - close all files."""
        self.close()

    def get_file_paths(self) -> dict[str, Path]:
        """Get paths to all output files.

        Returns:
            Dictionary mapping file type to path
        """
        return {
            "exclude": self.output_dir / f"Exclude-{self.file_stem}-{self.panel_name}.txt",
            "strand_flip": self.output_dir / f"Strand-Flip-{self.file_stem}-{self.panel_name}.txt",
            "force_allele": self.output_dir / f"Force-Allele1-{self.file_stem}-{self.panel_name}.txt",
            "position": self.output_dir / f"Position-{self.file_stem}-{self.panel_name}.txt",
            "chromosome": self.output_dir / f"Chromosome-{self.file_stem}-{self.panel_name}.txt",
            "id": self.output_dir / f"ID-{self.file_stem}-{self.panel_name}.txt",
            "freq_plot": self.output_dir / f"FreqPlot-{self.file_stem}-{self.panel_name}.txt",
        }

    def get_summary(self) -> dict[str, int]:
        """Get summary of files written.

        Returns:
            Dictionary mapping file type to line count
        """
        return {
            "exclude": self.exclude_count,
            "strand_flip": self.strand_flip_count,
            "force_allele": self.force_allele_count,
            "position_update": self.position_update_count,
            "chromosome_update": self.chromosome_update_count,
            "id_update": self.id_update_count,
            "freq_plot": self.freq_plot_count,
        }
