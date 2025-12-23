"""Configuration dataclass for the imputation harmonizer.

[claude-assisted] Configuration options for the parallel pre-imputation
harmonization pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Config:
    """Configuration for HRC/1000G/TOPMed pre-imputation check.

    Attributes:
        bim_file: Path to PLINK .bim or .pvar file
        frq_file: Path to PLINK .frq or .afreq frequency file
        ref_file: Path to reference panel file (HRC, 1000G, or TOPMed)
        panel: Reference panel type ("hrc", "1000g", or "topmed")
        population: 1000G population for frequency column (ignored for HRC/TOPMed)
        output_dir: Output directory for generated files
        freq_diff_threshold: Maximum allowed frequency difference (default 0.2)
        palindrome_maf_threshold: MAF threshold for excluding palindromic SNPs (default 0.4)
        verbose: Enable verbose logging
        keep_indels: Whether to keep indels (future feature, currently excluded)
        update_ids: Whether to generate ID update file
        include_x: Include X chromosome (1000G only; HRC has no X)
        output_format: Output format ("vcf", "plink1", or "plink2")
        max_workers: Maximum parallel workers for per-chromosome processing
        keep_temp_files: Keep temporary intermediate files
        plink_path: Path to PLINK2 executable (auto-detect if None)
    """

    bim_file: Path
    frq_file: Path
    ref_file: Path
    panel: Literal["hrc", "1000g", "topmed"]

    # Optional settings
    population: str = "ALL"
    output_dir: Path | None = None

    # Thresholds
    freq_diff_threshold: float = 0.2
    palindrome_maf_threshold: float = 0.4

    # Behavior flags
    verbose: bool = False
    keep_indels: bool = False
    update_ids: bool = False
    include_x: bool = False

    # Pipeline options
    output_format: Literal["vcf", "plink1", "plink2"] = "vcf"
    max_workers: int | None = None  # Default: min(cpu_count(), 22)
    keep_temp_files: bool = False
    plink_path: Path | None = None

    # Report options
    generate_report: bool = True
    report_file: Path | None = None  # Default: {output_dir}/{stem}-report.json

    # Chromosomes to process (default: autosomes 1-22)
    chromosomes: set[str] = field(
        default_factory=lambda: {str(i) for i in range(1, 23)}
    )

    def __post_init__(self) -> None:
        """Validate configuration and set defaults."""
        # Ensure paths are Path objects
        if isinstance(self.bim_file, str):
            self.bim_file = Path(self.bim_file)
        if isinstance(self.frq_file, str):
            self.frq_file = Path(self.frq_file)
        if isinstance(self.ref_file, str):
            self.ref_file = Path(self.ref_file)

        # Set output_dir to current directory if not specified
        if self.output_dir is None:
            self.output_dir = Path.cwd()
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)

        # Add X chromosome for 1000G if include_x is set
        if self.include_x and self.panel == "1000g":
            self.chromosomes = self.chromosomes | {"23", "X"}

        # Set default report file path
        if self.report_file is None and self.generate_report:
            self.report_file = self.output_dir / f"{self.file_stem}-report.json"
        elif isinstance(self.report_file, str):
            self.report_file = Path(self.report_file)

    @property
    def file_stem(self) -> str:
        """Get the stem (filename without extension) of the BIM file."""
        return self.bim_file.stem

    @property
    def panel_name(self) -> str:
        """Get display name for the reference panel."""
        names = {
            "hrc": "HRC",
            "1000g": "1000G",
            "topmed": "TOPMed",
        }
        return names.get(self.panel, self.panel.upper())

    def get_output_path(self, filename: str) -> Path:
        """Get full output path for a file.

        Args:
            filename: Name of the output file

        Returns:
            Full path to the output file
        """
        assert self.output_dir is not None  # Set in __post_init__
        return self.output_dir / filename

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        if not self.bim_file.exists():
            errors.append(f"BIM file not found: {self.bim_file}")

        if not self.frq_file.exists():
            errors.append(f"FRQ file not found: {self.frq_file}")

        if not self.ref_file.exists():
            errors.append(f"Reference file not found: {self.ref_file}")

        if self.output_dir is not None and not self.output_dir.exists():
            errors.append(f"Output directory does not exist: {self.output_dir}")

        if self.freq_diff_threshold < 0 or self.freq_diff_threshold > 1:
            errors.append(
                f"freq_diff_threshold must be between 0 and 1: {self.freq_diff_threshold}"
            )

        if self.palindrome_maf_threshold < 0 or self.palindrome_maf_threshold > 0.5:
            errors.append(
                f"palindrome_maf_threshold must be between 0 and 0.5: "
                f"{self.palindrome_maf_threshold}"
            )

        valid_populations = {"AFR", "AMR", "EAS", "EUR", "SAS", "ALL"}
        if self.panel == "1000g" and self.population not in valid_populations:
            errors.append(
                f"Invalid population '{self.population}'. "
                f"Valid options: {valid_populations}"
            )

        return errors
