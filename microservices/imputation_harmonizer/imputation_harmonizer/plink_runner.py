"""PLINK2 command execution for parallel chromosome pipeline.

[claude-assisted] Replaces Run-plink.sh with direct Python execution.
Uses PLINK2 for all operations and outputs VCF.GZ files per chromosome.

Pipeline:
1. Split input by chromosome (single pass via PLINK2 --split-chr)
2. Process each chromosome in parallel:
   - Generate frequency file
   - Check variants against reference
   - Write correction files
   - Apply corrections
   - Export to VCF.GZ
"""

import multiprocessing
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


@dataclass
class PlinkResult:
    """Result from a PLINK command execution.

    Attributes:
        success: Whether command succeeded
        command: Full command that was run
        stdout: Standard output
        stderr: Standard error
        output_prefix: Output file prefix
    """

    success: bool
    command: str
    stdout: str
    stderr: str
    output_prefix: Path | None


def find_plink2() -> Path | None:
    """Find PLINK2 executable in PATH.

    Searches for plink2 first, then falls back to plink.
    Prefers PLINK2 for better performance and features.

    Returns:
        Path to executable, or None if not found
    """
    for name in ["plink2", "plink"]:
        path = shutil.which(name)
        if path:
            return Path(path)
    return None


def find_plink1() -> Path | None:
    """Find PLINK 1.9 executable in PATH.

    Used for operations not supported in PLINK2 (like --flip).

    Returns:
        Path to PLINK 1.9 executable, or None if not found
    """
    for name in ["plink", "plink1.9"]:
        path = shutil.which(name)
        if path:
            # Verify it's actually PLINK 1.x
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                version = result.stdout.strip() or result.stderr.strip()
                # PLINK 1.x typically shows "PLINK v1.90" or similar
                if "v1." in version or "1.9" in version:
                    return Path(path)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
    return None


def verify_plink2_version(plink_path: Path) -> tuple[bool, str]:
    """Verify PLINK executable is version 2.x.

    Args:
        plink_path: Path to PLINK executable

    Returns:
        Tuple of (is_plink2, version_string)
    """
    try:
        result = subprocess.run(
            [str(plink_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip() or result.stderr.strip()
        is_plink2 = "PLINK v2" in version or "plink2" in version.lower()
        return is_plink2, version
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "unknown"


class PlinkPipeline:
    """Executes PLINK2 pipeline for variant corrections.

    Replaces the generated Run-plink.sh script with direct execution.

    Pipeline steps:
    1. Split by chromosome (single pass)
    2. Per-chromosome (parallel):
       - Generate frequency
       - Check variants
       - Apply corrections
       - Export VCF
    """

    def __init__(
        self,
        plink_path: Path | None = None,
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize pipeline.

        Args:
            plink_path: Path to PLINK2 executable (auto-detect if None)
            console: Rich console for output
            verbose: Print detailed progress

        Raises:
            RuntimeError: If PLINK2 not found
        """
        self.plink_path = plink_path or find_plink2()
        if self.plink_path is None:
            raise RuntimeError(
                "PLINK2 not found in PATH. Please install PLINK2 or specify --plink"
            )

        # Find PLINK 1.9 for operations not supported in PLINK2 (like --flip)
        self.plink1_path = find_plink1()

        self.console = console or Console()
        self.verbose = verbose

        # Verify PLINK version
        is_plink2, version = verify_plink2_version(self.plink_path)
        if not is_plink2:
            self.console.print(
                f"[yellow]Warning:[/yellow] PLINK appears to be v1.x ({version}). "
                f"Some features may not work correctly. PLINK2 is recommended."
            )

    def _run_plink(
        self,
        args: list[str],
        description: str = "",
    ) -> PlinkResult:
        """Execute a PLINK command.

        Args:
            args: Command arguments (without plink path)
            description: Description for logging

        Returns:
            PlinkResult with execution details
        """
        cmd = [str(self.plink_path)] + args
        cmd_str = " ".join(cmd)

        if self.verbose:
            self.console.print(f"[dim]Running: {cmd_str}[/dim]")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            # Find output prefix from --out argument
            output_prefix = None
            for i, arg in enumerate(args):
                if arg == "--out" and i + 1 < len(args):
                    output_prefix = Path(args[i + 1])
                    break

            success = result.returncode == 0
            if not success and self.verbose:
                self.console.print(f"[red]PLINK error:[/red] {result.stderr}")

            return PlinkResult(
                success=success,
                command=cmd_str,
                stdout=result.stdout,
                stderr=result.stderr,
                output_prefix=output_prefix,
            )
        except subprocess.TimeoutExpired:
            return PlinkResult(
                success=False,
                command=cmd_str,
                stdout="",
                stderr="Command timed out after 1 hour",
                output_prefix=None,
            )

    def _run_plink1(
        self,
        args: list[str],
        description: str = "",
    ) -> PlinkResult:
        """Execute a PLINK 1.9 command.

        Used for operations not supported in PLINK2 (like --flip).

        Args:
            args: Command arguments (without plink path)
            description: Description for logging

        Returns:
            PlinkResult with execution details

        Raises:
            RuntimeError: If PLINK 1.9 not available
        """
        if self.plink1_path is None:
            raise RuntimeError(
                "PLINK 1.9 not found in PATH. Required for strand flip operation."
            )

        cmd = [str(self.plink1_path)] + args
        cmd_str = " ".join(cmd)

        if self.verbose:
            self.console.print(f"[dim]Running (plink1): {cmd_str}[/dim]")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )

            output_prefix = None
            for i, arg in enumerate(args):
                if arg == "--out" and i + 1 < len(args):
                    output_prefix = Path(args[i + 1])
                    break

            success = result.returncode == 0
            if not success and self.verbose:
                self.console.print(f"[red]PLINK1 error:[/red] {result.stderr}")

            return PlinkResult(
                success=success,
                command=cmd_str,
                stdout=result.stdout,
                stderr=result.stderr,
                output_prefix=output_prefix,
            )
        except subprocess.TimeoutExpired:
            return PlinkResult(
                success=False,
                command=cmd_str,
                stdout="",
                stderr="Command timed out after 1 hour",
                output_prefix=None,
            )

    def split_by_chromosome(
        self,
        input_prefix: Path,
        work_dir: Path,
        input_format: Literal["plink1", "plink2"] = "plink1",
    ) -> list[str]:
        """Split input into per-chromosome files.

        Creates separate PLINK2 files for each chromosome found in the input
        by extracting each chromosome individually.

        Args:
            input_prefix: Input file prefix (without extension)
            work_dir: Working directory for split files
            input_format: Input format ("plink1" for .bed/.bim/.fam,
                         "plink2" for .pgen/.pvar/.psam)

        Returns:
            List of chromosome strings found (e.g., ["1", "2", ..., "22"])

        Raises:
            RuntimeError: If split fails
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        # Build input args based on format
        if input_format == "plink1":
            input_args = ["--bfile", str(input_prefix)]
        else:
            input_args = ["--pfile", str(input_prefix)]

        # First, detect which chromosomes exist in the input
        # Read the variant file to find unique chromosomes
        if input_format == "plink1":
            var_file = input_prefix.with_suffix(".bim")
        else:
            var_file = input_prefix.with_suffix(".pvar")

        chromosomes_found: set[str] = set()
        with open(var_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                chr_val = line.split()[0]
                # Only keep autosomes (1-22)
                if chr_val.isdigit() and 1 <= int(chr_val) <= 22:
                    chromosomes_found.add(chr_val)

        chromosomes = sorted(chromosomes_found, key=int)

        if not chromosomes:
            return []

        # Extract each chromosome to its own file
        for chr_num in chromosomes:
            output_prefix = work_dir / f"split.chr{chr_num}"

            args = input_args + [
                "--chr", chr_num,
                "--make-pgen",
                "--out", str(output_prefix),
            ]

            result = self._run_plink(args, f"Extracting chromosome {chr_num}")
            if not result.success:
                self.console.print(
                    f"[yellow]Warning:[/yellow] Failed to extract chr{chr_num}: {result.stderr}"
                )

        # Verify which chromosomes were actually created
        verified_chromosomes = []
        for chr_num in chromosomes:
            chr_pgen = work_dir / f"split.chr{chr_num}.pgen"
            if chr_pgen.exists():
                verified_chromosomes.append(chr_num)

        return verified_chromosomes

    def process_chromosome(
        self,
        chr_num: str,
        work_dir: Path,
        reference_file: Path,
        panel_type: str,
        population: str,
        freq_diff_threshold: float,
        palindrome_maf_threshold: float,
        output_format: Literal["vcf", "plink1", "plink2"],
        output_dir: Path,
        file_stem: str,
    ) -> Path | None:
        """Process single chromosome (runs as parallel worker).

        Performs:
        1. Generate frequency file
        2. Load reference for this chromosome
        3. Parse variants and check against reference
        4. Write correction files
        5. Apply corrections via PLINK2
        6. Export to final format

        Args:
            chr_num: Chromosome number (e.g., "1", "22")
            work_dir: Working directory with split files
            reference_file: Path to reference panel file
            panel_type: Panel type ("hrc", "1000g", "topmed")
            population: Population for 1000G
            freq_diff_threshold: Max frequency difference
            palindrome_maf_threshold: MAF threshold for palindromic
            output_format: Output format
            output_dir: Final output directory
            file_stem: Base filename for outputs

        Returns:
            Path to output file (VCF or PLINK), or None if failed
        """
        # Import here to avoid circular imports
        from imputation_harmonizer.checks.comparator import check_variants
        from imputation_harmonizer.config import Config
        from imputation_harmonizer.models import Statistics
        from imputation_harmonizer.parsers.afreq import parse_afreq
        from imputation_harmonizer.parsers.pvar import parse_pvar
        from imputation_harmonizer.reference.hrc import HRCPanel
        from imputation_harmonizer.reference.kg import KGPanel
        from imputation_harmonizer.reference.topmed import TOPMedPanel
        from imputation_harmonizer.writers.plink_files import PlinkFileWriter

        chr_prefix = work_dir / f"split.chr{chr_num}"
        chr_work = work_dir / f"chr{chr_num}"
        chr_work.mkdir(exist_ok=True)

        # Step 1: Generate frequency file for this chromosome
        freq_result = self._run_plink(
            ["--pfile", str(chr_prefix), "--freq", "--out", str(chr_work / "freq")],
            f"Generating frequencies for chr{chr_num}",
        )
        if not freq_result.success:
            return None

        freq_file = chr_work / "freq.afreq"

        # Step 2: Load reference panel for this chromosome only
        if panel_type == "hrc":
            reference = HRCPanel()
        elif panel_type == "topmed":
            reference = TOPMedPanel()
        else:
            reference = KGPanel()

        reference.load(
            filepath=reference_file,
            population=population,
            verbose=False,
            chromosome=chr_num,
        )

        # Step 3: Parse variants and frequencies
        # Note: chr_prefix is like "split.chr22", so we can't use with_suffix()
        # because it would replace ".chr22" with ".pvar". Use parent/name instead.
        pvar_file = chr_prefix.parent / f"{chr_prefix.name}.pvar"
        frequencies = parse_afreq(freq_file) if freq_file.exists() else {}

        # Create a temporary config for the checker
        config = Config(
            bim_file=pvar_file,
            frq_file=freq_file,
            ref_file=reference_file,
            panel=panel_type,  # type: ignore
            population=population,
            output_dir=chr_work,
            freq_diff_threshold=freq_diff_threshold,
            palindrome_maf_threshold=palindrome_maf_threshold,
        )

        stats = Statistics()
        panel_name = config.panel_name

        # Step 4: Check variants and write correction files
        with PlinkFileWriter(
            output_dir=chr_work,
            file_stem=f"chr{chr_num}",
            panel_name=panel_name,
        ) as writer:
            variants = parse_pvar(pvar_file, frequencies)
            for _, result in check_variants(variants, reference, config, stats):
                writer.write_result(result)

        # Clear reference to free memory
        reference.clear()

        # Step 5: Apply corrections via PLINK2
        exclude_file = chr_work / f"Exclude-chr{chr_num}-{panel_name}.txt"
        flip_file = chr_work / f"Strand-Flip-chr{chr_num}-{panel_name}.txt"
        force_file = chr_work / f"Force-Allele1-chr{chr_num}-{panel_name}.txt"

        current_prefix = chr_prefix
        step = 0

        # Exclude variants
        if exclude_file.exists() and exclude_file.stat().st_size > 0:
            step += 1
            next_prefix = chr_work / f"step{step}"
            self._run_plink(
                ["--pfile", str(current_prefix), "--exclude", str(exclude_file),
                 "--make-pgen", "--out", str(next_prefix)],
                f"Excluding variants for chr{chr_num}",
            )
            current_prefix = next_prefix

        # Flip strand - requires PLINK 1.9 (PLINK2 doesn't support --flip)
        if flip_file.exists() and flip_file.stat().st_size > 0:
            step += 1
            # Convert current pgen to bed for PLINK1
            bed_prefix = chr_work / f"step{step}_bed"
            self._run_plink(
                ["--pfile", str(current_prefix), "--make-bed",
                 "--out", str(bed_prefix)],
                f"Converting to BED for flip chr{chr_num}",
            )

            # Flip using PLINK 1.9
            flipped_prefix = chr_work / f"step{step}_flipped"
            self._run_plink1(
                ["--bfile", str(bed_prefix), "--flip", str(flip_file),
                 "--make-bed", "--out", str(flipped_prefix)],
                f"Flipping strand for chr{chr_num}",
            )

            # Convert back to pgen
            next_prefix = chr_work / f"step{step}"
            self._run_plink(
                ["--bfile", str(flipped_prefix), "--make-pgen",
                 "--out", str(next_prefix)],
                f"Converting back to PGEN for chr{chr_num}",
            )
            current_prefix = next_prefix

        # Force reference allele
        if force_file.exists() and force_file.stat().st_size > 0:
            step += 1
            next_prefix = chr_work / f"step{step}"
            self._run_plink(
                ["--pfile", str(current_prefix), "--ref-allele", "force", str(force_file),
                 "--make-pgen", "--out", str(next_prefix)],
                f"Forcing ref allele for chr{chr_num}",
            )
            current_prefix = next_prefix

        # Step 6: Export to final format
        output_file: Path
        if output_format == "vcf":
            output_file = output_dir / f"{file_stem}-chr{chr_num}.vcf.gz"
            self._run_plink(
                ["--pfile", str(current_prefix), "--export", "vcf", "bgz",
                 "--out", str(output_dir / f"{file_stem}-chr{chr_num}")],
                f"Exporting VCF for chr{chr_num}",
            )
        elif output_format == "plink2":
            output_file = output_dir / f"{file_stem}-chr{chr_num}.pgen"
            self._run_plink(
                ["--pfile", str(current_prefix), "--make-pgen",
                 "--out", str(output_dir / f"{file_stem}-chr{chr_num}")],
                f"Exporting PLINK2 for chr{chr_num}",
            )
        else:  # plink1
            output_file = output_dir / f"{file_stem}-chr{chr_num}.bed"
            self._run_plink(
                ["--pfile", str(current_prefix), "--make-bed",
                 "--out", str(output_dir / f"{file_stem}-chr{chr_num}")],
                f"Exporting PLINK1 for chr{chr_num}",
            )

        return output_file if output_file.exists() else None

    def run_parallel_pipeline(
        self,
        input_prefix: Path,
        output_dir: Path,
        reference_file: Path,
        panel_type: str,
        population: str,
        freq_diff_threshold: float,
        palindrome_maf_threshold: float,
        output_format: Literal["vcf", "plink1", "plink2"],
        file_stem: str,
        input_format: Literal["plink1", "plink2"] = "plink1",
        max_workers: int | None = None,
    ) -> list[Path]:
        """Apply corrections and export per-chromosome files.

        Uses the correction files already created by check_variants() in main.py.
        Pipeline:
        1. Apply all corrections to full dataset (exclude, flip, force-allele)
        2. Split by chromosome and export to final format

        Args:
            input_prefix: Input file prefix
            output_dir: Final output directory (contains correction files)
            reference_file: Path to reference panel (unused, for API compat)
            panel_type: Panel type (for filename)
            population: Population (unused, for API compat)
            freq_diff_threshold: (unused, for API compat)
            palindrome_maf_threshold: (unused, for API compat)
            output_format: Output format
            file_stem: Base filename
            input_format: Input format
            max_workers: (unused, for API compat)

        Returns:
            List of output file paths (one per chromosome)

        Raises:
            RuntimeError: If pipeline fails
        """
        # Set up working directory
        work_dir = output_dir / ".work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Get panel name for correction file naming
        panel_name = {"hrc": "HRC", "1000g": "1000G", "topmed": "TOPMed"}.get(
            panel_type, panel_type.upper()
        )

        self.console.print(f"\n[bold]Applying corrections and exporting...[/bold]")
        self.console.print(f"  Output format: {output_format}")

        # Build input args
        if input_format == "plink1":
            input_args = ["--bfile", str(input_prefix)]
        else:
            input_args = ["--pfile", str(input_prefix)]

        # Correction files from main.py check_variants pass
        exclude_file = output_dir / f"Exclude-{file_stem}-{panel_name}.txt"
        flip_file = output_dir / f"Strand-Flip-{file_stem}-{panel_name}.txt"
        force_file = output_dir / f"Force-Allele1-{file_stem}-{panel_name}.txt"

        # Step 1: Apply exclusions
        current_prefix = input_prefix
        step = 0

        if exclude_file.exists() and exclude_file.stat().st_size > 0:
            step += 1
            next_prefix = work_dir / f"step{step}"
            self._run_plink(
                input_args + ["--exclude", str(exclude_file),
                              "--make-pgen", "--out", str(next_prefix)],
                "Excluding variants",
            )
            current_prefix = next_prefix
            input_args = ["--pfile", str(current_prefix)]

        # Step 2: Apply strand flips (requires PLINK 1.9)
        if flip_file.exists() and flip_file.stat().st_size > 0:
            step += 1
            # Convert to BED for PLINK 1.9
            bed_prefix = work_dir / f"step{step}_bed"
            self._run_plink(
                input_args + ["--make-bed", "--out", str(bed_prefix)],
                "Converting to BED for flip",
            )

            # Flip using PLINK 1.9
            flipped_prefix = work_dir / f"step{step}_flipped"
            self._run_plink1(
                ["--bfile", str(bed_prefix), "--flip", str(flip_file),
                 "--make-bed", "--out", str(flipped_prefix)],
                "Flipping strand",
            )

            # Convert back to PGEN
            next_prefix = work_dir / f"step{step}"
            self._run_plink(
                ["--bfile", str(flipped_prefix), "--make-pgen",
                 "--out", str(next_prefix)],
                "Converting back to PGEN",
            )
            current_prefix = next_prefix
            input_args = ["--pfile", str(current_prefix)]

        # Step 3: Force reference allele
        if force_file.exists() and force_file.stat().st_size > 0:
            step += 1
            next_prefix = work_dir / f"step{step}"
            self._run_plink(
                input_args + ["--ref-allele", "force", str(force_file),
                              "--make-pgen", "--out", str(next_prefix)],
                "Forcing reference allele",
            )
            current_prefix = next_prefix
            input_args = ["--pfile", str(current_prefix)]

        # Step 4: Split by chromosome and export
        self.console.print("\n[bold]Exporting per-chromosome files...[/bold]")

        # Detect chromosomes from current PVAR
        pvar_file = current_prefix.parent / f"{current_prefix.name}.pvar"
        chromosomes: set[str] = set()
        with open(pvar_file) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                chr_val = line.split()[0]
                if chr_val.isdigit() and 1 <= int(chr_val) <= 22:
                    chromosomes.add(chr_val)

        sorted_chromosomes = sorted(chromosomes, key=int)
        output_files: list[Path] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("Exporting...", total=len(sorted_chromosomes))

            for chr_num in sorted_chromosomes:
                progress.update(task, description=f"Exporting chr{chr_num}...")

                output_stem = f"{file_stem}-chr{chr_num}"

                if output_format == "vcf":
                    output_file = output_dir / f"{output_stem}.vcf.gz"
                    self._run_plink(
                        input_args + ["--chr", chr_num, "--export", "vcf", "bgz",
                                      "--out", str(output_dir / output_stem)],
                        f"Exporting VCF for chr{chr_num}",
                    )
                elif output_format == "plink2":
                    output_file = output_dir / f"{output_stem}.pgen"
                    self._run_plink(
                        input_args + ["--chr", chr_num, "--make-pgen",
                                      "--out", str(output_dir / output_stem)],
                        f"Exporting PLINK2 for chr{chr_num}",
                    )
                else:  # plink1
                    output_file = output_dir / f"{output_stem}.bed"
                    self._run_plink(
                        input_args + ["--chr", chr_num, "--make-bed",
                                      "--out", str(output_dir / output_stem)],
                        f"Exporting PLINK1 for chr{chr_num}",
                    )

                if output_file.exists():
                    output_files.append(output_file)
                progress.advance(task)

        self.console.print(f"\n[green]Pipeline complete![/green]")
        self.console.print(f"  Output files: {len(output_files)}")

        return output_files

    def cleanup_temp_files(self, output_dir: Path) -> None:
        """Remove temporary files created during pipeline.

        Args:
            output_dir: Directory containing .work subdirectory
        """
        work_dir = output_dir / ".work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
