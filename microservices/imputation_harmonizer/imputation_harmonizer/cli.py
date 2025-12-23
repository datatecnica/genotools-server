"""Typer CLI for the imputation harmonizer.

[claude-assisted] Implements CLI matching the original Perl script's
command-line interface (lines 91-174), with added auto-frequency generation.

Usage:
    # Auto-generate frequency file (default behavior)
    imputation-harmonizer -b data.bim -r HRC.tab -p hrc

    # With explicit frequency file
    imputation-harmonizer -b data.bim -f data.frq -r HRC.tab -p hrc

    # 1000G with European population
    imputation-harmonizer -b data.bim -r 1000G.legend -p 1000g --pop EUR
"""

import shutil
import subprocess
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="imputation-harmonizer",
    help="Check PLINK files against HRC/1000G reference panels for pre-imputation QC",
    add_completion=False,
)

console = Console()


class Panel(str, Enum):
    """Reference panel type."""

    hrc = "hrc"
    kg = "1000g"
    topmed = "topmed"


class OutputFormat(str, Enum):
    """Output file format."""

    vcf = "vcf"
    plink1 = "plink1"
    plink2 = "plink2"


class Population(str, Enum):
    """1000G population for frequency column."""

    AFR = "AFR"
    AMR = "AMR"
    EAS = "EAS"
    EUR = "EUR"
    SAS = "SAS"
    ALL = "ALL"


def find_plink() -> str | None:
    """Find PLINK executable in PATH.

    Returns:
        Path to plink executable, or None if not found
    """
    return shutil.which("plink") or shutil.which("plink1.9") or shutil.which("plink2")


def generate_frequency_file(bim_file: Path, output_dir: Path) -> Path:
    """Generate .frq file using PLINK.

    Args:
        bim_file: Path to .bim file (used to derive .bed/.fam paths)
        output_dir: Directory for output files

    Returns:
        Path to generated .frq file

    Raises:
        RuntimeError: If PLINK is not found or fails
    """
    plink = find_plink()
    if not plink:
        raise RuntimeError(
            "PLINK not found in PATH. Please install PLINK or provide a .frq file with --freq"
        )

    # Derive the bfile prefix (without .bim extension)
    bfile = bim_file.parent / bim_file.stem

    # Output file path
    frq_prefix = output_dir / bim_file.stem
    frq_file = output_dir / f"{bim_file.stem}.frq"

    console.print(f"Generating frequency file using PLINK...")
    console.print(f"  PLINK: {plink}")
    console.print(f"  Input: {bfile}")
    console.print(f"  Output: {frq_file}")

    # Run PLINK --freq
    cmd = [
        plink,
        "--bfile", str(bfile),
        "--freq",
        "--out", str(frq_prefix),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if not frq_file.exists():
            raise RuntimeError(
                f"PLINK ran but .frq file not created. PLINK output:\n{result.stderr}"
            )
        console.print(f"  [green]Frequency file generated successfully[/green]\n")
        return frq_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"PLINK failed with exit code {e.returncode}:\n{e.stderr}"
        )
    except FileNotFoundError:
        raise RuntimeError(f"PLINK executable not found: {plink}")


@app.command()
def check(
    bim: Annotated[
        Path,
        typer.Option(
            "--bim", "-b",
            help="PLINK .bim file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    ref: Annotated[
        Path,
        typer.Option(
            "--ref", "-r",
            help="Reference panel file (HRC or 1000G)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    panel: Annotated[
        Panel,
        typer.Option(
            "--panel", "-p",
            help="Reference panel type: 'hrc' or '1000g'",
        ),
    ],
    freq: Annotated[
        Path | None,
        typer.Option(
            "--freq", "-f",
            help="PLINK .frq frequency file (auto-generated if not provided)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    skip_freq: Annotated[
        bool,
        typer.Option(
            "--skip-freq",
            help="Skip frequency check (use when .frq file unavailable and PLINK not installed)",
        ),
    ] = False,
    population: Annotated[
        Population,
        typer.Option(
            "--pop",
            help="1000G population for frequency column (ignored for HRC)",
        ),
    ] = Population.ALL,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir", "-o",
            help="Output directory (default: current directory)",
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    freq_diff: Annotated[
        float,
        typer.Option(
            "--freq-diff",
            help="Maximum allowed allele frequency difference (default: 0.2)",
            min=0.0,
            max=1.0,
        ),
    ] = 0.2,
    palindrome_maf: Annotated[
        float,
        typer.Option(
            "--palindrome-maf",
            help="MAF threshold for excluding palindromic SNPs (default: 0.4)",
            min=0.0,
            max=0.5,
        ),
    ] = 0.4,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v",
            help="Enable verbose logging",
        ),
    ] = False,
    include_x: Annotated[
        bool,
        typer.Option(
            "--include-x",
            help="Include X chromosome (1000G only; HRC has no X)",
        ),
    ] = False,
    plink_path: Annotated[
        str | None,
        typer.Option(
            "--plink",
            help="Path to PLINK2 executable (default: auto-detect from PATH)",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--output-format",
            help="Output format: 'vcf' (VCF.GZ), 'plink1' (.bed), or 'plink2' (.pgen)",
        ),
    ] = OutputFormat.vcf,
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers", "-j",
            help="Number of parallel workers for per-chromosome processing (default: CPU count)",
            min=1,
            max=22,
        ),
    ] = None,
    keep_temp: Annotated[
        bool,
        typer.Option(
            "--keep-temp",
            help="Keep temporary intermediate files",
        ),
    ] = False,
    report_file: Annotated[
        Path | None,
        typer.Option(
            "--report-file",
            help="Path for JSON report file (default: {stem}-report.json)",
        ),
    ] = None,
    no_report: Annotated[
        bool,
        typer.Option(
            "--no-report",
            help="Skip generating JSON report (saves memory for large datasets)",
        ),
    ] = False,
) -> None:
    """Check PLINK files against HRC, 1000G, or TOPMed reference panel.

    Compares variants in your GWAS dataset against a reference panel,
    applies corrections (strand flips, allele assignment, position updates),
    and outputs per-chromosome VCF.GZ files ready for imputation.

    Pipeline:
    1. Split input by chromosome (single pass)
    2. Process each chromosome in parallel:
       - Generate frequency file
       - Check variants against reference
       - Apply corrections via PLINK2
    3. Export to VCF.GZ (or PLINK1/PLINK2 format)

    Example usage:

        # Basic usage with HRC reference
        imputation-harmonizer -b data.bim -r HRC.r1.sites.tab -p hrc

        # TOPMed reference (gzipped)
        imputation-harmonizer -b data.bim -r TOPMed.tab.gz -p topmed

        # 1000G with European population
        imputation-harmonizer -b data.bim -r 1000G.legend -p 1000g --pop EUR

        # Control parallelism
        imputation-harmonizer -b data.bim -r ref.tab.gz -p hrc --workers 8

        # Output as PLINK2 format instead of VCF
        imputation-harmonizer -b data.bim -r ref.tab -p hrc --output-format plink2
    """
    from imputation_harmonizer.config import Config
    from imputation_harmonizer.main import run_check

    # Print banner
    console.print("\n")
    console.print(
        "[bold]HRC/1000G Pre-Imputation Checking Tool[/bold]",
        style="blue",
    )
    console.print("Python implementation v1.0.0\n")

    # Set default output directory
    if output_dir is None:
        output_dir = Path.cwd()

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle frequency file
    frq_file: Path | None = freq

    if skip_freq:
        console.print("[yellow]Warning:[/yellow] Skipping frequency checks (--skip-freq)")
        console.print("  Frequency-based exclusions will not be performed\n")
        # Create an empty frq file path that will result in empty frequencies dict
        frq_file = None
    elif freq is None:
        # Auto-generate frequency file
        try:
            frq_file = generate_frequency_file(bim, output_dir)
        except RuntimeError as e:
            console.print(f"[red]ERROR:[/red] {e}")
            console.print(
                "\n[yellow]Hint:[/yellow] Install PLINK, provide --freq, or use --skip-freq"
            )
            raise typer.Exit(code=1)

    # Create configuration
    config = Config(
        bim_file=bim,
        frq_file=frq_file if frq_file else bim.with_suffix(".frq"),  # Dummy path for skip_freq
        ref_file=ref,
        panel=panel.value,
        population=population.value,
        output_dir=output_dir,
        freq_diff_threshold=freq_diff if not skip_freq else 1.0,  # Disable freq check if skipping
        palindrome_maf_threshold=palindrome_maf,
        verbose=verbose,
        include_x=include_x,
        output_format=output_format.value,
        max_workers=workers,
        keep_temp_files=keep_temp,
        plink_path=Path(plink_path) if plink_path else None,
        generate_report=not no_report,
        report_file=report_file,
    )

    # Print options
    console.print("Options Set:")
    console.print(f"Reference Panel:             {config.panel_name}")
    console.print(f"Bim filename:                {config.bim_file}")
    console.print(f"Reference filename:          {config.ref_file}")
    if not skip_freq:
        console.print(f"Allele frequencies filename: {frq_file}")
    else:
        console.print(f"Allele frequencies:          [skipped]")

    if config.panel == "1000g":
        console.print(f"Population for 1000G:        {config.population}")

    if config.verbose:
        console.print("Verbose logging flag set")

    if not skip_freq:
        console.print(f"Frequency diff threshold:    {config.freq_diff_threshold}")
    console.print(f"Palindrome MAF threshold:    {config.palindrome_maf_threshold}")
    console.print("\n")

    # Validate configuration (skip frq validation if skip_freq)
    if not skip_freq:
        errors = config.validate()
        if errors:
            for error in errors:
                console.print(f"[red]ERROR:[/red] {error}")
            raise typer.Exit(code=1)

    # Print pipeline mode
    console.print(f"Output format:               {config.output_format}")
    if config.max_workers:
        console.print(f"Workers:                     {config.max_workers}")
    if config.generate_report:
        console.print(f"Report file:                 {config.report_file}")
    console.print("")

    # Run the check
    try:
        run_check(config, skip_freq=skip_freq)
    except Exception as e:
        console.print(f"[red]ERROR:[/red] {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
