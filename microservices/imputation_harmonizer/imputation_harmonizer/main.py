"""Main orchestration for the imputation harmonizer.

[claude-assisted] Implements the main run_check() function that coordinates
all components: loading reference, parsing files, checking variants,
running PLINK2 pipeline, and generating JSON report.
"""

from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from imputation_harmonizer.checks.comparator import check_variants
from imputation_harmonizer.config import Config
from imputation_harmonizer.models import Statistics
from imputation_harmonizer.parsers.bim import count_bim_variants, parse_bim
from imputation_harmonizer.parsers.frq import parse_frq
from imputation_harmonizer.reference.hrc import HRCPanel
from imputation_harmonizer.reference.kg import KGPanel
from imputation_harmonizer.reference.topmed import TOPMedPanel
from imputation_harmonizer.writers.log import print_summary, write_log_file
from imputation_harmonizer.writers.plink_files import PlinkFileWriter
from imputation_harmonizer.writers.report import ReportWriter

console = Console()


def run_check(
    config: Config,
    skip_freq: bool = False,
) -> None:
    """Run the full pre-imputation check and PLINK2 pipeline.

    Main entry point that coordinates:
    1. Loading reference panel (HRC, 1000G, or TOPMed)
    2. Parsing frequency file (unless skip_freq=True)
    3. Streaming and checking BIM file variants
    4. Writing correction files
    5. Running PLINK2 pipeline
    6. Writing JSON report (if enabled)

    Args:
        config: Configuration with file paths and thresholds
        skip_freq: If True, skip loading frequency file and use empty frequencies

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If file formats are invalid
        RuntimeError: If PLINK2 not found
    """
    stats = Statistics()

    # Initialize report writer if enabled
    report_writer = ReportWriter(config) if config.generate_report else None

    # Step 1: Load reference panel
    console.print(f"Reading {config.ref_file.name}")

    if config.panel == "hrc":
        reference = HRCPanel()
    elif config.panel == "topmed":
        reference = TOPMedPanel()
    else:
        reference = KGPanel()

    reference.load(
        filepath=config.ref_file,
        population=config.population,
        verbose=config.verbose,
    )

    console.print(f"Loaded {len(reference):,} variants from reference panel\n")

    # Step 2: Parse frequency file (or use empty dict if skipping)
    if skip_freq:
        console.print("Skipping frequency file (--skip-freq)\n")
        frequencies: dict[str, float] = {}
    else:
        console.print(f"Reading {config.frq_file.name}")
        frequencies = parse_frq(config.frq_file)
        console.print(f"Loaded {len(frequencies):,} frequencies\n")

    # Step 3: Count BIM variants for progress bar
    console.print(f"Processing {config.bim_file.name}")
    total_variants = count_bim_variants(config.bim_file)
    console.print(f"Total variants: {total_variants:,}\n")

    # Step 4: Process variants and write output files
    assert config.output_dir is not None  # Set in Config.__post_init__

    with PlinkFileWriter(
        output_dir=config.output_dir,
        file_stem=config.file_stem,
        panel_name=config.panel_name,
    ) as writer:
        # Create progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Checking variants...", total=total_variants)

            # Stream through BIM file and check each variant
            bim_variants = parse_bim(config.bim_file, frequencies)

            for bim_variant, result in check_variants(bim_variants, reference, config, stats):
                writer.write_result(result)
                if report_writer:
                    report_writer.add_result(result, bim_variant.chr, bim_variant.pos)
                progress.advance(task)

    # Step 5: Run PLINK2 pipeline
    from imputation_harmonizer.plink_runner import PlinkPipeline

    console.print("\n[bold]Running PLINK2 pipeline...[/bold]")

    pipeline = PlinkPipeline(
        plink_path=config.plink_path,
        console=console,
        verbose=config.verbose,
    )

    # Determine input format (currently assume PLINK1)
    input_prefix = config.bim_file.parent / config.bim_file.stem

    output_files = pipeline.run_parallel_pipeline(
        input_prefix=input_prefix,
        output_dir=config.output_dir,
        reference_file=config.ref_file,
        panel_type=config.panel,
        population=config.population,
        freq_diff_threshold=config.freq_diff_threshold,
        palindrome_maf_threshold=config.palindrome_maf_threshold,
        output_format=config.output_format,  # type: ignore
        file_stem=config.file_stem,
        input_format="plink1",
        max_workers=config.max_workers,
    )

    # Cleanup temp files by default
    if not config.keep_temp_files:
        pipeline.cleanup_temp_files(config.output_dir)

    # Step 6: Write log file
    log_path = write_log_file(
        output_dir=config.output_dir,
        file_stem=config.file_stem,
        panel_name=config.panel_name,
        config=config,
        stats=stats,
    )

    # Step 7: Write JSON report if enabled
    if report_writer and config.report_file:
        report_writer.write(
            output_path=config.report_file,
            stats=stats,
            output_files=output_files,
        )

    # Print summary
    print_summary(stats, config.panel_name)

    # Print output file information
    console.print("\n[bold]Output files generated:[/bold]")
    console.print(f"  Exclude file:       {writer.exclude_count:,} variants")
    console.print(f"  Strand flip file:   {writer.strand_flip_count:,} variants")
    console.print(f"  Force allele file:  {writer.force_allele_count:,} variants")
    console.print(f"  Position updates:   {writer.position_update_count:,} variants")
    console.print(f"  Chromosome updates: {writer.chromosome_update_count:,} variants")
    console.print(f"  ID updates:         {writer.id_update_count:,} variants")
    console.print(f"\n  Log file:           {log_path}")
    if config.generate_report and config.report_file:
        console.print(f"  Report file:        {config.report_file}")
    console.print(f"\n  Per-chromosome output files: {len(output_files)}")
    for f in sorted(output_files):
        console.print(f"    {f.name}")

    console.print("\n[green]Pipeline complete! Files ready for imputation.[/green]\n")

    # Clear reference panel to free memory
    reference.clear()
