"""Log file writer for statistics and summary.

[claude-assisted] Implements LOG file generation matching the original
Perl script's output (lines 506-523).
"""

from pathlib import Path

from imputation_harmonizer.config import Config
from imputation_harmonizer.models import Statistics


def write_log_file(
    output_dir: Path,
    file_stem: str,
    panel_name: str,
    config: Config,
    stats: Statistics,
) -> Path:
    """Write LOG file with run statistics.

    Args:
        output_dir: Directory for output files
        file_stem: Base filename (from BIM file)
        panel_name: Reference panel name ("HRC" or "1000G")
        config: Configuration used for the run
        stats: Statistics collected during processing

    Returns:
        Path to generated log file
    """
    log_path = output_dir / f"LOG-{file_stem}-{panel_name}.txt"

    with open(log_path, "w") as f:
        # Options used
        f.write("Options Set:\n")
        f.write(f"Reference Panel:             {panel_name}\n")
        f.write(f"Bim filename:                {config.bim_file}\n")
        f.write(f"Reference filename:          {config.ref_file}\n")
        f.write(f"Allele frequencies filename: {config.frq_file}\n")

        if config.panel == "1000g":
            f.write(f"Population for 1000G:        {config.population}\n")

        if config.verbose:
            f.write("Verbose logging flag set\n")

        f.write(f"Frequency diff threshold:    {config.freq_diff_threshold}\n")
        f.write(f"Palindrome MAF threshold:    {config.palindrome_maf_threshold}\n")
        f.write("\n\n")

        # Matching statistics
        f.write(f"Matching to {panel_name}\n\n")

        f.write("Position Matches\n")
        f.write(f" ID matches {panel_name} {stats.position_match_id_match}\n")
        f.write(f" ID Doesn't match {panel_name} {stats.position_match_id_mismatch}\n")
        f.write(f" Total Position Matches {stats.position_matches}\n")

        f.write("ID Match\n")
        f.write(f" Different position to {panel_name} {stats.id_match_position_mismatch}\n")

        f.write(f"No Match to {panel_name} {stats.no_match}\n")
        f.write(f"Skipped (X, XY, Y, MT) {stats.alt_chr_skipped}\n")
        f.write(f"Total in bim file {stats.total}\n")
        f.write(f"Total processed {stats.total_checked}\n\n")

        # Indels
        f.write(f"Indels (excluded) {stats.indels}\n\n")

        # Strand statistics
        f.write(f"SNPs not changed {stats.ref_alt_ok}\n")
        f.write(f"SNPs to change ref alt {stats.ref_alt_swap}\n")
        f.write(f"Strand ok {stats.strand_ok}\n")
        f.write(f"Total Strand ok {stats.strand_ok + stats.ref_alt_ok}\n")
        f.write(f"Total removed for allele Frequency diff > {config.freq_diff_threshold} {stats.freq_diff_excluded}\n")
        f.write(f"Palindromic SNPs with Freq > {config.palindrome_maf_threshold} {stats.palindromic_excluded}\n\n")

        f.write(f"Strand to change {stats.strand_flip}\n")
        f.write(f"Total checked {stats.total_checked}\n")
        f.write(f"Total checked Strand {stats.total_strand_checked}\n")
        f.write(f"Total removed for allele Frequency diff > {config.freq_diff_threshold} {stats.freq_diff_excluded}\n")
        f.write(f"Palindromic SNPs with Freq > {config.palindrome_maf_threshold} {stats.palindromic_excluded}\n\n")

        # Allele mismatches
        f.write(f"\nNon Matching alleles {stats.allele_mismatch}\n")
        f.write(f"ID and allele mismatching {stats.id_allele_mismatch}; where {panel_name} is . {stats.hrc_dot}\n")
        f.write(f"Duplicates removed {stats.duplicates}\n")

    return log_path


def print_summary(stats: Statistics, panel_name: str) -> None:
    """Print summary statistics to stdout.

    Args:
        stats: Statistics collected during processing
        panel_name: Reference panel name ("HRC" or "1000G")
    """
    print(f"\nMatching to {panel_name}")

    print("\nPosition Matches")
    print(f" ID matches {panel_name} {stats.position_match_id_match}")
    print(f" ID Doesn't match {panel_name} {stats.position_match_id_mismatch}")
    print(f" Total Position Matches {stats.position_matches}")

    print("ID Match")
    print(f" Different position to {panel_name} {stats.id_match_position_mismatch}")

    print(f"No Match to {panel_name} {stats.no_match}")
    print(f"Skipped (X, XY, Y, MT) {stats.alt_chr_skipped}")
    print(f"Total in bim file {stats.total}")
    print(f"Total processed {stats.total_checked}")

    print(f"\nIndels (excluded) {stats.indels}")

    print(f"\nSNPs not changed {stats.ref_alt_ok}")
    print(f"SNPs to change ref alt {stats.ref_alt_swap}")
    print(f"Strand ok {stats.strand_ok}")
    print(f"Strand to change {stats.strand_flip}")

    print(f"\nNon Matching alleles {stats.allele_mismatch}")
    print(f"Duplicates removed {stats.duplicates}")
