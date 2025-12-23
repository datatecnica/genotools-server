"""JSON report writer for pipeline output.

[claude-assisted] Creates a comprehensive JSON report containing metadata,
statistics, and full variant-level decisions for audit and reproducibility.
"""

import json
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from imputation_harmonizer.config import Config
from imputation_harmonizer.models import (
    AlleleAction,
    CheckResult,
    ExcludeReason,
    Statistics,
    StrandAction,
)


@dataclass
class VariantReport:
    """Single variant decision for JSON output.

    Attributes:
        id: Original variant ID from BIM file
        chr: Chromosome
        pos: Position
        action: "keep" or "exclude"
        exclude_reason: Reason for exclusion (if excluded)
        strand_flip: Whether strand needs to be flipped
        force_ref: Reference allele to force (if needed)
        update_position: New position (if position update needed)
        update_chromosome: New chromosome (if chromosome update needed)
        update_id: New ID from reference (if ID update needed)
        ref_freq: Reference allele frequency
        bim_freq: BIM file allele frequency
        freq_diff: Frequency difference (ref - bim)
        check_code: Check outcome code (1-6, matching Perl script)
        matched_by: How variant was matched ("position", "id", "none")
    """

    id: str
    chr: str
    pos: int
    action: str
    exclude_reason: str | None
    strand_flip: bool
    force_ref: str | None
    update_position: int | None
    update_chromosome: str | None
    update_id: str | None
    ref_freq: float | None
    bim_freq: float | None
    freq_diff: float | None
    check_code: int | None
    matched_by: str


@dataclass
class PipelineReport:
    """Complete pipeline report for JSON output.

    Attributes:
        metadata: Version, timestamp, input files, thresholds
        statistics: Counts from Statistics dataclass
        variants: List of all variant decisions
        output_files: List of output file paths
    """

    metadata: dict
    statistics: dict
    variants: list[VariantReport]
    output_files: list[str]


class ReportWriter:
    """Collects variant decisions and writes JSON report.

    Usage:
        writer = ReportWriter(config)
        for result in check_variants(...):
            writer.add_result(result, bim_variant)
        writer.write(output_path, stats, output_files)
    """

    def __init__(self, config: Config) -> None:
        """Initialize report writer.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.variants: list[VariantReport] = []
        self.start_time = datetime.now()

    def add_result(self, result: CheckResult, chr: str, pos: int) -> None:
        """Add a variant check result to the report.

        Args:
            result: CheckResult from variant comparison
            chr: Chromosome from BIM variant
            pos: Position from BIM variant
        """
        # Determine action
        action = "exclude" if result.exclude else "keep"

        # Convert exclude reason enum to string
        exclude_reason = None
        if result.exclude_reason is not None:
            exclude_reason = result.exclude_reason.name.lower()

        # Create variant report
        variant = VariantReport(
            id=result.snp_id,
            chr=chr,
            pos=pos,
            action=action,
            exclude_reason=exclude_reason,
            strand_flip=result.strand_action == StrandAction.FLIP,
            force_ref=result.force_ref_allele,
            update_position=result.update_position,
            update_chromosome=result.update_chromosome,
            update_id=result.update_id,
            ref_freq=result.ref_freq,
            bim_freq=result.bim_freq,
            freq_diff=result.freq_diff,
            check_code=result.check_code,
            matched_by=result.matched_by,
        )

        self.variants.append(variant)

    def write(
        self,
        output_path: Path,
        stats: Statistics,
        output_files: list[Path],
    ) -> None:
        """Write complete JSON report atomically.

        Writes to a temporary file first, then renames to prevent
        partial files on interruption.

        Args:
            output_path: Path for JSON report file
            stats: Pipeline statistics
            output_files: List of output VCF.GZ files created
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # Build metadata
        metadata = {
            "version": "1.0.0",
            "tool": "imputation-harmonizer",
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration,
            "input_files": {
                "bim_file": str(self.config.bim_file),
                "frq_file": str(self.config.frq_file) if self.config.frq_file else None,
                "ref_file": str(self.config.ref_file),
            },
            "reference_panel": self.config.panel_name,
            "population": self.config.population,
            "thresholds": {
                "freq_diff": self.config.freq_diff_threshold,
                "palindrome_maf": self.config.palindrome_maf_threshold,
            },
            "options": {
                "include_x": self.config.include_x,
                "output_format": self.config.output_format,
            },
        }

        # Build statistics dict
        statistics = {
            "total_variants": stats.total,
            "excluded": {
                "total": (
                    stats.indels
                    + stats.alt_chr_skipped
                    + stats.no_match
                    + stats.palindromic_excluded
                    + stats.freq_diff_excluded
                    + stats.allele_mismatch
                    + stats.duplicates
                ),
                "indels": stats.indels,
                "alt_chromosome": stats.alt_chr_skipped,
                "not_in_reference": stats.no_match,
                "palindromic_high_maf": stats.palindromic_excluded,
                "freq_diff_too_high": stats.freq_diff_excluded,
                "allele_mismatch": stats.allele_mismatch,
                "duplicate": stats.duplicates,
            },
            "strand_flipped": stats.strand_flip,
            "strand_ok": stats.strand_ok,
            "ref_alt_swapped": stats.ref_alt_swap,
            "ref_alt_ok": stats.ref_alt_ok,
            "matching": {
                "position_match_id_match": stats.position_match_id_match,
                "position_match_id_mismatch": stats.position_match_id_mismatch,
                "id_match_position_mismatch": stats.id_match_position_mismatch,
                "no_match": stats.no_match,
            },
        }

        # Build report
        report = PipelineReport(
            metadata=metadata,
            statistics=statistics,
            variants=self.variants,
            output_files=[str(f) for f in output_files],
        )

        # Convert to JSON-serializable dict
        report_dict = {
            "metadata": report.metadata,
            "statistics": report.statistics,
            "variants": [asdict(v) for v in report.variants],
            "output_files": report.output_files,
        }

        # Write atomically using temp file + rename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=output_path.parent,
            suffix=".json.tmp",
            delete=False,
        ) as tmp:
            json.dump(report_dict, tmp, indent=2)
            tmp_path = Path(tmp.name)

        # Atomic rename
        tmp_path.rename(output_path)
