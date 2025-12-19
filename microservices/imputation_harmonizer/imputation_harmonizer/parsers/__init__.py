"""Unified parser interface supporting PLINK1 and PLINK2 formats.

[claude-assisted] Provides format-agnostic parsing functions that auto-detect
PLINK1 (.bim/.frq) vs PLINK2 (.pvar/.afreq) based on file extension.
"""

from collections.abc import Iterator
from pathlib import Path

from imputation_harmonizer.format_detection import PlinkFormat, detect_plink_format
from imputation_harmonizer.models import BimVariant
from imputation_harmonizer.parsers.afreq import parse_afreq, parse_afreq_with_alleles
from imputation_harmonizer.parsers.bim import count_bim_variants, parse_bim
from imputation_harmonizer.parsers.frq import parse_frq, parse_frq_with_alleles
from imputation_harmonizer.parsers.pvar import count_pvar_variants, parse_pvar

__all__ = [
    # Unified interface
    "parse_variants",
    "parse_frequencies",
    "count_variants",
    # PLINK1 specific
    "parse_bim",
    "parse_frq",
    "parse_frq_with_alleles",
    "count_bim_variants",
    # PLINK2 specific
    "parse_pvar",
    "parse_afreq",
    "parse_afreq_with_alleles",
    "count_pvar_variants",
]


def parse_variants(
    filepath: Path,
    frequencies: dict[str, float] | None = None,
) -> Iterator[BimVariant]:
    """Parse variants from .bim or .pvar file (auto-detect format).

    Args:
        filepath: Path to variant file (.bim or .pvar, may be gzipped)
        frequencies: Optional frequency dictionary

    Yields:
        BimVariant for each variant

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized

    Example:
        >>> for v in parse_variants(Path("data.pvar"), freqs):
        ...     print(v.id, v.pos)
    """
    plink_format = detect_plink_format(filepath)
    if plink_format == PlinkFormat.PLINK1:
        yield from parse_bim(filepath, frequencies)
    else:
        yield from parse_pvar(filepath, frequencies)


def parse_frequencies(filepath: Path) -> dict[str, float]:
    """Parse frequencies from .frq or .afreq file (auto-detect format).

    Args:
        filepath: Path to frequency file (.frq or .afreq, may be gzipped)

    Returns:
        Dictionary mapping variant ID to frequency

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized
    """
    # Detect format from extension
    suffix = filepath.suffix.lower()
    if suffix == ".gz":
        # Check the extension before .gz
        base_suffix = filepath.with_suffix("").suffix.lower()
    else:
        base_suffix = suffix

    if base_suffix == ".frq":
        return parse_frq(filepath)
    elif base_suffix == ".afreq":
        return parse_afreq(filepath)
    else:
        raise ValueError(
            f"Unrecognized frequency file extension: {filepath.suffix}. "
            f"Expected .frq, .frq.gz, .afreq, or .afreq.gz"
        )


def count_variants(filepath: Path) -> int:
    """Count variants in .bim or .pvar file (auto-detect format).

    Args:
        filepath: Path to variant file (.bim or .pvar, may be gzipped)

    Returns:
        Number of variants

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not recognized
    """
    plink_format = detect_plink_format(filepath)
    if plink_format == PlinkFormat.PLINK1:
        return count_bim_variants(filepath)
    else:
        return count_pvar_variants(filepath)
