"""PLINK format auto-detection.

[claude-assisted] Detects PLINK1 vs PLINK2 format based on file extensions
and validates that required companion files exist.

PLINK1 format: .bed/.bim/.fam (binary) with .frq (frequency)
PLINK2 format: .pgen/.pvar/.psam (binary) with .afreq (frequency)
"""

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class PlinkFormat(Enum):
    """PLINK file format version."""

    PLINK1 = auto()  # .bed/.bim/.fam
    PLINK2 = auto()  # .pgen/.pvar/.psam


@dataclass(slots=True)
class PlinkFileSet:
    """A complete set of PLINK files.

    Attributes:
        format: PLINK1 or PLINK2
        prefix: Common prefix for all files (e.g., /data/study)
        variant_file: Path to .bim or .pvar
        freq_file: Path to .frq or .afreq (may be None)
        genotype_file: Path to .bed or .pgen
        sample_file: Path to .fam or .psam
    """

    format: PlinkFormat
    prefix: Path
    variant_file: Path
    freq_file: Path | None
    genotype_file: Path
    sample_file: Path


def _strip_gz(path: Path) -> Path:
    """Strip .gz extension if present."""
    if path.suffix == ".gz":
        return path.with_suffix("")
    return path


def detect_plink_format(variant_file: Path) -> PlinkFormat:
    """Detect PLINK format from variant file extension.

    Args:
        variant_file: Path to .bim, .bim.gz, .pvar, or .pvar.gz

    Returns:
        PlinkFormat.PLINK1 or PlinkFormat.PLINK2

    Raises:
        ValueError: If extension is not recognized

    Examples:
        >>> detect_plink_format(Path("data.bim"))
        PlinkFormat.PLINK1
        >>> detect_plink_format(Path("data.pvar.gz"))
        PlinkFormat.PLINK2
    """
    # Handle .gz extension
    base_path = _strip_gz(variant_file)
    suffix = base_path.suffix.lower()

    if suffix == ".bim":
        return PlinkFormat.PLINK1
    elif suffix == ".pvar":
        return PlinkFormat.PLINK2
    else:
        raise ValueError(
            f"Unrecognized variant file extension: {variant_file.suffix}. "
            f"Expected .bim, .bim.gz, .pvar, or .pvar.gz"
        )


def get_prefix(variant_file: Path) -> Path:
    """Get file prefix from variant file path.

    Strips .bim/.pvar and optional .gz extension.

    Args:
        variant_file: Path to variant file

    Returns:
        File prefix (directory + stem without extensions)

    Examples:
        >>> get_prefix(Path("/data/study.bim"))
        Path("/data/study")
        >>> get_prefix(Path("/data/study.pvar.gz"))
        Path("/data/study")
    """
    base_path = _strip_gz(variant_file)
    return base_path.with_suffix("")


def get_freq_file_extension(plink_format: PlinkFormat) -> str:
    """Get expected frequency file extension for format.

    Args:
        plink_format: PLINK1 or PLINK2

    Returns:
        ".frq" or ".afreq"
    """
    if plink_format == PlinkFormat.PLINK1:
        return ".frq"
    else:
        return ".afreq"


def resolve_plink_fileset(
    variant_file: Path,
    freq_file: Path | None = None,
) -> PlinkFileSet:
    """Resolve complete PLINK fileset from variant file.

    Given a .bim or .pvar file, finds the associated files
    (.bed/.fam or .pgen/.psam).

    Args:
        variant_file: Path to .bim or .pvar file
        freq_file: Optional explicit frequency file

    Returns:
        PlinkFileSet with all file paths

    Raises:
        FileNotFoundError: If required files don't exist
        ValueError: If variant file extension is not recognized
    """
    plink_format = detect_plink_format(variant_file)
    prefix = get_prefix(variant_file)

    if plink_format == PlinkFormat.PLINK1:
        genotype_file = prefix.with_suffix(".bed")
        sample_file = prefix.with_suffix(".fam")
        default_freq = prefix.with_suffix(".frq")
    else:
        genotype_file = prefix.with_suffix(".pgen")
        sample_file = prefix.with_suffix(".psam")
        default_freq = prefix.with_suffix(".afreq")

    # Validate required files exist
    if not variant_file.exists():
        raise FileNotFoundError(f"Variant file not found: {variant_file}")
    if not genotype_file.exists():
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_file}")

    # Resolve frequency file (explicit > default > None)
    resolved_freq: Path | None = None
    if freq_file is not None:
        if not freq_file.exists():
            raise FileNotFoundError(f"Frequency file not found: {freq_file}")
        resolved_freq = freq_file
    elif default_freq.exists():
        resolved_freq = default_freq

    return PlinkFileSet(
        format=plink_format,
        prefix=prefix,
        variant_file=variant_file,
        freq_file=resolved_freq,
        genotype_file=genotype_file,
        sample_file=sample_file,
    )


def find_variant_file(prefix: Path) -> Path | None:
    """Find variant file (.bim or .pvar) from prefix.

    Tries PLINK2 format first, then PLINK1.

    Args:
        prefix: File prefix (without extension)

    Returns:
        Path to variant file if found, None otherwise
    """
    # Try PLINK2 first
    for ext in [".pvar", ".pvar.gz"]:
        path = prefix.with_suffix(ext)
        if path.exists():
            return path

    # Try PLINK1
    for ext in [".bim", ".bim.gz"]:
        path = prefix.with_suffix(ext)
        if path.exists():
            return path

    return None
