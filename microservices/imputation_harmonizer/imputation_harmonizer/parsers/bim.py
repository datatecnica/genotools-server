"""PLINK BIM file parser.

[claude-assisted] Implements streaming BIM file parsing matching the original
Perl script's BIM processing logic. Supports gzipped files.

BIM file format (tab/space-separated, no header):
chromosome  rsID  genetic_distance  position  allele1  allele2
1           rs123 0                 10000     A        G
"""

from collections.abc import Iterator
from pathlib import Path

from imputation_harmonizer.io_utils import count_lines, smart_open
from imputation_harmonizer.models import BimVariant
from imputation_harmonizer.utils import normalize_chromosome


def parse_bim(
    filepath: Path,
    frequencies: dict[str, float] | None = None,
) -> Iterator[BimVariant]:
    """Stream variants from a PLINK BIM file.

    This is a generator that yields one variant at a time to avoid
    loading the entire file into memory (files can have 1M+ variants).

    Args:
        filepath: Path to PLINK .bim file
        frequencies: Optional dict mapping SNP IDs to frequencies (from .frq file)

    Yields:
        BimVariant for each line in the file

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If line format is invalid

    Example:
        >>> for variant in parse_bim(Path("data.bim")):
        ...     print(variant.id, variant.pos)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"BIM file not found: {filepath}")

    frequencies = frequencies or {}

    with smart_open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue

            # Split on whitespace (handles both tab and space)
            parts = line.split()

            if len(parts) < 6:
                raise ValueError(
                    f"Invalid BIM format at line {line_num}: expected 6 columns, "
                    f"got {len(parts)}"
                )

            chr_val = normalize_chromosome(parts[0])
            snp_id = parts[1]
            genetic_dist = float(parts[2])
            pos = int(parts[3])
            allele1 = parts[4].upper()  # Ensure uppercase
            allele2 = parts[5].upper()

            # Look up frequency from .frq file if available
            freq = frequencies.get(snp_id)

            yield BimVariant(
                chr=chr_val,
                id=snp_id,
                genetic_dist=genetic_dist,
                pos=pos,
                allele1=allele1,
                allele2=allele2,
                freq=freq,
            )


def count_bim_variants(filepath: Path) -> int:
    """Count the number of variants in a BIM file.

    Useful for progress reporting without loading full file.
    Supports gzipped files.

    Args:
        filepath: Path to PLINK .bim file (may be gzipped)

    Returns:
        Number of lines (variants) in the file
    """
    return count_lines(filepath)
