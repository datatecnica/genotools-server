"""PLINK2 .pvar file parser.

[claude-assisted] Implements streaming PVAR file parsing for PLINK2 format.
Supports gzipped files.

PVAR format (tab-separated, with header starting with #CHROM):
#CHROM  POS     ID      REF     ALT     [INFO]  [optional columns...]
1       10177   rs367896724     A       AC      .

Key differences from BIM:
- Has header line starting with #CHROM
- Column order: CHROM, POS, ID, REF, ALT (vs BIM: CHROM, ID, cM, POS, A1, A2)
- REF/ALT instead of A1/A2 (explicit strand)
- May have INFO column and other optional columns
"""

from collections.abc import Iterator
from pathlib import Path

from imputation_harmonizer.io_utils import smart_open
from imputation_harmonizer.models import BimVariant
from imputation_harmonizer.utils import normalize_chromosome


def parse_pvar(
    filepath: Path,
    frequencies: dict[str, float] | None = None,
) -> Iterator[BimVariant]:
    """Stream variants from a PLINK2 .pvar file.

    Yields BimVariant objects for compatibility with existing checking code.
    Maps REF->allele1, ALT->allele2.

    Args:
        filepath: Path to .pvar file (may be gzipped)
        frequencies: Optional frequency dict from .afreq file

    Yields:
        BimVariant for each data line

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns not found in header

    Note:
        genetic_dist is always 0.0 for .pvar (no cM column in PLINK2 format)

    Example:
        >>> for variant in parse_pvar(Path("data.pvar")):
        ...     print(variant.id, variant.pos, variant.allele1, variant.allele2)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"PVAR file not found: {filepath}")

    frequencies = frequencies or {}

    with smart_open(filepath) as f:
        # Find and parse header
        header_line: str | None = None
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#CHROM"):
                header_line = line
                break
            elif line.startswith("#"):
                # Skip other comment lines
                continue
            else:
                # No header found, file starts with data
                # Assume standard column order
                break

        # Parse header to find column indices
        if header_line:
            header = header_line.lstrip("#").split("\t")
            try:
                chrom_col = header.index("CHROM")
                pos_col = header.index("POS")
                id_col = header.index("ID")
                ref_col = header.index("REF")
                alt_col = header.index("ALT")
            except ValueError as e:
                raise ValueError(f"Required column not found in PVAR header: {e}")
        else:
            # Default column order per PLINK2 spec
            chrom_col, pos_col, id_col, ref_col, alt_col = 0, 1, 2, 3, 4

        # Process data lines
        line_num = 0
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            line_num += 1
            parts = line.split("\t")

            # Validate minimum columns
            max_col = max(chrom_col, pos_col, id_col, ref_col, alt_col)
            if len(parts) <= max_col:
                raise ValueError(
                    f"Invalid PVAR format at line {line_num}: "
                    f"expected at least {max_col + 1} columns, got {len(parts)}"
                )

            chr_val = normalize_chromosome(parts[chrom_col])
            pos = int(parts[pos_col])
            snp_id = parts[id_col]
            ref = parts[ref_col].upper()
            alt = parts[alt_col].upper()

            # Look up frequency from .afreq file if available
            freq = frequencies.get(snp_id)

            # Yield BimVariant with REF as allele1, ALT as allele2
            # genetic_dist is 0.0 since PVAR doesn't have cM column
            yield BimVariant(
                chr=chr_val,
                id=snp_id,
                genetic_dist=0.0,
                pos=pos,
                allele1=ref,
                allele2=alt,
                freq=freq,
            )


def count_pvar_variants(filepath: Path) -> int:
    """Count variants in a .pvar file (skipping header).

    Args:
        filepath: Path to .pvar file (may be gzipped)

    Returns:
        Number of variants (data lines, excluding header/comments)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"PVAR file not found: {filepath}")

    count = 0
    with smart_open(filepath) as f:
        for line in f:
            # Skip header and comment lines
            if not line.startswith("#"):
                count += 1
    return count
