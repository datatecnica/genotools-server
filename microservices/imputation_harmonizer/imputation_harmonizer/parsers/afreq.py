"""PLINK2 .afreq file parser.

[claude-assisted] Implements AFREQ file parsing for PLINK2 format.
Supports gzipped files.

AFREQ format (tab-separated, with header starting with #CHROM):
#CHROM  ID      REF     ALT     ALT_FREQS       OBS_CT
1       rs123   A       G       0.30            1000

Key differences from FRQ:
- Header starts with #CHROM
- ALT_FREQS is ALT allele frequency (not MAF)
- OBS_CT is observation count (like NCHROBS)
"""

from pathlib import Path

from imputation_harmonizer.io_utils import smart_open


def parse_afreq(filepath: Path) -> dict[str, float]:
    """Parse PLINK2 .afreq file into frequency dictionary.

    Args:
        filepath: Path to .afreq file (may be gzipped)

    Returns:
        Dictionary mapping variant ID to ALT allele frequency

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns not found in header

    Example:
        >>> freqs = parse_afreq(Path("data.afreq"))
        >>> freqs["rs123"]
        0.30
    """
    if not filepath.exists():
        raise FileNotFoundError(f"AFREQ file not found: {filepath}")

    frequencies: dict[str, float] = {}

    with smart_open(filepath) as f:
        # Find and parse header
        id_col: int | None = None
        freq_col: int | None = None

        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#CHROM") or line.startswith("#"):
                # Parse header
                header = line.lstrip("#").split("\t")
                try:
                    id_col = header.index("ID")
                except ValueError:
                    # Try SNP (alternate column name)
                    try:
                        id_col = header.index("SNP")
                    except ValueError:
                        raise ValueError("ID or SNP column not found in AFREQ header")

                # Try to find frequency column
                for freq_name in ["ALT_FREQS", "ALT_FREQ", "AF", "FREQ"]:
                    if freq_name in header:
                        freq_col = header.index(freq_name)
                        break

                if freq_col is None:
                    raise ValueError(
                        "Frequency column not found in AFREQ header. "
                        "Expected: ALT_FREQS, ALT_FREQ, AF, or FREQ"
                    )
                continue

            # If no header found, use default positions
            if id_col is None:
                # Default PLINK2 afreq format: #CHROM ID REF ALT ALT_FREQS OBS_CT
                id_col = 1
                freq_col = 4

            # Parse data line
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) <= max(id_col, freq_col):
                continue  # Skip malformed lines

            snp_id = parts[id_col]
            try:
                freq = float(parts[freq_col])
                frequencies[snp_id] = freq
            except ValueError:
                # Skip lines where frequency is not a valid float (e.g., "NA", ".")
                continue

    return frequencies


def parse_afreq_with_alleles(filepath: Path) -> dict[str, tuple[str, str, float]]:
    """Parse .afreq file including allele information.

    Args:
        filepath: Path to .afreq file (may be gzipped)

    Returns:
        Dict mapping ID to (REF, ALT, ALT_FREQ) tuples

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns not found in header
    """
    if not filepath.exists():
        raise FileNotFoundError(f"AFREQ file not found: {filepath}")

    frequencies: dict[str, tuple[str, str, float]] = {}

    with smart_open(filepath) as f:
        # Find and parse header
        id_col: int | None = None
        ref_col: int | None = None
        alt_col: int | None = None
        freq_col: int | None = None

        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#CHROM") or line.startswith("#"):
                # Parse header
                header = line.lstrip("#").split("\t")

                try:
                    id_col = header.index("ID")
                except ValueError:
                    try:
                        id_col = header.index("SNP")
                    except ValueError:
                        raise ValueError("ID or SNP column not found in AFREQ header")

                try:
                    ref_col = header.index("REF")
                    alt_col = header.index("ALT")
                except ValueError:
                    raise ValueError("REF or ALT column not found in AFREQ header")

                for freq_name in ["ALT_FREQS", "ALT_FREQ", "AF", "FREQ"]:
                    if freq_name in header:
                        freq_col = header.index(freq_name)
                        break

                if freq_col is None:
                    raise ValueError("Frequency column not found in AFREQ header")
                continue

            # If no header found, use default positions
            if id_col is None:
                id_col, ref_col, alt_col, freq_col = 1, 2, 3, 4

            # Parse data line
            if not line:
                continue

            parts = line.split("\t")
            max_col = max(id_col, ref_col, alt_col, freq_col)
            if len(parts) <= max_col:
                continue

            snp_id = parts[id_col]
            ref = parts[ref_col].upper()
            alt = parts[alt_col].upper()

            try:
                freq = float(parts[freq_col])
                frequencies[snp_id] = (ref, alt, freq)
            except ValueError:
                continue

    return frequencies
