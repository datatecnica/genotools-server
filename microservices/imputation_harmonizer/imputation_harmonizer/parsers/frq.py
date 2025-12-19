"""PLINK FRQ file parser.

[claude-assisted] Implements FRQ file parsing matching the original Perl
script's frequency loading (lines 198-205). Supports gzipped files.

FRQ file format (whitespace-separated, with header):
 CHR            SNP   A1   A2          MAF  NCHROBS
   1      rs367896724    A    C       0.0328     1000
"""

from pathlib import Path

from imputation_harmonizer.io_utils import smart_open


def parse_frq(filepath: Path) -> dict[str, float]:
    """Parse a PLINK FRQ file into a dictionary.

    The .frq file is typically small enough to load entirely into memory
    (unlike the .bim file which can be very large).

    Args:
        filepath: Path to PLINK .frq frequency file

    Returns:
        Dictionary mapping SNP IDs to allele frequencies

    Raises:
        FileNotFoundError: If file doesn't exist

    Example:
        >>> freqs = parse_frq(Path("data.frq"))
        >>> freqs["rs123"]
        0.328
    """
    if not filepath.exists():
        raise FileNotFoundError(f"FRQ file not found: {filepath}")

    frequencies: dict[str, float] = {}

    with smart_open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            # Strip leading/trailing whitespace
            line = line.strip()
            if not line:
                continue

            # Split on whitespace
            parts = line.split()

            # Skip header line (starts with CHR or contains non-numeric first column)
            # Perl: implicitly skips header by checking array indices
            if line_num == 1 and parts[0].upper() == "CHR":
                continue

            # Skip if first column is not a valid chromosome
            try:
                # Try to see if it's a header by checking if MAF column is numeric
                if len(parts) >= 5:
                    float(parts[4])
            except ValueError:
                # This is likely a header line
                continue

            if len(parts) < 5:
                continue  # Skip malformed lines

            # Perl: $af{$temp[1]} = $temp[4];
            # Column 1 = SNP ID, Column 4 = MAF
            snp_id = parts[1]
            try:
                maf = float(parts[4])
                frequencies[snp_id] = maf
            except ValueError:
                # Skip lines where MAF is not a valid float (e.g., "NA")
                continue

    return frequencies


def parse_frq_with_alleles(filepath: Path) -> dict[str, tuple[str, str, float]]:
    """Parse a PLINK FRQ file including allele information.

    Args:
        filepath: Path to PLINK .frq frequency file

    Returns:
        Dictionary mapping SNP IDs to (A1, A2, MAF) tuples

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not filepath.exists():
        raise FileNotFoundError(f"FRQ file not found: {filepath}")

    frequencies: dict[str, tuple[str, str, float]] = {}

    with smart_open(filepath) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Skip header
            if line_num == 1 and parts[0].upper() == "CHR":
                continue

            if len(parts) < 5:
                continue

            # CHR SNP A1 A2 MAF NCHROBS
            snp_id = parts[1]
            a1 = parts[2].upper()
            a2 = parts[3].upper()

            try:
                maf = float(parts[4])
                frequencies[snp_id] = (a1, a2, maf)
            except ValueError:
                continue

    return frequencies
