"""Utility functions for the imputation harmonizer.

[claude-assisted] Implements complement function and helpers matching
the original Perl script's strand flip logic (tr/ACGT/TGCA/).
"""

# Complement lookup table for DNA bases
# Matches Perl: $allele =~ tr/ACGT/TGCA/
COMPLEMENT: dict[str, str] = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",  # Unknown base stays as N
}


def complement(allele: str) -> str:
    """Get the complement of a single DNA base.

    Args:
        allele: Single DNA base (A, T, C, G, or N)

    Returns:
        Complementary base (A<->T, C<->G, N->N)

    Example:
        >>> complement("A")
        "T"
        >>> complement("G")
        "C"
    """
    return COMPLEMENT.get(allele, allele)


def complement_pair(a1: str, a2: str) -> tuple[str, str]:
    """Get complements of an allele pair.

    Args:
        a1: First allele
        a2: Second allele

    Returns:
        Tuple of complemented alleles

    Example:
        >>> complement_pair("A", "C")
        ("T", "G")
    """
    return complement(a1), complement(a2)


def is_palindromic(a1: str, a2: str) -> bool:
    """Check if a SNP is palindromic (A/T or G/C).

    Palindromic SNPs have alleles that are complements of each other,
    making strand determination ambiguous when MAF is close to 0.5.

    Args:
        a1: First allele
        a2: Second allele

    Returns:
        True if the SNP is palindromic (A/T, T/A, G/C, or C/G)

    Example:
        >>> is_palindromic("A", "T")
        True
        >>> is_palindromic("A", "G")
        False
    """
    return (a1, a2) in {("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")}


def is_indel(a1: str, a2: str) -> bool:
    """Check if a variant is an indel.

    Indels are identified by:
    - Allele is "-" (deletion marker)
    - Allele is "I" (insertion) or "D" (deletion)
    - Allele length > 1 (multi-base indel)

    Args:
        a1: First allele
        a2: Second allele

    Returns:
        True if the variant is an indel

    Note:
        HRC r1 has no indels, so these are excluded.
        Illumina format uses -/A but HRC uses T/TA for indels,
        so they will always fail allele matching anyway.
    """
    indel_markers = {"-", "I", "D"}
    return (
        a1 in indel_markers
        or a2 in indel_markers
        or len(a1) > 1
        or len(a2) > 1
    )


def sort_alleles(a1: str, a2: str) -> tuple[str, str]:
    """Sort alleles alphabetically.

    Used for creating consistent keys for duplicate detection.

    Args:
        a1: First allele
        a2: Second allele

    Returns:
        Tuple of alleles in sorted order

    Example:
        >>> sort_alleles("G", "A")
        ("A", "G")
    """
    if a1 <= a2:
        return a1, a2
    return a2, a1


def make_chrpos_key(chr_val: str, pos: int) -> str:
    """Create a chromosome:position key for lookups.

    Args:
        chr_val: Chromosome value
        pos: Base pair position

    Returns:
        Key in format "chr-pos"

    Example:
        >>> make_chrpos_key("1", 10000)
        "1-10000"
    """
    return f"{chr_val}-{pos}"


def make_allele_key(chr_val: str, pos: int, a1: str, a2: str) -> str:
    """Create a chromosome:position:alleles key for duplicate detection.

    Alleles are sorted for consistent key generation regardless of order.

    Args:
        chr_val: Chromosome value
        pos: Base pair position
        a1: First allele
        a2: Second allele

    Returns:
        Key in format "chr-pos-a1:a2" with sorted alleles

    Example:
        >>> make_allele_key("1", 10000, "G", "A")
        "1-10000-A:G"
    """
    sorted_a1, sorted_a2 = sort_alleles(a1, a2)
    return f"{chr_val}-{pos}-{sorted_a1}:{sorted_a2}"


def normalize_chromosome(chr_val: str) -> str:
    """Normalize chromosome value to consistent format.

    Handles variations like "chr1" -> "1", "01" -> "1".

    Args:
        chr_val: Chromosome value (may include "chr" prefix)

    Returns:
        Normalized chromosome value

    Example:
        >>> normalize_chromosome("chr1")
        "1"
        >>> normalize_chromosome("01")
        "1"
        >>> normalize_chromosome("X")
        "X"
    """
    # Remove 'chr' prefix if present
    if chr_val.lower().startswith("chr"):
        chr_val = chr_val[3:]

    # Remove leading zeros for numeric chromosomes
    if chr_val.isdigit():
        chr_val = str(int(chr_val))

    return chr_val


def extract_rsid(compound_id: str) -> str | None:
    """Extract rsID from a compound variant ID.

    1000G uses compound IDs like "rs123:10177:A:C".
    This extracts just the rsID portion.

    Args:
        compound_id: Variant ID (may be compound or simple)

    Returns:
        rsID if found, None otherwise

    Example:
        >>> extract_rsid("rs123:10177:A:C")
        "rs123"
        >>> extract_rsid("rs456")
        "rs456"
        >>> extract_rsid("10:12345:A:G")
        None
    """
    if not compound_id.startswith("rs"):
        return None

    # Split on colon and return first part
    parts = compound_id.split(":")
    return parts[0]
