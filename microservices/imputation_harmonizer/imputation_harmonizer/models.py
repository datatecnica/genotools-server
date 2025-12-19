"""Data models for the imputation harmonizer.

[claude-assisted] Implements data structures matching the original Perl script's
internal representations for variants, check results, and statistics.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal


class ExcludeReason(Enum):
    """Reasons for excluding a variant from the final dataset."""

    NOT_IN_REFERENCE = auto()
    INDEL = auto()
    PALINDROMIC_HIGH_MAF = auto()
    FREQ_DIFF_TOO_HIGH = auto()
    ALLELE_MISMATCH = auto()
    DUPLICATE = auto()
    ALT_CHROMOSOME = auto()  # X, Y, XY, MT for HRC; MT only for 1000G


class StrandAction(Enum):
    """Action needed for strand orientation."""

    NONE = auto()
    FLIP = auto()


class AlleleAction(Enum):
    """Action needed for allele assignment."""

    NONE = auto()
    FORCE_REF = auto()


@dataclass(slots=True)
class ReferenceVariant:
    """Variant from HRC or 1000G reference panel.

    Attributes:
        chr: Chromosome (string to handle "X", "Y", etc.)
        pos: Base pair position
        id: Variant identifier (rsID or compound ID)
        ref: Reference allele
        alt: Alternate allele
        alt_af: Alternate allele frequency in reference panel
    """

    chr: str
    pos: int
    id: str
    ref: str
    alt: str
    alt_af: float


@dataclass(slots=True)
class BimVariant:
    """Variant from PLINK .bim file.

    Attributes:
        chr: Chromosome
        id: Variant identifier (rsID)
        genetic_dist: Genetic distance (usually 0)
        pos: Base pair position
        allele1: First allele (usually minor allele in PLINK)
        allele2: Second allele (usually major allele in PLINK)
        freq: Allele frequency from .frq file (populated after parsing)
    """

    chr: str
    id: str
    genetic_dist: float
    pos: int
    allele1: str
    allele2: str
    freq: float | None = None


@dataclass
class CheckResult:
    """Result of checking a single variant against reference.

    Attributes:
        snp_id: Original variant ID from BIM file
        matched_by: How the variant was matched ("position", "id", or "none")
        ref_variant: Matched reference variant (if found)
        exclude: Whether to exclude this variant
        exclude_reason: Reason for exclusion (if excluded)
        strand_action: Strand flip action needed
        allele_action: Allele force action needed
        force_ref_allele: Reference allele to force (if needed)
        update_position: New position (if position update needed)
        update_chromosome: New chromosome (if chromosome update needed)
        update_id: New ID (if ID update needed)
        ref_freq: Reference frequency for plotting
        bim_freq: BIM file frequency for plotting
        freq_diff: Frequency difference (ref - bim)
        check_code: Check outcome code (1-6, matching original Perl script)
    """

    snp_id: str
    matched_by: Literal["position", "id", "none"]
    ref_variant: ReferenceVariant | None = None

    # Exclusion
    exclude: bool = False
    exclude_reason: ExcludeReason | None = None

    # Actions
    strand_action: StrandAction = StrandAction.NONE
    allele_action: AlleleAction = AlleleAction.NONE
    force_ref_allele: str | None = None

    # Position/ID updates
    update_position: int | None = None
    update_chromosome: str | None = None
    update_id: str | None = None

    # Frequency data for plotting
    ref_freq: float | None = None
    bim_freq: float | None = None
    freq_diff: float | None = None
    check_code: int | None = None


@dataclass
class Statistics:
    """Running statistics for the variant check process.

    Tracks counts matching the original Perl script's output statistics.
    """

    # Total counts
    total: int = 0
    indels: int = 0
    alt_chr_skipped: int = 0

    # Position/ID matching
    position_match_id_match: int = 0
    position_match_id_mismatch: int = 0
    id_match_position_mismatch: int = 0
    no_match: int = 0

    # Strand outcomes
    strand_ok: int = 0
    strand_flip: int = 0

    # Ref/alt outcomes
    ref_alt_ok: int = 0
    ref_alt_swap: int = 0

    # Exclusions
    palindromic_excluded: int = 0
    freq_diff_excluded: int = 0
    allele_mismatch: int = 0
    duplicates: int = 0

    # ID mismatches where HRC ID is '.'
    id_allele_mismatch: int = 0
    hrc_dot: int = 0

    @property
    def position_matches(self) -> int:
        """Total position matches (ID match + ID mismatch)."""
        return self.position_match_id_match + self.position_match_id_mismatch

    @property
    def total_checked(self) -> int:
        """Total variants that were checked (excluding alt chr)."""
        return (
            self.position_match_id_match
            + self.position_match_id_mismatch
            + self.id_match_position_mismatch
            + self.no_match
        )

    @property
    def total_strand_checked(self) -> int:
        """Total variants checked for strand."""
        return self.strand_ok + self.strand_flip


@dataclass
class StrandCheckResult:
    """Result from strand and allele checking.

    Used internally by check_strand_and_alleles function.
    """

    exclude: bool = False
    exclude_reason: ExcludeReason | None = None
    strand_action: StrandAction = StrandAction.NONE
    allele_action: AlleleAction = AlleleAction.NONE
    force_ref_allele: str | None = None
    ref_freq: float | None = None
    freq_diff: float | None = None
    check_code: int = 0
