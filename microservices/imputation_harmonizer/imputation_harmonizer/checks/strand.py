"""Strand and allele checking logic.

[claude-assisted] Implements strand checking matching the original Perl script's
check_strand() function (lines 595-723).

The check has 6 possible outcomes:
1. Strand OK, ref/alt OK - no action needed
2. Strand OK, ref/alt swapped - force reference allele
3. Strand flipped, ref/alt OK - flip strand
4. Strand flipped, ref/alt swapped - flip strand + force reference allele
5. Palindromic (A/T, G/C) with MAF > 0.4 - exclude (cannot resolve strand)
6. Frequency difference > 0.2 - exclude (possible allele mismatch)
0. No allele match - exclude (allele mismatch)
"""

from imputation_harmonizer.models import (
    AlleleAction,
    ExcludeReason,
    StrandAction,
    StrandCheckResult,
)
from imputation_harmonizer.utils import complement_pair, is_palindromic


def check_strand_and_alleles(
    ref_alleles: tuple[str, str],
    bim_alleles: tuple[str, str],
    ref_alt_af: float,
    bim_af: float,
    palindrome_maf_threshold: float = 0.4,
    freq_diff_threshold: float = 0.2,
) -> StrandCheckResult:
    """Check strand orientation and allele assignment against reference.

    Compares alleles from BIM file against reference panel, detecting
    strand flips and ref/alt swaps. Excludes palindromic SNPs with
    MAF > threshold and variants with frequency difference > threshold.

    Args:
        ref_alleles: (ref, alt) tuple from reference panel
        bim_alleles: (a1, a2) tuple from BIM file
        ref_alt_af: Alternate allele frequency in reference panel
        bim_af: Allele frequency from dataset .frq file
        palindrome_maf_threshold: MAF threshold for excluding palindromic SNPs (default 0.4)
        freq_diff_threshold: Max allowed frequency difference (default 0.2)

    Returns:
        StrandCheckResult with action needed and metadata

    Note:
        Frequency comparison logic:
        - Cases 1 & 3 (ref/alt OK): compare ref_freq (1-alt_af) with bim_af
        - Cases 2 & 4 (ref/alt swapped): compare alt_af with bim_af
    """
    ref_a, alt_a = ref_alleles
    bim_a1, bim_a2 = bim_alleles

    # Calculate MAF for palindromic check
    # Perl: if ($altaf > 0.5) { $maf = 1 - $altaf; } else { $maf = $altaf; }
    maf = min(ref_alt_af, 1 - ref_alt_af)

    # Check palindromic SNPs first (absolute exclusion if MAF > threshold)
    # Perl: if ($maf > 0.4 and ($a1 eq 'A:T' or $a1 eq 'T:A' or $a1 eq 'G:C' or $a1 eq 'C:G'))
    if is_palindromic(ref_a, alt_a) and maf > palindrome_maf_threshold:
        return StrandCheckResult(
            exclude=True,
            exclude_reason=ExcludeReason.PALINDROMIC_HIGH_MAF,
            check_code=5,
        )

    # Get complemented BIM alleles for strand flip detection
    # Perl: $a2 =~ tr/ACGTN/TGCAN/;
    bim_c1, bim_c2 = complement_pair(bim_a1, bim_a2)

    # Initialize result
    result = StrandCheckResult()

    # Case 1: Strand OK, ref/alt OK
    # Perl: if ($alleles1[0] eq $alleles2[0] and $alleles1[1] eq $alleles2[1])
    if ref_a == bim_a1 and alt_a == bim_a2:
        # Reference frequency = 1 - alt_af
        ref_freq = 1 - ref_alt_af
        diff = ref_freq - bim_af

        result.strand_action = StrandAction.NONE
        result.allele_action = AlleleAction.NONE
        result.ref_freq = ref_freq
        result.freq_diff = diff
        result.check_code = 1

    # Case 2: Strand OK, ref/alt swapped
    # Perl: elsif ($alleles1[0] eq $alleles2[1] and $alleles1[1] eq $alleles2[0])
    elif ref_a == bim_a2 and alt_a == bim_a1:
        # Alleles are swapped, compare alt_af with bim_af
        diff = ref_alt_af - bim_af

        result.strand_action = StrandAction.NONE
        result.allele_action = AlleleAction.FORCE_REF
        result.force_ref_allele = ref_a
        result.ref_freq = ref_alt_af
        result.freq_diff = diff
        result.check_code = 2

    # Case 3: Strand flipped, ref/alt OK
    # Perl: elsif ($alleles1[0] eq $allelesflip[0] and $alleles1[1] eq $allelesflip[1])
    elif ref_a == bim_c1 and alt_a == bim_c2:
        # Strand is flipped, ref/alt assignment OK
        ref_freq = 1 - ref_alt_af
        diff = ref_freq - bim_af

        result.strand_action = StrandAction.FLIP
        result.allele_action = AlleleAction.NONE
        result.ref_freq = ref_freq
        result.freq_diff = diff
        result.check_code = 3

    # Case 4: Strand flipped, ref/alt swapped
    # Perl: elsif ($alleles1[0] eq $allelesflip[1] and $alleles1[1] eq $allelesflip[0])
    elif ref_a == bim_c2 and alt_a == bim_c1:
        # Strand is flipped and alleles are swapped
        diff = ref_alt_af - bim_af

        result.strand_action = StrandAction.FLIP
        result.allele_action = AlleleAction.FORCE_REF
        result.force_ref_allele = ref_a
        result.ref_freq = ref_alt_af
        result.freq_diff = diff
        result.check_code = 4

    # Case 0: No match - allele mismatch
    else:
        return StrandCheckResult(
            exclude=True,
            exclude_reason=ExcludeReason.ALLELE_MISMATCH,
            check_code=0,
        )

    # Check frequency difference (applies to all non-excluded cases)
    # Perl: if ($diff > 0.2) { print E "$id\n"; $check = 6; $allelediff++; }
    if result.freq_diff is not None:
        freq_diff_abs = abs(result.freq_diff)
        if freq_diff_abs > freq_diff_threshold:
            return StrandCheckResult(
                exclude=True,
                exclude_reason=ExcludeReason.FREQ_DIFF_TOO_HIGH,
                ref_freq=result.ref_freq,
                freq_diff=result.freq_diff,
                check_code=6,
            )

    return result
