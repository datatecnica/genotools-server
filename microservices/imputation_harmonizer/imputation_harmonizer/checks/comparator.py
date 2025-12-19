"""Main variant comparison logic.

[claude-assisted] Implements the main comparison loop matching the original
Perl script's variant processing (lines 298-496).

Processing flow:
1. Skip non-standard chromosomes (X, Y, XY, MT for HRC)
2. Detect and exclude indels
3. Try position match, then ID match
4. Handle position/chromosome updates
5. Check for duplicates
6. Check strand and alleles
"""

from collections.abc import Iterator

from imputation_harmonizer.checks.strand import check_strand_and_alleles
from imputation_harmonizer.config import Config
from imputation_harmonizer.models import (
    AlleleAction,
    BimVariant,
    CheckResult,
    ExcludeReason,
    Statistics,
    StrandAction,
)
from imputation_harmonizer.reference.base import ReferencePanel
from imputation_harmonizer.utils import (
    complement_pair,
    is_indel,
    make_allele_key,
    make_chrpos_key,
    sort_alleles,
)


def check_variants(
    bim_variants: Iterator[BimVariant],
    reference: ReferencePanel,
    config: Config,
    stats: Statistics | None = None,
) -> Iterator[CheckResult]:
    """Compare BIM variants against reference panel.

    Main comparison loop that yields CheckResult for each variant.

    Args:
        bim_variants: Iterator of BimVariant from BIM file
        reference: Loaded reference panel (HRC or 1000G)
        config: Configuration with thresholds and settings
        stats: Optional Statistics object to update (mutated in place)

    Yields:
        CheckResult for each variant processed

    Note:
        The stats object is mutated in place as variants are processed.
        This allows real-time tracking of processing statistics.
    """
    if stats is None:
        stats = Statistics()

    # Track seen variants for duplicate detection
    # Key format: "chr-pos-sorted_a1:sorted_a2"
    seen: set[str] = set()

    # Build set of valid chromosomes
    valid_chromosomes = config.chromosomes.copy()
    if config.include_x and config.panel == "1000g":
        valid_chromosomes.add("23")
        valid_chromosomes.add("X")

    for variant in bim_variants:
        stats.total += 1

        # Create chr:pos key for this variant
        chrpos = make_chrpos_key(variant.chr, variant.pos)

        # Skip non-standard chromosomes
        # Perl: if ($temp[0] <= 22 or ($kgflag and $temp[0] == 23))
        if variant.chr not in valid_chromosomes:
            stats.alt_chr_skipped += 1
            yield CheckResult(
                snp_id=variant.id,
                matched_by="none",
                exclude=True,
                exclude_reason=ExcludeReason.ALT_CHROMOSOME,
            )
            continue

        # Check for indels
        # Perl: if ($temp[4] eq '-' or $temp[5] eq '-' or ...)
        if is_indel(variant.allele1, variant.allele2):
            stats.indels += 1
            yield CheckResult(
                snp_id=variant.id,
                matched_by="none",
                exclude=True,
                exclude_reason=ExcludeReason.INDEL,
            )
            continue

        # Create allele keys for duplicate detection
        sorted_a1, sorted_a2 = sort_alleles(variant.allele1, variant.allele2)
        allele_key = make_allele_key(variant.chr, variant.pos, sorted_a1, sorted_a2)

        # Also create complement key for duplicate detection
        comp_a1, comp_a2 = complement_pair(variant.allele1, variant.allele2)
        sorted_comp_a1, sorted_comp_a2 = sort_alleles(comp_a1, comp_a2)
        comp_key = make_allele_key(variant.chr, variant.pos, sorted_comp_a1, sorted_comp_a2)

        # Try position match first
        ref_var = reference.get_by_position(variant.chr, variant.pos)
        matched_by: str = "position" if ref_var else "none"

        # Initialize result for this variant
        result = CheckResult(
            snp_id=variant.id,
            matched_by=matched_by,
            ref_variant=ref_var,
        )

        # If position match found
        if ref_var:
            # Check for duplicate at this position
            # Perl: if ($seen{$ChrPosAlleles})
            if allele_key in seen:
                stats.duplicates += 1
                result.exclude = True
                result.exclude_reason = ExcludeReason.DUPLICATE
                yield result
                continue

            # Mark as seen
            seen.add(allele_key)
            seen.add(comp_key)

            # Check ID match/mismatch
            if ref_var.id == variant.id:
                # Perl: $idmatch++
                stats.position_match_id_match += 1
            else:
                # Position matches but ID differs
                # Check if this is a mismapped SNP
                ref_chrpos = reference.get_chrpos_for_id(variant.id)

                if (
                    ref_chrpos
                    and ref_var.id != "."
                    and chrpos != ref_chrpos
                ):
                    # SNP exists elsewhere in reference - mismapped
                    # Update position to reference position
                    stats.id_match_position_mismatch += 1

                    # Parse chr-pos from reference lookup
                    ref_chr, ref_pos_str = ref_chrpos.split("-")
                    ref_pos = int(ref_pos_str)

                    # Create new allele keys for duplicate check at new position
                    new_allele_key = make_allele_key(ref_chr, ref_pos, sorted_a1, sorted_a2)
                    new_comp_key = make_allele_key(ref_chr, ref_pos, sorted_comp_a1, sorted_comp_a2)

                    # Check if updating position would create duplicate
                    if new_allele_key in seen or new_comp_key in seen:
                        stats.duplicates += 1
                        result.exclude = True
                        result.exclude_reason = ExcludeReason.DUPLICATE
                        yield result
                        continue

                    # Mark new position as seen
                    seen.add(new_allele_key)
                    seen.add(new_comp_key)

                    # Update result with position correction
                    result.update_position = ref_pos
                    if variant.chr != ref_chr:
                        result.update_chromosome = ref_chr

                    # Get reference variant at new position for strand check
                    ref_var = reference.get_by_position(ref_chr, ref_pos)
                    result.ref_variant = ref_var
                else:
                    # Just different name, same position
                    stats.position_match_id_mismatch += 1
                    result.update_id = ref_var.id

        # If no position match, try ID match
        elif reference.has_id(variant.id):
            matched_by = "id"
            result.matched_by = "id"

            # Get reference position for this ID
            ref_chrpos = reference.get_chrpos_for_id(variant.id)
            if ref_chrpos:
                ref_chr, ref_pos_str = ref_chrpos.split("-")
                ref_pos = int(ref_pos_str)

                # Create allele key at reference position for duplicate check
                ref_allele_key = make_allele_key(ref_chr, ref_pos, sorted_a1, sorted_a2)
                ref_comp_key = make_allele_key(ref_chr, ref_pos, sorted_comp_a1, sorted_comp_a2)

                if ref_allele_key in seen:
                    stats.duplicates += 1
                    result.exclude = True
                    result.exclude_reason = ExcludeReason.DUPLICATE
                    yield result
                    continue

                # Mark as seen at reference position
                seen.add(ref_allele_key)
                seen.add(ref_comp_key)

                # Update position/chromosome
                # Perl: print C "$temp[1]\t$ChrPosRef[0]\n"; print P "$temp[1]\t$ChrPosRef[1]\n";
                result.update_chromosome = ref_chr
                result.update_position = ref_pos
                stats.id_match_position_mismatch += 1

                # Get reference variant for strand check
                ref_var = reference.get_by_id(variant.id)
                result.ref_variant = ref_var

        # No match on position or ID
        else:
            stats.no_match += 1
            result.exclude = True
            result.exclude_reason = ExcludeReason.NOT_IN_REFERENCE
            yield result
            continue

        # Perform strand and allele check
        if ref_var is None:
            # Should not happen, but handle gracefully
            result.exclude = True
            result.exclude_reason = ExcludeReason.NOT_IN_REFERENCE
            yield result
            continue

        strand_result = check_strand_and_alleles(
            ref_alleles=(ref_var.ref, ref_var.alt),
            bim_alleles=(variant.allele1, variant.allele2),
            ref_alt_af=ref_var.alt_af,
            bim_af=variant.freq or 0.0,
            palindrome_maf_threshold=config.palindrome_maf_threshold,
            freq_diff_threshold=config.freq_diff_threshold,
        )

        # Update result with strand check results
        result.exclude = strand_result.exclude
        result.exclude_reason = strand_result.exclude_reason
        result.strand_action = strand_result.strand_action
        result.allele_action = strand_result.allele_action
        result.force_ref_allele = strand_result.force_ref_allele
        result.ref_freq = strand_result.ref_freq
        result.bim_freq = variant.freq
        result.freq_diff = strand_result.freq_diff
        result.check_code = strand_result.check_code

        # Update statistics
        if strand_result.exclude:
            reason = strand_result.exclude_reason
            if reason == ExcludeReason.PALINDROMIC_HIGH_MAF:
                stats.palindromic_excluded += 1
            elif reason == ExcludeReason.FREQ_DIFF_TOO_HIGH:
                stats.freq_diff_excluded += 1
            elif reason == ExcludeReason.ALLELE_MISMATCH:
                stats.allele_mismatch += 1
        else:
            # Strand statistics
            if strand_result.strand_action == StrandAction.NONE:
                stats.strand_ok += 1
            else:
                stats.strand_flip += 1

            # Ref/alt statistics
            if strand_result.allele_action == AlleleAction.NONE:
                stats.ref_alt_ok += 1
            else:
                stats.ref_alt_swap += 1

        yield result
