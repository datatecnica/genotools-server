"""
Tests for app/models/harmonization.py

Covers HarmonizationAction enum properties, HarmonizationRecord Pydantic
validators and properties, and HarmonizationStats computed fields.
No mocking required — pure model tests.
"""

import pytest
from pydantic import ValidationError

from app.models.harmonization import (
    HarmonizationAction,
    HarmonizationRecord,
    HarmonizationStats,
)


# ---------------------------------------------------------------------------
# HarmonizationAction
# ---------------------------------------------------------------------------

class TestHarmonizationAction:

    def test_requires_genotype_transform_true_for_swap(self):
        assert HarmonizationAction.SWAP.requires_genotype_transform is True

    def test_requires_genotype_transform_true_for_flip_swap(self):
        assert HarmonizationAction.FLIP_SWAP.requires_genotype_transform is True

    @pytest.mark.parametrize("action", [
        HarmonizationAction.EXACT,
        HarmonizationAction.FLIP,
    ])
    def test_requires_genotype_transform_false_for_non_swap(self, action):
        assert action.requires_genotype_transform is False

    def test_is_valid_false_for_invalid(self):
        assert HarmonizationAction.INVALID.is_valid is False

    def test_is_valid_false_for_ambiguous(self):
        assert HarmonizationAction.AMBIGUOUS.is_valid is False

    @pytest.mark.parametrize("action", [
        HarmonizationAction.EXACT,
        HarmonizationAction.SWAP,
        HarmonizationAction.FLIP,
        HarmonizationAction.FLIP_SWAP,
    ])
    def test_is_valid_true_for_valid_actions(self, action):
        assert action.is_valid is True


# ---------------------------------------------------------------------------
# HarmonizationRecord
# ---------------------------------------------------------------------------

def _make_record(**overrides):
    """Factory for HarmonizationRecord with sensible defaults."""
    defaults = dict(
        snp_list_id="chr1:100000:A:C",
        pgen_variant_id="1:100000:A:C",
        chromosome="1",
        position=100000,
        snp_list_a1="A",
        snp_list_a2="C",
        pgen_a1="A",
        pgen_a2="C",
        harmonization_action=HarmonizationAction.EXACT,
        genotype_transform=None,
        file_path="/fake/file.pgen",
        data_type="NBA",
        ancestry="EUR",
    )
    defaults.update(overrides)
    return HarmonizationRecord(**defaults)


class TestHarmonizationRecord:

    def test_normalize_chromosome_removes_chr_prefix(self):
        record = _make_record(chromosome="chr1")
        assert record.chromosome == "1"

    def test_normalize_chromosome_uppercases_x(self):
        record = _make_record(chromosome="chrx")
        assert record.chromosome == "X"

    def test_normalize_chromosome_mt(self):
        record = _make_record(chromosome="chrmt")
        assert record.chromosome == "MT"

    def test_normalize_allele_uppercases_and_strips(self):
        record = _make_record(snp_list_a1=" a ")
        assert record.snp_list_a1 == "A"

    def test_normalize_allele_uppercase_only(self):
        record = _make_record(snp_list_a2="c")
        assert record.snp_list_a2 == "C"

    def test_variant_key_format(self):
        record = _make_record(chromosome="1", position=100000,
                               snp_list_a1="A", snp_list_a2="C")
        assert record.variant_key == "1:100000:A:C"

    def test_requires_transformation_true_for_swap(self, sample_harmonization_records):
        # sample_harmonization_records[2] is SWAP
        swap_record = sample_harmonization_records[2]
        assert swap_record.requires_transformation is True

    def test_requires_transformation_false_for_exact(self, sample_harmonization_records):
        exact_record = sample_harmonization_records[0]
        assert exact_record.requires_transformation is False

    def test_is_strand_ambiguous_at_pair(self):
        record = _make_record(snp_list_a1="A", snp_list_a2="T")
        assert record.is_strand_ambiguous is True

    def test_is_strand_ambiguous_cg_pair(self):
        record = _make_record(snp_list_a1="C", snp_list_a2="G")
        assert record.is_strand_ambiguous is True

    def test_is_not_strand_ambiguous_ac_pair(self):
        record = _make_record(snp_list_a1="A", snp_list_a2="C")
        assert record.is_strand_ambiguous is False

    def test_missing_required_field_raises_validation_error(self):
        with pytest.raises(ValidationError):
            HarmonizationRecord(
                # snp_list_id deliberately omitted
                pgen_variant_id="1:100000:A:C",
                chromosome="1",
                position=100000,
                snp_list_a1="A",
                snp_list_a2="C",
                pgen_a1="A",
                pgen_a2="C",
                harmonization_action=HarmonizationAction.EXACT,
                file_path="/fake/file.pgen",
                data_type="NBA",
            )


# ---------------------------------------------------------------------------
# HarmonizationStats
# ---------------------------------------------------------------------------

def _make_invalid_record():
    return _make_record(harmonization_action=HarmonizationAction.INVALID,
                        snp_list_id="chr9:999999:A:G", pgen_variant_id="9:999999:A:G")


class TestHarmonizationStats:

    def test_update_from_records_counts_correctly(self, sample_harmonization_records):
        # sample_harmonization_records = [EXACT, FLIP, SWAP]
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(sample_harmonization_records)

        assert stats.exact_matches == 1
        assert stats.flipped_strand == 1
        assert stats.swapped_alleles == 1
        assert stats.flip_and_swap == 0
        assert stats.invalid_variants == 0

    def test_harmonized_variants_sum(self, sample_harmonization_records):
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(sample_harmonization_records)
        assert stats.harmonized_variants == stats.exact_matches + stats.flipped_strand + stats.swapped_alleles

    def test_harmonization_rate_correct(self, sample_harmonization_records):
        # 3 records; to get rate = 0.75, add one invalid record
        all_records = sample_harmonization_records + [_make_invalid_record()]
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(all_records)

        assert stats.total_variants == 4
        assert abs(stats.harmonization_rate - 0.75) < 1e-9

    def test_harmonization_rate_zero_total(self):
        stats = HarmonizationStats(total_variants=0)
        assert stats.harmonization_rate == 0.0  # no ZeroDivisionError

    def test_failure_rate_nonzero_with_invalid(self, sample_harmonization_records):
        all_records = sample_harmonization_records + [_make_invalid_record()]
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(all_records)
        assert stats.failure_rate > 0

    def test_summary_dict_has_expected_keys(self, sample_harmonization_records):
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(sample_harmonization_records)
        expected_keys = {
            "total_variants", "harmonized", "harmonization_rate",
            "exact_matches", "swapped_alleles", "flipped_strand",
            "flip_and_swap", "invalid", "ambiguous", "failure_rate"
        }
        assert expected_keys.issubset(stats.summary_dict.keys())

    def test_update_from_records_sets_total_variants(self, sample_harmonization_records):
        stats = HarmonizationStats(total_variants=0)
        stats.update_from_records(sample_harmonization_records)
        assert stats.total_variants == len(sample_harmonization_records)
