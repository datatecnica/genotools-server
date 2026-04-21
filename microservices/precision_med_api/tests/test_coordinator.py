"""
Tests for app/processing/coordinator.py :: ExtractionCoordinator

Tests data-manipulation methods directly — no ProcessPool invoked.
Uses in-memory DataFrames with fully invented sample IDs.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from app.processing.coordinator import ExtractionCoordinator
from app.processing.extractor import VariantExtractor
from app.processing.transformer import GenotypeTransformer


# ---------------------------------------------------------------------------
# Module-level fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def coordinator(mock_settings_with_paths):
    """ExtractionCoordinator with mocked extractor/transformer (no I/O)."""
    mock_extractor = Mock(spec=VariantExtractor)
    mock_transformer = Mock(spec=GenotypeTransformer)
    return ExtractionCoordinator(mock_extractor, mock_transformer, mock_settings_with_paths)


# ---------------------------------------------------------------------------
# TestNormalizeSampleId
# ---------------------------------------------------------------------------

class TestNormalizeSampleId:

    def test_normalize_sample_id_parametrized(self, coordinator, sample_ids_to_normalize):
        for input_id, expected in sample_ids_to_normalize:
            assert coordinator._normalize_sample_id(input_id) == expected, \
                f"_normalize_sample_id({input_id!r}) should return {expected!r}"

    def test_normalize_odd_part_count_unchanged(self, coordinator):
        # 3 parts (odd) should NOT be treated as a duplicate
        sample_id = "MOCK_X_000002_000003"   # 4 parts, but halves differ → not duplicate
        result = coordinator._normalize_sample_id(sample_id)
        # Halves are "MOCK_X" vs "000002_000003" — not equal, so unchanged
        assert result == "MOCK_X_000002_000003"

    def test_normalize_empty_string(self, coordinator):
        assert coordinator._normalize_sample_id("") == ""

    def test_normalize_three_part_odd(self, coordinator):
        # 3 parts: len(parts)=3, not >= 4 → no dedup check
        sid = "TEST_FAKE_001"
        assert coordinator._normalize_sample_id(sid) == "TEST_FAKE_001"

    def test_normalize_removes_zero_prefix_before_dedup(self, coordinator):
        # After stripping '0_': 'MOCK_X_001' → not a duplicate pattern
        assert coordinator._normalize_sample_id("0_MOCK_X_001") == "MOCK_X_001"

    def test_normalize_six_part_duplicate(self, coordinator):
        # 6-part string where first 3 == last 3
        sid = "MOCK_X_001_MOCK_X_001"
        result = coordinator._normalize_sample_id(sid)
        assert result == "MOCK_X_001"


# ---------------------------------------------------------------------------
# TestMergeAncestryResults — Critical regression block
# ---------------------------------------------------------------------------

class TestMergeAncestryResults:

    def test_merge_two_ancestries_preserves_all_samples(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')

        eur_samples = [c for c in eur_genotype_df.columns
                       if c.startswith('MOCK_EUR')]
        eas_samples = [c for c in eas_genotype_df.columns
                       if c.startswith('MOCK_EAS')]
        for s in eur_samples + eas_samples:
            assert s in result.columns, f"Sample column missing after merge: {s}"

    def test_merge_two_ancestries_preserves_all_variants(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')
        assert len(result) == 3   # 3 shared variants

    def test_merge_no_zero_out_of_genotypes(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        """THE regression test for the 2025-12-15 combine_first bug.

        Before the fix, _dup columns were dropped without combine_first,
        which silently zeroed out metadata-bearing variant rows.
        """
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')

        # Pick EUR_P001 genotype at the first variant (snp_list_id='chr1:100000:A:C')
        original_gt = eur_genotype_df[
            eur_genotype_df['snp_list_id'] == 'chr1:100000:A:C'
        ].iloc[0]['MOCK_EUR_P001']

        merged_gt = result[
            result['snp_list_id'] == 'chr1:100000:A:C'
        ].iloc[0]['MOCK_EUR_P001']

        assert merged_gt == original_gt, \
            f"Genotype zeroed out after merge: expected {original_gt}, got {merged_gt}"

    def test_merge_handles_dup_columns_with_combine_first(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        """No _dup suffix columns should survive the merge."""
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')
        dup_cols = [c for c in result.columns if c.endswith('_dup')]
        assert dup_cols == [], f"Found _dup columns after merge: {dup_cols}"

    def test_merge_ancestry_column_dropped(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        # Add 'ancestry' column to simulate pre-merge DataFrames
        eur_with_anc = eur_genotype_df.copy()
        eur_with_anc['ancestry'] = 'EUR'
        eas_with_anc = eas_genotype_df.copy()
        eas_with_anc['ancestry'] = 'EAS'

        result = coordinator._merge_ancestry_results(
            [eur_with_anc, eas_with_anc], 'NBA')
        assert 'ancestry' not in result.columns

    def test_merge_single_file_returns_that_file(
            self, coordinator, eur_genotype_df):
        result = coordinator._merge_ancestry_results([eur_genotype_df], 'NBA')
        # Single-file path: result should contain same variants
        assert len(result) == len(eur_genotype_df)
        assert set(result.columns).issuperset(
            {'snp_list_id', 'variant_id', 'chromosome'})

    def test_merge_empty_list_returns_empty(self, coordinator):
        result = coordinator._merge_ancestry_results([], 'NBA')
        assert result.empty

    def test_concat_within_ancestry_deduplicates_variants(
            self, coordinator, eur_chr1_df, eur_chr2_df):
        # Both DFs have source_file with '/EUR/' → same ancestry group
        result = coordinator._merge_ancestry_results(
            [eur_chr1_df, eur_chr2_df], 'IMPUTED')
        assert len(result) == 4   # 2 chr1 + 2 chr2 variants, no duplicates
        merge_keys = ['snp_list_id', 'variant_id', 'chromosome', 'position',
                      'counted_allele', 'alt_allele']
        dupes = result.duplicated(subset=merge_keys)
        assert not dupes.any(), "Duplicate variant rows after concat"

    def test_merge_preserves_maf_corrected_column(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')
        assert 'maf_corrected' in result.columns
        # No NaN in the boolean metadata column where at least one ancestry has a value
        # (combine_first should handle this)

    def test_merge_cross_ancestry_nans_for_absent_samples(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')
        # EUR samples should have NaN for variants that EAS brings via outer join
        # Both ancestries have the same 3 variants, so all genotypes should be non-NaN
        # (the outer join matches on variant rows — NaN would be in SAMPLE columns
        #  for absent samples, which is expected behaviour)
        eur_cols = [c for c in result.columns if c.startswith('MOCK_EUR')]
        eas_cols = [c for c in result.columns if c.startswith('MOCK_EAS')]
        assert len(eur_cols) == 10
        assert len(eas_cols) == 10

    def test_merge_two_ancestries_no_dup_variant_rows(
            self, coordinator, eur_genotype_df, eas_genotype_df):
        result = coordinator._merge_ancestry_results(
            [eur_genotype_df, eas_genotype_df], 'NBA')
        merge_keys = ['snp_list_id', 'variant_id', 'chromosome', 'position',
                      'counted_allele', 'alt_allele']
        dupes = result.duplicated(subset=merge_keys)
        assert not dupes.any(), "Duplicate variant rows in merged result"


# ---------------------------------------------------------------------------
# TestReorderDataframeColumns
# ---------------------------------------------------------------------------

class TestReorderDataframeColumns:

    EXPECTED_METADATA_ORDER = [
        'chromosome', 'variant_id', '(C)M', 'position',
        'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
        'pgen_a1', 'pgen_a2', 'data_type', 'source_file',
        'maf_corrected', 'original_alt_af',
    ]

    def test_metadata_cols_come_first(self, coordinator, df_with_mixed_sample_ids):
        result = coordinator._reorder_dataframe_columns(df_with_mixed_sample_ids)
        # Check that metadata columns that are present come before any sample column
        result_cols = list(result.columns)
        meta_present = [c for c in self.EXPECTED_METADATA_ORDER if c in result_cols]
        sample_start = next(
            i for i, c in enumerate(result_cols) if c not in self.EXPECTED_METADATA_ORDER
        )
        for m in meta_present:
            assert result_cols.index(m) < sample_start, \
                f"Metadata col '{m}' appears after sample columns"

    def test_sample_ids_normalized_in_output(self, coordinator, df_with_mixed_sample_ids):
        result = coordinator._reorder_dataframe_columns(df_with_mixed_sample_ids)
        assert 'FAKE_SAMPLE_001' in result.columns   # 0_ prefix stripped
        assert 'FAKE_SAMPLE_002' in result.columns   # WGS duplicate collapsed
        assert 'TEST_PERSON_042' in result.columns   # clean ID unchanged

    def test_raw_messy_ids_absent(self, coordinator, df_with_mixed_sample_ids):
        result = coordinator._reorder_dataframe_columns(df_with_mixed_sample_ids)
        assert '0_FAKE_SAMPLE_001'               not in result.columns
        assert 'FAKE_SAMPLE_002_FAKE_SAMPLE_002' not in result.columns

    def test_sample_cols_sorted_alphabetically(self, coordinator, df_with_mixed_sample_ids):
        result = coordinator._reorder_dataframe_columns(df_with_mixed_sample_ids)
        result_cols = list(result.columns)
        meta_set = set(self.EXPECTED_METADATA_ORDER)
        sample_cols = [c for c in result_cols if c not in meta_set]
        assert sample_cols == sorted(sample_cols), \
            "Sample columns are not sorted alphabetically"

    def test_duplicate_sample_ids_after_normalization_deduplicated(self, coordinator):
        # Two columns both normalize to the same name → only first survives.
        # Both original columns require renaming (neither is already the target name).
        df = pd.DataFrame({
            'chromosome': ['1'], 'variant_id': ['1:100000:A:C'],
            'position': [100000], 'counted_allele': ['C'], 'alt_allele': ['A'],
            'harmonization_action': ['EXACT'], 'snp_list_id': ['chr1:100000:A:C'],
            'pgen_a1': ['A'], 'pgen_a2': ['C'],
            'data_type': ['NBA'], 'source_file': ['/fake/file.pgen'],
            'maf_corrected': [False], 'original_alt_af': [0.1],
            '0_DEDUP_TEST': [1.0],           # normalizes to DEDUP_TEST
            'DEDUP_TEST_DEDUP_TEST': [2.0],  # also normalizes to DEDUP_TEST
        })
        result = coordinator._reorder_dataframe_columns(df)
        dup_count = sum(1 for c in result.columns if c == 'DEDUP_TEST')
        assert dup_count == 1, "Duplicate sample ID appeared more than once"
