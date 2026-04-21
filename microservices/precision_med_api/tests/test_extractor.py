"""
Tests for app/processing/extractor.py :: VariantExtractor

PLINK binary is never invoked: _check_plink_availability is patched to return
False, which triggers the built-in _simulate_plink_extraction() fallback.
File I/O uses tmp_path or pre-built fixtures — no real data accessed.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from app.processing.extractor import VariantExtractor
from app.models.harmonization import HarmonizationAction


# ---------------------------------------------------------------------------
# Module-level fixture: one extractor per test
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor(mock_settings_with_paths):
    """VariantExtractor with mock settings (no real file paths needed)."""
    return VariantExtractor(mock_settings_with_paths)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_cols(df):
    """Return sample column names (not in the standard metadata set)."""
    META = {'chromosome', 'variant_id', '(C)M', 'position', 'counted_allele',
            'alt_allele', 'harmonization_action', 'snp_list_id',
            'pgen_a1', 'pgen_a2', 'data_type', 'source_file',
            'maf_corrected', 'original_alt_af'}
    return [c for c in df.columns if c not in META]


def _make_plan_row(pgen_variant_id, snp_list_id, action, transform,
                   snp_list_a1, snp_list_a2, pgen_a1, pgen_a2):
    return {
        'pgen_variant_id': pgen_variant_id,
        'snp_list_id': snp_list_id,
        'harmonization_action': action,
        'genotype_transform': transform,
        'snp_list_a1': snp_list_a1,
        'snp_list_a2': snp_list_a2,
        'pgen_a1': pgen_a1,
        'pgen_a2': pgen_a2,
    }


# ---------------------------------------------------------------------------
# TestApplyMAFCorrection — Critical regression block
# ---------------------------------------------------------------------------

class TestApplyMAFCorrection:

    def test_high_af_variant_gets_corrected(self, extractor, high_af_genotype_df):
        result = extractor._apply_maf_correction(high_af_genotype_df)

        row1 = result.iloc[0]
        assert row1['maf_corrected']
        # Original: [2,2,2,2,2,2,2,1,0,0]  → corrected: [0,0,0,0,0,0,0,1,2,2]
        sc = _sample_cols(result)
        gts = [float(x) for x in row1[sc]]
        assert gts == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0]

    def test_high_af_corrected_alleles_swapped(self, extractor, high_af_genotype_df):
        original_counted = high_af_genotype_df.iloc[0]['counted_allele']
        original_alt     = high_af_genotype_df.iloc[0]['alt_allele']

        result = extractor._apply_maf_correction(high_af_genotype_df)
        row1 = result.iloc[0]
        assert row1['counted_allele'] == original_alt
        assert row1['alt_allele'] == original_counted

    def test_low_af_variant_not_corrected(self, extractor, high_af_genotype_df):
        result = extractor._apply_maf_correction(high_af_genotype_df)

        row2 = result.iloc[1]
        assert not row2['maf_corrected']
        # Genotypes [1,1,1,0,0,0,0,0,0,0] should be unchanged
        sc = _sample_cols(result)
        gts = [float(x) for x in row2[sc]]
        assert gts == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_original_alt_af_always_recorded(self, extractor, high_af_genotype_df):
        result = extractor._apply_maf_correction(high_af_genotype_df)
        assert result['original_alt_af'].notna().all()

    def test_original_alt_af_value_for_high_af(self, extractor, high_af_genotype_df):
        result = extractor._apply_maf_correction(high_af_genotype_df)
        assert abs(result.iloc[0]['original_alt_af'] - 0.75) < 1e-9

    def test_boundary_af_exactly_05_not_corrected(self, extractor):
        # 5 samples with genotype 1, 5 with genotype 0 → AF = 5/20 = 0.25, not > 0.5
        # To get AF = 0.5: 5 homs + 0 others → alt_count = 10, total = 20 → 0.5
        samples = [f"S{i}" for i in range(10)]
        base = {
            'chromosome': ['1'], 'variant_id': ['1:100000:A:C'],
            'position': [100000], 'counted_allele': ['C'], 'alt_allele': ['A'],
            'harmonization_action': ['EXACT'], 'snp_list_id': ['chr1:100000:A:C'],
            'pgen_a1': ['A'], 'pgen_a2': ['C'],
        }
        for i, s in enumerate(samples):
            # 5 samples genotype 2, 5 samples genotype 0 → alt = 10, total = 20 → AF = 0.5
            base[s] = [2.0 if i < 5 else 0.0]
        df = pd.DataFrame(base)

        result = extractor._apply_maf_correction(df)
        # Condition is strictly > 0.5, so AF == 0.5 should NOT be corrected
        assert not result.iloc[0]['maf_corrected']

    def test_all_missing_genotypes_no_flip(self, extractor):
        samples = ['S1', 'S2', 'S3']
        base = {
            'chromosome': ['1'], 'variant_id': ['1:100000:A:C'],
            'position': [100000], 'counted_allele': ['C'], 'alt_allele': ['A'],
            'harmonization_action': ['EXACT'], 'snp_list_id': ['chr1:100000:A:C'],
            'pgen_a1': ['A'], 'pgen_a2': ['C'],
            'S1': [np.nan], 'S2': [np.nan], 'S3': [np.nan],
        }
        df = pd.DataFrame(base)
        result = extractor._apply_maf_correction(df)
        assert not result.iloc[0]['maf_corrected']

    def test_empty_dataframe_returns_empty(self, extractor):
        result = extractor._apply_maf_correction(pd.DataFrame())
        assert result.empty

    def test_maf_correction_adds_both_metadata_columns(self, extractor, high_af_genotype_df):
        result = extractor._apply_maf_correction(high_af_genotype_df)
        assert 'maf_corrected' in result.columns
        assert 'original_alt_af' in result.columns


# ---------------------------------------------------------------------------
# TestHarmonizeExtractedGenotypes — Allele counting regression block
# ---------------------------------------------------------------------------

class TestHarmonizeExtractedGenotypes:
    """harmonization_records is a plan DataFrame (from _harmonization_records_to_plan_df),
    not a list of HarmonizationRecord objects."""

    def _make_raw_df(self, variant_id, sample_values):
        """Build a minimal raw genotype DataFrame (1 variant row)."""
        row = {
            'chromosome': [variant_id.split(':')[0]],
            'variant_id': [variant_id],
            'position':   [int(variant_id.split(':')[1])],
            'counted_allele': [variant_id.split(':')[2]],
            'alt_allele':     [variant_id.split(':')[3]],
        }
        for name, val in sample_values.items():
            row[name] = [val]
        return pd.DataFrame(row)

    def test_exact_action_flips_genotypes_to_count_alt(self, extractor):
        # THE allele counting fix: raw 0 (homozygous REF) → 2 (homozygous pathogenic ALT)
        raw = self._make_raw_df('1:100000:A:C',
                                {'MOCK_S1': 0.0, 'MOCK_S2': 1.0, 'MOCK_S3': 2.0})
        plan = pd.DataFrame([_make_plan_row(
            '1:100000:A:C', 'chr1:100000:A:C',
            HarmonizationAction.EXACT, None,
            'A', 'C', 'A', 'C'
        )])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        assert not result.empty
        assert result.iloc[0]['MOCK_S1'] == 2.0   # 0 → 2 (hom REF → hom ALT)
        assert result.iloc[0]['MOCK_S2'] == 1.0   # 1 stays 1
        assert result.iloc[0]['MOCK_S3'] == 0.0   # 2 → 0

    def test_flip_action_flips_genotypes_to_count_alt(self, extractor):
        raw = self._make_raw_df('1:200000:T:G',
                                {'MOCK_S1': 0.0, 'MOCK_S2': 1.0, 'MOCK_S3': 2.0})
        plan = pd.DataFrame([_make_plan_row(
            '1:200000:T:G', 'chr1:200000:T:G',
            HarmonizationAction.FLIP, None,
            'T', 'G', 'G', 'T'
        )])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        assert result.iloc[0]['MOCK_S1'] == 2.0
        assert result.iloc[0]['MOCK_S3'] == 0.0

    def test_counted_allele_is_alt_not_ref(self, extractor):
        raw = self._make_raw_df('1:100000:A:C', {'MOCK_S1': 0.0})
        plan = pd.DataFrame([_make_plan_row(
            '1:100000:A:C', 'chr1:100000:A:C',
            HarmonizationAction.EXACT, None,
            'A', 'C', 'A', 'C'    # snp_list_a1='A' (REF), snp_list_a2='C' (ALT)
        )])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        assert result.iloc[0]['counted_allele'] == 'C'   # ALT, not REF
        assert result.iloc[0]['alt_allele']     == 'A'   # REF

    def test_swap_action_applies_2_minus_x_transform(self, extractor):
        raw = self._make_raw_df('2:300000:G:C',
                                {'MOCK_S1': 0.0, 'MOCK_S2': 1.0, 'MOCK_S3': 2.0})
        plan = pd.DataFrame([_make_plan_row(
            '2:300000:G:C', 'chr2:300000:C:G',
            HarmonizationAction.SWAP, '2-x',
            'C', 'G', 'G', 'C'
        )])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        # 2-x: 0→2, 1→1, 2→0
        assert result.iloc[0]['MOCK_S1'] == 2.0
        assert result.iloc[0]['MOCK_S2'] == 1.0
        assert result.iloc[0]['MOCK_S3'] == 0.0

    def test_multiple_snp_list_ids_per_pgen_variant(self, extractor):
        raw = self._make_raw_df('1:100000:A:C', {'MOCK_S1': 0.0})
        plan = pd.DataFrame([
            _make_plan_row('1:100000:A:C', 'chr1:100000:A:C',
                           HarmonizationAction.EXACT, None, 'A', 'C', 'A', 'C'),
            _make_plan_row('1:100000:A:C', 'chr1:100000:A:C_ALIAS',
                           HarmonizationAction.EXACT, None, 'A', 'C', 'A', 'C'),
        ])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        # One PGEN variant maps to 2 SNP list IDs → 2 output rows
        assert len(result) == 2

    def test_missing_genotypes_preserved(self, extractor):
        raw = self._make_raw_df('1:100000:A:C', {'MOCK_S1': np.nan, 'MOCK_S2': 1.0})
        plan = pd.DataFrame([_make_plan_row(
            '1:100000:A:C', 'chr1:100000:A:C',
            HarmonizationAction.EXACT, None, 'A', 'C', 'A', 'C'
        )])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        assert pd.isna(result.iloc[0]['MOCK_S1'])

    def test_harmonization_metadata_columns_added(self, extractor):
        raw = self._make_raw_df('1:100000:A:C', {'MOCK_S1': 0.0})
        plan = pd.DataFrame([_make_plan_row(
            '1:100000:A:C', 'chr1:100000:A:C',
            HarmonizationAction.EXACT, None, 'A', 'C', 'A', 'C'
        )])
        result = extractor._harmonize_extracted_genotypes(raw, plan)
        for col in ('harmonization_action', 'snp_list_id', 'pgen_a1', 'pgen_a2'):
            assert col in result.columns, f"Missing column: {col}"

    def test_empty_raw_df_returns_empty(self, extractor):
        plan = pd.DataFrame([_make_plan_row(
            '1:100000:A:C', 'chr1:100000:A:C',
            HarmonizationAction.EXACT, None, 'A', 'C', 'A', 'C'
        )])
        result = extractor._harmonize_extracted_genotypes(pd.DataFrame(), plan)
        assert result.empty

    def test_empty_plan_returns_raw_df_unchanged(self, extractor):
        # When the harmonization plan is empty, the code short-circuits and
        # returns the raw DataFrame as-is (no snp_list_id or metadata added).
        raw = self._make_raw_df('1:100000:A:C', {'MOCK_S1': 0.0})
        result = extractor._harmonize_extracted_genotypes(raw, pd.DataFrame())
        assert 'snp_list_id' not in result.columns


# ---------------------------------------------------------------------------
# TestNormalizeSampleIdsFromPsam
# ---------------------------------------------------------------------------

class TestNormalizeSampleIdsFromPsam:

    def test_renames_fid_iid_to_iid(self, extractor, psam_tmp_file):
        df = pd.DataFrame({
            'chromosome': ['1'],
            'FAKEFAM001_TESTIND001': [1.0],
            'OTHER_COL': [0.0],
        })
        result = extractor._normalize_sample_ids_from_psam(df, psam_tmp_file)
        assert 'TESTIND001' in result.columns
        assert 'FAKEFAM001_TESTIND001' not in result.columns

    def test_ignores_non_psam_columns(self, extractor, psam_tmp_file):
        df = pd.DataFrame({
            'chromosome': ['1'],
            'FAKEFAM001_TESTIND001': [1.0],
            'UNRELATED_COL': [99.0],
        })
        result = extractor._normalize_sample_ids_from_psam(df, psam_tmp_file)
        assert 'UNRELATED_COL' in result.columns

    def test_handles_missing_psam_gracefully(self, extractor):
        df = pd.DataFrame({'chromosome': ['1'], 'SAMPLE_X': [0.0]})
        result = extractor._normalize_sample_ids_from_psam(df, '/nonexistent/path.psam')
        # Returns DataFrame unchanged, no exception
        assert 'SAMPLE_X' in result.columns

    def test_handles_hash_fid_header(self, extractor, tmp_path):
        # psam_tmp_file uses #FID header; verify it still works
        psam_path = str(tmp_path / "hashfid.psam")
        with open(psam_path, 'w') as f:
            f.write("#FID\tIID\tSEX\nFAKEFAM002\tTESTIND010\t1\n")
        df = pd.DataFrame({'chromosome': ['1'], 'FAKEFAM002_TESTIND010': [2.0]})
        result = extractor._normalize_sample_ids_from_psam(df, psam_path)
        assert 'TESTIND010' in result.columns


# ---------------------------------------------------------------------------
# TestReadTrawFile
# ---------------------------------------------------------------------------

class TestReadTrawFile:

    def test_reads_traw_renames_columns(self, extractor, traw_tmp_file):
        result = extractor._read_traw_file(traw_tmp_file)
        assert 'chromosome' in result.columns
        assert 'variant_id' in result.columns
        assert 'position' in result.columns
        for old_col in ('CHR', 'SNP', 'POS', 'COUNTED', 'ALT'):
            assert old_col not in result.columns

    def test_reads_traw_with_psam_normalizes_sample_ids(self, extractor, tmp_path, psam_tmp_file):
        traw_data = pd.DataFrame({
            'CHR': ['1'], 'SNP': ['1:100000:A:C'], 'CM': [0],
            'POS': [100000], 'COUNTED': ['C'], 'ALT': ['A'],
            'FAKEFAM001_TESTIND001': [1.0],
        })
        traw_path = str(tmp_path / "fid_iid_test.traw")
        traw_data.to_csv(traw_path, sep='\t', index=False)

        result = extractor._read_traw_file(traw_path, psam_path=psam_tmp_file)
        assert 'TESTIND001' in result.columns
        assert 'FAKEFAM001_TESTIND001' not in result.columns

    def test_returns_correct_row_count(self, extractor, traw_tmp_file):
        result = extractor._read_traw_file(traw_tmp_file)
        assert len(result) == 3   # sample_traw_data has 3 variants


# ---------------------------------------------------------------------------
# TestSimulatePlinkExtraction
# ---------------------------------------------------------------------------

class TestSimulatePlinkExtraction:

    def test_simulate_returns_correct_shapes(self, extractor):
        variant_ids = [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',
            'chr3:500000:G:A',
        ]
        result = extractor._simulate_plink_extraction('/fake/path', variant_ids)
        assert len(result) == 5
        sample_columns = _sample_cols(result)
        assert len(sample_columns) == 1000

    def test_simulate_genotype_values_in_valid_range(self, extractor):
        variant_ids = ['chr1:100000:A:C', 'chr2:300000:C:G']
        result = extractor._simulate_plink_extraction('/fake/path', variant_ids)
        sc = _sample_cols(result)
        for _, row in result.iterrows():
            vals = pd.to_numeric(row[sc], errors='coerce').dropna()
            assert ((vals == 0.0) | (vals == 1.0) | (vals == 2.0)).all(), \
                "Non-{0,1,2} genotype value found"
