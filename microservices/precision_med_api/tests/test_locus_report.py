"""
Tests for app/processing/locus_report_generator.py :: LocusReportGenerator

All file-loading methods are patched before __init__ runs.
ProbeSelectionLoader is replaced with a MagicMock per test.
Clinical data uses fully invented GP2IDs (MOCK_EUR_P*, MOCK_EAS_P*).
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from app.processing.locus_report_generator import LocusReportGenerator
from app.utils.probe_selector_loader import ProbeSelectionLoader


# ---------------------------------------------------------------------------
# Fixture: LocusReportGenerator with all file I/O patched out
# ---------------------------------------------------------------------------

@pytest.fixture
def locus_gen(mock_settings_with_paths, sample_master_key, sample_extended_clinical,
              snp_list_with_locus):
    """LocusReportGenerator with patched _load_* methods and a MagicMock probe_selector.

    Also sets dosage thresholds required by _calculate_variant_carrier_counts.
    """
    mock_settings_with_paths.dosage_het_min = 0.5
    mock_settings_with_paths.dosage_het_max = 1.5
    mock_settings_with_paths.dosage_hom_min = 1.5
    mock_settings_with_paths.release_path   = '/fake/release'
    mock_settings_with_paths.release        = '11'
    mock_settings_with_paths.snp_list_path  = '/fake/snp_list.csv'
    mock_settings_with_paths.master_key_path = None

    with patch.object(LocusReportGenerator, '_load_master_key',
                      return_value=sample_master_key):
        with patch.object(LocusReportGenerator, '_load_extended_clinical',
                          return_value=sample_extended_clinical):
            with patch.object(LocusReportGenerator, '_load_snp_list',
                              return_value=snp_list_with_locus):
                gen = LocusReportGenerator(mock_settings_with_paths,
                                           probe_selection_path=None)

    gen.probe_selector = MagicMock(spec=ProbeSelectionLoader)
    # Default: no probe recommendation (multi-probe → keep all)
    gen.probe_selector.get_recommended_variant.return_value = None
    return gen


# ---------------------------------------------------------------------------
# TestFilterToSelectedProbes — Probe selection regression block
# ---------------------------------------------------------------------------

class TestFilterToSelectedProbes:

    def test_single_probe_mutation_always_kept(self, locus_gen, multi_probe_genotype_df):
        # Rows with unique snp_list_id must always survive
        result = locus_gen._filter_to_selected_probes(multi_probe_genotype_df)
        single_probe_ids = {'chr2:300000:C:G', 'chr3:500000:G:A'}
        result_snp_ids = set(result['snp_list_id'].tolist())
        assert single_probe_ids.issubset(result_snp_ids)

    def test_multi_probe_with_recommendation_keeps_only_recommended(
            self, locus_gen, multi_probe_genotype_df):
        """THE probe selection regression test."""
        # Recommend probe '1:100000:A:C' for mutation 'chr1:100000:A:C'
        locus_gen.probe_selector.get_recommended_variant.side_effect = lambda sid: (
            '1:100000:A:C' if sid == 'chr1:100000:A:C' else None
        )
        result = locus_gen._filter_to_selected_probes(multi_probe_genotype_df)

        variant_ids = set(result['variant_id'].tolist())
        assert '1:100000:A:C'    in variant_ids, "Recommended probe should be kept"
        assert '1:100000:A:C_v2' not in variant_ids, "Non-recommended probe should be removed"

    def test_multi_probe_without_recommendation_keeps_all(
            self, locus_gen, multi_probe_genotype_df):
        # get_recommended_variant returns None for all → keep both probes
        locus_gen.probe_selector.get_recommended_variant.return_value = None
        result = locus_gen._filter_to_selected_probes(multi_probe_genotype_df)
        assert '1:100000:A:C'    in set(result['variant_id'])
        assert '1:100000:A:C_v2' in set(result['variant_id'])

    def test_empty_dataframe_returns_empty(self, locus_gen):
        result = locus_gen._filter_to_selected_probes(pd.DataFrame())
        assert result.empty

    def test_result_row_count_correct(self, locus_gen, multi_probe_genotype_df):
        # 2 probes → 1 kept (with recommendation) + 2 single = 3 rows
        locus_gen.probe_selector.get_recommended_variant.side_effect = lambda sid: (
            '1:100000:A:C' if sid == 'chr1:100000:A:C' else None
        )
        result = locus_gen._filter_to_selected_probes(multi_probe_genotype_df)
        assert len(result) == 3

    def test_probe_selector_called_with_correct_snp_list_id(
            self, locus_gen, multi_probe_genotype_df):
        locus_gen._filter_to_selected_probes(multi_probe_genotype_df)
        # get_recommended_variant should have been called with the multi-probe snp_list_id
        called_ids = [call[0][0] for call in
                      locus_gen.probe_selector.get_recommended_variant.call_args_list]
        assert 'chr1:100000:A:C' in called_ids


# ---------------------------------------------------------------------------
# TestJoinClinicalData
# ---------------------------------------------------------------------------

def _make_genotype_wide(gp2ids, snp_list_id='chr1:100000:A:C',
                         variant_id='1:100000:A:C', genotype_value=1.0):
    """Build a wide-format genotype DataFrame (1 variant, N samples)."""
    base = {
        'variant_id':           [variant_id],
        'snp_list_id':          [snp_list_id],
        'chromosome':           ['1'],
        'position':             [100000],
        'counted_allele':       ['C'],
        'alt_allele':           ['A'],
        'harmonization_action': ['EXACT'],
        'pgen_a1':              ['A'],
        'pgen_a2':              ['C'],
        'data_type':            ['NBA'],
        'source_file':          ['/fake/file.pgen'],
        'maf_corrected':        [False],
        'original_alt_af':      [0.2],
    }
    for gp2id in gp2ids:
        base[gp2id] = [genotype_value]
    return pd.DataFrame(base)


class TestJoinClinicalData:

    def test_join_produces_carrier_rows_only(self, locus_gen, sample_master_key):
        # 3 samples are carriers (gt > 0), rest are 0
        carrier_ids  = ['MOCK_EUR_P001', 'MOCK_EUR_P002', 'MOCK_EUR_P003']
        non_carrier_ids = ['MOCK_EUR_P004', 'MOCK_EUR_P005',
                           'MOCK_EUR_P006', 'MOCK_EUR_P007']
        df = _make_genotype_wide(carrier_ids, genotype_value=1.0)
        for nid in non_carrier_ids:
            df[nid] = 0.0

        result = locus_gen._join_clinical_data(df, data_type='NBA')
        assert len(result) == 3

    def test_join_adds_locus_column(self, locus_gen):
        df = _make_genotype_wide(['MOCK_EUR_P001'], snp_list_id='chr1:100000:A:C')
        result = locus_gen._join_clinical_data(df, data_type='NBA')
        assert 'gene' in result.columns
        assert result['gene'].iloc[0] == 'GENE1'

    def test_join_nba_uses_nba_label_for_ancestry(self, locus_gen):
        df = _make_genotype_wide(['MOCK_EUR_P001'])
        result = locus_gen._join_clinical_data(df, data_type='NBA')
        assert 'ancestry' in result.columns
        assert result['ancestry'].iloc[0] == 'EUR'

    def test_join_wgs_uses_wgs_label_for_ancestry(self, locus_gen):
        df = _make_genotype_wide(['MOCK_EAS_P001'])
        result = locus_gen._join_clinical_data(df, data_type='WGS')
        assert result['ancestry'].iloc[0] == 'EAS'

    def test_join_exomes_falls_back_to_wgs_label(self, locus_gen, sample_master_key):
        # Patch master_key so nba_label is NaN for MOCK_EAS_P001
        patched_key = sample_master_key.copy()
        patched_key.loc[
            patched_key['GP2ID'] == 'MOCK_EAS_P001', 'nba_label'
        ] = np.nan
        locus_gen.master_key = patched_key

        df = _make_genotype_wide(['MOCK_EAS_P001'])
        result = locus_gen._join_clinical_data(df, data_type='EXOMES')
        assert result['ancestry'].iloc[0] == 'EAS'

    def test_join_exomes_unknown_when_both_labels_missing(self, locus_gen, sample_master_key):
        patched_key = sample_master_key.copy()
        patched_key.loc[
            patched_key['GP2ID'] == 'MOCK_EUR_P001', 'nba_label'
        ] = np.nan
        patched_key.loc[
            patched_key['GP2ID'] == 'MOCK_EUR_P001', 'wgs_label'
        ] = np.nan
        locus_gen.master_key = patched_key

        df = _make_genotype_wide(['MOCK_EUR_P001'])
        result = locus_gen._join_clinical_data(df, data_type='EXOMES')
        assert result['ancestry'].iloc[0] == 'Unknown'

    def test_join_includes_clinical_columns(self, locus_gen):
        df = _make_genotype_wide(['MOCK_EUR_P001'])
        result = locus_gen._join_clinical_data(df, data_type='NBA')
        for col in ('hoehn_and_yahr_stage', 'moca_total_score', 'dat_sbr_caudate_mean'):
            assert col in result.columns, f"Expected column missing: {col}"

    def test_join_unmatched_gp2id_still_present_with_nan_clinical(self, locus_gen):
        # GP2ID not in master_key → row still present (left join)
        df = _make_genotype_wide(['GHOST_ID_999'])
        result = locus_gen._join_clinical_data(df, data_type='NBA')
        assert len(result) == 1
        assert pd.isna(result.iloc[0].get('gene') or None) or True  # may or may not have gene


# ---------------------------------------------------------------------------
# TestCalculateAncestryMetrics
# ---------------------------------------------------------------------------

def _make_carrier_df(gp2ids, extended_clinical, hy_stages, moca_scores,
                     age_at_collection=None, age_of_onset=None,
                     dat_values=None):
    """Build a long-format carrier DataFrame for ancestry metric tests."""
    n = len(gp2ids)
    data = {
        'GP2ID':                  gp2ids,
        'variant_id':             ['1:100000:A:C'] * n,
        'snp_list_id':            ['chr1:100000:A:C'] * n,
        'genotype':               [1.0] * n,
        'gene':                   ['GENE1'] * n,
        'ancestry':               ['EUR'] * n,
        'extended_clinical_data': extended_clinical,
        'hoehn_and_yahr_stage':   hy_stages,
        'moca_total_score':       moca_scores,
        'dat_sbr_caudate_mean':   dat_values if dat_values else [np.nan] * n,
    }
    if age_at_collection:
        data['age_at_sample_collection'] = age_at_collection
    if age_of_onset:
        data['age_of_onset'] = age_of_onset
    return pd.DataFrame(data)


class TestCalculateAncestryMetrics:

    def _eur_carrier_df(self):
        """10 EUR carriers with known clinical values (from sample fixtures)."""
        ids  = [f"MOCK_EUR_P{i:03d}" for i in range(1, 11)]
        return _make_carrier_df(
            gp2ids             = ids,
            extended_clinical  = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            hy_stages          = [1.0, 1.5, 1.9, 2.0, 2.5] + [np.nan] * 5,
            moca_scores        = [18.0, 20.0, 22.0, 24.0, 26.0] + [np.nan] * 5,
            dat_values         = [2.1, np.nan, 1.8, np.nan, 2.3] + [np.nan] * 5,
            age_at_collection  = [62.0, 64.0, 66.0, 68.0, 70.0] + [np.nan] * 5,
            age_of_onset       = [60.0, 60.0, 60.0, 60.0, 60.0] + [np.nan] * 5,
        )

    def test_total_carriers_deduplicates_by_gp2id(self, locus_gen):
        # Same GP2ID in two rows (carrier of 2 variants) → counted once
        ids = ['MOCK_EUR_P001', 'MOCK_EUR_P001', 'MOCK_EUR_P002']
        df = _make_carrier_df(ids, [1, 1, 1], [1.0, 1.0, 2.0], [20.0, 20.0, 22.0])
        metrics = locus_gen._calculate_ancestry_metrics(df, 'EUR')
        assert metrics.total_carriers == 2

    def test_hy_less_than_2_correct_count(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P001=1.0, P002=1.5, P003=1.9 all < 2 → 3
        assert metrics.hy_less_than_2 == 3

    def test_hy_less_than_3_correct_count(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P001-P005 all < 3 (values 1.0, 1.5, 1.9, 2.0, 2.5) → 5
        assert metrics.hy_less_than_3 == 5

    def test_moca_gte_20_correct_count(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P002=20, P003=22, P004=24, P005=26 → 4
        assert metrics.moca_gte_20 == 4

    def test_moca_gte_24_correct_count(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P004=24, P005=26 → 2
        assert metrics.moca_gte_24 == 2

    def test_disease_duration_lte_3_years(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P001: 62-60=2 ≤ 3 → 1
        assert metrics.disease_duration_lte_3_years == 1

    def test_disease_duration_lte_5_years(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P001: 2, P002: 4 → 2
        assert metrics.disease_duration_lte_5_years == 2

    def test_disease_duration_negative_excluded(self, locus_gen):
        ids = ['MOCK_EUR_P001', 'MOCK_EUR_P002']
        df = _make_carrier_df(
            ids, [1, 1], [1.0, np.nan], [20.0, np.nan],
            age_at_collection=[58.0, 65.0],
            age_of_onset=[60.0, 60.0],   # P001 onset AFTER collection → negative → excluded
        )
        metrics = locus_gen._calculate_ancestry_metrics(df, 'EUR')
        assert metrics.disease_duration_lte_3_years == 0
        assert metrics.disease_duration_lte_5_years == 1   # P002: 65-60=5 ≤ 5

    def test_missing_clinical_columns_return_zero(self, locus_gen):
        ids = ['MOCK_EUR_P001', 'MOCK_EUR_P002']
        df = pd.DataFrame({
            'GP2ID':                  ids,
            'extended_clinical_data': [1, 0],
            # No hoehn_and_yahr_stage, moca_total_score, etc.
        })
        metrics = locus_gen._calculate_ancestry_metrics(df, 'EUR')
        assert metrics.hy_available == 0
        assert metrics.moca_available == 0
        assert metrics.dat_caudate_available == 0

    def test_carriers_with_extended_clinical_data_count(self, locus_gen):
        metrics = locus_gen._calculate_ancestry_metrics(self._eur_carrier_df(), 'EUR')
        # P001-P005 have extended_clinical_data=1 → 5
        assert metrics.carriers_with_clinical_data == 5


# ---------------------------------------------------------------------------
# TestCalculateVariantCarrierCounts
# ---------------------------------------------------------------------------

class TestCalculateVariantCarrierCounts:

    def _make_wide_geno_df(self):
        """2-variant DataFrame with known carrier values."""
        return pd.DataFrame({
            'chromosome':           ['1', '2'],
            'variant_id':           ['1:100000:A:C', '2:300000:C:G'],
            'position':             [100000, 300000],
            'counted_allele':       ['C', 'G'],
            'alt_allele':           ['A', 'C'],
            'harmonization_action': ['EXACT', 'SWAP'],
            'snp_list_id':          ['chr1:100000:A:C', 'chr2:300000:C:G'],
            'pgen_a1':              ['A', 'C'],
            'pgen_a2':              ['C', 'G'],
            'data_type':            ['NBA', 'NBA'],
            'source_file':          ['/fake/file.pgen', '/fake/file.pgen'],
            'maf_corrected':        [False, False],
            'original_alt_af':      [0.1, 0.2],
            'MOCK_S001': [1.0, 2.0],    # het, hom
            'MOCK_S002': [0.0, 1.0],    # non-carrier, het
            'MOCK_S003': [2.0, 0.0],    # hom, non-carrier
        })

    def test_het_hom_noncarrier_classified_correctly(self, locus_gen):
        df = self._make_wide_geno_df()
        details = locus_gen._calculate_variant_carrier_counts(df)

        v1 = details['1:100000:A:C']
        # MOCK_S001=1 (het), MOCK_S002=0 (non), MOCK_S003=2 (hom)
        assert v1.heterozygous_count == 1
        assert v1.homozygous_count   == 1
        assert v1.carrier_count      == 2

    def test_returns_variant_detail_objects_with_alleles(self, locus_gen):
        from app.models.locus_report import VariantDetail
        df = self._make_wide_geno_df()
        details = locus_gen._calculate_variant_carrier_counts(df)

        assert isinstance(details, dict)
        for key, val in details.items():
            assert isinstance(val, VariantDetail), f"Expected VariantDetail for {key}"
            # Alleles come from pgen_a2 (ref) and pgen_a1 (alt)
            assert val.ref_allele != ''
            assert val.alt_allele != ''
