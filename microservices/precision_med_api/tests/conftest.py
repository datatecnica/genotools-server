"""
Pytest configuration and shared fixtures for the test suite.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock

from app.core.config import Settings
from app.models.harmonization import HarmonizationRecord, HarmonizationAction


@pytest.fixture
def test_settings():
    """Create test settings with temporary paths."""
    settings = Settings()
    # Override paths for testing
    settings.mnt_path = tempfile.mkdtemp()
    return settings


@pytest.fixture
def sample_snp_list():
    """Create a sample SNP list for testing."""
    return pd.DataFrame({
        'variant_id': [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',  # Ambiguous
            'chr3:500000:G:A'
        ],
        'chromosome': ['1', '1', '2', '2', '3'],
        'position': [100000, 200000, 300000, 400000, 500000],
        'ref': ['A', 'T', 'C', 'A', 'G'],
        'alt': ['C', 'G', 'G', 'T', 'A'],
        'gene': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'],
        'rsid': ['rs1', 'rs2', 'rs3', 'rs4', 'rs5'],
        'hg38': [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',
            'chr3:500000:G:A'
        ]
    })


@pytest.fixture
def sample_pvar_data():
    """Create sample PVAR data for testing."""
    return pd.DataFrame({
        'CHROM': ['1', '1', '2', '2', '3'],
        'POS': [100000, 200000, 300000, 400000, 500000],
        'ID': [
            '1:100000:A:C',      # Exact match
            '1:200000:G:T',      # Flip
            '2:300000:G:C',      # Swap
            '2:400000:A:T',      # Ambiguous
            '3:500000:T:C'       # Flip+Swap
        ],
        'REF': ['A', 'G', 'G', 'A', 'T'],
        'ALT': ['C', 'T', 'C', 'T', 'C']
    })


@pytest.fixture
def sample_harmonization_records():
    """Create sample harmonization records for testing."""
    return [
        HarmonizationRecord(
            snp_list_id='chr1:100000:A:C',
            pgen_variant_id='1:100000:A:C',
            chromosome='1',
            position=100000,
            snp_list_a1='A',
            snp_list_a2='C',
            pgen_a1='A',
            pgen_a2='C',
            harmonization_action=HarmonizationAction.EXACT,
            genotype_transform=None,
            file_path='/test/file.pgen',
            data_type='NBA',
            ancestry='EUR'
        ),
        HarmonizationRecord(
            snp_list_id='chr1:200000:T:G',
            pgen_variant_id='1:200000:G:T',
            chromosome='1',
            position=200000,
            snp_list_a1='T',
            snp_list_a2='G',
            pgen_a1='G',
            pgen_a2='T',
            harmonization_action=HarmonizationAction.FLIP,
            genotype_transform=None,
            file_path='/test/file.pgen',
            data_type='NBA',
            ancestry='EUR'
        ),
        HarmonizationRecord(
            snp_list_id='chr2:300000:C:G',
            pgen_variant_id='2:300000:G:C',
            chromosome='2',
            position=300000,
            snp_list_a1='C',
            snp_list_a2='G',
            pgen_a1='G',
            pgen_a2='C',
            harmonization_action=HarmonizationAction.SWAP,
            genotype_transform='2-x',
            file_path='/test/file.pgen',
            data_type='NBA',
            ancestry='EUR'
        )
    ]


@pytest.fixture
def sample_genotype_data():
    """Create sample genotype data for testing."""
    np.random.seed(42)
    
    # Create realistic genotype data
    n_variants = 5
    n_samples = 100
    
    # Generate genotypes with realistic allele frequencies
    genotype_matrix = []
    sample_ids = [f"SAMPLE_{i:06d}" for i in range(n_samples)]
    
    for i in range(n_variants):
        # Random allele frequency between 0.01 and 0.5
        freq = np.random.uniform(0.01, 0.5)
        
        # Hardy-Weinberg proportions
        p_aa = (1 - freq) ** 2
        p_ab = 2 * freq * (1 - freq)
        p_bb = freq ** 2
        
        genotypes = np.random.choice([0, 1, 2], size=n_samples, p=[p_aa, p_ab, p_bb])
        
        # Add some missing data (2% missing rate)
        missing_mask = np.random.random(n_samples) < 0.02
        genotypes = genotypes.astype(float)
        genotypes[missing_mask] = np.nan
        
        genotype_matrix.append(genotypes)
    
    # Create DataFrame
    data = {
        'chromosome': ['1', '1', '2', '2', '3'],
        'variant_id': [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',
            'chr3:500000:G:A'
        ],
        'position': [100000, 200000, 300000, 400000, 500000],
        'counted_allele': ['C', 'G', 'G', 'T', 'A'],  # Now counts ALT allele (allele of interest)
        'alt_allele': ['A', 'T', 'C', 'A', 'G'],      # REF allele
        'harmonization_action': ['EXACT', 'FLIP', 'SWAP', 'AMBIGUOUS', 'FLIP_SWAP'],
        'snp_list_id': [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',
            'chr3:500000:G:A'
        ],
        'data_type': ['NBA'] * 5,
        'ancestry': ['EUR'] * 5
    }
    
    # Add sample genotype columns
    for i, sample_id in enumerate(sample_ids):
        data[sample_id] = [genotype_matrix[j][i] for j in range(n_variants)]
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_plink_files(tmp_path):
    """Create mock PLINK files for testing."""
    # Create PVAR file
    pvar_content = """##fileformat=VCFv4.2
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO
1	100000	1:100000:A:C	A	C	.	.	.
1	200000	1:200000:G:T	G	T	.	.	.
2	300000	2:300000:G:C	G	C	.	.	.
"""
    pvar_file = tmp_path / "test.pvar"
    pvar_file.write_text(pvar_content)
    
    # Create PSAM file
    psam_content = """#FID	IID	PAT	MAT	SEX	PHENO
FAM001	SAMPLE_000001	0	0	1	-9
FAM002	SAMPLE_000002	0	0	2	-9
FAM003	SAMPLE_000003	0	0	1	-9
"""
    psam_file = tmp_path / "test.psam"
    psam_file.write_text(psam_content)
    
    # Create dummy PGEN file
    pgen_file = tmp_path / "test.pgen"
    pgen_file.write_bytes(b"dummy_pgen_content")
    
    return {
        'pvar': str(pvar_file),
        'psam': str(psam_file),
        'pgen': str(pgen_file),
        'base': str(tmp_path / "test")
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


@pytest.fixture
def mock_settings_with_paths(tmp_path):
    """Create mock settings with valid file paths."""
    settings = Mock(spec=Settings)
    
    # Set up mock paths
    base_path = tmp_path / "mock_data"
    base_path.mkdir()
    
    settings.mnt_path = str(tmp_path)
    settings.release = "10"
    settings.ANCESTRIES = ["EUR", "EAS", "AFR"]
    settings.CHROMOSOMES = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
    
    # Mock path methods
    settings.get_cache_path.return_value = str(base_path / "cache")
    settings.get_output_path.return_value = str(base_path / "output")
    settings.get_nba_path.return_value = str(base_path / "nba" / "EUR_test")
    settings.get_wgs_path.return_value = str(base_path / "wgs" / "wgs_test")
    settings.get_imputed_path.return_value = str(base_path / "imputed" / "chr1_EUR_test")
    
    # Mock file listing methods
    settings.list_available_ancestries.return_value = ["EUR", "EAS"]
    settings.list_available_chromosomes.return_value = ["1", "2", "22"]
    
    return settings


def create_test_pvar_file(file_path: str, variants_data: list):
    """Helper function to create PVAR test files."""
    with open(file_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for variant in variants_data:
            f.write(f"{variant['chrom']}\t{variant['pos']}\t{variant['id']}\t{variant['ref']}\t{variant['alt']}\t.\t.\t.\n")


def create_test_traw_file(file_path: str, genotype_data: pd.DataFrame):
    """Helper function to create TRAW test files."""
    genotype_data.to_csv(file_path, sep='\t', index=False, na_rep='NA')


@pytest.fixture
def sample_traw_data():
    """Create sample TRAW format data for testing."""
    return pd.DataFrame({
        'CHR': ['1', '1', '2'],
        'SNP': ['1:100000:A:C', '1:200000:T:G', '2:300000:C:G'],
        'CM': [0, 0, 0],
        'POS': [100000, 200000, 300000],
        'COUNTED': ['C', 'G', 'G'],  # Now counts ALT allele (allele of interest)
        'ALT': ['A', 'T', 'C'],      # REF allele
        'SAMPLE_000001': [0, 1, 2],
        'SAMPLE_000002': [1, 0, 1],
        'SAMPLE_000003': [2, 2, 0]
    })


# ============================================================
# Multi-ancestry genotype DataFrames (for merge/concat tests)
# ============================================================

def _make_ancestry_genotype_df(sample_prefix: str, source_file: str,
                                n_samples: int = 10) -> pd.DataFrame:
    """Helper: build a 3-variant genotype DataFrame for one ancestry."""
    samples = [f"{sample_prefix}{i:03d}" for i in range(1, n_samples + 1)]
    base = {
        'snp_list_id': ['chr1:100000:A:C', 'chr1:200000:T:G', 'chr2:300000:C:G'],
        'variant_id':  ['1:100000:A:C',    '1:200000:T:G',    '2:300000:C:G'],
        'chromosome':  ['1', '1', '2'],
        'position':    [100000, 200000, 300000],
        'counted_allele': ['C', 'G', 'G'],
        'alt_allele':     ['A', 'T', 'C'],
        'harmonization_action': ['EXACT', 'FLIP', 'SWAP'],
        'pgen_a1': ['A', 'T', 'C'],
        'pgen_a2': ['C', 'G', 'G'],
        'data_type': ['NBA', 'NBA', 'NBA'],
        'source_file': [source_file] * 3,
        'maf_corrected': [False, False, False],
        'original_alt_af': [0.2, 0.15, 0.1],
    }
    # Add deterministic genotype values for each sample
    np.random.seed(0)
    for j, s in enumerate(samples):
        base[s] = [float((j + i) % 3) for i in range(3)]
    return pd.DataFrame(base)


@pytest.fixture
def eur_genotype_df():
    """3 variants × 10 invented EUR samples for multi-ancestry merge tests."""
    return _make_ancestry_genotype_df(
        sample_prefix='MOCK_EUR_P',
        source_file='/fake/EUR/EUR_release11.pgen'
    )


@pytest.fixture
def eas_genotype_df():
    """Same 3 variants × 10 invented EAS samples for multi-ancestry merge tests."""
    return _make_ancestry_genotype_df(
        sample_prefix='MOCK_EAS_P',
        source_file='/fake/EAS/EAS_release11.pgen'
    )


@pytest.fixture
def eur_chr1_df():
    """2 chr1 variants × 5 EUR samples — tests concat-within-ancestry."""
    samples = [f"MOCK_EUR_P{i:03d}" for i in range(1, 6)]
    base = {
        'snp_list_id': ['chr1:100000:A:C', 'chr1:200000:T:G'],
        'variant_id':  ['1:100000:A:C',    '1:200000:T:G'],
        'chromosome':  ['1', '1'],
        'position':    [100000, 200000],
        'counted_allele': ['C', 'G'],
        'alt_allele':     ['A', 'T'],
        'harmonization_action': ['EXACT', 'FLIP'],
        'pgen_a1': ['A', 'T'],
        'pgen_a2': ['C', 'G'],
        'data_type': ['IMPUTED', 'IMPUTED'],
        'source_file': ['/fake/EUR/chr1_EUR_release11.pgen'] * 2,
        'maf_corrected': [False, False],
        'original_alt_af': [0.2, 0.15],
    }
    for j, s in enumerate(samples):
        base[s] = [float(j % 3), float((j + 1) % 3)]
    return pd.DataFrame(base)


@pytest.fixture
def eur_chr2_df():
    """2 chr2 variants × 5 EUR samples (same samples as eur_chr1_df)."""
    samples = [f"MOCK_EUR_P{i:03d}" for i in range(1, 6)]
    base = {
        'snp_list_id': ['chr2:300000:C:G', 'chr2:400000:A:T'],
        'variant_id':  ['2:300000:C:G',    '2:400000:A:T'],
        'chromosome':  ['2', '2'],
        'position':    [300000, 400000],
        'counted_allele': ['G', 'T'],
        'alt_allele':     ['C', 'A'],
        'harmonization_action': ['SWAP', 'AMBIGUOUS'],
        'pgen_a1': ['C', 'A'],
        'pgen_a2': ['G', 'T'],
        'data_type': ['IMPUTED', 'IMPUTED'],
        'source_file': ['/fake/EUR/chr2_EUR_release11.pgen'] * 2,
        'maf_corrected': [False, False],
        'original_alt_af': [0.1, 0.05],
    }
    for j, s in enumerate(samples):
        base[s] = [float((j + 2) % 3), float(j % 2)]
    return pd.DataFrame(base)


# ============================================================
# Sample ID normalization fixtures
# ============================================================

@pytest.fixture
def sample_ids_to_normalize():
    """Pairs of (input_id, expected_normalized_id) covering all normalization branches."""
    return [
        ("0_FAKE_SAMPLE_001",              "FAKE_SAMPLE_001"),   # 0_ prefix strip
        ("FAKE_SAMPLE_001_FAKE_SAMPLE_001", "FAKE_SAMPLE_001"),   # WGS duplicate (4 parts)
        ("TEST_PERSON_042",                 "TEST_PERSON_042"),   # clean passthrough
        ("MOCK_X_001_MOCK_X_001",           "MOCK_X_001"),        # 4-part duplicate
        ("0_TEST_PERSON_999",               "TEST_PERSON_999"),   # 0_ on longer ID
        ("TEST_PERSON_001",                 "TEST_PERSON_001"),   # clean passthrough
    ]


@pytest.fixture
def df_with_mixed_sample_ids():
    """2-variant DataFrame whose sample columns need normalization."""
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
        '0_FAKE_SAMPLE_001':             [0.0, 1.0],
        'FAKE_SAMPLE_002_FAKE_SAMPLE_002': [1.0, 2.0],
        'TEST_PERSON_042':               [2.0, 0.0],
    })


# ============================================================
# Clinical data fixtures (fully invented IDs)
# ============================================================

@pytest.fixture
def sample_master_key():
    """20 rows of fake clinical metadata for testing locus reports.

    GP2IDs are completely invented. Age values chosen so that
    disease durations (age_at_sample - age_of_onset) are exactly
    2, 4, 6, 8, 10 years for EUR P001-P005 to allow exact assertions.
    """
    eur_ids = [f"MOCK_EUR_P{i:03d}" for i in range(1, 11)]
    eas_ids = [f"MOCK_EAS_P{i:03d}" for i in range(1, 11)]

    # EUR: P001-P005 have age data; P006-P010 do not
    eur_age_collection = [62.0, 64.0, 66.0, 68.0, 70.0] + [np.nan] * 5
    eur_age_onset      = [60.0, 60.0, 60.0, 60.0, 60.0] + [np.nan] * 5
    eur_ext_clinical   = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    eas_age_collection = [np.nan] * 10
    eas_age_onset      = [np.nan] * 10
    eas_ext_clinical   = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    return pd.DataFrame({
        'GP2ID':                    eur_ids + eas_ids,
        'nba_label':                ['EUR'] * 10 + ['EAS'] * 10,
        'wgs_label':                ['EUR'] * 10 + ['EAS'] * 10,
        'extended_clinical_data':   eur_ext_clinical + eas_ext_clinical,
        'age_at_sample_collection': eur_age_collection + eas_age_collection,
        'age_of_onset':             eur_age_onset + eas_age_onset,
    })


@pytest.fixture
def sample_extended_clinical():
    """20 rows of fake clinical phenotype data (baseline visits).

    Values chosen so exact assertions pass for EUR carriers:
      H&Y < 2  : P001, P002, P003         (3 carriers)
      H&Y < 3  : P001, P002, P003, P004, P005 (5 carriers)
      MoCA ≥ 20: P002, P003, P004, P005   (4 carriers)
      MoCA ≥ 24: P004, P005               (2 carriers)
    """
    eur_ids = [f"MOCK_EUR_P{i:03d}" for i in range(1, 11)]
    eas_ids = [f"MOCK_EAS_P{i:03d}" for i in range(1, 11)]

    eur_hy    = [1.0, 1.5, 1.9, 2.0, 2.5, np.nan, np.nan, np.nan, np.nan, np.nan]
    eur_moca  = [18.0, 20.0, 22.0, 24.0, 26.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    eur_dat   = [2.1, np.nan, 1.8, np.nan, 2.3, np.nan, np.nan, np.nan, np.nan, np.nan]

    eas_hy   = [np.nan] * 10
    eas_moca = [np.nan] * 10
    eas_dat  = [np.nan] * 10

    return pd.DataFrame({
        'GP2ID':                 eur_ids + eas_ids,
        'visit_month':           [0] * 20,
        'Phenotype':             ['PD'] * 20,
        'hoehn_and_yahr_stage':  eur_hy + eas_hy,
        'moca_total_score':      eur_moca + eas_moca,
        'dat_sbr_caudate_mean':  eur_dat + eas_dat,
    })


@pytest.fixture
def snp_list_with_locus():
    """SNP list with locus annotations for locus report testing."""
    return pd.DataFrame({
        'snp_list_id': [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',
            'chr3:500000:G:A',
        ],
        'gene': ['GENE1', 'GENE1', 'GENE2', 'GENE2', 'GENE3'],
        'hg38': [
            'chr1:100000:A:C',
            'chr1:200000:T:G',
            'chr2:300000:C:G',
            'chr2:400000:A:T',
            'chr3:500000:G:A',
        ],
    })


# ============================================================
# MAF correction test data
# ============================================================

@pytest.fixture
def high_af_genotype_df():
    """2-variant DataFrame for MAF correction tests.

    Variant 1: genotypes [2,2,2,2,2,2,2,1,0,0] → AF = 15/20 = 0.75 → SHOULD be corrected.
    Variant 2: genotypes [1,1,1,0,0,0,0,0,0,0] → AF = 3/20  = 0.15 → should NOT be corrected.
    """
    samples = [f"MOCK_S{i:03d}" for i in range(1, 11)]
    row1_gts = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0, 0.0]
    row2_gts = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    base = {
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
    }
    for i, s in enumerate(samples):
        base[s] = [row1_gts[i], row2_gts[i]]
    return pd.DataFrame(base)


# ============================================================
# PLINK file fixtures (tmp_path-based, no real sample IDs)
# ============================================================

@pytest.fixture
def traw_tmp_file(tmp_path, sample_traw_data):
    """Write sample_traw_data to a tmp TRAW file; return its path."""
    traw_path = tmp_path / "test.traw"
    sample_traw_data.to_csv(str(traw_path), sep='\t', index=False, na_rep='NA')
    return str(traw_path)


@pytest.fixture
def psam_tmp_file(tmp_path):
    """Write a fake PSAM file mapping FAKEFAM001_TESTIND00N → TESTIND00N."""
    lines = ["#FID\tIID\tSEX"]
    for i in range(1, 4):
        lines.append(f"FAKEFAM001\tTESTIND00{i}\t1")
    psam_path = tmp_path / "test.psam"
    psam_path.write_text("\n".join(lines) + "\n")
    return str(psam_path)


# ============================================================
# Multi-probe fixture for probe selection tests
# ============================================================

@pytest.fixture
def multi_probe_genotype_df():
    """4-row DataFrame with 2 probes for one mutation and 2 single-probe mutations."""
    samples = ['MOCK_S001', 'MOCK_S002', 'MOCK_S003']
    return pd.DataFrame({
        'snp_list_id': [
            'chr1:100000:A:C',   # probe 1 of 2
            'chr1:100000:A:C',   # probe 2 of 2 (same snp_list_id)
            'chr2:300000:C:G',   # single probe
            'chr3:500000:G:A',   # single probe
        ],
        'variant_id': [
            '1:100000:A:C',
            '1:100000:A:C_v2',
            '2:300000:C:G',
            '3:500000:G:A',
        ],
        'chromosome': ['1', '1', '2', '3'],
        'position':   [100000, 100000, 300000, 500000],
        'MOCK_S001':  [1.0, 2.0, 0.0, 1.0],
        'MOCK_S002':  [0.0, 0.0, 1.0, 2.0],
        'MOCK_S003':  [2.0, 1.0, 2.0, 0.0],
    })