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
        'counted_allele': ['A', 'T', 'C', 'A', 'G'],
        'alt_allele': ['C', 'G', 'G', 'T', 'A'],
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
        'COUNTED': ['A', 'T', 'C'],
        'ALT': ['C', 'G', 'G'],
        'SAMPLE_000001': [0, 1, 2],
        'SAMPLE_000002': [1, 0, 1],
        'SAMPLE_000003': [2, 2, 0]
    })