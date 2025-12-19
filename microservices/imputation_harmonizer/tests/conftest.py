"""Pytest fixtures for imputation_harmonizer tests."""

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_hrc(fixtures_dir: Path) -> Path:
    """Path to minimal HRC reference file."""
    return fixtures_dir / "mini_hrc.tab"


@pytest.fixture
def mini_1000g(fixtures_dir: Path) -> Path:
    """Path to minimal 1000G reference file."""
    return fixtures_dir / "mini_1000g.legend"


@pytest.fixture
def sample_bim(fixtures_dir: Path) -> Path:
    """Path to sample BIM file."""
    return fixtures_dir / "sample.bim"


@pytest.fixture
def sample_frq(fixtures_dir: Path) -> Path:
    """Path to sample FRQ file."""
    return fixtures_dir / "sample.frq"


@pytest.fixture
def sample_files(tmp_path: Path) -> dict[str, Path]:
    """Create minimal test files for integration testing.

    Test cases:
    - rs123: Exact match (A/G in both), freq matches -> no action
    - rs456: Exact match (C/T in both), freq matches -> no action
    - rs789: Palindromic A/T with MAF 0.45 > 0.4 -> exclude
    - rs101: Strand flip needed (T/C in BIM is complement of A/G in ref)
    - rs102: Ref/alt swapped (G/A in BIM vs A/G in ref) -> force allele
    """
    # Create sample BIM file
    bim = tmp_path / "test.bim"
    bim.write_text(
        "1\trs123\t0\t10000\tA\tG\n"  # Match: A/G with ref A/G
        "1\trs456\t0\t20000\tC\tT\n"  # Match: C/T with ref C/T
        "1\trs789\t0\t30000\tA\tT\n"  # Palindromic, high MAF
        "1\trs101\t0\t40000\tT\tC\n"  # Flip: T/C is complement of A/G
        "1\trs102\t0\t50000\tG\tA\n"  # Swap: G/A is swap of A/G
    )

    # Create sample FRQ file
    # For case 1 (strand OK, ref/alt OK): bim_af should be close to ref_freq = 1 - alt_af
    # For case 2 (strand OK, swapped): bim_af should be close to alt_af
    # For case 3 (flipped, OK): bim_af should be close to ref_freq = 1 - alt_af
    # For case 4 (flipped, swapped): bim_af should be close to alt_af
    frq = tmp_path / "test.frq"
    frq.write_text(
        "CHR\tSNP\tA1\tA2\tMAF\tNCHROBS\n"
        "1\trs123\tA\tG\t0.70\t1000\n"  # ref_freq = 1 - 0.30 = 0.70
        "1\trs456\tC\tT\t0.75\t1000\n"  # ref_freq = 1 - 0.25 = 0.75
        "1\trs789\tA\tT\t0.45\t1000\n"  # High MAF palindromic (excluded anyway)
        "1\trs101\tT\tC\t0.72\t1000\n"  # Flip case: ref_freq = 1 - 0.28 = 0.72
        "1\trs102\tG\tA\t0.30\t1000\n"  # Swap case: compare to alt_af = 0.30
    )

    # Create sample HRC reference file
    ref = tmp_path / "ref.tab"
    ref.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tAC\tAN\tAF\n"
        "1\t10000\trs123\tA\tG\t300\t1000\t0.30\n"
        "1\t20000\trs456\tC\tT\t250\t1000\t0.25\n"
        "1\t30000\trs789\tA\tT\t450\t1000\t0.45\n"  # High MAF palindromic
        "1\t40000\trs101\tA\tG\t280\t1000\t0.28\n"  # T/C -> A/G (complement)
        "1\t50000\trs102\tA\tG\t300\t1000\t0.30\n"  # G/A -> A/G (swapped)
    )

    return {"bim": bim, "frq": frq, "ref": ref, "dir": tmp_path}
