#!/usr/bin/env python3
"""
SNP List Coverage Profiling Script (Parallelized)

Profiles variant coverage across all data types (WGS, IMPUTED, NBA, EXOMES)
by merging pvar files with the SNP list on chromosome and position.

Optimized for parallel execution on multi-core machines (e.g., e2-standard-32).

Usage:
    python profile_snp_coverage.py
    python profile_snp_coverage.py --output-dir ./coverage_reports/
    python profile_snp_coverage.py --workers 16  # Limit workers

Can also be imported and used programmatically:
    from profile_snp_coverage import run_coverage_profiling
    results = run_coverage_profiling(snp_list_path, output_dir, settings)
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Optional, Any
import multiprocessing as mp
import time
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CoverageResult:
    """Coverage results for a single data type"""
    data_type: str
    exact_matches: Set[str]  # Set of variant_ids that matched exactly
    position_matches: Set[str]  # Set of variant_ids that matched by position


@dataclass
class LocusCoverage:
    """Coverage statistics for a single locus"""
    locus: str
    chromosome: str
    total_variants: int
    wgs_exact: int
    wgs_position: int
    imputed_exact: int
    imputed_position: int
    nba_exact: int
    nba_position: int
    exomes_exact: int
    exomes_position: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed percentages"""
        d = asdict(self)
        total = self.total_variants
        if total > 0:
            d['wgs_exact_pct'] = self.wgs_exact / total * 100
            d['wgs_position_pct'] = self.wgs_position / total * 100
            d['imputed_exact_pct'] = self.imputed_exact / total * 100
            d['imputed_position_pct'] = self.imputed_position / total * 100
            d['nba_exact_pct'] = self.nba_exact / total * 100
            d['nba_position_pct'] = self.nba_position / total * 100
            d['exomes_exact_pct'] = self.exomes_exact / total * 100
            d['exomes_position_pct'] = self.exomes_position / total * 100
        else:
            for dt in ['wgs', 'imputed', 'nba', 'exomes']:
                d[f'{dt}_exact_pct'] = 0.0
                d[f'{dt}_position_pct'] = 0.0
        return d


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_chr(chrom) -> str:
    """Normalize chromosome format to 'chrN'"""
    chrom_str = str(chrom)
    if chrom_str.startswith('chr'):
        return chrom_str
    return f"chr{chrom_str}"


def read_pvar(path: Path) -> pd.DataFrame:
    """Read pvar file, skipping ## header lines"""
    with open(path, 'r') as f:
        header_lines = sum(1 for line in f if line.startswith('##'))
    return pd.read_csv(path, sep='\t', skiprows=header_lines, low_memory=False)


def load_and_normalize_snp_list(snp_list_path: str) -> pd.DataFrame:
    """Load and normalize the SNP list for coverage profiling"""
    logger.info(f"Loading SNP list from {snp_list_path}")
    snp_list = pd.read_csv(snp_list_path)

    # Filter to variants with hg38 coordinates
    snp_list = snp_list[snp_list['hg38'].notna()].copy()

    # Build normalized variant_id (chr:pos:ref:alt)
    snp_list['variant_id'] = snp_list['hg38'].apply(
        lambda x: f"chr{x}" if not str(x).startswith('chr') else x
    )

    # Extract chromosome and position
    snp_list['chr'] = snp_list['variant_id'].str.split(':').str[0]
    snp_list['pos'] = snp_list['variant_id'].str.split(':').str[1]

    # Create position key for position-only matching
    snp_list['pos_key'] = snp_list['chr'] + ':' + snp_list['pos']

    logger.info(f"  Loaded {len(snp_list)} variants across {snp_list['locus'].nunique()} loci")
    return snp_list


def get_pvar_path(
    base_path: Path,
    data_type: str,
    ancestry: str,
    chrom: str,
    release: str = "11"
) -> Optional[Path]:
    """Get the pvar file path for a given data type, ancestry, and chromosome"""
    # Data type paths (relative to base_path)
    data_paths = {
        "WGS": "wgs/deepvariant_joint_calling/plink",
        "IMPUTED": "imputed_genotypes",
        "NBA": "raw_genotypes",
        "EXOMES": "clinical_exomes/plink",
    }

    base = base_path / data_paths.get(data_type, "")

    if data_type == "WGS":
        path = base / ancestry / f"{chrom}_{ancestry}_release{release}.pvar"
    elif data_type == "IMPUTED":
        path = base / ancestry / f"{chrom}_{ancestry}_release{release}_vwb.pvar"
    elif data_type == "NBA":
        path = base / ancestry / f"{ancestry}_release{release}_vwb.pvar"
    elif data_type == "EXOMES":
        path = base / f"{chrom}.pvar"
    else:
        return None

    return path if path.exists() else None


# =============================================================================
# Parallel Processing Functions
# =============================================================================

# Module-level variables for worker processes (set by initializer)
_worker_base_path = None
_worker_release = None


def _init_worker(base_path: str, release: str):
    """Initialize worker process with shared configuration"""
    global _worker_base_path, _worker_release
    _worker_base_path = Path(base_path)
    _worker_release = release


def check_single_file(args: Tuple) -> Tuple[str, str, Set[str], Set[str]]:
    """
    Worker function to check a single pvar file.

    Args:
        args: Tuple of (data_type, chrom, snp_variants, snp_positions, snp_pos_to_variants, ancestry)

    Returns:
        Tuple of (data_type, chrom, exact_matches, position_matches)
    """
    data_type, chrom, snp_variants, snp_positions, snp_pos_to_variants, ancestry = args

    pvar_path = get_pvar_path(_worker_base_path, data_type, ancestry, chrom, _worker_release)

    if pvar_path is None:
        return (data_type, chrom, set(), set())

    try:
        pvar = read_pvar(pvar_path)
    except Exception as e:
        return (data_type, chrom, set(), set())

    # Normalize chromosome format in pvar
    pvar['chr'] = pvar['#CHROM'].apply(normalize_chr)

    # Filter to target chromosome (for NBA which has all chromosomes)
    if data_type == "NBA":
        pvar = pvar[pvar['chr'] == chrom]

    if len(pvar) == 0:
        return (data_type, chrom, set(), set())

    # Create variant_id (chr:pos:ref:alt) and pos_key (chr:pos)
    pvar['variant_id'] = pvar.apply(
        lambda r: f"{r['chr']}:{r['POS']}:{r['REF']}:{r['ALT']}", axis=1
    )
    pvar['pos_key'] = pvar['chr'] + ':' + pvar['POS'].astype(str)

    # Find matches
    pvar_variants = set(pvar['variant_id'])
    pvar_positions = set(pvar['pos_key'])

    # Exact matches
    exact_match_ids = pvar_variants & snp_variants

    # Position matches - get variant_ids for positions that exist
    position_match_positions = pvar_positions & snp_positions
    position_match_ids = set()
    for pos in position_match_positions:
        if pos in snp_pos_to_variants:
            position_match_ids.update(snp_pos_to_variants[pos])

    return (data_type, chrom, exact_match_ids, position_match_ids)


def profile_all_parallel(
    snp_list: pd.DataFrame,
    base_path: Path,
    ancestry: str,
    max_workers: int,
    release: str = "11",
    data_types: List[str] = None
) -> Dict[str, CoverageResult]:
    """Profile all data types in parallel"""
    if data_types is None:
        data_types = ['WGS', 'IMPUTED', 'NBA', 'EXOMES']

    logger.info(f"Profiling {len(data_types)} data types in parallel with {max_workers} workers...")
    start_time = time.time()

    # Prepare SNP list data per chromosome for workers
    chromosomes = sorted(snp_list['chr'].unique())

    # Build tasks list - all (data_type, chrom) combinations
    tasks = []

    for data_type in data_types:
        for chrom in chromosomes:
            snp_list_chr = snp_list[snp_list['chr'] == chrom]
            if len(snp_list_chr) == 0:
                continue

            # Prepare serializable data for workers
            snp_variants = set(snp_list_chr['variant_id'])
            snp_positions = set(snp_list_chr['pos_key'])

            # Build pos_key -> variant_ids mapping
            snp_pos_to_variants = {}
            for _, row in snp_list_chr.iterrows():
                pos_key = row['pos_key']
                if pos_key not in snp_pos_to_variants:
                    snp_pos_to_variants[pos_key] = []
                snp_pos_to_variants[pos_key].append(row['variant_id'])

            tasks.append((
                data_type, chrom, snp_variants, snp_positions,
                snp_pos_to_variants, ancestry
            ))

    logger.info(f"  Created {len(tasks)} tasks across {len(data_types)} data types")

    # Results storage
    results = {dt: {'exact': set(), 'position': set()} for dt in data_types}
    completed = 0

    # Run in parallel with initializer
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(str(base_path), release)
    ) as executor:
        futures = {executor.submit(check_single_file, task): task for task in tasks}

        for future in as_completed(futures):
            data_type, chrom, exact, position = future.result()
            results[data_type]['exact'].update(exact)
            results[data_type]['position'].update(position)

            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                elapsed = time.time() - start_time
                logger.info(f"  Progress: {completed}/{len(tasks)} tasks ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    logger.info(f"Completed all profiling in {elapsed:.1f} seconds")

    # Convert to CoverageResult objects
    coverage_results = {}
    for data_type in data_types:
        exact_count = len(results[data_type]['exact'])
        position_count = len(results[data_type]['position'])
        logger.info(f"  {data_type}: {exact_count} exact, {position_count} position matches")

        coverage_results[data_type] = CoverageResult(
            data_type=data_type,
            exact_matches=results[data_type]['exact'],
            position_matches=results[data_type]['position']
        )

    return coverage_results


# =============================================================================
# Aggregation and Reporting Functions
# =============================================================================

def aggregate_by_locus(
    snp_list: pd.DataFrame,
    coverage_results: Dict[str, CoverageResult]
) -> List[LocusCoverage]:
    """Aggregate coverage statistics by locus"""
    locus_stats = []

    # Get available data types from results
    available_types = set(coverage_results.keys())

    for locus in sorted(snp_list['locus'].unique()):
        locus_variants = snp_list[snp_list['locus'] == locus]
        locus_variant_ids = set(locus_variants['variant_id'])
        locus_chr = locus_variants['chr'].iloc[0]

        def get_count(dt, match_type):
            if dt not in available_types:
                return 0
            matches = getattr(coverage_results[dt], f'{match_type}_matches')
            return len(locus_variant_ids & matches)

        stats = LocusCoverage(
            locus=locus,
            chromosome=locus_chr,
            total_variants=len(locus_variant_ids),
            wgs_exact=get_count('WGS', 'exact'),
            wgs_position=get_count('WGS', 'position'),
            imputed_exact=get_count('IMPUTED', 'exact'),
            imputed_position=get_count('IMPUTED', 'position'),
            nba_exact=get_count('NBA', 'exact'),
            nba_position=get_count('NBA', 'position'),
            exomes_exact=get_count('EXOMES', 'exact'),
            exomes_position=get_count('EXOMES', 'position'),
        )
        locus_stats.append(stats)

    return locus_stats


def create_variant_coverage_df(
    snp_list: pd.DataFrame,
    coverage_results: Dict[str, CoverageResult]
) -> pd.DataFrame:
    """Create per-variant coverage DataFrame"""
    df = snp_list[['variant_id', 'snp_name', 'locus', 'chr', 'pos', 'rsid']].copy()

    for data_type, result in coverage_results.items():
        df[f'{data_type}_exact'] = df['variant_id'].isin(result.exact_matches)
        df[f'{data_type}_position'] = df['variant_id'].isin(result.position_matches)

    return df


def create_summary_dict(
    locus_stats: List[LocusCoverage],
    coverage_results: Dict[str, CoverageResult]
) -> Dict[str, Any]:
    """Create summary dictionary for JSON output"""
    total_variants = sum(s.total_variants for s in locus_stats)

    summary = {
        'total_variants': total_variants,
        'total_loci': len(locus_stats),
        'by_data_type': {}
    }

    for data_type in ['WGS', 'IMPUTED', 'NBA', 'EXOMES']:
        if data_type in coverage_results:
            result = coverage_results[data_type]
            exact = len(result.exact_matches)
            position = len(result.position_matches)
            summary['by_data_type'][data_type] = {
                'exact_matches': exact,
                'position_matches': position,
                'exact_pct': exact / total_variants * 100 if total_variants > 0 else 0,
                'position_pct': position / total_variants * 100 if total_variants > 0 else 0,
            }

    return summary


def save_outputs(
    locus_stats: List[LocusCoverage],
    variant_df: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Path,
    output_name: str = None
) -> Dict[str, str]:
    """Save coverage reports to files. Returns dict of output file paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{output_name}_" if output_name else ""

    # Locus coverage CSV
    locus_df = pd.DataFrame([s.to_dict() for s in locus_stats])
    locus_path = output_dir / f"{prefix}coverage_by_locus.csv"
    locus_df.to_csv(locus_path, index=False)
    logger.info(f"Saved locus coverage to: {locus_path}")

    # Variant coverage CSV
    variant_path = output_dir / f"{prefix}coverage_by_variant.csv"
    variant_df.to_csv(variant_path, index=False)
    logger.info(f"Saved variant coverage to: {variant_path}")

    # Summary JSON
    summary_path = output_dir / f"{prefix}coverage_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved coverage summary to: {summary_path}")

    return {
        'coverage_by_locus': str(locus_path),
        'coverage_by_variant': str(variant_path),
        'coverage_summary': str(summary_path),
    }


# =============================================================================
# Main API Function (for pipeline integration)
# =============================================================================

def run_coverage_profiling(
    snp_list_path: str,
    output_dir: str,
    base_path: str,
    release: str = "11",
    ancestry: str = "EUR",
    max_workers: int = None,
    output_name: str = None,
    data_types: List[str] = None
) -> Optional[Dict[str, str]]:
    """
    Run coverage profiling and return output file paths.

    This is the main entry point for pipeline integration.

    Args:
        snp_list_path: Path to SNP list CSV
        output_dir: Directory for output files
        base_path: Base path to release data (e.g., /path/to/release11)
        release: Release version string (e.g., "11")
        ancestry: Ancestry to check (default: EUR)
        max_workers: Number of parallel workers (default: CPU count)
        output_name: Prefix for output files (optional)
        data_types: List of data types to check (default: all four)

    Returns:
        Dictionary mapping output types to file paths, or None if failed
    """
    try:
        # Determine workers
        if max_workers is None:
            max_workers = mp.cpu_count()

        logger.info(f"Starting coverage profiling with {max_workers} workers")

        # Load SNP list
        snp_list = load_and_normalize_snp_list(snp_list_path)

        # Profile all data types
        coverage_results = profile_all_parallel(
            snp_list=snp_list,
            base_path=Path(base_path),
            ancestry=ancestry,
            max_workers=max_workers,
            release=release,
            data_types=data_types
        )

        # Aggregate by locus
        locus_stats = aggregate_by_locus(snp_list, coverage_results)

        # Create variant-level DataFrame
        variant_df = create_variant_coverage_df(snp_list, coverage_results)

        # Create summary
        summary = create_summary_dict(locus_stats, coverage_results)

        # Save outputs
        output_files = save_outputs(
            locus_stats=locus_stats,
            variant_df=variant_df,
            summary=summary,
            output_dir=Path(output_dir),
            output_name=output_name
        )

        logger.info("Coverage profiling completed successfully")
        return output_files

    except Exception as e:
        logger.error(f"Coverage profiling failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# =============================================================================
# CLI Functions
# =============================================================================

def print_locus_report(locus_stats: List[LocusCoverage]):
    """Print formatted locus coverage report to console"""
    print("\n" + "=" * 80)
    print("LOCUS COVERAGE REPORT")
    print("=" * 80)

    # Sort by total variants descending
    sorted_stats = sorted(locus_stats, key=lambda x: x.total_variants, reverse=True)

    for stats in sorted_stats:
        print(f"\n{stats.locus} ({stats.chromosome}, {stats.total_variants} variants)")

        def fmt_pct(n, total):
            pct = (n / total * 100) if total > 0 else 0
            return f"{n:3d}/{total:3d} ({pct:5.1f}%)"

        print(f"  WGS:     {fmt_pct(stats.wgs_exact, stats.total_variants)} exact, "
              f"{fmt_pct(stats.wgs_position, stats.total_variants)} position")
        print(f"  IMPUTED: {fmt_pct(stats.imputed_exact, stats.total_variants)} exact, "
              f"{fmt_pct(stats.imputed_position, stats.total_variants)} position")
        print(f"  NBA:     {fmt_pct(stats.nba_exact, stats.total_variants)} exact, "
              f"{fmt_pct(stats.nba_position, stats.total_variants)} position")
        print(f"  EXOMES:  {fmt_pct(stats.exomes_exact, stats.total_variants)} exact, "
              f"{fmt_pct(stats.exomes_position, stats.total_variants)} position")


def print_summary_table(locus_stats: List[LocusCoverage]):
    """Print summary table of coverage by data type"""
    print("\n" + "=" * 80)
    print("SUMMARY BY DATA TYPE")
    print("=" * 80)

    total_variants = sum(s.total_variants for s in locus_stats)

    for data_type in ['WGS', 'IMPUTED', 'NBA', 'EXOMES']:
        exact = sum(getattr(s, f'{data_type.lower()}_exact') for s in locus_stats)
        position = sum(getattr(s, f'{data_type.lower()}_position') for s in locus_stats)

        exact_pct = (exact / total_variants * 100) if total_variants > 0 else 0
        pos_pct = (position / total_variants * 100) if total_variants > 0 else 0

        print(f"{data_type:8s}: {exact:4d}/{total_variants} ({exact_pct:5.1f}%) exact, "
              f"{position:4d}/{total_variants} ({pos_pct:5.1f}%) position")


def main():
    parser = argparse.ArgumentParser(
        description="Profile SNP list coverage across data types (parallelized)"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('.'),
        help='Output directory for CSV reports (default: current directory)'
    )
    parser.add_argument(
        '--base-path', '-b',
        type=Path,
        default=Path("/home/vitaled2/gcs_mounts/gp2tier2_vwb/release11"),
        help='Base path to release data'
    )
    parser.add_argument(
        '--snp-list', '-s',
        type=Path,
        default=Path("/home/vitaled2/gcs_mounts/genotools_server/precision_med/summary_data/precision_med_snp_list.csv"),
        help='Path to SNP list CSV'
    )
    parser.add_argument(
        '--release', '-r',
        type=str,
        default="11",
        help='Release version (default: 11)'
    )
    parser.add_argument(
        '--ancestry', '-a',
        type=str,
        default="EUR",
        help='Ancestry to check (default: EUR)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--output-name', '-n',
        type=str,
        default=None,
        help='Prefix for output files (optional)'
    )
    args = parser.parse_args()

    # Setup logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Determine number of workers
    max_workers = args.workers or mp.cpu_count()
    print(f"Using {max_workers} parallel workers")

    # Load SNP list
    snp_list = load_and_normalize_snp_list(str(args.snp_list))

    # Profile all data types in parallel
    coverage_results = profile_all_parallel(
        snp_list=snp_list,
        base_path=args.base_path,
        ancestry=args.ancestry,
        max_workers=max_workers,
        release=args.release
    )

    # Aggregate by locus
    locus_stats = aggregate_by_locus(snp_list, coverage_results)

    # Create variant-level DataFrame
    variant_df = create_variant_coverage_df(snp_list, coverage_results)

    # Create summary
    summary = create_summary_dict(locus_stats, coverage_results)

    # Print reports to console
    print_summary_table(locus_stats)
    print_locus_report(locus_stats)

    # Save outputs
    save_outputs(
        locus_stats=locus_stats,
        variant_df=variant_df,
        summary=summary,
        output_dir=args.output_dir,
        output_name=args.output_name
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
