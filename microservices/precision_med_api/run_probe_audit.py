#!/usr/bin/env python3
"""
Run full NBA probe audit against WGS ground truth.

Validates ALL NBA probes (not just SNP list variants) against WGS data
to identify probe quality issues and cross-ancestry discrepancies.

Usage:
    python run_probe_audit.py --release 11 --ancestries EUR AFR
    python run_probe_audit.py --release 11  # All ancestries
"""

import sys
import os
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import Settings
from app.core.logging_config import setup_logging, get_progress_logger, get_log_file_path
from app.processing.probe_auditor import ProbeAuditor, compare_across_ancestries


def audit_single_ancestry(args: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
    """
    Audit a single ancestry (worker function for parallel execution).

    Args:
        args: Tuple of (ancestry, release)

    Returns:
        Tuple of (ancestry, result_dict)
    """
    ancestry, release = args

    try:
        # Each worker creates its own Settings and Auditor
        settings = Settings(release=release)
        auditor = ProbeAuditor(settings)

        result = auditor.run_audit(ancestry)
        return (ancestry, result)

    except Exception as e:
        import traceback
        return (ancestry, {
            'ancestry': ancestry,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def parse_args():
    """Parse command line arguments."""
    default_settings = Settings()
    all_ancestries = default_settings.ANCESTRIES

    parser = argparse.ArgumentParser(
        description='Run full NBA probe audit against WGS ground truth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Audit all ancestries for release 11
    python run_probe_audit.py --release 11

    # Audit specific ancestries
    python run_probe_audit.py --release 11 --ancestries EUR AFR AAC

    # Custom output location
    python run_probe_audit.py --release 11 --output /path/to/audit_results
        """
    )

    parser.add_argument(
        '--release',
        type=str,
        required=True,
        help='GP2 release version (required, e.g., 11)'
    )
    parser.add_argument(
        '--ancestries',
        type=str,
        nargs='+',
        default=all_ancestries,
        help=f'Ancestries to audit (default: all {len(all_ancestries)} ancestries from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for audit results (default: config-based results directory)'
    )
    parser.add_argument(
        '--job-name',
        type=str,
        default='probe_audit',
        help='Job name for output files (default: probe_audit)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum workers for parallel processing (default: auto-detect based on CPU cores)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing across ancestries (default: True)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        default=False,
        help='Disable parallel processing (run sequentially)'
    )

    return parser.parse_args()


def save_audit_report(report: Dict[str, Any], output_path: str) -> str:
    """Save audit report to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    return output_path


def main():
    args = parse_args()

    # Initialize settings
    settings = Settings(release=args.release)

    # Determine output directory
    if args.output is None:
        output_dir = settings.results_path
    else:
        output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = setup_logging(log_dir=output_dir, job_name=args.job_name)
    logger = logging.getLogger(__name__)
    progress = get_progress_logger()

    try:
        # Determine parallelization settings
        use_parallel = args.parallel and not args.no_parallel
        max_workers = args.max_workers
        if max_workers is None:
            # Use up to 2 workers by default (conservative for memory)
            # Each worker can use ~20-30GB memory at peak
            max_workers = min(2, len(args.ancestries), multiprocessing.cpu_count())

        progress.info(f"NBA Probe Audit - Release {args.release}")
        progress.info(f"Ancestries: {', '.join(args.ancestries)}")
        progress.info(f"Output: {output_dir}")
        progress.info(f"Parallel: {use_parallel} (max_workers={max_workers})")

        start_time = time.time()

        # Run audit for each ancestry
        ancestry_results = {}
        total_multi_probe = 0

        if use_parallel and len(args.ancestries) > 1:
            # Parallel execution using ProcessPoolExecutor
            progress.info(f"\nStarting parallel audit with {max_workers} workers...")

            # Prepare arguments for workers
            worker_args = [(ancestry, args.release) for ancestry in args.ancestries]

            completed = 0
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all jobs
                future_to_ancestry = {
                    executor.submit(audit_single_ancestry, arg): arg[0]
                    for arg in worker_args
                }

                # Collect results as they complete
                for future in as_completed(future_to_ancestry):
                    ancestry = future_to_ancestry[future]
                    completed += 1

                    try:
                        anc, result = future.result()
                        ancestry_results[anc] = result

                        if result['success']:
                            multi_probe_count = len(result.get('multi_probe_mutations', {}))
                            total_multi_probe += multi_probe_count
                            progress.info(f"  [{completed}/{len(args.ancestries)}] {anc}: "
                                          f"{result['shared_positions']} shared, "
                                          f"{multi_probe_count} multi-probe mutations")
                        else:
                            progress.warning(f"  [{completed}/{len(args.ancestries)}] {anc}: "
                                             f"FAILED - {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        logger.error(f"Worker failed for {ancestry}: {e}")
                        ancestry_results[ancestry] = {
                            'ancestry': ancestry,
                            'success': False,
                            'error': str(e)
                        }

        else:
            # Sequential execution
            auditor = ProbeAuditor(settings)

            for i, ancestry in enumerate(args.ancestries, 1):
                progress.info(f"\n[{i}/{len(args.ancestries)}] Auditing {ancestry}...")
                ancestry_start = time.time()

                try:
                    result = auditor.run_audit(ancestry)
                    ancestry_results[ancestry] = result

                    if result['success']:
                        multi_probe_count = len(result.get('multi_probe_mutations', {}))
                        total_multi_probe += multi_probe_count
                        elapsed = time.time() - ancestry_start

                        progress.info(f"  {ancestry}: {result['shared_positions']} shared positions, "
                                      f"{multi_probe_count} multi-probe mutations ({elapsed:.1f}s)")
                    else:
                        progress.warning(f"  {ancestry}: FAILED - {result.get('error', 'Unknown error')}")

                except Exception as e:
                    logger.error(f"Failed to audit {ancestry}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    ancestry_results[ancestry] = {
                        'ancestry': ancestry,
                        'success': False,
                        'error': str(e)
                    }

        # Cross-ancestry comparison
        progress.info("\nComparing probe recommendations across ancestries...")
        cross_ancestry = compare_across_ancestries(ancestry_results)

        discrepant_count = cross_ancestry.get('discrepant_probes_count', 0)
        concordant_count = cross_ancestry.get('concordant_probes_count', 0)

        if discrepant_count > 0:
            progress.info(f"  Cross-ancestry discrepancies: {discrepant_count} probes")
        progress.info(f"  Concordant across ancestries: {concordant_count} probes")

        # Build final report
        end_time = time.time()
        elapsed_total = end_time - start_time

        report = {
            'job_id': args.job_name,
            'release': args.release,
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': elapsed_total,
            'summary': {
                'ancestries_audited': len([r for r in ancestry_results.values() if r.get('success')]),
                'total_multi_probe_mutations': total_multi_probe,
                'cross_ancestry_discrepancies': discrepant_count,
                'cross_ancestry_concordant': concordant_count
            },
            'by_ancestry': ancestry_results,
            'cross_ancestry_analysis': cross_ancestry
        }

        # Save report
        report_filename = f"{args.job_name}_release{args.release}.json"
        report_path = os.path.join(output_dir, report_filename)
        save_audit_report(report, report_path)

        # Print summary
        minutes = int(elapsed_total // 60)
        seconds = int(elapsed_total % 60)

        progress.info(f"\nProbe audit complete in {minutes}m {seconds}s")
        progress.info(f"Report saved: {report_path}")

        # Show discrepancies if any
        if discrepant_count > 0:
            progress.info(f"\n=== Cross-Ancestry Discrepancies ({discrepant_count}) ===")
            for disc in cross_ancestry.get('discrepant_probes', [])[:10]:  # Show first 10
                mutation_id = disc['mutation_id']
                recs = disc['recommendations_by_ancestry']
                rec_summary = ", ".join([f"{anc}:{r['best_probe']}" for anc, r in recs.items()])
                progress.info(f"  {mutation_id}: {rec_summary}")

            if discrepant_count > 10:
                progress.info(f"  ... and {discrepant_count - 10} more (see full report)")

        # Show log file location
        log_path = get_log_file_path()
        if log_path:
            progress.info(f"\nDetailed logs: {log_path}")

        return 0

    except Exception as e:
        progress.error(f"Probe audit failed: {e}")
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
