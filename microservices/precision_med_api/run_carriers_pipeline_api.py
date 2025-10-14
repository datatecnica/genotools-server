#!/usr/bin/env python3
"""
Run carriers pipeline via API endpoint.

This script provides the same interface as run_carriers_pipeline.py but
executes the pipeline through the REST API instead of directly calling
the coordinator. Requires the API server to be running (python start_api.py).

Quick Start:
    # Terminal 1: Start API server
    python start_api.py

    # Terminal 2: Submit jobs via API
    python run_carriers_pipeline_api.py --ancestries AAC AFR
    python run_carriers_pipeline_api.py --job-name my_analysis
    python run_carriers_pipeline_api.py --skip-extraction

Benefits over Direct CLI:
    ✅ Remote execution without SSH
    ✅ Job tracking with unique IDs
    ✅ Non-blocking submission (--no-follow)
    ✅ Multiple concurrent jobs
    ✅ API server can run on different machine
"""

import sys
import argparse
import time
import logging
import requests
from typing import Optional, Dict, Any
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import Settings


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments (matches CLI script interface)."""
    # Get default ancestries from settings
    default_settings = Settings()
    all_ancestries = default_settings.ANCESTRIES

    parser = argparse.ArgumentParser(
        description='Run carriers pipeline via API endpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation (2 ancestries)
    python run_carriers_pipeline_via_api.py --ancestries AAC AFR

    # Full pipeline with all ancestries
    python run_carriers_pipeline_via_api.py --job-name full_analysis

    # Skip extraction for rapid postprocessing
    python run_carriers_pipeline_via_api.py --job-name existing --skip-extraction

    # Custom API host/port
    python run_carriers_pipeline_via_api.py --api-host localhost --api-port 8001
        """
    )

    # Pipeline arguments (match CLI script)
    parser.add_argument(
        '--job-name',
        type=str,
        default='carriers_analysis',
        help='Job name for output files (default: carriers_analysis)'
    )
    parser.add_argument(
        '--ancestries',
        type=str,
        nargs='+',
        default=None,  # None = use all ancestries
        help=f'Ancestries to process (default: all {len(all_ancestries)} ancestries)'
    )
    parser.add_argument(
        '--data-types',
        type=str,
        nargs='+',
        choices=['NBA', 'WGS', 'IMPUTED'],
        default=['NBA', 'WGS', 'IMPUTED'],
        help='Data types to process (default: all)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=True,
        help='Enable parallel processing (default: True)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_false',
        dest='parallel',
        help='Disable parallel processing'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Maximum workers (default: auto-detect)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        default=True,
        help='Use performance optimizations (default: True)'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_false',
        dest='optimize',
        help='Disable performance optimizations'
    )
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        default=False,
        help='Skip extraction phase if results already exist (default: False)'
    )
    parser.add_argument(
        '--skip-probe-selection',
        action='store_true',
        default=False,
        help='Skip probe selection phase (default: False)'
    )
    parser.add_argument(
        '--skip-locus-reports',
        action='store_true',
        default=False,
        help='Skip locus report generation (default: False)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory (default: config-based results path)'
    )

    # API-specific arguments
    parser.add_argument(
        '--api-host',
        type=str,
        default='localhost',
        help='API server host (default: localhost)'
    )
    parser.add_argument(
        '--api-port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=5,
        help='Status polling interval in seconds (default: 5)'
    )
    parser.add_argument(
        '--max-wait',
        type=int,
        default=3600,
        help='Maximum wait time in seconds (default: 3600 = 1 hour)'
    )
    parser.add_argument(
        '--no-follow',
        action='store_true',
        help='Submit job and exit without waiting for completion'
    )

    return parser.parse_args()


def check_api_health(api_base_url: str) -> bool:
    """Check if API server is healthy."""
    try:
        response = requests.get(f"{api_base_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            logger.info(f"✅ API server is healthy (version {health.get('version', 'unknown')})")
            return True
        else:
            logger.error(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to API server: {e}")
        logger.error(f"   Make sure the API server is running: python start_api.py")
        return False


def submit_pipeline_job(api_base_url: str, args) -> Optional[Dict[str, Any]]:
    """Submit pipeline job to API."""
    # Build request payload
    request_data = {
        "job_name": args.job_name,
        "ancestries": args.ancestries,  # None = all ancestries
        "data_types": args.data_types,
        "parallel": args.parallel,
        "max_workers": args.max_workers,
        "optimize": args.optimize,
        "skip_extraction": args.skip_extraction,
        "skip_probe_selection": args.skip_probe_selection,
        "skip_locus_reports": args.skip_locus_reports,
        "output_dir": args.output_dir
    }

    logger.info("=== Pipeline Configuration ===")
    logger.info(f"📋 Job name: {args.job_name}")
    logger.info(f"📊 Data types: {args.data_types}")
    logger.info(f"🌍 Ancestries: {args.ancestries or 'all'}")
    logger.info(f"⚡ Parallel: {args.parallel}")
    logger.info(f"👥 Max workers: {args.max_workers or 'auto-detect'}")
    logger.info(f"🚀 Optimize: {args.optimize}")
    logger.info(f"📋 Skip extraction: {args.skip_extraction}")
    logger.info(f"🔬 Skip probe selection: {args.skip_probe_selection}")
    logger.info(f"📊 Skip locus reports: {args.skip_locus_reports}")
    logger.info(f"📁 Output dir: {args.output_dir or 'config-based'}")
    logger.info("")

    try:
        logger.info("🚀 Submitting pipeline job to API...")
        response = requests.post(
            f"{api_base_url}/pipeline",
            json=request_data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info(f"✅ Job submitted successfully!")
                logger.info(f"📋 Job ID: {result['job_id']}")
                logger.info(f"📊 Status: {result['status']}")
                logger.info(f"💬 Message: {result.get('message', 'N/A')}")
                return result
            else:
                logger.error(f"❌ Job submission failed: {result.get('message', 'Unknown error')}")
                return None
        else:
            logger.error(f"❌ API request failed: {response.status_code}")
            logger.error(f"   Response: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to submit job: {e}")
        return None


def poll_job_status(api_base_url: str, job_id: str, poll_interval: int, max_wait: int):
    """Poll job status until completion or timeout."""
    max_polls = max_wait // poll_interval
    start_time = time.time()

    logger.info("")
    logger.info("⏳ Waiting for job completion...")
    logger.info(f"   Polling every {poll_interval}s (max {max_wait}s)")
    logger.info("")

    for poll_count in range(1, max_polls + 1):
        try:
            response = requests.get(
                f"{api_base_url}/pipeline/{job_id}",
                timeout=10
            )

            if response.status_code != 200:
                logger.error(f"❌ Status check failed: {response.status_code}")
                return None

            status_data = response.json()
            status = status_data.get("status")
            progress = status_data.get("progress", "N/A")

            elapsed = time.time() - start_time
            logger.info(f"[{poll_count:3d}/{max_polls}] Status: {status:12s} | Elapsed: {elapsed:6.1f}s | {progress}")

            # Check if completed or failed
            if status == "completed":
                logger.info("")
                logger.info("✅ Job completed successfully!")

                # Fetch full results
                try:
                    results_response = requests.get(
                        f"{api_base_url}/pipeline/{job_id}/results",
                        timeout=10
                    )
                    if results_response.status_code == 200:
                        return results_response.json()
                except Exception as e:
                    logger.warning(f"⚠️ Could not fetch results: {e}")

                return status_data

            elif status == "failed":
                logger.error("")
                logger.error("❌ Job failed!")
                if "error" in status_data:
                    logger.error(f"   Error: {status_data['error']}")
                return status_data

            # Continue polling
            time.sleep(poll_interval)

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Status check error: {e}")
            time.sleep(poll_interval)

    logger.warning(f"⚠️ Job did not complete within {max_wait}s")
    logger.warning(f"   Job is still running. Check status manually:")
    logger.warning(f"   curl {api_base_url}/pipeline/{job_id}")
    return None


def print_results(results: Dict[str, Any]):
    """Print job results in a user-friendly format."""
    logger.info("")
    logger.info("=== Pipeline Results ===")

    if results.get("success"):
        logger.info(f"✅ Pipeline completed successfully!")
        logger.info(f"📋 Job ID: {results.get('job_id')}")
        logger.info(f"⏱️  Execution time: {results.get('execution_time_seconds', 0):.1f}s")

        # Output files
        output_files = results.get("output_files", {})
        if output_files:
            logger.info(f"📁 Generated {len(output_files)} output files:")
            for file_type, file_path in output_files.items():
                logger.info(f"   {file_type}: {file_path}")

        # Summary
        summary = results.get("summary", {})
        if summary:
            logger.info("")
            logger.info("📊 Summary:")
            for key, value in summary.items():
                if isinstance(value, dict):
                    logger.info(f"   {key}:")
                    for sub_key, sub_value in value.items():
                        logger.info(f"      {sub_key}: {sub_value}")
                else:
                    logger.info(f"   {key}: {value}")
    else:
        logger.error("❌ Pipeline failed!")
        errors = results.get("errors", [])
        for error in errors:
            logger.error(f"   Error: {error}")


def main():
    """Main execution."""
    args = parse_args()

    # Build API base URL
    api_base_url = f"http://{args.api_host}:{args.api_port}/api/v1/carriers"

    logger.info("=== Precision Medicine Carriers Pipeline (via API) ===")
    logger.info(f"📡 API endpoint: {api_base_url}")
    logger.info("")

    # Check API health
    if not check_api_health(api_base_url):
        logger.error("")
        logger.error("💡 Start the API server first:")
        logger.error("   python start_api.py")
        return 1

    logger.info("")

    # Submit job
    submission_result = submit_pipeline_job(api_base_url, args)
    if not submission_result:
        return 1

    job_id = submission_result.get("job_id")

    # Exit if not following
    if args.no_follow:
        logger.info("")
        logger.info("✅ Job submitted. Not waiting for completion (--no-follow)")
        logger.info(f"   Check status: curl {api_base_url}/pipeline/{job_id}")
        return 0

    # Poll for completion
    results = poll_job_status(
        api_base_url,
        job_id,
        args.poll_interval,
        args.max_wait
    )

    if results:
        print_results(results)
        return 0 if results.get("success") else 1
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
