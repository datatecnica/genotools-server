#!/usr/bin/env python3
"""
Simple script to run precision medicine recruitment analysis via API endpoint.

This script demonstrates how to use the /recruitment_analysis endpoint
programmatically instead of using curl.

Usage:
    # Defaults from server config (no explicit carriers paths)
    python run_precision_med_recruitment.py --release 10

    # Dry run
    python run_precision_med_recruitment.py --release 10 --dry-run

    # Custom output directory
    python run_precision_med_recruitment.py --release 10 --output-dir ~/custom_output

    # Provide a single carriers dataset explicitly
    python run_precision_med_recruitment.py --release 10 \
      --carriers-int /path/release10_carriers_int.parquet \
      --carriers-string /path/release10_carriers_string.parquet \
      --carriers-var-info /path/release10_var_info.parquet

    # Call API twice (e.g., to run with provided paths and then defaults)
    python run_precision_med_recruitment.py --release 10 \
      --carriers-int /path/release10_carriers_int.parquet \
      --carriers-string /path/release10_carriers_string.parquet \
      --carriers-var-info /path/release10_var_info.parquet
"""

import argparse
import requests
import json
import sys
import os
from typing import Dict, Any


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Precision Medicine Recruitment Analysis via API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_precision_med_recruitment.py --release 10
    python run_precision_med_recruitment.py --release 10 --dry-run
    python run_precision_med_recruitment.py --release 10 --mnt-path ~/custom_mounts
    python run_precision_med_recruitment.py --release 10 --output-dir ~/custom_output
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--release',
        type=str,
        required=True,
        help='GP2 release version (e.g., "10", "9")'
    )
    
    # Optional output
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results files (auto-generated if not provided)'
    )
    
    # Optional single carriers dataset overrides (generic)
    parser.add_argument('--carriers-int', type=str, help='Path to carriers int parquet')
    parser.add_argument('--carriers-string', type=str, help='Path to carriers string parquet')
    parser.add_argument('--carriers-var-info', type=str, help='Path to carriers variant info parquet')

    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check paths and data availability without running analysis'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default="http://localhost:8000",
        help='API base URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def call_recruitment_analysis_api(
    api_url: str,
    release: str,
    output_dir: str = None,
    dry_run: bool = False,
    verbose: bool = False,
    carriers_paths: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """
    Call the recruitment analysis API endpoint.
    
    Args:
        api_url: Base URL for the API
        release: GP2 release version
        mnt_path: Base mount path for data files
        output_dir: Optional output directory
        dry_run: Whether to perform a dry run
        verbose: Enable verbose output
        
    Returns:
        API response as dictionary
        
    Raises:
        requests.RequestException: If API call fails
    """
    endpoint = f"{api_url}/recruitment_analysis"
    
    # Prepare request payload
    payload = {
        "release": release,
        "dry_run": dry_run
    }
    
    if output_dir:
        payload["output_dir"] = output_dir

    # Attach carriers paths if provided (map to API model keys)
    if carriers_paths:
        if carriers_paths.get('carriers_int'):
            payload['carriers_int'] = carriers_paths['carriers_int']
        if carriers_paths.get('carriers_string'):
            payload['carriers_string'] = carriers_paths['carriers_string']
        if carriers_paths.get('carriers_var_info'):
            payload['carriers_var_info'] = carriers_paths['carriers_var_info']
    
    if verbose:
        print(f"🌐 Calling API: {endpoint}")
        print(f"📦 Payload: {json.dumps(payload, indent=2)}")
    
    # Make API request
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout for analysis
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        raise requests.RequestException(
            f"❌ Could not connect to API at {api_url}. "
            "Make sure the FastAPI server is running with: uvicorn main:app --reload"
        )
    except requests.exceptions.Timeout:
        raise requests.RequestException(
            "❌ API request timed out. The analysis may be taking longer than expected."
        )
    except requests.exceptions.HTTPError as e:
        error_detail = "Unknown error"
        try:
            error_response = response.json()
            error_detail = error_response.get("detail", str(e))
        except:
            error_detail = str(e)
        
        raise requests.RequestException(f"❌ API request failed: {error_detail}")


def print_analysis_results(response: Dict[str, Any], verbose: bool = False):
    """Print formatted analysis results."""
    status = response.get("status", "unknown")
    message = response.get("message", "No message")
    
    print(f"\n📊 Analysis Status: {status}")
    print(f"💬 Message: {message}")
    
    # Print configuration
    if "config" in response:
        config = response["config"]
        print(f"\n📋 Configuration:")
        print(f"  Release: {config.get('release')}")
        print(f"  Mount Path: {config.get('mnt_path')}")
        print(f"  Output Directory: {config.get('output_dir')}")
    
    # Print dry run results
    if status == "dry_run_success":
        if "validated_paths" in response:
            print(f"\n✅ Validated Paths:")
            for path_name in response["validated_paths"]:
                print(f"  - {path_name}")
        return
    
    # Print analysis summary
    if "summary" in response:
        summary = response["summary"]
        print(f"\n📈 Analysis Summary:")
        print(f"  Total Loci: {summary.get('total_loci', 0)}")
        print(f"  Loci Analyzed: {summary.get('loci_analyzed', [])}")
        print(f"  Exported Files: {summary.get('exported_files_count', 0)}")
        
        # Print carrier statistics
        if "carrier_statistics" in summary:
            print(f"\n🧬 Carrier Statistics:")
            for locus, stats in summary["carrier_statistics"].items():
                print(f"  {locus}:")
                print(f"    Total Carriers: {stats.get('total_carriers', 0)}")
                print(f"    Recruitment Analyzed: {stats.get('recruitment_analyzed', False)}")
    
    # Print output directory
    if "output_directory" in response:
        print(f"\n📁 Output Directory: {response['output_directory']}")
    
    # Print exported files (verbose mode)
    if verbose and "exported_files" in response:
        exported_files = response["exported_files"]
        if exported_files:
            print(f"\n📄 Exported Files:")
            for file_type, filepath in exported_files.items():
                print(f"  {file_type}: {filepath}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🚀 Precision Medicine Recruitment Analysis via API")
    print("=" * 60)
    
    # Compute default output dirs for nba and wgs if none provided
    base_default = os.path.expanduser(f"~/gcs_mounts/clinical_trial_output/release{args.release}")
    nba_default_dir = os.path.join(base_default, "nba")
    wgs_default_dir = os.path.join(base_default, "wgs")
    output_dir = os.path.expanduser(args.output_dir) if args.output_dir else None
    
    print(f"📋 Release: {args.release}")
    if args.dry_run:
        print("🔍 Mode: Dry Run (path validation only)")
    else:
        print("🔬 Mode: Full Analysis")
    
    # Determine which runs to perform: always run twice
    runs: list[tuple[str, Dict[str, str] | None]] = []

    # First run: if generic carriers provided, use them; else use defaults
    first_paths = None
    if any([args.carriers_int, args.carriers_string, args.carriers_var_info]):
        first_paths = {
            'carriers_int': os.path.expanduser(args.carriers_int) if args.carriers_int else None,
            'carriers_string': os.path.expanduser(args.carriers_string) if args.carriers_string else None,
            'carriers_var_info': os.path.expanduser(args.carriers_var_info) if args.carriers_var_info else None,
        }
    runs.append(("NBA", first_paths))

    # Second run: always run defaults (server-side config)
    runs.append(("WGS", None))

    overall_success = True
    try:
        for label, carriers_paths in runs:
            print(f"\n=== Running analysis: {label} ===")
            # Choose default output per run if user did not specify --output-dir
            run_output_dir = output_dir
            if run_output_dir is None:
                run_output_dir = nba_default_dir if label == "NBA" else wgs_default_dir
            # Ensure parent exists for cleanliness (server also ensures it)
            try:
                os.makedirs(run_output_dir, exist_ok=True)
            except Exception:
                pass
            response = call_recruitment_analysis_api(
                api_url=args.api_url,
                release=args.release,
                output_dir=run_output_dir,
                dry_run=args.dry_run,
                verbose=args.verbose,
                carriers_paths=carriers_paths,
            )
            print_analysis_results(response, args.verbose)

        print("\n✅ All requested analyses completed!")
        sys.exit(0 if overall_success else 1)

    except requests.RequestException as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
