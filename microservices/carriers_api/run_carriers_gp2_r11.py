#!/usr/bin/env python3
"""
Wrapper script to run carrier analysis on GP2 Release 11 cohort data return files.
This script adapts the main carrier pipeline to work with the specific file structure
and naming patterns of GP2 R11 cohort data return.
"""

import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='Run carrier analysis on GP2 Release 11 cohort data return files'
    )
    parser.add_argument('--mnt-dir', 
                       default='/home/vitaled2/gcs_mounts',
                       help='Mount directory path (default: /home/vitaled2/gcs_mounts)')
    parser.add_argument('--api-url', 
                       default='http://localhost:8000',
                       help='API base URL (default: http://localhost:8000)')
    parser.add_argument('--cleanup', 
                       type=bool, 
                       default=True,
                       help='Enable cleanup of existing files (default: True)')
    parser.add_argument('--combine-only', 
                       action='store_true',
                       help='Only run combination step (skip individual processing)')
    
    args = parser.parse_args()
    
    # Build the command for the main carrier pipeline
    cmd = [
        sys.executable, 'run_carriers.py',
        '--mnt-dir', args.mnt_dir,
        '--release', '11',
        '--api-url', args.api_url,
        '--cleanup', str(args.cleanup),
        '--data-type', 'nba',
        '--carriers-dir', f'{args.mnt_dir}/gp2_release11/cohort_data_return/variant_reports',
        '--release-dir', f'{args.mnt_dir}/gp2_release11'
    ]
    
    if args.combine_only:
        cmd.append('--combine-only')
    
    print("Running GP2 Release 11 carrier analysis with command:")
    print(' '.join(cmd))
    print()
    
    # Execute the main script
    try:
        result = subprocess.run(cmd, check=True)
        print("\nGP2 Release 11 carrier analysis completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nError running carrier analysis: {e}")
        return e.returncode
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())