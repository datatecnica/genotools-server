import argparse
import os
import requests
from src.core.pipeline_config import PipelineConfig
from src.core.api_utils import post_to_api

def process_nba_data(config: PipelineConfig):
    """Prepare browser files for GP2's NBA data using the API"""
    print("\n=== Preparing NBA Files ===")
    
    nba_out_dir = f'{config.browser_base_dir}/nba/release{config.release}'
    os.makedirs(nba_out_dir, exist_ok=True) # can later replace with file manager functionality    

    payload = {
        "release_num": config.release, 
        "master_path": f'{config.master_key_dir}/master_key_release{config.release}_final_vwb.csv',
        "gt_path": f'{config.gt_base_dir}/GP2_r{config.release}_final_post_genotools.json',
        "out_dir": nba_out_dir
    }

    post_to_api(f"{config.api_base_url}/prep_browser", payload)
    
    return nba_out_dir


def main():
    parser = argparse.ArgumentParser(description='Prepare all necessary files for the cohort browser.')
    parser.add_argument('--mnt-dir', default='~/gcs_mounts', help='Mount directory path')
    parser.add_argument('--release', default=10, help='Release version')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')

    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        mnt_dir=args.mnt_dir,
        release=args.release,
        api_base_url=args.api_url
    )
    
    # Check API health
    response = requests.get(f"{config.api_base_url}/health")
    print(f"Health check: {response.json()}")
    
    # Process NBA data
    results = {}
    results['nba'] = process_nba_data(config)
    
    # Print final summary
    print("\n=== All processing complete! ===")

if __name__ == "__main__":
    main()