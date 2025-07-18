import argparse
import requests
from src.core.pipeline_config import PipelineConfig
from src.core.file_manager import FileManager
from src.core.api_utils import post_to_api
from src.core.manager import CarrierAnalysisManager

# gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
# gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
# python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

def process_population_data(config: PipelineConfig, file_manager: FileManager):
    """Process carrier data for each population using the API"""
    print("\n=== Processing Array Data (NBA) ===")
    
    nba_output_dir = f'{config.carriers_base_dir}/nba/release{config.release}'
    
    for label in config.labels:
        print(f"\nProcessing {label}...")
        label_output_dir = f'{nba_output_dir}/{label}'
        
        # Prepare directory
        file_manager.ensure_directory(label_output_dir)
        file_manager.cleanup_directory(label_output_dir, config.cleanup_enabled)
        
        # Build and send API request
        payload = {
            "geno_path": f"{config.release_base_dir}/{label}/{label}_release{config.release}_vwb",
            "snplist_path": config.snplist_path,
            "out_path": f"{label_output_dir}/{label}_release{config.release}",
            "release_version": config.release
        }
        
        post_to_api(f"{config.api_base_url}/process_carriers", payload)
    
    return nba_output_dir


def process_wgs_data(config: PipelineConfig, file_manager: FileManager):
    """Process WGS carrier data using the API"""
    print("\n=== Processing WGS Data ===")
    
    wgs_output_dir = f'{config.carriers_base_dir}/wgs/release{config.release}'
    
    file_manager.ensure_directory(wgs_output_dir)
    file_manager.cleanup_directory(wgs_output_dir, config.cleanup_enabled)
    
    payload = {
        "geno_path": f"{config.wgs_raw_dir}/R{config.release}_wgs_carrier_vars",
        "snplist_path": config.snplist_path,
        "out_path": f"{wgs_output_dir}/release{config.release}",
        "release_version": config.release
    }
    
    post_to_api(f"{config.api_base_url}/process_carriers", payload)
    
    return wgs_output_dir


def process_imputed_data(config: PipelineConfig, file_manager: FileManager):
    """Process imputed carrier data for each ancestry using the API"""
    print("\n=== Processing Imputed Data ===")
    
    imputed_output_dir = f'{config.carriers_base_dir}/imputed/release{config.release}'
    
    for label in config.labels:
        print(f"\nProcessing imputed data for {label}...")
        label_output_dir = f'{imputed_output_dir}/{label}'
        
        # Prepare directory
        file_manager.ensure_directory(label_output_dir)
        file_manager.cleanup_directory(label_output_dir, config.cleanup_enabled)
        
        # Build and send API request for imputed processing
        payload = {
            "ancestry": label,
            "imputed_base_dir": config.imputed_base_dir,
            "snplist_path": config.snplist_path,
            "out_path": f"{label_output_dir}/{label}_release{config.release}",
            "release_version": config.release
        }
        
        post_to_api(f"{config.api_base_url}/process_imputed_carriers", payload)
    
    return imputed_output_dir


def combine_population_data(config: PipelineConfig, output_dir: str, data_type: str = "nba"):
    """Combine data outputs using the CarrierCombiner"""
    print(f"\n=== Combining {data_type.upper()} Data Outputs ===")
    
    # Build results_by_label dict for CarrierCombiner
    results_by_label = {}
    for label in config.labels:
        results_by_label[label] = {
            'var_info': f"{output_dir}/{label}/{label}_release{config.release}_var_info.parquet",
            'carriers_string': f"{output_dir}/{label}/{label}_release{config.release}_carriers_string.parquet",
            'carriers_int': f"{output_dir}/{label}/{label}_release{config.release}_carriers_int.parquet"
        }
    
    # Create output directory and manager
    combined_output_dir = f'{output_dir}/combined'
    FileManager.ensure_directory(combined_output_dir)
    
    manager = CarrierAnalysisManager()
    combined_output_path = f'{combined_output_dir}/{data_type}_release{config.release}_combined'
    
    # Note: pass None for key_file since we don't use study info, and track_probe_usage=True
    combined_files = manager.carrier_combiner.combine_carrier_files(
        results_by_label, 
        key_file=None, 
        out_path=combined_output_path,
        track_probe_usage=True
    )
    
    return combined_files

def main():
    parser = argparse.ArgumentParser(description='Process carrier data for multiple populations')
    parser.add_argument('--mnt-dir', default='/home/vitaled2/gcs_mounts', help='Mount directory path')
    parser.add_argument('--release', default='10', help='Release version')
    parser.add_argument('--api-url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--cleanup', type=bool, default=True, help='Enable cleanup of existing files')
    parser.add_argument('--carriers-dir', help='Override carriers base directory')
    parser.add_argument('--release-dir', help='Override release data directory')
    parser.add_argument('--wgs-dir', help='Override WGS raw data directory')
    parser.add_argument('--imputed-dir', help='Override imputed data base directory')
    
    # Data type selection arguments
    parser.add_argument('--data-type', choices=['nba', 'wgs', 'imputed', 'all'], default='all',
                       help='Type of data to process (default: all)')
    parser.add_argument('--combine-only', action='store_true',
                       help='Only run combination step (skip individual processing)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PipelineConfig(
        mnt_dir=args.mnt_dir,
        release=args.release,
        api_base_url=args.api_url,
        cleanup_enabled=args.cleanup,
        carriers_base_dir=args.carriers_dir,
        release_base_dir=args.release_dir,
        wgs_raw_dir=args.wgs_dir,
        imputed_base_dir=args.imputed_dir
    )
    
    # Check API health
    response = requests.get(f"{config.api_base_url}/health")
    print(f"Health check: {response.json()}")
    print(f"Cleanup enabled: {config.cleanup_enabled}")
    print(f"Data type: {args.data_type}")
    
    # Initialize file manager
    file_manager = FileManager()

    ################# Hardcode directories for testing combine_population_data #################
    # nba_output_dir = f'{config.mnt_dir}/genotools_server/carriers/nba/release{config.release}'
    # wgs_output_dir = f'{config.mnt_dir}/genotools_server/carriers/wgs/release{config.release}'
    ############################################################################################
    
    # Process different data types based on selection
    results = {}
    
    if not args.combine_only:
        if args.data_type in ['nba', 'all']:
            nba_output_dir = process_population_data(config, file_manager)
            results['nba'] = nba_output_dir
        
        if args.data_type in ['wgs', 'all']:
            wgs_output_dir = process_wgs_data(config, file_manager)
            results['wgs'] = wgs_output_dir
        
        if args.data_type in ['imputed', 'all']:
            imputed_output_dir = process_imputed_data(config, file_manager)
            results['imputed'] = imputed_output_dir
    else:
        # For combine-only mode, construct the expected directories
        if args.data_type in ['nba', 'all']:
            results['nba'] = f'{config.carriers_base_dir}/nba/release{config.release}'
        if args.data_type in ['wgs', 'all']:
            results['wgs'] = f'{config.carriers_base_dir}/wgs/release{config.release}'
        if args.data_type in ['imputed', 'all']:
            results['imputed'] = f'{config.carriers_base_dir}/imputed/release{config.release}'
    
    # Combine results for data types that support combination (NBA and Imputed)
    combined_results = {}
    
    if 'nba' in results:
        combined_results['nba'] = combine_population_data(config, results['nba'], 'nba')
    
    if 'imputed' in results:
        combined_results['imputed'] = combine_population_data(config, results['imputed'], 'imputed')
    
    # Print final summary
    print("\n=== All processing complete! ===")
    
    for data_type, output_dir in results.items():
        print(f"{data_type.upper()} individual files saved in: {output_dir}")
    
    for data_type, combined_files in combined_results.items():
        print(f"{data_type.upper()} combined files saved in: {combined_files}")


if __name__ == "__main__":
    main()
