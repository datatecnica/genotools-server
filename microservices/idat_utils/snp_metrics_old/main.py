#!/usr/bin/env python3
import os
import argparse
import time
from snp_metrics import process_idat_files, extract_vcf_columns

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process IDAT files to generate SNP metrics or extract metadata from VCF files.')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parser for process command
    process_parser = subparsers.add_parser('process', help='Process IDAT files')
    # Required arguments for process
    process_parser.add_argument('--idat_path', required=True, help='Path to directory containing IDAT files')
    process_parser.add_argument('--output_directory', required=True, help='Directory to output processed files')
    process_parser.add_argument('--bpm', required=True, help='Path to BPM file')
    process_parser.add_argument('--bpm_csv', required=True, help='Path to BPM CSV file')
    process_parser.add_argument('--egt', required=True, help='Path to EGT file')
    process_parser.add_argument('--ref_fasta', required=True, help='Path to reference FASTA file')
    process_parser.add_argument('--iaap', required=True, help='Path to IAAP CLI executable')
    process_parser.add_argument('--bcftools_plugins_path', required=True, help='Path to bcftools plugins directory')
    # Optional arguments for process
    process_parser.add_argument('--cleanup', action='store_true', help='Delete intermediate files after processing')
    process_parser.add_argument('--debug', action='store_true', help='Print detailed debugging information')
    process_parser.add_argument('--bcftools_threads', type=int, help='Maximum number of threads for bcftools commands')
    
    # Parser for extract command
    extract_parser = subparsers.add_parser('extract', help='Extract metadata from VCF file')
    # Required arguments for extract
    extract_parser.add_argument('--vcf_path', required=True, help='Path to VCF file')
    extract_parser.add_argument('--output_path', required=True, help='Path to output file or directory')
    # Optional arguments for extract
    extract_parser.add_argument('--columns', choices=['all', 'metadata', 'sample'], default='metadata', 
                              help='Columns to extract (default: metadata)')
    extract_parser.add_argument('--output_format', choices=['csv', 'parquet'], default='parquet',
                              help='Output format (default: parquet)')
    extract_parser.add_argument('--partition', action='store_true', 
                              help='Partition output by chromosome (for parquet format)')
    extract_parser.add_argument('--num_rows', type=int, default=None,
                              help='Number of rows to extract (default: all)')
    
    return parser.parse_args()

def process_command(args):
    """Run the IDAT processing pipeline."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_directory, exist_ok=True)
    
    print(f"Processing IDAT directory: {args.idat_path}")
    print(f"Output directory: {args.output_directory}")
    
    # Run process_idat_files function with provided arguments
    success = process_idat_files(
        idat_path=args.idat_path,
        output_directory=args.output_directory,
        bpm=args.bpm,
        bpm_csv=args.bpm_csv,
        egt=args.egt,
        ref_fasta=args.ref_fasta,
        iaap=args.iaap,
        bcftools_plugins_path=args.bcftools_plugins_path,
        cleanup_intermediate_files=args.cleanup,
        debug=args.debug,
        bcftools_max_threads=args.bcftools_threads
    )
    
    if success:
        barcode = os.path.basename(args.idat_path)
        print(f"Successfully processed IDAT files for {barcode}")
        print(f"Results are available in: {os.path.join(args.output_directory, barcode)}")
        
        if not args.cleanup:
            tmp_dir = os.path.join(args.output_directory, f"tmp_{barcode}")
            print(f"Intermediate files are stored in: {tmp_dir}")
    else:
        print("Failed to process IDAT files")
        return 1
    
    return 0

def extract_command(args):
    """Extract metadata from VCF file."""
    print(f"Extracting data from VCF file: {args.vcf_path}")
    print(f"Output path: {args.output_path}")
    print(f"Columns to extract: {args.columns}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        # Run extract_vcf_columns function with provided arguments
        result_df = extract_vcf_columns(
            vcf_file=args.vcf_path,
            output_path=args.output_path,
            num_rows=args.num_rows,
            columns=args.columns,
            output_format=args.output_format,
            partition_by_chromosome=args.partition
        )
        
        # Print a summary of the extraction
        print(f"Successfully extracted {len(result_df)} rows and {len(result_df.columns)} columns")
        if len(result_df) > 0:
            print(f"Columns: {', '.join(result_df.columns)}")
        print(f"Results saved to: {args.output_path}")
        return 0
    except Exception as e:
        print(f"Error extracting data from VCF file: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Run the main program."""
    # Track overall script runtime
    script_start_time = time.time()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Print start time
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute the appropriate command
    if args.command == 'process':
        result = process_command(args)
    elif args.command == 'extract':
        result = extract_command(args)
    else:
        print("No command specified. Use 'process' or 'extract'.")
        print("For help, run: python main.py -h")
        result = 1
    
    # Calculate total runtime
    script_duration = time.time() - script_start_time
    hours, remainder = divmod(script_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print("Script Execution Summary")
    print("="*50)
    print(f"Command: {args.command if hasattr(args, 'command') else 'None'}")
    print("-"*50)
    print(f"Total Runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (HH:MM:SS)")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(script_start_time))}")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    return result

if __name__ == "__main__":
    exit(main())
