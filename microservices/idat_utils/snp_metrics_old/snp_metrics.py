import os
import subprocess
import numpy as np
import pandas as pd
import psutil
import glob
import gzip
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
import argparse
# Suppress copy warning.
pd.options.mode.chained_assignment = None


def convert_idat_to_gtc(iaap, bpm, egt, barcode_out_path, idat_path, debug=False):
    """Convert IDAT files to GTC format.
    
    Args:
        iaap: Path to IAAP CLI executable
        bpm: Path to BPM file
        egt: Path to EGT file
        barcode_out_path: Output directory for GTC files
        idat_path: Directory containing IDAT files
        debug: Whether to print detailed debugging information (default: False)
    
    Returns:
        List of GTC file paths if successful, False otherwise
    """
    if debug:
        print("\n=== Starting IDAT to GTC conversion debugging ===")
        
        # Verify input files exist with detailed output
        print("\nChecking input files:")
        for file_path, desc in [
            (iaap, "IAAP CLI"),
            (bpm, "BPM file"),
            (egt, "EGT file"),
            (idat_path, "IDAT directory"),
            (barcode_out_path, "Output directory")
        ]:
            exists = os.path.exists(file_path)
            print(f"- {desc}: {file_path}")
            print(f"  Exists: {exists}")
            if exists:
                if os.path.isfile(file_path):
                    print(f"  Size: {os.path.getsize(file_path)} bytes")
                    print(f"  Permissions: {oct(os.stat(file_path).st_mode)[-3:]}")
                elif os.path.isdir(file_path):
                    print(f"  Is directory: True")
                    print(f"  Permissions: {oct(os.stat(file_path).st_mode)[-3:]}")
    else:
        # Simple validation without verbose output
        for file_path, desc in [
            (iaap, "IAAP CLI"),
            (bpm, "BPM file"),
            (egt, "EGT file"),
            (idat_path, "IDAT directory")
        ]:
            if not os.path.exists(file_path):
                print(f"ERROR: {desc} not found: {file_path}")
                return False
    
    # Check IDAT files
    red_idat = glob.glob(os.path.join(idat_path, "*_Red.idat"))
    grn_idat = glob.glob(os.path.join(idat_path, "*_Grn.idat"))
    
    if debug:
        print("\nScanning for IDAT files:")
        print(f"Red IDAT files found ({len(red_idat)}):")
        for f in red_idat:
            print(f"- {os.path.basename(f)} ({os.path.getsize(f)} bytes)")
        
        print(f"\nGreen IDAT files found ({len(grn_idat)}):")
        for f in grn_idat:
            print(f"- {os.path.basename(f)} ({os.path.getsize(f)} bytes)")
    
    if not red_idat or not grn_idat:
        print("\nERROR: Missing IDAT files!")
        return False
    
    if debug:
        # Check output directory
        print(f"\nChecking output directory: {barcode_out_path}")
        print(f"Exists: {os.path.exists(barcode_out_path)}")
        print(f"Is directory: {os.path.isdir(barcode_out_path)}")
        print(f"Permissions: {oct(os.stat(barcode_out_path).st_mode)[-3:]}")
    
    # Get initial GTC files
    initial_gtc_files = set(glob.glob(os.path.join(barcode_out_path, "*.gtc")))
    if debug:
        print(f"\nExisting GTC files: {len(initial_gtc_files)}")
    
    # Set environment and run command
    env = os.environ.copy()
    env["DOTNET_SYSTEM_GLOBALIZATION_INVARIANT"] = "1"
    
    # Build and run command
    idat_to_gtc_cmd = f"{iaap} gencall {bpm} {egt} {barcode_out_path} -f {idat_path} -g -t 8"
    if debug:
        print(f"\nExecuting command:\n{idat_to_gtc_cmd}")
    else:
        print(f"Converting IDAT files to GTC format...")
    
    try:
        result = subprocess.run(
            idat_to_gtc_cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            env=env,
            text=True  # This will decode output as text
        )
        
        if debug:
            print("\nCommand output:")
            print("=== STDOUT ===")
            print(result.stdout)
            print("=== STDERR ===")
            print(result.stderr)
            print(f"Return code: {result.returncode}")
        
        if result.returncode != 0:
            error_msg = f"IDAT to GTC conversion failed with exit code {result.returncode}"
            if debug:
                print(f"\nERROR: {error_msg}")
            else:
                print(f"ERROR: {error_msg}")
                print(f"STDERR: {result.stderr}")
            return False
        
    except Exception as e:
        error_msg = f"Exception while running IDAT to GTC conversion: {str(e)}"
        if debug:
            print(f"\nERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        else:
            print(f"ERROR: {error_msg}")
        return False
    
    # Check for new GTC files
    all_gtc_files = glob.glob(os.path.join(barcode_out_path, "*.gtc"))
    new_gtc_files = [f for f in all_gtc_files if f not in initial_gtc_files]
    
    if debug:
        print(f"\nGTC files after conversion: {len(all_gtc_files)}")
        print(f"New GTC files generated: {len(new_gtc_files)}")
    
    if not new_gtc_files:
        error_msg = "No new GTC files were generated"
        if debug:
            print(f"\nERROR: {error_msg}")
            # List all files in output directory to help debug
            print("\nContents of output directory:")
            for f in os.listdir(barcode_out_path):
                print(f"- {f}")
        else:
            print(f"ERROR: {error_msg}")
        return False
    
    if debug:
        print("\nSuccessfully generated GTC files:")
        for f in new_gtc_files:
            print(f"- {os.path.basename(f)}")
        print("\n=== IDAT to GTC conversion complete ===")
    else:
        print(f"Successfully created {len(new_gtc_files)} GTC files")
    
    return new_gtc_files

def gtc_to_vcf(gtc_directory, vcf_directory, bpm, bpm_csv, egt, ref_fasta, out_tmp, bcftools_plugins_path, threads=8, memory="4G"):
    """Convert all GTC files in a directory to multiple VCF files (one per sample)."""
    
    ram_tmp = "/dev/shm/vcf_tmp" if os.path.exists("/dev/shm") else out_tmp
    os.makedirs(ram_tmp, exist_ok=True)
    os.makedirs(vcf_directory, exist_ok=True)
    
    # First, generate a combined VCF in a temporary location
    temp_vcf = os.path.join(out_tmp, "temp_combined.vcf.gz")
    
    cmd = f"""export BCFTOOLS_PLUGINS={bcftools_plugins_path} && \
bcftools +gtc2vcf \
--no-version -Ov \
--bpm {bpm} \
--csv {bpm_csv} \
--egt {egt} \
--gtcs {gtc_directory} \
--fasta-ref {ref_fasta} | \
bcftools norm \
-m \
-both \
-Ou \
--no-version \
-c w \
-f {ref_fasta} \
--threads {threads} | \
bcftools sort \
-T {ram_tmp}/ \
-m {memory} | \
bcftools view \
--threads {threads} \
-Oz -l 1 \
-o {temp_vcf}"""
    
    print(f"Processing all GTC files in: {gtc_directory}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"Command failed: {result.stderr.decode('utf-8')}")
        print(f"Error message: {result.stderr.decode('utf-8')}")
        return False
    
    # Index the combined VCF
    index_cmd = f"bcftools index --tbi {temp_vcf}"
    subprocess.run(index_cmd, shell=True)
    
    # Get sample names from VCF
    sample_ids = [x for x in get_vcf_names(temp_vcf) 
                  if x not in ['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','FORMAT']]
    
    print(f"Splitting combined VCF into {len(sample_ids)} sample files")
    
    # Split the combined VCF by sample
    vcf_files = []
    for sample_id in sample_ids:
        sample_vcf = os.path.join(vcf_directory, f"{sample_id}.vcf.gz")
        split_cmd = f"bcftools view -Oz -s {sample_id} -o {sample_vcf} {temp_vcf}"
        split_result = subprocess.run(split_cmd, shell=True)
        
        if split_result.returncode != 0:
            print(f"Failed to extract sample {sample_id}")
            continue
            
        vcf_files.append(sample_vcf)
    
    # Clean up temporary combined VCF
    os.remove(temp_vcf)
    if os.path.exists(temp_vcf + ".tbi"):
        os.remove(temp_vcf + ".tbi")
    
    print(f"Created {len(vcf_files)} sample VCF files in {vcf_directory}")
    return vcf_files

def get_vcf_names(vcf_path):
    """Get column names from VCF file."""
    opener = gzip.open if vcf_path.endswith('.gz') else open
    mode = 'rt' if vcf_path.endswith('.gz') else 'r'
    
    with opener(vcf_path, mode) as ifile:
            for line in ifile:
                if line.startswith("#CHROM"):
                    vcf_names = [x.strip('\n') for x in line.split('\t')]
                    break
    return vcf_names

def extract_info(info, idx, pattern):
    """Extract information from VCF INFO field."""
    split_info = info.split(";")
    if idx < len(split_info):
        return split_info[idx].replace(pattern, "")
    return None

def merge_chromosome(chrom_chunks):
    """Helper function to merge parquet chunks for a specific chromosome."""
    chrom, chunk_list = chrom_chunks
    merged_df = pd.concat([
        pd.read_parquet(f).query(f"chromosome == '{chrom}'")
        for f, _ in chunk_list
    ])
    return chrom, merged_df

def merge_parquet_chunks(chunk_pattern, output_directory):
    chunk_files = sorted(glob.glob(chunk_pattern))
    if not chunk_files:
        return False
    
    # Group chunks by chromosome for parallel processing
    chunks_by_chrom = {}
    for chunk_file in chunk_files:
        df = pd.read_parquet(chunk_file)
        for chrom in df['chromosome'].unique():
            if chrom not in chunks_by_chrom:
                chunks_by_chrom[chrom] = []
            chunks_by_chrom[chrom].append((chunk_file, chrom))
    
    # Merge chromosomes in parallel
    with ProcessPoolExecutor() as executor:
        futures = []
        for chrom, chunks in chunks_by_chrom.items():
            future = executor.submit(merge_chromosome, (chrom, chunks))
            futures.append(future)
        
        # Write results as they complete - without .parquet extension
        for future in futures:
            chrom, merged_df = future.result()
            # Remove .parquet extension
            output_file = os.path.join(output_directory, f"chromosome_{chrom}")
            merged_df.to_parquet(output_file, compression='snappy', index=False)

def process_idat_files(idat_path, output_directory, bpm, bpm_csv, egt, ref_fasta, iaap, bcftools_plugins_path, cleanup_intermediate_files=True, debug=False, bcftools_max_threads=None):
    """Process a single IDAT directory to generate SNP metrics.
    
    Args:
        idat_path: Path to directory containing IDAT files
        output_directory: Directory to output processed files
        bpm: Path to BPM file
        bpm_csv: Path to BPM CSV file
        egt: Path to EGT file
        ref_fasta: Path to reference FASTA file
        iaap: Path to IAAP CLI executable
        bcftools_plugins_path: Path to bcftools plugins directory
        cleanup_intermediate_files: Whether to delete GTC and VCF files after processing (default: True)
        debug: Whether to print detailed debugging information (default: False)
        bcftools_max_threads: Maximum number of threads for bcftools (default: None, uses all available)
    """
    # Start timing the entire pipeline
    pipeline_start_time = time.time()
    
    # Verify the IDAT directory exists and contains IDAT files
    if not os.path.isdir(idat_path):
        print(f"IDAT directory not found: {idat_path}")
        return False
        
    if not any(f.endswith('.idat') for f in os.listdir(idat_path)):
        print(f"No IDAT files found in {idat_path}")
        return False
    
    # Get barcode from directory name
    barcode = os.path.basename(idat_path)
    barcode_out_path = os.path.join(output_directory, barcode)
    os.makedirs(barcode_out_path, exist_ok=True)
    
    # Create temporary directory to store intermediate files
    out_tmp = os.path.join(output_directory, f"tmp_{barcode}")
    os.makedirs(out_tmp, exist_ok=True)
    
    print(f"Intermediate files will be stored in: {out_tmp}")
    if not cleanup_intermediate_files:
        print(f"Intermediate files will be kept for future use")
    
    # Calculate resource allocation
    cpu_count = os.cpu_count()
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Calculate bcftools threads based on CPU count
    if bcftools_max_threads is None:
        bcftools_threads = min(cpu_count, 8)  # Default cap at 8 threads
    else:
        bcftools_threads = min(cpu_count, bcftools_max_threads)
    
    # Step 1: Convert IDAT files to GTC (write to temp directory)
    step1_start_time = time.time()
    print(f"Step 1: Converting IDAT to GTC for {barcode}...")
    gtc_files = convert_idat_to_gtc(iaap, bpm, egt, out_tmp, idat_path, debug=debug)
    if not gtc_files:
        print(f"Failed to convert IDAT to GTC for {barcode}")
        return False
    
    step1_duration = time.time() - step1_start_time
    # print(f"Step 1 complete: Created {len(gtc_files)} GTC files in {step1_duration:.2f} seconds")
    
    # Step 2: Process all GTC files and create per-sample VCF files
    step2_start_time = time.time()
    print(f"Step 2: Converting GTC files to per-sample VCF files...")
    vcf_files = gtc_to_vcf(out_tmp, out_tmp, bpm, bpm_csv, egt, ref_fasta, out_tmp, 
                     bcftools_plugins_path, threads=bcftools_threads, 
                     memory=f"{int(total_memory_gb * 0.7)}G")
    
    if not vcf_files:
        print(f"Failed to convert GTC files to VCF for {barcode}")
        return False
    
    step2_duration = time.time() - step2_start_time
    # glob vcf_files for testing
    vcf_files = glob.glob(os.path.join(out_tmp, "*.vcf.gz"))
    print(vcf_files)
    print(f"Step 2 complete: Created {len(vcf_files)} VCF files in {step2_duration:.2f} seconds")
    
    # Step 3: Process each sample VCF file in parallel
    step3_start_time = time.time()
    print(f"Step 3: Processing {len(vcf_files)} sample VCF files for {barcode}...")
    
    # Calculate max workers based on available memory and CPU
    memory_per_worker = 2  # Assume each worker needs ~2GB
    workers_by_memory = int(total_memory_gb / memory_per_worker * 0.7)  # Use 70% of available memory
    max_workers = min(
        cpu_count,  # Don't exceed CPU count
        workers_by_memory,  # Don't exceed memory capacity
        len(vcf_files),  # Don't exceed number of files
        8  # Hard cap at 8 workers
    )
    
    print(f"Resource allocation:")
    print(f"- Total Memory: {total_memory_gb:.1f}GB")
    print(f"- Max Workers: {max_workers}")
    
    # Adjust chunk size based on available memory per worker
    memory_per_chunk = total_memory_gb / max_workers * 0.3  # Use 30% of worker's memory for chunks
    chunk_size = int(min(
        50000 * (memory_per_chunk / 2),  # Scale chunk size with available memory
        200000  # Hard cap at 200K rows
    ))
    print(f"- Chunk size: {chunk_size}")
    
    success = True
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Group VCF files by size for better load balancing
        vcf_sizes = [(f, os.path.getsize(f)) for f in vcf_files]
        vcf_sizes.sort(key=lambda x: x[1], reverse=True)  # Process largest files first
        
        # Process files in batches to control memory usage
        batch_size = max_workers
        batch_count = 0
        for i in range(0, len(vcf_sizes), batch_size):
            batch_count += 1
            batch = vcf_sizes[i:i + batch_size]
            batch_start_time = time.time()
            print(f"Processing batch {batch_count} ({len(batch)} files)...")
            
            futures = []
            
            for vcf_file, _ in batch:
                sample_id = os.path.basename(vcf_file).replace('.vcf.gz', '')
                # Use barcode_out_path instead of creating separate sample directories
                sample_output_dir = barcode_out_path
                
                temp_output_path = os.path.join(out_tmp, f"temp_{sample_id}")
                
                future = executor.submit(
                    process_single_sample_vcf, 
                    vcf_file,
                    temp_output_path,
                    chunk_size,
                    sample_output_dir,
                    sample_id
                )
                futures.append((sample_id, future))
            
            # Wait for current batch to complete before starting next batch
            for sample_id, future in futures:
                try:
                    if not future.result():
                        print(f"Failed to process SNP data for sample {sample_id}")
                        success = False
                except Exception as e:
                    print(f"Error processing sample {sample_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    success = False
            
            batch_duration = time.time() - batch_start_time
            print(f"Batch {batch_count} completed in {batch_duration:.2f} seconds")
            
            # Force garbage collection between batches
            import gc
            gc.collect()
    
    step3_duration = time.time() - step3_start_time
    
    if not success:
        print(f"Some samples failed to process for {barcode}")
        return False
    
    print(f"Step 3 complete: Processed VCF files in {step3_duration:.2f} seconds")
    
    # Step 4: Clean up temporary files
    step4_start_time = time.time()
    print(f"Step 4: Cleaning up temporary files...")
    
    try:
        # Remove chunk files which are always temporary
        chunk_files = glob.glob(os.path.join(out_tmp, "temp_*"))
        for file_path in chunk_files:
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
        print(f"Temporary chunk files removed")
        
        # If cleanup is requested, remove the entire temp directory
        if cleanup_intermediate_files:
            print(f"Cleaning up all intermediate files from {out_tmp}")
            shutil.rmtree(out_tmp)
            print(f"All intermediate files removed")
        else:
            print(f"Keeping intermediate GTC and VCF files in {out_tmp} for future use")
    except Exception as e:
        print(f"Warning: Error during cleanup: {str(e)}")
    
    step4_duration = time.time() - step4_start_time
    print(f"Step 4 complete: Cleanup finished in {step4_duration:.2f} seconds")
    
    # Calculate and display total pipeline runtime
    pipeline_duration = time.time() - pipeline_start_time
    
    # Convert seconds to hours, minutes, seconds format
    hours, remainder = divmod(pipeline_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "="*50)
    print(f"Pipeline Execution Summary for {barcode}")
    print("="*50)
    print(f"Step 1 (IDAT to GTC): {step1_duration:.2f} seconds")
    print(f"Step 2 (GTC to VCF): {step2_duration:.2f} seconds")
    print(f"Step 3 (VCF Processing): {step3_duration:.2f} seconds")
    print(f"Step 4 (Cleanup): {step4_duration:.2f} seconds")
    print("-"*50)
    print(f"Total Runtime: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (HH:MM:SS)")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pipeline_start_time))}")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pipeline_start_time + pipeline_duration))}")
    print("="*50)
    
    return True


def process_single_sample_vcf(vcf_file, out_path, chunk_size=100000, final_output_dir=None, sample_id=None):
    """Process a single-sample VCF file."""
    print(f"Processing sample VCF file: {vcf_file}")
    start_time = time.time()
    
    # Get sample ID from filename if not provided
    if sample_id is None:
        sample_id = os.path.basename(vcf_file).replace('.vcf.gz', '')
    
    # Create output directory if needed
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Open file for reading
    opener = gzip.open if vcf_file.endswith('.gz') else open
    mode = 'rt' if vcf_file.endswith('.gz') else 'r'
    
    # Get column names
    names = get_vcf_names(vcf_file)
    
    # Ensure we have the right sample column
    metadata_cols = ['#CHROM','POS','ID','REF','ALT','QUAL','FILTER','INFO','FORMAT']
    sample_ids = [x for x in names if x not in metadata_cols]
    
    if len(sample_ids) != 1:
        print(f"Expected single sample VCF, but found {len(sample_ids)} samples: {sample_ids}")
        return False
    
    # Count total lines for progress reporting
    total_lines = 0
    with opener(vcf_file, mode) as f:
        for line in f:
            if not line.startswith('#'):
                total_lines += 1
    
    print(f"Total data lines to process: {total_lines}")
    
    # Process file in chunks
    chunk_count = 0
    
    with opener(vcf_file, mode) as f:
        # Skip header lines
        for line in f:
            if line.startswith('#') and not line.startswith('#CHROM'):
                continue
            elif line.startswith('#CHROM'):
                # Found column headers
                break
        
        # Process data lines in chunks
        current_chunk = []
        processed_lines = 0
        
        # Add parallel chunk processing
        def process_chunk(chunk_data):
            chunk_df = pd.DataFrame(chunk_data, columns=names)
            processed_chunk = process_single_sample_chunk(chunk_df, sample_ids[0])
            return processed_chunk
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=4) as chunk_executor:
            current_chunks = []
            chunk_futures = []
            
            for line in f:
                line_data = line.strip().split('\t')
                current_chunks.append(line_data)
                processed_lines += 1
                
                if len(current_chunks) >= chunk_size:
                    # Submit chunk for processing
                    chunk_future = chunk_executor.submit(process_chunk, current_chunks)
                    chunk_futures.append((chunk_count, chunk_future))
                    current_chunks = []
                    chunk_count += 1
                
            # Process remaining chunks
            if current_chunks:
                chunk_future = chunk_executor.submit(process_chunk, current_chunks)
                chunk_futures.append((chunk_count, chunk_future))
            
            # Write results as they complete - Use extension-less temp files
            for chunk_idx, future in chunk_futures:
                try:
                    processed_chunk = future.result()
                    # Remove .parquet extension
                    chunk_file = f"{out_path}_chunk_{chunk_idx}"
                    # Write to temporary file with .tmp suffix only
                    temp_file = f"{chunk_file}.tmp"
                    processed_chunk.to_parquet(temp_file)
                    # Then rename (atomic operation on most file systems)
                    os.rename(temp_file, chunk_file)
                    
                    # After writing - verify without extension
                    verification_df = pd.read_parquet(chunk_file)
                    if len(verification_df) != len(processed_chunk):
                        raise ValueError("Data integrity check failed")
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
                    # Implement retry logic or fallback
    
    print(f"Sample {sample_id} - Processed {chunk_count + 1} chunks in {time.time() - start_time:.2f} seconds")
    
    # Merge all chunk files
    if out_path:
        # If final_output_dir is specified, use that instead of the default merged location
        output_dir = final_output_dir if final_output_dir else f"{out_path}_merged"
        
        # Use sample_id in the output file name without .parquet extension
        output_file = os.path.join(output_dir, f"{sample_id}")
        
        # Modified merge function call to create single file per sample
        merge_sample_chunks(f"{out_path}_chunk_*", output_file, sample_id)
    
    return True

def process_single_sample_chunk(chunk_df, sample_id):
    """Process a single chunk of a single-sample VCF data."""
    
    # Fix metadata columns to match actual dataframe columns
    chrom_col = '#CHROM'
    if '#CHROM' not in chunk_df.columns and 'CHROM' in chunk_df.columns:
        chrom_col = 'CHROM'
    elif '#CHROM' not in chunk_df.columns:
        # Look for a column that might be CHROM (like chr1)
        chrom_cols = [c for c in chunk_df.columns if c.startswith('chr')]
        if chrom_cols:
            chrom_col = chrom_cols[0]
    
    # Extract necessary columns
    sample_data = chunk_df[[chrom_col, 'POS', 'ID', 'FORMAT', sample_id]].copy()
    
    # Rename CHROM column if needed
    if chrom_col != 'CHROM':
        sample_data = sample_data.rename(columns={chrom_col: 'CHROM'})
    
    # Split metrics column
    metric_cols = ['GT','GQ','IGC','BAF','LRR','NORMX','NORMY','R','THETA','X','Y']
    sample_data[metric_cols] = sample_data[sample_id].str.split(':', expand=True, n=10)
    
    # Clean up chromosome column (still needed for merging, will be excluded later)
    sample_data['CHROM'] = sample_data['CHROM'].astype(str).str.replace('chr','')
    
    # Map GT values using -9 for missing instead of NaN
    numeric_gt_map = {'0/0': 0, '0/1': 1, '1/1': 2, './.': -9}
    sample_data['GT'] = sample_data['GT'].map(numeric_gt_map)
    
    # Handle any missing values not caught by the mapping
    sample_data['GT'] = sample_data['GT'].fillna(-9)
    
    # Convert numeric columns
    for col in ['BAF', 'LRR', 'R', 'THETA']:
        sample_data[col] = pd.to_numeric(sample_data[col], errors='coerce')
    
    # Add IID column (replacing Sample_ID)
    sample_data['IID'] = sample_id
    
    # Rename columns
    final_df = sample_data.rename(columns={
        'ID': 'snpID',
        'CHROM': 'chromosome',
        'POS': 'position'
    })
    
    # Set correct types - now GT is integer
    final_df = final_df.astype({
        'chromosome': str,
        'position': int,
        'snpID': str,
        'IID': str,
        'BAF': float,
        'LRR': float,
        'R': float,
        'THETA': float,
        'GT': int  # Using -9 for missing
    })
    
    # Select only required columns for output
    required_columns = ['snpID', 'BAF', 'LRR', 'R', 'THETA', 'GT']
    
    # We still need chromosome and position for merging but they'll be excluded in final output
    working_columns = ['chromosome', 'position'] + required_columns
    
    return final_df[working_columns]

def extract_vcf_columns(vcf_file, output_path=None, num_rows=10, columns="all", 
                        output_format="csv", partition_by_chromosome=False):
    """
    Extract rows and specific columns from a VCF file, split INFO and FORMAT fields.
    Uses a predefined list of columns for consistent extraction.
    
    Args:
        vcf_file: Path to VCF file
        output_path: Optional path to save the extracted data
        num_rows: Number of rows to extract (default: 10, use None for all rows)
        columns: Columns to extract. Options:
            - "all": Extract all columns (default)
            - "metadata": Extract only metadata columns
            - "sample": Extract only sample-specific columns
            - List of specific column names to extract
        output_format: Format to save the output (default: "csv")
            - "csv": Save as CSV file
            - "parquet": Save as Parquet file/directory
        partition_by_chromosome: Whether to partition Parquet output by chromosome 
            (default: False, only applies when output_format="parquet")
        
    Returns:
        DataFrame containing the extracted data with selected columns
    """
    # Define metadata columns - use the exact list provided
    metadata_cols_list = [
        'ID', 'ASSAY_TYPE', 'devR_AB', 'POS', 'FRAC_T', 'FRAC_G', 
        'meanTHETA_BB', 'meanR_AB', 'devTHETA_AB', 'GC', 'N_AA', 
        'QUAL', 'Orig_Score', 'FRAC_C', 'GenTrain_Score', 'devR_BB', 
        'NORM_ID', 'devR_AA', 'FILTER', 'Intensity_Threshold', 
        'meanR_AA', 'CHROM', 'devTHETA_AA', 'ALLELE_A', 'N_AB', 
        'meanR_BB', 'meanTHETA_AA', 'meanTHETA_AB', 'REF', 
        'devTHETA_BB', 'N_BB', 'ALLELE_B', 'FRAC_A', 'BEADSET_ID', 
        'ALT', 'Cluster_Sep', 'a1', 'a2'  # Added a1 and a2
    ]
    
    # Define sample-specific columns
    sample_specific_list = ['IID', 'GT', 'GQ', 'IGC', 'BAF', 'LRR', 'NORMX', 'NORMY', 'R', 'THETA', 'X', 'Y']
    
    # Print extraction info
    if num_rows is None:
        print(f"Extracting ALL rows from VCF: {vcf_file}")
    else:
        print(f"Extracting first {num_rows} rows from VCF: {vcf_file}")
    
    start_time = time.time()
    
    # Extract sample ID from the filename
    sample_id = os.path.basename(vcf_file).replace('.vcf.gz', '')
    
    # --- OPTIMIZATION 1: Use chunked reading for large files ---
    chunk_size = 100000  # Adjust based on memory availability
    
    # Open file for reading
    opener = gzip.open if vcf_file.endswith('.gz') else open
    mode = 'rt' if vcf_file.endswith('.gz') else 'r'
    
    # Get column names from VCF
    vcf_names = get_vcf_names(vcf_file)
    vcf_metadata_cols = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
    vcf_sample_cols = [x for x in vcf_names if x not in vcf_metadata_cols]
    
    # Only use chunked processing if reading all rows or a large number
    if num_rows is None or num_rows > chunk_size:
        result_df = None
        current_chunk = []
        total_rows_read = 0
        
        with opener(vcf_file, mode) as f:
            # Skip header lines
            for line in f:
                if line.startswith('#') and not line.startswith('#CHROM'):
                    continue
                elif line.startswith('#CHROM'):
                    # Found column headers
                    break
            
            # Process data in chunks
            for line in f:
                line_data = line.strip().split('\t')
                current_chunk.append(line_data)
                total_rows_read += 1
                
                # Process chunk when it reaches the desired size or at end of desired rows
                if len(current_chunk) >= chunk_size or (num_rows is not None and total_rows_read >= num_rows):
                    # Convert chunk to DataFrame
                    chunk_df = pd.DataFrame(current_chunk, columns=vcf_names)
                    
                    # Process this chunk
                    processed_chunk = _process_vcf_chunk(
                        chunk_df, 
                        vcf_sample_cols, 
                        sample_id,
                        faster=True
                    )
                    
                    # Append to result or create new result
                    if result_df is None:
                        result_df = processed_chunk
                    else:
                        result_df = pd.concat([result_df, processed_chunk], ignore_index=True)
                    
                    # Report progress
                    print(f"Processed {total_rows_read} rows ({len(current_chunk)} in this chunk)")
                    
                    # Clear memory
                    current_chunk = []
                    
                    # Break if we've reached the desired number of rows
                    if num_rows is not None and total_rows_read >= num_rows:
                        break
            
            # Process any remaining rows
            if current_chunk:
                chunk_df = pd.DataFrame(current_chunk, columns=vcf_names)
                processed_chunk = _process_vcf_chunk(
                    chunk_df, 
                    vcf_sample_cols, 
                    sample_id,
                    faster=True
                )
                
                if result_df is None:
                    result_df = processed_chunk
                else:
                    result_df = pd.concat([result_df, processed_chunk], ignore_index=True)
                
                print(f"Processed {total_rows_read} total rows")
    else:
        # For small numbers of rows, use the original approach without chunking
        current_chunk = []
        row_count = 0
        
        with opener(vcf_file, mode) as f:
            # Skip header lines
            for line in f:
                if line.startswith('#') and not line.startswith('#CHROM'):
                    continue
                elif line.startswith('#CHROM'):
                    # Found column headers
                    break
            
            # Read specified number of data lines
            for line in f:
                if row_count >= num_rows:
                    break
                    
                line_data = line.strip().split('\t')
                current_chunk.append(line_data)
                row_count += 1
        
        # Create DataFrame from chunk
        if not current_chunk:
            print("No data rows found in VCF file")
            return pd.DataFrame()
            
        chunk_df = pd.DataFrame(current_chunk, columns=vcf_names)
        result_df = _process_vcf_chunk(
            chunk_df, 
            vcf_sample_cols, 
            sample_id,
            faster=True
        )
    
    # If we got no results, return empty dataframe
    if result_df is None or result_df.empty:
        print("No data rows processed")
        return pd.DataFrame()
    
    print(f"Read {len(result_df)} total rows")
    
    # Define columns to always include based on the mode
    always_include = []
    if columns != "metadata":  # Only include ID and IID for non-metadata extractions
        always_include = ['ID', 'IID']
    else:
        # For metadata, only include ID
        always_include = ['ID']
    
    # Add a1 and a2 to always include if they exist
    if 'a1' in result_df.columns:
        always_include.append('a1')
    if 'a2' in result_df.columns:
        always_include.append('a2')
    
    # Filter columns based on user selection
    if columns == "all":
        # Keep all columns
        filtered_df = result_df
    elif columns == "metadata":
        # Filter to only metadata columns
        available_metadata = [col for col in metadata_cols_list if col in result_df.columns]
        
        # Always include specified columns if they exist (except IID for metadata)
        for col in always_include:
            if col not in available_metadata and col in result_df.columns:
                available_metadata.append(col)
                
        filtered_df = result_df[available_metadata]
    elif columns == "sample":
        # Filter to only sample-specific columns
        available_sample_cols = [col for col in sample_specific_list if col in result_df.columns]
        
        # Always include specified columns if they exist
        for col in always_include:
            if col not in available_sample_cols and col in result_df.columns:
                available_sample_cols.append(col)
                
        filtered_df = result_df[available_sample_cols]
    elif isinstance(columns, list):
        # Filter to user-specified columns
        available_columns = [col for col in columns if col in result_df.columns]
        
        # Always include ID if it exists
        if 'ID' not in available_columns and 'ID' in result_df.columns:
            available_columns.append('ID')
        
        # Check if we need to include the IID column (not for metadata)
        if columns != "metadata":
            has_sample_column = any(col in sample_specific_list for col in available_columns)
            if has_sample_column and 'IID' not in available_columns and 'IID' in result_df.columns:
                available_columns.append('IID')
        
        # Always include a1 and a2 if they exist and not already specified
        if 'a1' not in available_columns and 'a1' in result_df.columns:
            available_columns.append('a1')
        if 'a2' not in available_columns and 'a2' in result_df.columns:
            available_columns.append('a2')
            
        filtered_df = result_df[available_columns]
    else:
        print(f"Warning: Unrecognized columns option '{columns}'. Returning all columns.")
        filtered_df = result_df
    
    print(f"Extracted and processed {len(filtered_df)} rows with {len(filtered_df.columns)} columns in {time.time() - start_time:.2f} seconds")
    filtered_df = filtered_df.rename(columns={'ID': 'snpID'})
    # Save result if a path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Handle different output formats
        if output_format.lower() == "csv":
            filtered_df.to_csv(output_path, index=False)
            print(f"Saved VCF data to CSV: {output_path}")
        elif output_format.lower() == "parquet":
            if partition_by_chromosome:
                if 'CHROM' not in filtered_df.columns:
                    print("Warning: Cannot partition by chromosome because 'CHROM' column is not in the filtered data.")
                    # Remove .parquet extension if it exists
                    clean_path = output_path.replace('.parquet', '') if output_path.endswith('.parquet') else output_path
                    filtered_df.to_parquet(clean_path, index=False)
                    print(f"Saved VCF data to Parquet (unpartitioned): {clean_path}")
                else:
                    # Remove .parquet extension if it exists
                    parquet_dir = output_path.replace('.parquet', '') if output_path.endswith('.parquet') else output_path
                    os.makedirs(parquet_dir, exist_ok=True)
                    
                    filtered_df.to_parquet(
                        parquet_dir,
                        partition_cols=['CHROM'],
                        compression='snappy',
                        index=False
                    )
                    print(f"Saved VCF data to Parquet (partitioned by chromosome): {parquet_dir}")
            else:
                # Remove .parquet extension if it exists
                clean_path = output_path.replace('.parquet', '') if output_path.endswith('.parquet') else output_path
                filtered_df.to_parquet(clean_path, compression='snappy', index=False)
                print(f"Saved VCF data to Parquet: {clean_path}")
        else:
            print(f"Warning: Unrecognized output format '{output_format}'. Data not saved.")
    
    return filtered_df

# Helper function to process a chunk of VCF data
def _process_vcf_chunk(chunk_df, vcf_sample_cols, sample_id, faster=True):
    """
    Process a single chunk of VCF data using a predefined set of metadata columns.
    
    Args:
        chunk_df: DataFrame containing the chunk of VCF data
        vcf_sample_cols: List of sample columns in the VCF
        sample_id: Sample ID string
        faster: Whether to use faster processing methods (not used when relying on predefined columns)
        
    Returns:
        Processed DataFrame
    """
    # Fix CHROM column name if needed
    chrom_col = '#CHROM'
    if '#CHROM' not in chunk_df.columns and 'CHROM' in chunk_df.columns:
        chrom_col = 'CHROM'
    elif '#CHROM' not in chunk_df.columns:
        chrom_cols = [c for c in chunk_df.columns if c.startswith('chr')]
        if chrom_cols:
            chrom_col = chrom_cols[0]
    
    # Rename chromosome column for consistency
    if chrom_col != 'CHROM':
        chunk_df = chunk_df.rename(columns={chrom_col: 'CHROM'})
    
    # Clean up chromosome column
    chunk_df['CHROM'] = chunk_df['CHROM'].astype(str).str.replace('chr', '')
    
    # Convert numeric columns to appropriate types
    if 'POS' in chunk_df.columns:
        chunk_df['POS'] = pd.to_numeric(chunk_df['POS'], errors='coerce')
    
    # Keep track of columns to drop
    columns_to_drop = []
    
    # Extract INFO fields using the predefined list of keys we know exist in this data
    if 'INFO' in chunk_df.columns:
        # Predefined list of INFO keys we want to extract
        info_keys = [
            'ASSAY_TYPE', 'devR_AB', 'FRAC_T', 'FRAC_G', 'meanTHETA_BB', 
            'meanR_AB', 'devTHETA_AB', 'GC', 'N_AA', 'Orig_Score', 
            'FRAC_C', 'GenTrain_Score', 'devR_BB', 'NORM_ID', 
            'devR_AA', 'Intensity_Threshold', 'meanR_AA', 'devTHETA_AA', 
            'ALLELE_A', 'N_AB', 'meanR_BB', 'meanTHETA_AA', 
            'meanTHETA_AB', 'devTHETA_BB', 'N_BB', 'ALLELE_B', 
            'FRAC_A', 'BEADSET_ID', 'Cluster_Sep'
        ]
        
        # Create a column for each INFO key
        for key in info_keys:
            pattern = f"{key}="
            # Use vectorized extraction
            chunk_df[key] = chunk_df['INFO'].str.extract(f"{pattern}([^;]+)", expand=False)
        
        # Mark INFO column for deletion
        columns_to_drop.append('INFO')
    
    # Process FORMAT column and sample genotype data
    if 'FORMAT' in chunk_df.columns and vcf_sample_cols:
        # Get FORMAT fields from the first row
        if chunk_df['FORMAT'].notna().any():
            format_fields = chunk_df['FORMAT'].iloc[0].split(':')
            
            for sample in vcf_sample_cols:
                if sample in chunk_df.columns:
                    # Use str.split with expand=True directly
                    format_data = chunk_df[sample].str.split(':', expand=True)
                    
                    # Add columns for each format field
                    for i, field in enumerate(format_fields):
                        if i < format_data.shape[1]:  # Only if data exists
                            chunk_df[field] = format_data[i]
                    
                    columns_to_drop.append(sample)
        
        columns_to_drop.append('FORMAT')
    
    # Drop processed columns
    if columns_to_drop:
        chunk_df = chunk_df.drop(columns=columns_to_drop)
    

    if 'IID' not in chunk_df.columns:
        chunk_df['IID'] = sample_id
    # chunk_df['IID'] = chunk_df['IID']
    
    # Create a1 and a2 columns that are properly aligned to A and B alleles
    if all(col in chunk_df.columns for col in ['REF', 'ALT', 'ALLELE_A', 'ALLELE_B']):
        # Initialize a1 and a2 with REF values (the default)
        chunk_df['a1'] = chunk_df['REF']
        chunk_df['a2'] = chunk_df['REF']
        
        # Update a1 based on ALLELE_A
        # If ALLELE_A = 1, then a1 should be ALT
        mask_a1 = chunk_df['ALLELE_A'].astype(str) == '1'
        if mask_a1.any():
            chunk_df.loc[mask_a1, 'a1'] = chunk_df.loc[mask_a1, 'ALT']
        
        # Update a2 based on ALLELE_B
        # If ALLELE_B = 1, then a2 should be ALT
        mask_a2 = chunk_df['ALLELE_B'].astype(str) == '1'
        if mask_a2.any():
            chunk_df.loc[mask_a2, 'a2'] = chunk_df.loc[mask_a2, 'ALT']
    
    return chunk_df

def merge_sample_chunks(chunk_pattern, output_file, sample_id):
    """Merge all chunks for a sample into a single parquet file, partitioned by chromosome."""
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    if not chunk_files:
        print(f"No chunk files found for sample {sample_id}")
        print(f"Looking for pattern: {chunk_pattern}")
        
        # Create an empty output file with the correct structure
        # We need to include chromosome for partitioning
        empty_df = pd.DataFrame({
            'chromosome': [],
            'snpID': [],
            'BAF': [],
            'LRR': [],
            'R': [],
            'THETA': [],
            'GT': []
        })
        
        # Set the correct types
        empty_df = empty_df.astype({
            'chromosome': str,
            'snpID': str,
            'BAF': float,
            'LRR': float,
            'R': float,
            'THETA': float,
            'GT': int
        })
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write the empty DataFrame with chromosome partitioning
        empty_df.to_parquet(output_file, compression='snappy', index=False, 
                           partition_cols=['chromosome'])
        print(f"Created empty partitioned output for sample {sample_id}: {output_file}")
        return True
    
    print(f"Merging {len(chunk_files)} chunk files for sample {sample_id}")
    
    try:
        # Read and concatenate all chunk files
        dfs = []
        for chunk_file in chunk_files:
            df = pd.read_parquet(chunk_file)
            dfs.append(df)
        
        if not dfs:
            print(f"No data found in chunk files for sample {sample_id}")
            return False
            
        # Concatenate all dataframes
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Remove position but keep chromosome for partitioning
        # Add final_df containing only the columns we need plus chromosome
        final_columns = ['chromosome', 'snpID', 'BAF', 'LRR', 'R', 'THETA', 'GT']
        final_df = merged_df[final_columns]
        
        # Write with chromosome partitioning
        # Use a temporary directory with a unique name to avoid conflicts
        import uuid
        temp_dir = f"{output_file}_{uuid.uuid4().hex}.tmp"
        
        # Ensure the temp directory doesn't exist before creating it
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(temp_dir), exist_ok=True)
        
        # Write to the temporary directory with partitioning
        final_df.to_parquet(
            temp_dir, 
            compression='snappy',
            index=False,
            partition_cols=['chromosome']
        )
        
        # Check if output_file is a directory or a file
        # If it's a file, we need to handle differently than if it's a directory
        if os.path.exists(output_file):
            if os.path.isdir(output_file):
                # It's a directory, remove it
                shutil.rmtree(output_file)
            else:
                # It's a file, just remove the file
                os.remove(output_file)
                
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Move the temp directory to the final location
        shutil.move(temp_dir, output_file)
        
        print(f"Successfully created partitioned files for sample {sample_id}: {output_file}")
        
        # Verify at least one of the partition files
        chroms = merged_df['chromosome'].unique()
        if len(chroms) > 0:
            # Check the first chromosome's partition
            part_path = os.path.join(output_file, f"chromosome={chroms[0]}")
            if os.path.exists(part_path):
                # List all partitions in the directory
                parts = os.listdir(output_file)
                print(f"Created {len(parts)} chromosome partitions")
            else:
                print(f"Warning: Expected partition not found at {part_path}")
        
        # Remove chunk files after successful merge
        for chunk_file in chunk_files:
            os.remove(chunk_file)
            
        return True
    except Exception as e:
        print(f"Error merging chunks for sample {sample_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


####### example for extract_vcf_columns #######
####### the following is used to extract all metadata from a single-sample vcf #######
# vcf_path = f"/path/to/vcf/{barcode}.vcf.gz"
# parquet_path_out = f"/path/to/output/metadata"

# full_metadata = extract_vcf_columns(
#     vcf_path,
#     output_path=parquet_path_out,
#     num_rows=None,
#     columns="metadata",
#     output_format="parquet",
#     partition_by_chromosome=True
# )

