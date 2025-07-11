''' 
the following was run in a vm with 8 cores and 16GB of memory
the data for a single barcode was copied from gcs storage to the vm
this also contains a bunch of random checks to ensure data integrity
'''

import numpy as np
import pandas as pd
import os
import time
from snp_metrics_old.snp_metrics import extract_vcf_columns

output_directory = "/home/vitaled2/gp2-microservices/services/idat_utils/data/output"
vcf_path = f"/{output_directory}/snp_metrics/tmp_207847320055/207847320055_R01C01.vcf.gz"

full_metadata = extract_vcf_columns(
    vcf_path,
    output_path=f"{output_directory}/NBA_metadata",
    num_rows=None,
    columns="metadata",
    output_format="parquet",
    partition_by_chromosome=True
)

# parquet_path = f"{output_directory}/NBA_metadata"
# df = pd.read_parquet(parquet_path)
# display(df)

# test_data = extract_vcf_columns(
#     vcf_file=vcf_path,
#     num_rows=1000,
#     columns="metadata"
# )
# display(test_data)

# now check that all columns are the same across samples

# Output directory
output_directory = "/home/vitaled2/gp2-microservices/services/idat_utils/data/output"
comparison_dir = f"{output_directory}/metadata_comparison"
os.makedirs(comparison_dir, exist_ok=True)

# Sample paths
sample_paths = [
    f"/{output_directory}/207847320055/sample_vcfs/207847320055_R01C01.vcf.gz",
    f"/{output_directory}/207847320055/sample_vcfs/207847320055_R02C01.vcf.gz",
    f"/{output_directory}/207847320055/sample_vcfs/207847320055_R03C01.vcf.gz",
    f"/{output_directory}/207847320055/sample_vcfs/207847320055_R04C01.vcf.gz"
]

# Process each sample
sample_dfs = []
sample_names = []

print(f"Processing {len(sample_paths)} samples...")
start_time = time.time()

for i, vcf_path in enumerate(sample_paths):
    sample_id = os.path.basename(vcf_path).replace('.vcf.gz', '')
    sample_names.append(sample_id)
    
    print(f"\nProcessing sample {i+1}/{len(sample_paths)}: {sample_id}")
    output_path = f"{comparison_dir}/{sample_id}_metadata"
    
    # Extract metadata
    metadata_df = extract_vcf_columns(
        vcf_file=vcf_path,
        output_path=output_path,
        num_rows=None,  # Process all rows
        columns="metadata",
        output_format="parquet",
        partition_by_chromosome=False  # Use False for easier comparison
    )
    
    # Load the parquet file to ensure we're comparing the actual saved data
    loaded_df = pd.read_parquet(output_path)
    sample_dfs.append(loaded_df)
    
    print(f"Sample {sample_id}: {len(loaded_df)} rows, {len(loaded_df.columns)} columns")
    print(f"Columns: {loaded_df.columns.tolist()}")

print(f"\nAll samples processed in {time.time() - start_time:.2f} seconds")

# 1. Compare column lists between all samples
all_columns_sets = [set(df.columns) for df in sample_dfs]
common_columns = all_columns_sets[0]
for cols in all_columns_sets[1:]:
    common_columns = common_columns.intersection(cols)

print(f"\nAll samples have {len(common_columns)} common columns")
missing_columns = set()
for i, cols in enumerate(all_columns_sets):
    diff = cols.difference(common_columns)
    if diff:
        print(f"Sample {sample_names[i]} has {len(diff)} unique columns: {diff}")
        missing_columns.update(diff)

# 2. Compare values in common columns
print("\nComparing values across samples...")

# Function to compare dataframes
def compare_dfs(dfs, names):
    # Sort all dataframes the same way for reliable comparison
    sorted_dfs = []
    for df in dfs:
        # We'll sort by CHROM, POS, ID which should create a reliable ordering
        if all(col in df.columns for col in ['CHROM', 'POS', 'ID']):
            sorted_df = df.sort_values(['CHROM', 'POS', 'ID']).reset_index(drop=True)
        else:
            # Fallback sort by position if standardized columns not available
            sorted_df = df.sort_values(df.columns[0]).reset_index(drop=True)
        sorted_dfs.append(sorted_df)
    
    # Check if row counts match
    row_counts = [len(df) for df in sorted_dfs]
    if len(set(row_counts)) > 1:
        print(f"WARNING: Samples have different row counts: {list(zip(names, row_counts))}")
        # Find minimum row count for comparison
        min_rows = min(row_counts)
        # Trim dataframes to same size
        sorted_dfs = [df.iloc[:min_rows] for df in sorted_dfs]
        print(f"Trimming all dataframes to {min_rows} rows for comparison")
    
    # Compare each column individually
    identical_columns = []
    different_columns = []
    
    for col in common_columns:
        columns_identical = True
        # Compare first df to all others
        first_col = sorted_dfs[0][col]
        
        for i in range(1, len(sorted_dfs)):
            other_col = sorted_dfs[i][col]
            
            # Try standard comparison first
            try:
                if not first_col.equals(other_col):
                    columns_identical = False
                    break
            except:
                # Handle categorical columns or other types that might cause issues
                if not (first_col.astype(str) == other_col.astype(str)).all():
                    columns_identical = False
                    break
        
        if columns_identical:
            identical_columns.append(col)
        else:
            different_columns.append(col)
    
    return identical_columns, different_columns

identical_cols, different_cols = compare_dfs(sample_dfs, sample_names)

print(f"\nResults of comparison:")
print(f"- {len(identical_cols)} columns are identical across all samples")
print(f"- {len(different_cols)} columns differ between samples")

if different_cols:
    print(f"\nColumns with differences: {different_cols}")
else:
    print("\nAll common columns have identical values across samples!")

print("\nComparison complete!")




output_directory = "/home/vitaled2/gp2-microservices/services/idat_utils/data/output"
vcf_path1 = f"/{output_directory}/207847320055/sample_vcfs/207847320055_R01C01.vcf.gz"
sample_df1 = extract_vcf_columns(vcf_path1, num_rows=1000)

vcf_path2 = f"/{output_directory}/207847320055/sample_vcfs/207847320055_R02C01.vcf.gz"
sample_df2 = extract_vcf_columns(vcf_path2, num_rows=1000)

vcf_path3 = f"/{output_directory}/207847320055/sample_vcfs/207847320055_R03C01.vcf.gz"
sample_df3 = extract_vcf_columns(vcf_path3, num_rows=1000)

vcf_path4 = f"/{output_directory}/207847320055/sample_vcfs/207847320055_R04C01.vcf.gz"
sample_df4 = extract_vcf_columns(vcf_path4, num_rows=1000)

def compare_columns_across_samples(dfs, column_names):
    identical_columns = []
    
    for col in column_names:
        if all(col in df.columns for df in dfs):  # Ensure column exists in all DFs
            # Extract the column from each DataFrame
            column_series = [df[col] for df in dfs]
            
            # Check if all values are identical for this column
            identical = True
            for i in range(1, len(column_series)):
                if not column_series[0].equals(column_series[i]):
                    identical = False
                    break
            
            if identical:
                identical_columns.append(col)
    
    return identical_columns

# Get common columns among all samples
common_columns = set(sample_df1.columns)
for df in [sample_df2, sample_df3, sample_df4]:
    common_columns &= set(df.columns)

# Find which common columns have identical values
identical_columns = compare_columns_across_samples(
    [sample_df1, sample_df2, sample_df3, sample_df4],
    common_columns
)

print(f"Columns with identical values across all samples: {identical_columns}")

# metadata_cols = [
#     'ID',
#     'ASSAY_TYPE', 
#     'devR_AB', 
#     'POS', 
#     'FRAC_T', 
#     'FRAC_G', 
#     'meanTHETA_BB', 
#     'meanR_AB', 
#     'devTHETA_AB', 
#     'GC', 
#     'N_AA', 
#     'QUAL', 
#     'Orig_Score', 
#     'FRAC_C', 
#     'GenTrain_Score', 
#     'devR_BB', 
#     'NORM_ID', 
#     'devR_AA', 
#     'FILTER', 
#     'Intensity_Threshold', 
#     'meanR_AA', 
#     'CHROM', 
#     'devTHETA_AA', 
#     'ALLELE_A', 
#     'ID', 
#     'N_AB', 
#     'meanR_BB', 
#     'meanTHETA_AA', 
#     'meanTHETA_AB', 
#     'REF', 
#     'devTHETA_BB', 
#     'N_BB', 
#     'ALLELE_B', 
#     'FRAC_A', 
#     'BEADSET_ID', 
#     'ALT', 
#     'Cluster_Sep'
#     ]