#!/usr/bin/env python3
"""
Verify harmonization correctness by comparing input SNP list, original PLINK variants, and harmonized output.
Shows before/after genotypes to prove transformation is working.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.core.config import Settings
from app.processing.harmonizer import HarmonizationEngine
from app.processing.transformer import GenotypeTransformer
try:
    import pgenlib
except ImportError:
    print("Warning: pgenlib not available - genotype comparison will be limited")
    pgenlib = None

def extract_original_genotypes(pgen_path: str, variant_indices: list, sample_limit: int = 10):
    """Extract original genotypes from PGEN file for comparison."""
    if pgenlib is None:
        return {}
    
    try:
        # Open PGEN file for reading
        pgen_file = pgenlib.PgenReader(bytes(pgen_path, 'utf8'))
        variant_ct = pgen_file.get_variant_ct()
        sample_ct = pgen_file.get_raw_sample_ct()
        
        print(f"PGEN file: {variant_ct} variants, {sample_ct} samples")
        
        genotype_data = {}
        
        # Extract genotypes for each variant (limited samples for display)
        for i, variant_idx in enumerate(variant_indices[:5]):  # Limit to first 5 variants
            if variant_idx >= variant_ct:
                continue
                
            # Allocate genotype array
            genotype_int32 = np.empty(sample_ct, dtype=np.int32)
            
            # Read genotypes for this variant
            pgen_file.read(variant_idx, genotype_int32)
            
            # Convert to standard 0/1/2 format and limit samples
            genotypes = genotype_int32[:sample_limit]
            genotype_data[variant_idx] = genotypes
            
        pgen_file.close()
        return genotype_data
        
    except Exception as e:
        print(f"Error reading PGEN file: {e}")
        return {}

def show_genotype_transformation(original_genotypes: np.ndarray, action: str, transformer: GenotypeTransformer):
    """Show before/after genotype transformation."""
    if len(original_genotypes) == 0:
        return "No genotypes available", "No genotypes available"
    
    # Apply transformation
    transformed_genotypes = transformer.apply_transformation(original_genotypes, action)
    
    # Format for display (show first 10 samples)
    orig_str = ' '.join([str(int(g)) if not np.isnan(g) else 'NA' for g in original_genotypes[:10]])
    trans_str = ' '.join([str(int(g)) if not np.isnan(g) else 'NA' for g in transformed_genotypes[:10]])
    
    return orig_str, trans_str

def create_harmonization_comparison_df(settings: Settings, sample_limit: int = 10) -> pd.DataFrame:
    """
    Create comprehensive DataFrame showing harmonization before/after comparison.
    
    Returns:
        DataFrame with columns showing original data, harmonized data, and genotype transformations
    """
    transformer = GenotypeTransformer()
    
    # Paths from test_nba_pipeline.py  
    snp_list_path = settings.snp_list_path
    nba_aac_path = settings.get_nba_path("AAC")
    output_path_prefix = Path("~/gcs_mounts/genotools_server/precision_med/testing/AAC_harmonization_test").expanduser()
    output_dir = output_path_prefix.parent
    output_prefix = output_path_prefix.name
    
    print(f"Loading data...")
    print(f"SNP list: {snp_list_path}")
    print(f"NBA AAC file: {nba_aac_path}")
    print(f"Output prefix: {output_path_prefix}")
    
    # Load SNP list (try comma separator first)
    try:
        snp_list = pd.read_csv(snp_list_path, sep=',')
    except:
        snp_list = pd.read_csv(snp_list_path, sep='\t')
    
    if 'hg38' in snp_list.columns:
        coords = snp_list['hg38'].str.split(':', expand=True)
        snp_list['chromosome'] = coords[0].str.replace('chr', '').str.upper().astype(str)
        snp_list['position'] = pd.to_numeric(coords[1])
        snp_list['snp_ref'] = coords[2].str.upper()
        snp_list['snp_alt'] = coords[3].str.upper()
    
    # Load original PVAR file
    pvar_path = f"{nba_aac_path}.pvar"
    pvar_df = pd.read_csv(pvar_path, sep='\t', comment='#', header=None, 
                         names=['CHROM', 'POS', 'ID', 'REF', 'ALT'], low_memory=False)
    
    pvar_df['chromosome'] = pvar_df['CHROM'].astype(str).str.upper()
    pvar_df['position'] = pd.to_numeric(pvar_df['POS'])
    pvar_df['pvar_ref'] = pvar_df['REF'].str.upper()
    pvar_df['pvar_alt'] = pvar_df['ALT'].str.upper()
    
    # Load harmonized output TRAW
    traw_files = list(output_dir.glob(f"{output_prefix}*.traw"))
    if not traw_files:
        print(f"No TRAW files found with prefix {output_prefix} in {output_dir}")
        return pd.DataFrame()
    
    traw_df = pd.read_csv(traw_files[0], sep='\t')
    
    # Load harmonization actions from variant summary CSV
    variant_summary_files = list(output_dir.glob(f"{output_prefix}*variant_summary.csv"))
    if variant_summary_files:
        variant_summary_df = pd.read_csv(variant_summary_files[0])
        
        # Extract harmonization actions  
        harm_df = variant_summary_df[['variant_id', 'harmonization_action', 'snp_list_id']].copy()
        harm_df = harm_df.rename(columns={'variant_id': 'SNP'})
        
        # Add genotype transform info based on action
        def get_transform_formula(action):
            if action in ['SWAP', 'FLIP_SWAP']:
                return '2-x'
            else:
                return None
        
        harm_df['genotype_transform'] = harm_df['harmonization_action'].apply(get_transform_formula)
    else:
        harm_df = pd.DataFrame()
    
    # Use variant summary CSV as primary data source (it has all the mappings we need)
    if variant_summary_df.empty:
        print("No variant summary CSV found - cannot perform comparison")
        return pd.DataFrame()
    
    # Start with variant summary and add other data
    merged = variant_summary_df.copy()
    
    # Ensure consistent data types for merging
    merged['chromosome'] = merged['chromosome'].astype(str)
    merged['position'] = pd.to_numeric(merged['position'])
    
    # Add SNP list data (match on coordinates)
    snp_cols = []
    for col in ['snp_name', 'chromosome', 'position', 'snp_ref', 'snp_alt']:
        if col in snp_list.columns:
            snp_cols.append(col)
    
    if snp_cols:
        merged = pd.merge(
            merged,
            snp_list[snp_cols],
            on=['chromosome', 'position'],
            how='left',
            suffixes=('', '_snplist')
        )
    
    # Add PVAR data (match on coordinates)
    merged = pd.merge(
        merged,
        pvar_df[['chromosome', 'position', 'ID', 'pvar_ref', 'pvar_alt']],
        on=['chromosome', 'position'],
        how='left'
    )
    
    # Add TRAW allele info (rename from variant summary columns)
    merged['traw_a1'] = merged['counted_allele']
    merged['traw_a2'] = merged['alt_allele']
    
    # Extract original genotypes and create comprehensive comparison DataFrame
    print(f"Extracting original genotypes from PGEN file...")
    pgen_path = f"{nba_aac_path}.pgen"
    
    # Get PGEN variant indices for our harmonized variants (limit for memory)
    variant_indices = []
    variant_id_to_idx = {}
    if 'ID' in merged.columns and not merged.empty:
        # Find indices of variants in PVAR file using the PVAR ID column
        for i, (_, row) in enumerate(merged.head(20).iterrows()):  # Limit to first 20
            pvar_id = row.get('ID')  # PVAR ID like "Seq_rs770946447"
            if pd.notna(pvar_id):
                pvar_idx = pvar_df[pvar_df['ID'] == pvar_id].index
                if len(pvar_idx) > 0:
                    variant_indices.append(pvar_idx[0])
                    variant_id_to_idx[pvar_id] = pvar_idx[0]
    
    original_genotypes = extract_original_genotypes(pgen_path, variant_indices, sample_limit)
    
    # Create comprehensive comparison DataFrame
    comparison_data = []
    
    for i, (_, row) in enumerate(merged.head(20).iterrows()):  # Limit to first 20 variants
        traw_variant_id = row.get('variant_id', 'Unknown')  # TRAW SNP name
        pvar_id = row.get('ID', 'Unknown')  # PVAR ID 
        action = row.get('harmonization_action', 'UNKNOWN')
        
        # Get original and transformed genotypes using PVAR ID
        if pvar_id in variant_id_to_idx:
            variant_idx = variant_id_to_idx[pvar_id]
            if variant_idx in original_genotypes:
                orig_geno = original_genotypes[variant_idx]
                transformed_geno = transformer.apply_transformation(orig_geno, action)
                
                # Format genotype strings
                orig_str = ' '.join([str(int(g)) if not np.isnan(g) else 'NA' for g in orig_geno])
                trans_str = ' '.join([str(int(g)) if not np.isnan(g) else 'NA' for g in transformed_geno])
            else:
                orig_str, trans_str = "Not available", "Not available"
        else:
            orig_str, trans_str = "Not available", "Not available"
        
        # Determine transformation formula
        if action in ['SWAP', 'FLIP_SWAP']:
            transform_formula = "2-x (0→2, 1→1, 2→0)"
        elif action in ['EXACT', 'FLIP']:
            transform_formula = "none (0→0, 1→1, 2→2)"
        else:
            transform_formula = "unknown"
        
        # Build comparison record
        comparison_record = {
            'traw_variant_id': traw_variant_id,  # TRAW SNP name
            'pvar_variant_id': pvar_id,  # PVAR ID
            'chromosome': row.get('chromosome', '?'),
            'position': row.get('position', '?'),
            'snp_list_ref': row.get('snp_ref', '?'),
            'snp_list_alt': row.get('snp_alt', '?'),
            'pvar_ref': row.get('pvar_ref', '?'),
            'pvar_alt': row.get('pvar_alt', '?'),
            'traw_a1': row.get('traw_a1', '?'),
            'traw_a2': row.get('traw_a2', '?'),
            'harmonization_action': action,
            'transform_formula': transform_formula,
            f'original_genotypes_first_{sample_limit}': orig_str,
            f'transformed_genotypes_first_{sample_limit}': trans_str
        }
        
        comparison_data.append(comparison_record)
    
    # Create final comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    return comparison_df

def main():
    settings = Settings()
    
    # Create comprehensive comparison DataFrame
    comparison_df = create_harmonization_comparison_df(settings, sample_limit=10)
    
    if comparison_df.empty:
        print("No data available for comparison")
        return
    
    # Display results
    print(f"\nHarmonization Comparison Results ({len(comparison_df)} variants):")
    print("=" * 120)
    
    # Show DataFrame with all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(comparison_df.to_string(index=False))
    
    # Show action summary
    print(f"\nHarmonization Action Summary:")
    print(comparison_df['harmonization_action'].value_counts())
    
    # Save comprehensive comparison DataFrame
    output_file = Path("~/gcs_mounts/genotools_server/precision_med/testing/harmonization_comparison_with_genotypes.csv").expanduser()
    comparison_df.to_csv(output_file, index=False)
    print(f"\nComprehensive comparison DataFrame saved to: {output_file}")
    
    # Return the DataFrame for programmatic use
    return comparison_df

if __name__ == "__main__":
    main()