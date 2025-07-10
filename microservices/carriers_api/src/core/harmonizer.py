import pandas as pd
import os
import tempfile
from typing import Optional, List
from src.core.plink_operations import (
    ExtractSnpsCommand, 
    FrequencyCommand, 
    SwapAllelesCommand, 
    UpdateAllelesCommand, 
    ExportCommand, 
    CopyFilesCommand
)


class AlleleHarmonizer:
    def harmonize_and_extract(self, 
                             geno_path: str, 
                             reference_path: Optional[str], 
                             plink_out: str,
                             additional_args: List[str] = None) -> str:
        """
        Harmonize alleles if reference provided, then extract SNPs and execute PLINK operations.
        
        Args:
            geno_path: Path to PLINK file prefix
            reference_path: Path to reference allele file
            plink_out: Output path prefix
            additional_args: Optional list of additional PLINK arguments
            
        Returns:
            str: Path to the subset SNP list file with matched IDs
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            current_geno = geno_path
            
            # Step 1: Find common SNPs between genotype and reference
            common_snps_path = self._find_common_snps(geno_path, reference_path, os.path.join(tmpdir, "common_snps"))
            
            # Step 2: Extract only the common SNPs (for efficiency with large files)
            # Also standardizes chromosome format during extraction
            extracted_prefix = os.path.join(tmpdir, "extracted")
            extract_cmd = ExtractSnpsCommand(geno_path, common_snps_path, extracted_prefix, output_chr='M')
            extract_cmd.execute()
            
            # Step 3: Harmonize alleles on the smaller extracted dataset
            harmonized_prefix = os.path.join(tmpdir, "harmonized")
            match_info_path = os.path.join(tmpdir, "common_snps_match_info.tsv")
            self._harmonize_alleles(extracted_prefix, reference_path, harmonized_prefix, match_info_path)
            current_geno = harmonized_prefix
            
            # Step 4: Build and execute PLINK command with all operations
            export_cmd = ExportCommand(
                pfile=current_geno, 
                out=plink_out, 
                additional_args=additional_args
            )
            export_cmd.execute()
            
            # Create subset SNP list with matched IDs
            subset_snp_path = f"{plink_out}_subset_snps.csv"
            
            # Read the match info to get the mapping between genotype IDs and reference variants
            match_info_path = os.path.join(tmpdir, "common_snps_match_info.tsv")
            match_info = pd.read_csv(match_info_path, sep='\t')
            
            # Read original reference SNP list
            ref_df = pd.read_csv(reference_path, dtype={'chrom': str})
            
            # Use hg38 as variant_id (with uppercase alleles)
            ref_df['hg38'] = ref_df['hg38'].astype(str).str.strip().str.replace(' ', '')
            ref_df['variant_id'] = ref_df['hg38'].str.upper()
            
            # Merge match info with reference data to get subset of matched variants
            subset_snps = match_info.merge(ref_df, left_on='variant_id_ref', right_on='variant_id', how='inner')
            
            # Keep original reference columns plus the genotype ID
            ref_cols = list(ref_df.columns)
            if 'id' not in ref_cols:
                ref_cols.append('id')
            
            # Rename id_geno to id for consistency
            subset_snps = subset_snps.rename(columns={'id_geno': 'id'})
            
            # Select relevant columns, keeping all from original reference
            cols_to_keep = [col for col in ref_cols if col in subset_snps.columns]
            subset_snps = subset_snps[cols_to_keep]
            
            # Save the subset SNP list
            subset_snps.to_csv(subset_snp_path, index=False)
            
            return subset_snp_path
    
    def _find_common_snps(self, pfile: str, reference: str, out: str, chunk_size: int = 500000) -> str:
        """
        Find SNPs common between the PLINK file and reference using exact allele matching.
        This preserves functional variants that differ at the same position.
        
        Args:
            pfile: Path to PLINK file prefix
            reference: Path to TSV file with columns chrom, pos, a1, a2
            out: Output path prefix
            chunk_size: Number of variants to process at once (default 100,000)
            
        Returns:
            str: Path to file containing common SNP IDs
        """
        pvar_path = f"{pfile}.pvar"
        
        # Read and prepare reference data
        ref_df = pd.read_csv(reference, dtype={'chrom': str})
        ref_df['hg38'] = ref_df['hg38'].astype(str).str.strip().str.replace(' ', '')
        hg38_parts = ref_df['hg38'].str.split(':')
        ref_df['chrom'] = hg38_parts.str[0]
        ref_df['pos'] = hg38_parts.str[1]
        ref_df['a1'] = hg38_parts.str[2].str.upper()
        ref_df['a2'] = hg38_parts.str[3].str.upper()
        
        # CRITICAL CHANGE: Create exact variant IDs instead of just position IDs
        ref_df['exact_variant_id'] = (ref_df['chrom'] + ':' + ref_df['pos'] + 
                                      ':' + ref_df['a1'] + ':' + ref_df['a2'])
        ref_df['variant_id'] = ref_df['hg38'].str.upper()
        
        # For efficient lookup, create both exact matches and flip/swap variants
        ref_exact_variants = set(ref_df['exact_variant_id'].values)
        
        # Also create flipped and swapped versions for matching
        complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
        ref_variants_all = set()
        ref_variant_mapping = {}  # Maps genotype variant to reference info
        
        for _, row in ref_df.iterrows():
            exact_id = row['exact_variant_id']
            ref_variants_all.add(exact_id)
            ref_variant_mapping[exact_id] = {
                'variant_id_ref': row['variant_id'],
                'match_type': 'exact',
                'ref_row': row
            }
            
            # Add swapped version
            swap_id = f"{row['chrom']}:{row['pos']}:{row['a2']}:{row['a1']}"
            ref_variants_all.add(swap_id)
            ref_variant_mapping[swap_id] = {
                'variant_id_ref': row['variant_id'],
                'match_type': 'swap',
                'ref_row': row
            }
            
            # Add flipped versions
            a1_flip = complement.get(row['a1'], row['a1'])
            a2_flip = complement.get(row['a2'], row['a2'])
            flip_id = f"{row['chrom']}:{row['pos']}:{a1_flip}:{a2_flip}"
            flip_swap_id = f"{row['chrom']}:{row['pos']}:{a2_flip}:{a1_flip}"
            
            ref_variants_all.add(flip_id)
            ref_variant_mapping[flip_id] = {
                'variant_id_ref': row['variant_id'],
                'match_type': 'flip',
                'ref_row': row
            }
            
            ref_variants_all.add(flip_swap_id)
            ref_variant_mapping[flip_swap_id] = {
                'variant_id_ref': row['variant_id'],
                'match_type': 'flip_swap',
                'ref_row': row
            }
        
        print(f"Looking for {len(ref_exact_variants)} exact variants (with {len(ref_variants_all)} total including harmonized versions)")
        
        all_matches = []
        
        # Process pvar in chunks with improved parsing and performance
        def get_chunk_reader(pvar_path, chunk_size):
            """Get appropriate chunk reader for pvar file format"""
            # First, detect the file format by reading a few lines
            with open(pvar_path, 'r') as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()
            
            if first_line.startswith('#CHROM') or first_line.startswith('CHROM'):
                # Header format - use standard column names
                return pd.read_csv(pvar_path, sep='\t', chunksize=chunk_size, dtype={'#CHROM': str})
            else:
                # Headerless format - skip comment lines and use only first 5 columns
                return pd.read_csv(pvar_path, sep='\t', comment='#', header=None,
                                  names=['chrom', 'pos', 'id', 'a1', 'a2'],
                                  usecols=[0, 1, 2, 3, 4],
                                  dtype={'chrom': str}, chunksize=chunk_size)
        
        try:
            chunk_reader = get_chunk_reader(pvar_path, chunk_size)
            
            for i, chunk in enumerate(chunk_reader):
                # Standardize column names if needed
                if chunk.columns[0] in ['#CHROM', 'CHROM']:
                    chunk.columns = ['chrom', 'pos', 'id', 'a1', 'a2'] + list(chunk.columns[5:])
                
                # PERFORMANCE OPTIMIZATION: Pre-filter chunk by exact variant IDs
                # This dramatically reduces the number of variants we need to process
                # Clean and standardize data first
                chunk['chrom'] = chunk['chrom'].astype(str).str.strip()
                chunk['pos'] = chunk['pos'].astype(str).str.strip()
                chunk['id'] = chunk['id'].astype(str).str.strip()
                chunk['a1'] = chunk['a1'].astype(str).str.strip().str.upper()
                chunk['a2'] = chunk['a2'].astype(str).str.strip().str.upper()
                
                chunk['exact_variant_id'] = (chunk['chrom'] + ':' + chunk['pos'] + ':' + 
                                            chunk['a1'] + ':' + chunk['a2'])
                
                # Only process variants that exist in our reference set
                chunk_filtered = chunk[chunk['exact_variant_id'].isin(ref_variants_all)]
                
                if not chunk_filtered.empty:
                    # Process only the filtered chunk
                    chunk_matches = []
                    for _, row in chunk_filtered.iterrows():
                        geno_variant_id = row['exact_variant_id']
                        if geno_variant_id in ref_variant_mapping:
                            match_info = ref_variant_mapping[geno_variant_id]
                            match_row = {
                                'id_geno': row['id'],
                                'variant_id_ref': match_info['variant_id_ref'],
                                'match_type': match_info['match_type'],
                                'a1_ref': match_info['ref_row']['a1'],
                                'a2_ref': match_info['ref_row']['a2']
                            }
                            # Add snp_name_ref if available
                            if 'snp_name' in match_info['ref_row']:
                                match_row['snp_name_ref'] = match_info['ref_row']['snp_name']
                            chunk_matches.append(match_row)
                    
                    if chunk_matches:
                        all_matches.append(pd.DataFrame(chunk_matches))
                        print(f"Processed chunk {i+1}, found {len(chunk_matches)} matches (filtered {len(chunk)} -> {len(chunk_filtered)} variants)")
                else:
                    print(f"Processed chunk {i+1}, no matches found (filtered {len(chunk)} -> 0 variants)")
                
        except Exception as e:
            print(f"Error processing pvar file: {e}")
            raise
        
        # Combine all matches
        if all_matches:
            true_matches = pd.concat(all_matches, ignore_index=True)
            # REMOVED: drop_duplicates(subset=['id_geno']) - this was losing functional variants!
            # Each match represents a distinct functional variant, even if they share genotype IDs
            print(f"Total variant matches found: {len(true_matches)}")
        else:
            true_matches = pd.DataFrame()
            print("No matches found")
        
        # Ensure id_geno exists in true_matches
        if not true_matches.empty and 'id_geno' not in true_matches.columns:
            print(f"ERROR: id_geno column not found. Available columns: {list(true_matches.columns)}")
            if 'id' in true_matches.columns:
                true_matches = true_matches.rename(columns={'id': 'id_geno'})
            else:
                raise ValueError("Cannot find variant ID column in merged data")
        
        # Write common SNP IDs to file
        # For PLINK extraction, we only need unique genotype variant IDs
        common_snps_path = f"{out}.txt"
        if not true_matches.empty:
            unique_geno_ids = true_matches['id_geno'].drop_duplicates()
            unique_geno_ids.to_csv(common_snps_path, index=False, header=False)
            print(f"Writing {len(unique_geno_ids)} unique genotype variant IDs for PLINK extraction")
        else:
            # Create empty file if no matches
            with open(common_snps_path, 'w') as f:
                pass
        
        # Save match information for harmonization step
        match_info_path = f"{out}_match_info.tsv"
        if not true_matches.empty:
            # Debug: print available columns
            print(f"DEBUG: Available columns in true_matches: {list(true_matches.columns)}")
            
            match_info_cols = ['id_geno', 'variant_id_ref', 'match_type']
            # Add reference columns that exist
            for col in ['a1_ref', 'a2_ref', 'snp_name_ref']:
                if col in true_matches.columns:
                    match_info_cols.append(col)
            
            # Filter to only include columns that actually exist
            existing_match_info_cols = [col for col in match_info_cols if col in true_matches.columns]
            print(f"DEBUG: Using columns for match info: {existing_match_info_cols}")
            
            true_matches[existing_match_info_cols].to_csv(match_info_path, sep='\t', index=False)
        else:
            # Create empty match info file
            pd.DataFrame(columns=['id_geno', 'variant_id_ref', 'match_type']).to_csv(match_info_path, sep='\t', index=False)
        
        return common_snps_path
    
    def _extract_snps(self, pfile: str, snps_file: str, out: str) -> None:
        """
        Extract specified SNPs from PLINK file.
        
        Args:
            pfile: Path to PLINK file prefix
            snps_file: Path to file with SNP IDs to extract
            out: Output path prefix
        """
        extract_cmd = ExtractSnpsCommand(pfile, snps_file, out)
        extract_cmd.execute()
    
    def _harmonize_alleles(self, pfile: str, reference: str, out: str, match_info_path: str) -> None:
        """
        Harmonize alleles in PLINK files to match reference alleles for carrier screening.
        Preserves biological meaning by keeping alleles as specified in reference (a2 = allele of interest).
        
        Args:
            pfile: Path to PLINK file prefix
            reference: Path to TSV file with columns chrom, pos, a1, a2
            out: Output path prefix
            match_info_path: Path to match information file
        """
        # Read match information
        match_info = pd.read_csv(match_info_path, sep='\t')
        
        # Extract SNP IDs by match type
        swap_mask = match_info['match_type'].isin(['swap', 'flip_swap'])
        flip_mask = match_info['match_type'].isin(['flip', 'flip_swap'])
        
        with tempfile.TemporaryDirectory() as nested_tmpdir:
            current_pfile = pfile
            update_alleles_path = os.path.join(nested_tmpdir, "update_alleles.txt")
            swap_path = os.path.join(nested_tmpdir, "swap_snps.txt")
            updated_pfile = os.path.join(nested_tmpdir, "updated")
            
            # Handle flipping
            flip_snps = match_info[flip_mask]
            if not flip_snps.empty:
                # Read pvar to get current alleles
                pvar_path = f"{pfile}.pvar"
                try:
                    # Try reading with header first (PLINK2 format)
                    pvar = pd.read_csv(pvar_path, sep='\t', dtype={'#CHROM': str})
                    # Rename columns to standard names
                    pvar.columns = ['chrom', 'pos', 'id', 'a1', 'a2'] + list(pvar.columns[5:])
                except:
                    # Fall back to headerless format
                    pvar = pd.read_csv(pvar_path, sep='\t', comment='#', header=None,
                                      names=['chrom', 'pos', 'id', 'a1', 'a2'],
                                      usecols=[0, 1, 2, 3, 4],
                                      dtype={'chrom': str})
                
                # Merge with flip SNPs to get alleles
                flip_snps_with_alleles = flip_snps.merge(pvar, left_on='id_geno', right_on='id')
                
                # Prepare update alleles file
                complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
                update_alleles = flip_snps_with_alleles[['id', 'a1', 'a2']].copy()
                update_alleles['new_a1'] = update_alleles['a1'].map(lambda x: complement.get(x, x))
                update_alleles['new_a2'] = update_alleles['a2'].map(lambda x: complement.get(x, x))
                
                # Write the update alleles file with 5 columns: ID, old-A1, old-A2, new-A1, new-A2
                update_alleles[['id', 'a1', 'a2', 'new_a1', 'new_a2']].to_csv(
                    update_alleles_path, sep='\t', index=False, header=False)
                
                # Execute update alleles command
                update_cmd = UpdateAllelesCommand(pfile, update_alleles_path, updated_pfile)
                update_cmd.execute()
                current_pfile = updated_pfile
            
            # Handle swapping
            swap_snps = match_info[swap_mask]
            if not swap_snps.empty:
                # For swap operations, we need to know which allele to make REF
                # We'll use the reference a1 as the target REF allele
                swap_snps[['id_geno', 'a1_ref']].to_csv(swap_path, sep='\t', index=False, header=False)
                reference_adjusted = os.path.join(nested_tmpdir, "reference_adjusted")
                
                swap_cmd = SwapAllelesCommand(current_pfile, swap_path, reference_adjusted)
                swap_cmd.execute()
                current_pfile = reference_adjusted
            
            # CARRIER SCREENING MODIFICATION: Skip frequency-based allele swapping
            # For carrier screening, we preserve the biological meaning of alleles as defined
            # in the reference SNP list (where a2 is always the allele of interest).
            # Frequency-based swapping would break the biological interpretation.
            
            # Just copy the harmonized files without frequency-based swapping
            copy_cmd = CopyFilesCommand(current_pfile, out)
            copy_cmd.execute() 