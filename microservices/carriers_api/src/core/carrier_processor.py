import pandas as pd
from typing import Dict, Tuple, List
from src.core.harmonizer import AlleleHarmonizer
from src.core.data_repository import DataRepository
from src.core.genotype_converter import GenotypeConverter
from src.core.carrier_validator import CarrierValidator


class CarrierProcessorFactory:
    @staticmethod
    def create_variant_processor(harmonizer: AlleleHarmonizer, data_repo: DataRepository) -> 'VariantProcessor':
        """Create a VariantProcessor instance"""
        return VariantProcessor(harmonizer, data_repo)
    
    @staticmethod
    def create_carrier_extractor(variant_processor: 'VariantProcessor', 
                                genotype_converter: GenotypeConverter,
                                data_repo: DataRepository) -> 'CarrierExtractor':
        """Create a CarrierExtractor instance"""
        return CarrierExtractor(variant_processor, genotype_converter, data_repo)
    
    @staticmethod
    def create_carrier_combiner(data_repo: DataRepository) -> 'CarrierCombiner':
        """Create a CarrierCombiner instance"""
        return CarrierCombiner(data_repo)
    
    @staticmethod
    def create_validator(data_repo: DataRepository, 
                         genotype_converter: GenotypeConverter) -> 'CarrierValidator':
        """Create a CarrierValidator instance"""
        return CarrierValidator(data_repo, genotype_converter)


# Processing Components (utilize Strategy pattern internally)
class VariantProcessor:
    def __init__(self, harmonizer: AlleleHarmonizer, data_repo: DataRepository):
        self.harmonizer = harmonizer
        self.data_repo = data_repo
    
    def extract_variants(self, geno_path: str, reference_path: str, plink_out: str) -> Tuple[pd.DataFrame, str]:
        """
        Extract variant statistics using plink2 from genotype data.
        
        Args:
            geno_path: Path to the plink2 pfile prefix (without extensions)
            reference_path: Path to reference allele file for harmonization
            plink_out: Prefix for output files
            
        Returns:
            Tuple[pd.DataFrame, str]: Combined variant statistics with frequency/missingness data and path to subset SNP file
        """
        # Use the updated harmonize_and_extract method that returns subset SNP path
        subset_snp_path = self.harmonizer.harmonize_and_extract(geno_path, reference_path, plink_out)
        
        # Initialize empty DataFrame for results
        var_stats = pd.DataFrame()
        
        try:
            # Read frequency data
            freq = self.data_repo.read_csv(f"{plink_out}.afreq", sep='\t')
            freq.rename(columns={'ID':'SNP'}, inplace=True)
            var_stats = freq
            
            # Read variant missingness data
            try:
                vmiss = self.data_repo.read_csv(f"{plink_out}.vmiss", sep='\t')
                vmiss.rename(columns={'ID':'SNP'}, inplace=True)
                
                # Merge frequency and missingness data
                var_stats = freq.merge(vmiss[['SNP', 'F_MISS']], on='SNP', how='left')
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"Warning: Missing data file {plink_out}.vmiss not found or empty")
                # var_stats remains freq-only
                
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Warning: Frequency data file {plink_out}.afreq not found or empty")
            # Check if at least we have missingness data
            try:
                vmiss = self.data_repo.read_csv(f"{plink_out}.vmiss", sep='\t')
                vmiss.rename(columns={'ID':'SNP'}, inplace=True)
                var_stats = vmiss
                print(f"Found missingness data but no frequency data")
            except (FileNotFoundError, pd.errors.EmptyDataError):
                print(f"Error: Neither frequency nor missingness data available")
                # Return empty DataFrame with expected column
                var_stats = pd.DataFrame(columns=['SNP'])
        
        return var_stats, subset_snp_path


class CarrierExtractor:
    def __init__(self, variant_processor: VariantProcessor, 
                 genotype_converter: GenotypeConverter,
                 data_repo: DataRepository):
        self.variant_processor = variant_processor
        self.genotype_converter = genotype_converter
        self.data_repo = data_repo

    def extract_carriers(self, geno_path: str, snplist_path: str, out_path: str) -> Dict[str, str]:
        """
        Extract carrier information for given SNPs from a PLINK2 dataset.
        
        Args:
            geno_path: Path to PLINK2 files prefix (without .pgen/.pvar/.psam extension)
            snplist_path: Path to file containing SNP information
            out_path: Output path prefix for generated files
        
        Returns:
            dict: Paths to generated carrier files
        """
        # Read SNP information to get the id to snp_name mapping
        snp_list_df = self.data_repo.read_csv(snplist_path)
        
        # Prepare prefix for PLINK output
        plink_out = f"{out_path}_snps"
        
        # Extract variant statistics using the SNP list as reference and get subset SNP path
        var_stats, subset_snp_path = self.variant_processor.extract_variants(geno_path, snplist_path, plink_out)
        
        # Read the subset SNP list that contains matched IDs
        subset_snp_df = self.data_repo.read_csv(subset_snp_path)
        
        # Read and process traw data
        traw = self.data_repo.read_csv(f"{plink_out}.traw", sep='\t')
        
        # Use the subset SNP dataframe for merging instead of the original
        traw_merged = subset_snp_df[['id','chrom','pos','a1','a2']].merge(traw, how='left', left_on='id', right_on='SNP')
        
        # Add snp_name to traw_merged if available in subset_snp_df
        if 'snp_name' in subset_snp_df.columns:
            traw_merged = traw_merged.merge(subset_snp_df[['id', 'snp_name']], on='id', how='left')
            # Define column sets including snp_name
            colnames = ['id', 'snp_name', 'chrom', 'pos', 'a1', 'a2','CHR', 'SNP', '(C)M', 'POS', 'COUNTED', 'ALT']
        else:
            # Define column sets without snp_name
            colnames = ['id', 'chrom', 'pos', 'a1', 'a2','CHR', 'SNP', '(C)M', 'POS', 'COUNTED', 'ALT']
            
        var_cols = [x for x in colnames if x not in ['id']]
        sample_cols = list(traw_merged.drop(columns=colnames).columns)
        
        # Process final traw data
        traw_final = traw_merged.loc[:, colnames + sample_cols]
        
        # Create string format output - use snp_name for genotype conversion if available
        traw_out = traw_final.copy()
        if 'snp_name' in traw_final.columns:
            # Use snp_name for genotype naming
            traw_out[sample_cols] = traw_out.apply(
                lambda row: [self.genotype_converter.convert(row[col], row['snp_name']) for col in sample_cols],
                axis=1,
                result_type='expand'
            )
        else:
            # Fall back to using id for genotype naming
            traw_out[sample_cols] = traw_out.apply(
                lambda row: [self.genotype_converter.convert(row[col], row['id']) for col in sample_cols],
                axis=1,
                result_type='expand'
            )
        
        # Create comprehensive variant information file by combining subset_snp metadata with PLINK statistics
        # 
        # HARMONIZATION EXPLANATION:
        # During harmonization, alleles are matched between input reference and genotype data:
        # - Original input alleles (ref_allele_orig, alt_allele_orig) are from your input SNP list
        # - PLINK alleles (plink_counted, plink_alt) are from the harmonized genotype data
        # - These may differ due to strand flips, allele swaps, or reference/alt designation
        # - PLINK uses "COUNTED" allele for frequency calculations (equivalent to REF)
        # - Genotype calls in carriers files are relative to the harmonized PLINK alleles
        comprehensive_var_info = subset_snp_df.copy()
        
        # Add PLINK statistics (frequency and missingness data)
        if not var_stats.empty and 'SNP' in var_stats.columns:
            comprehensive_var_info = comprehensive_var_info.merge(var_stats, left_on='id', right_on='SNP', how='left')
            # Remove redundant SNP column
            comprehensive_var_info.drop(columns=['SNP'], inplace=True, errors='ignore')
        
        # Remove any chromosome columns from comprehensive_var_info before merging to avoid conflicts
        # Only keep the 'chrom' column from traw data
        chr_cols_to_remove = ['chrom', 'CHR', '#CHROM']
        comprehensive_var_info.drop(columns=chr_cols_to_remove, inplace=True, errors='ignore')
        
        # Add only the chromosome column from traw metadata (the only column we want to keep)
        if 'CHR' in traw_final.columns:
            plink_metadata = traw_final[['id', 'CHR']].copy()
            plink_metadata.rename(columns={'CHR': 'chrom'}, inplace=True)
            comprehensive_var_info = comprehensive_var_info.merge(plink_metadata, on='id', how='left')
        
        # Clean up any remaining unwanted columns
        columns_to_drop = ['variant_id']
        comprehensive_var_info.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        
        # Remove any remaining allele columns that might exist
        redundant_allele_cols = ['REF', 'ALT', 'PROVISIONAL_REF?']
        comprehensive_var_info.drop(columns=redundant_allele_cols, inplace=True, errors='ignore')
            
        # Ensure pos is integer if it exists
        if 'pos' in comprehensive_var_info.columns:
            comprehensive_var_info['pos'] = comprehensive_var_info['pos'].astype(int)
        
        # Reorder columns for better readability: id before snp_name_alt, chrom before pos
        desired_column_order = [
            'id', 'snp_name', 'snp_name_alt', 'locus', 'rsid', 
            'hg38', 'hg19', 'chrom', 'pos', 'a1', 'a2', 'COUNTED',
            'ancestry', 'submitter_email', 'precision_medicine', 'pipeline'
            ]
        
        frequency_columns = ['ALT_FREQS', 'OBS_CT', 'F_MISS']
        
        # Build final column order: desired columns first, then frequency columns, then any remaining columns
        available_desired_cols = [col for col in desired_column_order if col in comprehensive_var_info.columns]
        available_freq_cols = [col for col in frequency_columns if col in comprehensive_var_info.columns]
        remaining_cols = [col for col in comprehensive_var_info.columns if col not in available_desired_cols + available_freq_cols]
        
        final_column_order = available_desired_cols + available_freq_cols + remaining_cols
        comprehensive_var_info = comprehensive_var_info[final_column_order]
        
        # Save the comprehensive variant information file (replaces both var_info and subset_snps)
        var_info_output_path = f"{out_path}_var_info.csv"
        self.data_repo.write_csv(comprehensive_var_info, var_info_output_path, index=False)

        # Process and save string format
        carriers_string = traw_out.drop(columns=var_cols).set_index('id').T.reset_index()
        carriers_string.columns.name = None
        carriers_string = carriers_string.fillna('./.')
        carriers_string = carriers_string.astype(str)
        carriers_string.rename(columns={'index':'IID'}, inplace=True)
        carriers_string.loc[:,'IID'] = carriers_string.loc[:,'IID'].str.replace('0_', '')
        
        # Keep original probe IDs as column names (removed renaming to snp_name)
        # if 'snp_name' in subset_snp_df.columns:
        #     id_to_snp_name = subset_snp_df.set_index('id')['snp_name'].to_dict()
        #     carriers_string = carriers_string.rename(columns=id_to_snp_name)
        
        self.data_repo.write_csv(carriers_string, f"{out_path}_carriers_string.csv", index=False)
        
        # Process and save integer format
        carriers_int = traw_final.drop(columns=var_cols).set_index('id').T.reset_index()
        carriers_int.columns.name = None
        carriers_int.rename(columns={'index':'IID'}, inplace=True)
        carriers_int.loc[:,'IID'] = carriers_int.loc[:,'IID'].str.replace('0_', '')
        
        # Keep original probe IDs as column names (removed renaming to snp_name)
        # if 'snp_name' in subset_snp_df.columns:
        #     carriers_int = carriers_int.rename(columns=id_to_snp_name)
        
        self.data_repo.write_csv(carriers_int, f"{out_path}_carriers_int.csv", index=False)
        
        return {
            'var_info': f"{out_path}_var_info.csv",
            'carriers_string': f"{out_path}_carriers_string.csv",
            'carriers_int': f"{out_path}_carriers_int.csv"
        }


class CarrierCombiner:
    def __init__(self, data_repo: DataRepository):
        self.data_repo = data_repo

    def combine_carrier_files(self, results_by_label: Dict[str, Dict[str, str]], 
                             key_file: str, out_path: str, track_probe_usage: bool = True) -> Dict[str, str]:
        """
        Combine carrier files from multiple ancestry labels into consolidated output files.
        Updated to incorporate improved logic from combine_array_data_outputs().
        
        Args:
            results_by_label: Dictionary mapping ancestry labels to their extract_carriers results
            key_file: Path to key file containing study information (None if not using study info)
            out_path: Path prefix for output files
            track_probe_usage: Whether to track which variants were used per population
        
        Returns:
            dict: Paths to generated output files
        """
        print(f"\n=== Combining Array Data Outputs ===")
        
        # Initialize combined dataframes
        combined_carriers_string = pd.DataFrame()
        combined_carriers_int = pd.DataFrame() 
        combined_var_info = pd.DataFrame()
        
        # Track which var_info columns we've seen to handle population-specific frequency columns
        var_info_base = None
        
        for i, (label, results) in enumerate(results_by_label.items()):
            print(f"Processing {label} for combination...")
            
            # Read the files
            var_info = self.data_repo.read_csv(results['var_info'])
            carriers_string = self.data_repo.read_csv(results['carriers_string'])
            carriers_int = self.data_repo.read_csv(results['carriers_int'])
            
            # Process var_info (combine frequency information from all populations)
            if var_info_base is None:
                # First population - use as base and drop frequency columns
                var_info_base = var_info.copy()
                freq_cols = ['ALT_FREQS', 'OBS_CT']
                if 'F_MISS' in var_info_base.columns:
                    freq_cols.append('F_MISS')
                var_info_base = var_info_base.drop(columns=freq_cols, errors='ignore')
                combined_var_info = var_info_base.copy()
            
            # Add population-specific frequency columns
            combined_var_info[f'ALT_FREQS_{label}'] = var_info['ALT_FREQS']
            combined_var_info[f'OBS_CT_{label}'] = var_info['OBS_CT']
            if 'F_MISS' in var_info.columns:
                combined_var_info[f'F_MISS_{label}'] = var_info['F_MISS']
            
            # Track which variant IDs were used in carriers data for this ancestry
            if track_probe_usage:
                # Mark all variants as used (removed F_MISS prioritization)
                combined_var_info[f'{label}_probe_used'] = True
                
                # Add detailed logging
                used_ids = combined_var_info[combined_var_info[f'{label}_probe_used']]['id'].tolist()
                unused_ids = combined_var_info[~combined_var_info[f'{label}_probe_used']]['id'].tolist()
                
                print(f"{label}: {len(used_ids)} variants used in carriers data, {len(unused_ids)} variants not used")
            
            # Process carriers data (add ancestry column and combine)
            carriers_string['ancestry'] = label
            carriers_int['ancestry'] = label
            
            # Clean IID column (remove '0_' prefix if present)
            carriers_string['IID'] = carriers_string['IID'].str.replace('0_', '')
            carriers_int['IID'] = carriers_int['IID'].str.replace('0_', '')
            
            # Combine with previous data
            combined_carriers_string = pd.concat([combined_carriers_string, carriers_string], ignore_index=True)
            combined_carriers_int = pd.concat([combined_carriers_int, carriers_int], ignore_index=True)
        
        # Calculate aggregate statistics for var_info
        self._calculate_aggregate_statistics(combined_var_info)
        
        # Clean up unnecessary columns
        self._cleanup_columns(combined_var_info)
        
        # Prepare final output format
        combined_carriers_string_final, combined_carriers_int_final = self._prepare_final_output(
            combined_carriers_string, combined_carriers_int, key_file)
        
        # Save combined files
        return self._save_combined_files(combined_var_info, combined_carriers_string_final, 
                                       combined_carriers_int_final, out_path)
    
    def _calculate_aggregate_statistics(self, var_info: pd.DataFrame) -> None:
        """Calculate aggregate statistics with improved weighted averages"""
        print("Calculating aggregate statistics...")
        
        # Get frequency and observation columns
        freq_cols = [col for col in var_info.columns if col.startswith('ALT_FREQS_')]
        obs_cols = [col for col in var_info.columns if col.startswith('OBS_CT_')]
        miss_cols = [col for col in var_info.columns if col.startswith('F_MISS_')]
        
        # Calculate total observations
        var_info['OBS_CT'] = var_info[obs_cols].sum(axis=1)
        
        # Calculate weighted average allele frequency
        weighted_sum = pd.Series(0, index=var_info.index)
        for i, freq_col in enumerate(freq_cols):
            matching_obs_col = obs_cols[i]
            weighted_sum += var_info[freq_col] * var_info[matching_obs_col]
        
        # Avoid division by zero
        var_info['ALT_FREQS'] = weighted_sum / var_info['OBS_CT'].replace(0, float('nan'))
        
        # Calculate weighted average missingness (improved from original)
        if miss_cols:
            miss_weighted_sum = pd.Series(0, index=var_info.index)
            for i, miss_col in enumerate(miss_cols):
                matching_obs_col = obs_cols[i]
                miss_weighted_sum += var_info[miss_col] * var_info[matching_obs_col]
            
            # Weighted average missingness
            var_info['F_MISS'] = miss_weighted_sum / var_info['OBS_CT'].replace(0, float('nan'))
        
        # Handle NaN values in frequency columns
        self._handle_nan_values(var_info, freq_cols, obs_cols, miss_cols)
    
    def _handle_nan_values(self, var_info: pd.DataFrame, freq_cols: List[str], 
                          obs_cols: List[str], miss_cols: List[str]) -> None:
        """Handle NaN values in frequency columns after calculations"""
        print("Handling NaN values in frequency columns...")
        
        # Set NaN to 0.0 for ALT_FREQS columns when corresponding OBS_CT > 0
        for i, freq_col in enumerate(freq_cols):
            matching_obs_col = obs_cols[i]
            nan_mask = var_info[freq_col].isna()
            valid_obs_mask = var_info[matching_obs_col] > 0
            fix_mask = nan_mask & valid_obs_mask
            
            if fix_mask.sum() > 0:
                print(f"Setting {fix_mask.sum()} NaN values to 0.0 in {freq_col} where {matching_obs_col} > 0")
                var_info.loc[fix_mask, freq_col] = 0.0
        
        # Set NaN to 0.0 for F_MISS columns when corresponding OBS_CT > 0
        for i, miss_col in enumerate(miss_cols):
            matching_obs_col = obs_cols[i]
            nan_mask = var_info[miss_col].isna()
            valid_obs_mask = var_info[matching_obs_col] > 0
            fix_mask = nan_mask & valid_obs_mask
            
            if fix_mask.sum() > 0:
                print(f"Setting {fix_mask.sum()} NaN values to 0.0 in {miss_col} where {matching_obs_col} > 0")
                var_info.loc[fix_mask, miss_col] = 0.0
        
        # Also handle the aggregate ALT_FREQS and F_MISS columns
        for col_name in ['ALT_FREQS', 'F_MISS']:
            if col_name in var_info.columns:
                nan_mask = var_info[col_name].isna()
                valid_obs_mask = var_info['OBS_CT'] > 0
                fix_mask = nan_mask & valid_obs_mask
                
                if fix_mask.sum() > 0:
                    print(f"Setting {fix_mask.sum()} NaN values to 0.0 in aggregate {col_name} where OBS_CT > 0")
                    var_info.loc[fix_mask, col_name] = 0.0
    
    def _cleanup_columns(self, var_info: pd.DataFrame) -> None:
        """Remove unnecessary columns like F_MISS_SIMPLE_AVG and redundant chromosome/allele columns"""
        columns_to_remove = ['F_MISS_SIMPLE_AVG', 'chrom.1', 'CHR', '#CHROM', 'REF', 'ALT', 'COUNTED', 'PROVISIONAL_REF?']
        var_info.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        print(f"Cleaned up unnecessary columns: {[col for col in columns_to_remove if col in var_info.columns]}")
    
    def _prepare_final_output(self, carriers_string: pd.DataFrame, carriers_int: pd.DataFrame, key_file: str) -> tuple:
        """Prepare final output format for carriers data"""
        # Reorder carriers dataframes columns 
        variant_columns = [col for col in carriers_string.columns if col not in ['IID', 'ancestry']]
        carriers_string_final = carriers_string[['IID', 'ancestry'] + variant_columns].copy()
        carriers_int_final = carriers_int[['IID', 'ancestry'] + variant_columns].copy()
        
        # Fill missing values in carriers data
        carriers_string_final[variant_columns] = carriers_string_final[variant_columns].fillna('./.')
        
        # Add study information if key file is provided
        if key_file is not None:
            key = self.data_repo.read_csv(key_file)
            carriers_string_merge = carriers_string_final.merge(key[['IID','study']], how='left', on='IID')
            carriers_string_final = carriers_string_merge[['IID', 'study', 'ancestry'] + variant_columns]
            
            carriers_int_merge = carriers_int_final.merge(key[['IID','study']], how='left', on='IID')
            carriers_int_final = carriers_int_merge[['IID', 'study', 'ancestry'] + variant_columns]
        
        return carriers_string_final, carriers_int_final
    
    def _save_combined_files(self, var_info: pd.DataFrame, carriers_string: pd.DataFrame,
                           carriers_int: pd.DataFrame, out_path: str) -> Dict[str, str]:
        """Save combined files and return their paths"""
        # Define output paths
        carriers_string_path = f"{out_path}_string.csv"
        carriers_int_path = f"{out_path}_int.csv"
        var_info_path = f"{out_path}_info.csv"
        
        # Save combined files
        print(f"Saving combined files to {out_path}...")
        self.data_repo.write_csv(var_info, var_info_path, index=False)
        self.data_repo.write_csv(carriers_string, carriers_string_path, index=False)
        self.data_repo.write_csv(carriers_int, carriers_int_path, index=False)
        
        print(f"✓ Combined var_info: {var_info_path}")
        print(f"✓ Combined carriers_string: {carriers_string_path}")
        print(f"✓ Combined carriers_int: {carriers_int_path}")
        
        return {
            'var_info': var_info_path,
            'carriers_string': carriers_string_path,
            'carriers_int': carriers_int_path
        }