import pandas as pd
import polars as pl
import os
import tempfile
import time
import logging
from typing import Optional, List, Set
from src.core.plink_operations import (
    ExtractSnpsCommand, 
    FrequencyCommand, 
    SwapAllelesCommand, 
    UpdateAllelesCommand, 
    ExportCommand, 
    CopyFilesCommand
)
from src.core.variant_cache import VariantIndexCache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.pipeline_config import PipelineConfig

from src.utils.performance_metrics import track_performance, record_cache_hit, record_cache_miss, record_items_processed

logger = logging.getLogger(__name__)


class AlleleHarmonizer:
    def __init__(self, config: 'PipelineConfig'):
        """
        Initialize AlleleHarmonizer with pipeline configuration
        
        Args:
            config: Pipeline configuration containing optimization settings
        """
        self.config = config
        self.enable_caching = config.enable_variant_caching
        
        if self.enable_caching:
            self.cache = VariantIndexCache(config.variant_cache_dir)
        else:
            self.cache = None

    def harmonize_and_extract(self, geno_path: str, reference_path: str, plink_out: str) -> str:
        """
        Main method to harmonize alleles between genotype and reference files.
        Returns the path to a harmonized subset file suitable for carrier extraction.
        """
        tmpdir = tempfile.mkdtemp()
        
        try:
            
            # Step 1: Find common SNPs between genotype and reference
            common_snps_path = self._find_common_snps(geno_path, reference_path, os.path.join(tmpdir, "common_snps"))
            
            # Step 2: Extract only the common SNPs (for efficiency with large files)
            # Also standardizes chromosome format during extraction
            extracted_prefix = os.path.join(tmpdir, "extracted")
            extract_cmd = ExtractSnpsCommand(geno_path, common_snps_path, extracted_prefix, output_chr='M')
            try:
                extract_cmd.execute()
            except ValueError as e:
                if "No variants found after extraction" in str(e):
                    print(f"No target variants found in {geno_path}")
                    return None
                else:
                    raise
            
            # Step 3: Harmonize alleles on the smaller extracted dataset
            harmonized_prefix = os.path.join(tmpdir, "harmonized")
            match_info_path = os.path.join(tmpdir, "common_snps_match_info.tsv")
            self._harmonize_alleles(extracted_prefix, reference_path, harmonized_prefix, match_info_path)
            current_geno = harmonized_prefix
            
            # Step 4: Build and execute PLINK command with all operations
            export_cmd = ExportCommand(
                pfile=current_geno, 
                out=plink_out
            )
            export_cmd.execute()
            
            # Create subset SNP list with matched IDs
            subset_snp_path = f"{plink_out}_subset_snps.csv"
            
            # Read the match info to get the mapping between genotype IDs and reference variants
            match_info = pd.read_csv(match_info_path, sep='\t')
            
            # Read original reference SNP list
            ref_df = pd.read_csv(reference_path, dtype={'chrom': str})
            
            # Use hg38 as variant_id (with uppercase alleles)
            ref_df['hg38'] = ref_df['hg38'].astype(str).str.strip().str.replace(' ', '')
            ref_df['variant_id'] = ref_df['hg38'].str.upper()
            
            # Parse hg38 to create chrom, pos, a1, a2 columns (needed for downstream processing)
            hg38_parts = ref_df['hg38'].str.split(':')
            ref_df['chrom'] = hg38_parts.str[0]
            ref_df['pos'] = hg38_parts.str[1]
            ref_df['a1'] = hg38_parts.str[2].str.upper()
            ref_df['a2'] = hg38_parts.str[3].str.upper()
            
            # Merge match info with reference data to get subset of matched variants
            subset_snps = match_info.merge(ref_df, left_on='variant_id_ref', right_on='variant_id', how='inner')
            
            # Keep original reference columns plus the genotype ID and parsed columns
            ref_cols = list(ref_df.columns)
            subset_cols = ref_cols + ['id_geno']
            subset_snps = subset_snps[subset_cols]
            
            # Rename id_geno to id for consistency with carrier processor expectations
            subset_snps = subset_snps.rename(columns={'id_geno': 'id'})
            
            # Save the subset SNP list
            subset_snps.to_csv(subset_snp_path, index=False)
            
            return subset_snp_path
            
        finally:
            # Clean up temporary directory
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
    
    def _find_common_snps(self, pfile: str, reference: str, out: str, chunk_size: int = 500000) -> str:
        """
        Find SNPs common between the PLINK file and reference using exact allele matching
        with Polars and variant caching.
        
        Args:
            pfile: Path to PLINK file prefix
            reference: Path to TSV file with columns chrom, pos, a1, a2
            out: Output path prefix
            chunk_size: Unused (kept for compatibility)
            
        Returns:
            str: Path to file containing common SNP IDs
        """
        # Check for existing cached results (quick win caching)
        cache_path = f"{out}_cache.parquet"
        match_info_cache_path = f"{out}_match_info_cache.tsv"
        common_snps_path = f"{out}.txt"
        
        if self._should_use_cache(cache_path, pfile, reference):
            logger.info(f"Using cached results from {cache_path}")
            return self._load_from_cache(cache_path, match_info_cache_path, common_snps_path)
        
        # Proceed with normal processing
        logger.info("No valid cache found, processing from scratch")
        
        with track_performance("find_common_snps", pfile=pfile, reference=reference):
            return self._process_snp_matching(pfile, reference, out, cache_path, match_info_cache_path)
    
    def _process_snp_matching(self, pfile: str, reference: str, out: str, cache_path: str, match_info_cache_path: str) -> str:
        """Core SNP matching logic with performance tracking"""
        pvar_path = f"{pfile}.pvar"
        
        # Read and prepare reference data with Polars for speed
        with track_performance("read_reference_data", reference=reference):
            logger.info(f"Reading reference data from {reference}")
            
            try:
                ref_df = pl.read_csv(reference, dtypes={'chrom': pl.Utf8})
            except:
                # Fallback to pandas if polars fails
                ref_pandas = pd.read_csv(reference, dtype={'chrom': str})
                ref_df = pl.from_pandas(ref_pandas)
            
            # Clean and parse reference data
            ref_df = ref_df.with_columns([
                pl.col('hg38').cast(pl.Utf8).str.strip_chars().str.replace(' ', ''),
            ])
            
            # Parse hg38 coordinates
            ref_df = ref_df.with_columns([
                pl.col('hg38').str.split(':').list.get(0).alias('chrom'),
                pl.col('hg38').str.split(':').list.get(1).alias('pos'),
                pl.col('hg38').str.split(':').list.get(2).str.to_uppercase().alias('a1'),
                pl.col('hg38').str.split(':').list.get(3).str.to_uppercase().alias('a2'),
                pl.col('hg38').str.to_uppercase().alias('variant_id')
            ])
            
            # Create exact variant IDs for matching
            ref_df = ref_df.with_columns([
                (pl.col('chrom') + ':' + pl.col('pos') + ':' + 
                 pl.col('a1') + ':' + pl.col('a2')).alias('exact_variant_id')
            ])
            
            record_items_processed("read_reference_data", len(ref_df))
            logger.info(f"Processed {len(ref_df)} reference variants")
        
        # Create comprehensive variant set including harmonized versions
        with track_performance("create_variant_mappings"):
            ref_variants_all, ref_variant_mapping = self._create_variant_mappings(ref_df)
            record_items_processed("create_variant_mappings", len(ref_variants_all))
            logger.info(f"Created {len(ref_variants_all)} variant mappings")
        
        # Use caching if enabled
        if self.enable_caching and self.cache:
            # Try to extract ancestry/chromosome info from path for better cache keys
            ancestry, chromosome, release = self._extract_path_info(pfile)
            
            with track_performance("variant_cache_harmonization", ancestry=ancestry, chromosome=chromosome):
                logger.info("Using enhanced variant cache with harmonization support")
                
                # Use the new harmonization-aware method that handles all 4 arrangements
                true_matches = self.cache.get_matching_variant_ids(
                    pvar_path, ref_df.to_pandas(), ancestry, chromosome, release
                )
                
                record_cache_hit("variant_cache_harmonization")
                record_items_processed("variant_cache_harmonization", len(true_matches))
                logger.info(f"Cache-based harmonization found {len(true_matches)} variant matches")
                
                # Cache handled everything - true_matches is ready to use
        else:
            # Fallback to direct polars processing (still much faster than original pandas chunking)
            with track_performance("direct_pvar_processing"):
                logger.info("Processing pvar file directly with polars (no cache)")
                matching_variants_pd = self._process_pvar_direct(pvar_path, ref_variants_all)
                record_cache_miss("direct_pvar_processing")
                record_items_processed("direct_pvar_processing", len(matching_variants_pd))
                logger.info(f"Direct processing found {len(matching_variants_pd)} matching variants")
            
            # Process matches to get final results for non-cached path
            with track_performance("process_final_matches"):
                true_matches = self._process_matches(matching_variants_pd, ref_variant_mapping)
                record_items_processed("process_final_matches", len(true_matches))
        
        # Write outputs
        with track_performance("write_output_files"):
            common_snps_path = self._write_output_files(true_matches, out)
        
        # Cache results for future use (quick win caching)
        if self.config.enable_file_caching:
            try:
                self._save_to_cache(true_matches, cache_path, match_info_cache_path)
                logger.info(f"Results cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
        
        return common_snps_path
    
    def _should_use_cache(self, cache_path: str, pfile: str, reference: str) -> bool:
        """Check if cache exists and is valid (simple timestamp-based validation)"""
        if not os.path.exists(cache_path):
            return False
        
        try:
            # Check if cache is newer than input files
            cache_mtime = os.path.getmtime(cache_path)
            pvar_mtime = os.path.getmtime(f"{pfile}.pvar")
            ref_mtime = os.path.getmtime(reference)
            
            return cache_mtime >= max(pvar_mtime, ref_mtime)
        except OSError:
            return False
    
    def _load_from_cache(self, cache_path: str, match_info_cache_path: str, common_snps_path: str) -> str:
        """Load results from cache and recreate output files"""
        try:
            # Load cached matches
            true_matches = pd.read_parquet(cache_path)
            
            # Recreate output files
            self._write_output_files(true_matches, common_snps_path.replace('.txt', ''))
            
            logger.info(f"Loaded {len(true_matches)} cached matches")
            return common_snps_path
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            raise
    
    def _save_to_cache(self, true_matches: pd.DataFrame, cache_path: str, match_info_cache_path: str) -> None:
        """Save results to cache"""
        if not true_matches.empty:
            # Save matches to parquet for fast loading
            true_matches.to_parquet(cache_path, index=False)
            
            # Also save match info for reference
            match_info_cols = ['id_geno', 'variant_id_ref', 'match_type']
            for col in ['a1_ref', 'a2_ref', 'snp_name_ref']:
                if col in true_matches.columns:
                    match_info_cols.append(col)
            
            existing_cols = [col for col in match_info_cols if col in true_matches.columns]
            true_matches[existing_cols].to_csv(match_info_cache_path, sep='\t', index=False)
    
    def _extract_path_info(self, pfile: str) -> tuple:
        """Extract ancestry, chromosome, and release info from path for cache keys"""
        path_parts = pfile.split('/')
        filename = path_parts[-1] if path_parts else pfile
        
        ancestry = None
        chromosome = None
        release = None
        
        # Try to extract from common naming patterns
        if 'chr' in filename:
            parts = filename.split('_')
            for part in parts:
                if part.startswith('chr'):
                    chromosome = part.replace('chr', '')
                elif part.startswith('release'):
                    release = part.replace('release', '')
                elif len(part) == 3 and part.isupper():  # Ancestry codes like AAC, AFR
                    ancestry = part
        
        return ancestry, chromosome, release
    
    def _create_variant_mappings(self, ref_df: pl.DataFrame) -> tuple:
        """Create comprehensive variant mappings including harmonized versions"""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        ref_variants_all = set()
        ref_variant_mapping = {}
        
        # Convert to pandas for easier iteration (small reference set)
        ref_pandas = ref_df.to_pandas()
        
        for _, row in ref_pandas.iterrows():
            exact_id = row['exact_variant_id']
            
            # Exact match
            ref_variants_all.add(exact_id)
            ref_variant_mapping[exact_id] = {
                'variant_id_ref': row['variant_id'],
                'match_type': 'exact',
                'ref_row': row
            }
            
            # Swapped version
            swap_id = f"{row['chrom']}:{row['pos']}:{row['a2']}:{row['a1']}"
            ref_variants_all.add(swap_id)
            ref_variant_mapping[swap_id] = {
                'variant_id_ref': row['variant_id'],
                'match_type': 'swap',
                'ref_row': row
            }
            
            # Flipped versions
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
        
        return ref_variants_all, ref_variant_mapping
    
    def _process_pvar_direct(self, pvar_path: str, ref_variants_all: Set[str]) -> pd.DataFrame:
        """Direct polars processing when cache is not available"""
        try:
            # Use polars scan for memory-efficient processing
            matching_df = (
                pl.scan_csv(
                    pvar_path,
                    separator='\t',
                    comment_prefix='#',
                    has_header=False,
                    new_columns=['chrom', 'pos', 'id', 'a1', 'a2', 'rest']
                )
                .select(['chrom', 'pos', 'id', 'a1', 'a2'])
                .with_columns([
                    pl.col('chrom').cast(pl.Utf8).str.strip_chars(),
                    pl.col('id').cast(pl.Utf8).str.strip_chars(),
                    pl.col('a1').cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                    pl.col('a2').cast(pl.Utf8).str.strip_chars().str.to_uppercase()
                ])
                .with_columns([
                    # Create all 4 variant arrangements for matching (like cache does)
                    (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
                     pl.col('a1') + ':' + pl.col('a2')).alias('variant_exact'),
                    (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
                     pl.col('a2') + ':' + pl.col('a1')).alias('variant_swap')
                ])
            )
            
            # Add flipped variants
            matching_df = self._add_flipped_variants_direct(matching_df)
            
            # Filter for matches against any of the 4 arrangements
            matching_df = (
                matching_df
                .filter(
                    pl.col('variant_exact').is_in(ref_variants_all) |
                    pl.col('variant_swap').is_in(ref_variants_all) |
                    pl.col('variant_flip').is_in(ref_variants_all) |
                    pl.col('variant_flip_swap').is_in(ref_variants_all)
                )
                .collect()
            )
            
            return matching_df.to_pandas()
            
        except Exception as e:
            logger.warning(f"Polars processing failed: {e}, falling back to pandas")
            # Fallback to pandas chunking (original method but simplified)
            return self._pandas_fallback(pvar_path, ref_variants_all)
    
    def _add_flipped_variants_direct(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add flipped variants for direct processing (same as cache method)"""
        df = df.with_columns([
            # Create flipped a1 (A<->T, C<->G)
            pl.col('a1')
              .str.replace_all('A', '1')  # Temp placeholder
              .str.replace_all('T', 'A')
              .str.replace_all('1', 'T')
              .str.replace_all('C', '2')  # Temp placeholder  
              .str.replace_all('G', 'C')
              .str.replace_all('2', 'G')
              .alias('a1_flip'),
            
            # Create flipped a2 (A<->T, C<->G)
            pl.col('a2')
              .str.replace_all('A', '1')  # Temp placeholder
              .str.replace_all('T', 'A') 
              .str.replace_all('1', 'T')
              .str.replace_all('C', '2')  # Temp placeholder
              .str.replace_all('G', 'C')
              .str.replace_all('2', 'G')
              .alias('a2_flip')
        ])
        
        # Create flipped variant keys
        df = df.with_columns([
            # Flipped variant (complement of both alleles)
            (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
             pl.col('a1_flip') + ':' + pl.col('a2_flip')).alias('variant_flip'),
            
            # Flip-swapped variant (complement + swap)
            (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
             pl.col('a2_flip') + ':' + pl.col('a1_flip')).alias('variant_flip_swap')
        ])
        
        # Drop temporary columns
        df = df.drop(['a1_flip', 'a2_flip'])
        
        return df
    
    def _pandas_fallback(self, pvar_path: str, ref_variants_all: Set[str]) -> pd.DataFrame:
        """Pandas fallback processing"""
        try:
            # Read with pandas
            pvar_df = pd.read_csv(
                pvar_path, 
                sep='\t', 
                comment='#', 
                header=None,
                names=['chrom', 'pos', 'id', 'a1', 'a2'],
                usecols=[0, 1, 2, 3, 4],
                dtype={'chrom': str}
            )
            
            # Clean and standardize
            pvar_df['chrom'] = pvar_df['chrom'].astype(str).str.strip()
            pvar_df['id'] = pvar_df['id'].astype(str).str.strip()
            pvar_df['a1'] = pvar_df['a1'].astype(str).str.strip().str.upper()
            pvar_df['a2'] = pvar_df['a2'].astype(str).str.strip().str.upper()
            pvar_df['variant_key'] = (pvar_df['chrom'] + ':' + pvar_df['pos'].astype(str) + ':' + 
                                     pvar_df['a1'] + ':' + pvar_df['a2'])
            
            # Filter to matching variants
            return pvar_df[pvar_df['variant_key'].isin(ref_variants_all)]
            
        except Exception as e:
            logger.error(f"Pandas fallback failed: {e}")
            return pd.DataFrame()
    
    def _process_matches(self, matching_variants_pd: pd.DataFrame, ref_variant_mapping: dict) -> pd.DataFrame:
        """Process matching variants to create final match results"""
        if matching_variants_pd.empty:
            logger.info("No matches found")
            return pd.DataFrame()
        
        all_matches = []
        
        # Check which variant arrangement columns are available
        variant_cols = ['variant_exact', 'variant_swap', 'variant_flip', 'variant_flip_swap']
        available_cols = [col for col in variant_cols if col in matching_variants_pd.columns]
        
        if not available_cols:
            # Fallback to old column names for backward compatibility
            variant_key_col = 'variant_key' if 'variant_key' in matching_variants_pd.columns else 'exact_variant_id'
            available_cols = [variant_key_col]
        
        for _, row in matching_variants_pd.iterrows():
            # Check each variant arrangement to find the match
            matched = False
            for variant_col in available_cols:
                if variant_col in row:
                    geno_variant_id = row[variant_col]
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
                        all_matches.append(match_row)
                        matched = True
                        break
        
        if all_matches:
            true_matches = pd.DataFrame(all_matches)
            logger.info(f"Total variant matches found: {len(true_matches)}")
            return true_matches
        else:
            return pd.DataFrame()
    
    def _write_output_files(self, true_matches: pd.DataFrame, out: str) -> str:
        """Write output files for PLINK and match information"""
        common_snps_path = f"{out}.txt"
        match_info_path = f"{out}_match_info.tsv"
        
        if not true_matches.empty:
            # Write unique genotype variant IDs for PLINK extraction
            unique_geno_ids = true_matches['id_geno'].drop_duplicates()
            unique_geno_ids.to_csv(common_snps_path, index=False, header=False)
            logger.info(f"Writing {len(unique_geno_ids)} unique genotype variant IDs for PLINK extraction")
            
            # Write match information
            match_info_cols = ['id_geno', 'variant_id_ref', 'match_type']
            for col in ['a1_ref', 'a2_ref', 'snp_name_ref']:
                if col in true_matches.columns:
                    match_info_cols.append(col)
            
            existing_cols = [col for col in match_info_cols if col in true_matches.columns]
            true_matches[existing_cols].to_csv(match_info_path, sep='\t', index=False)
        else:
            # Create empty files
            with open(common_snps_path, 'w') as f:
                pass
            pd.DataFrame(columns=['id_geno', 'variant_id_ref', 'match_type']).to_csv(
                match_info_path, sep='\t', index=False
            )
        
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