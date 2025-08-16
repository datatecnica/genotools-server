import os
import pandas as pd
import polars as pl
from typing import Set, Optional, Dict
import time
from pathlib import Path
import logging
from src.utils.performance_metrics import track_performance, record_cache_hit, record_cache_miss, record_items_processed

logger = logging.getLogger(__name__)


class VariantIndexCache:
    """
    Permanent variant database for PLINK .pvar files to accelerate repeated variant lookups.
    
    Cache key format: {ancestry}_chr{chrom}_r{release}.parquet
    Stores: variant_id, chrom, pos, a1, a2, and computed variant_key
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize permanent variant database
        
        Args:
            cache_dir: Directory to store variant database files. If None, uses ~/.genotools_cache
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.genotools_cache/variant_index")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, pvar_path: str, ancestry: str = None, 
                           chromosome: str = None, release: str = None) -> str:
        """
        Generate unique cache key for pvar file
        
        Args:
            pvar_path: Path to .pvar file
            ancestry: Ancestry label (e.g., 'AAC', 'AFR')
            chromosome: Chromosome (e.g., '1', '22', 'X')
            release: Release version (e.g., '10')
            
        Returns:
            str: Cache filename
        """
        if ancestry and chromosome and release:
            return f"{ancestry}_chr{chromosome}_r{release}.parquet"
        else:
            # Fallback for single files - use basename without hash
            basename = Path(pvar_path).stem
            return f"{basename}.parquet"
    
    def _build_variant_index(self, pvar_path: str) -> pl.DataFrame:
        """
        Build variant index from .pvar file using polars for speed
        
        Args:
            pvar_path: Path to .pvar file
            
        Returns:
            pl.DataFrame: Variant index with columns [chrom, pos, id, a1, a2, variant_key]
        """
        logger.info(f"Building variant index for {pvar_path}")
        start_time = time.time()
        
        try:
            # Use polars lazy reading for memory efficiency
            # First check if file has header to determine read method
            with open(pvar_path, 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('#CHROM') or first_line.startswith('CHROM'):
                # Header format - read with header and proper dtypes
                chrom_col = '#CHROM' if first_line.startswith('#CHROM') else 'CHROM'
                df = (
                    pl.scan_csv(
                        pvar_path,
                        separator='\t',
                        dtypes={chrom_col: pl.Utf8, 'POS': pl.Int64, 'ID': pl.Utf8, 'REF': pl.Utf8, 'ALT': pl.Utf8}
                    )
                    .select([
                        pl.col(chrom_col).alias('chrom'),
                        pl.col('POS').alias('pos'),
                        pl.col('ID').alias('id'),
                        pl.col('REF').alias('a1'),
                        pl.col('ALT').alias('a2')
                    ])
                )
            else:
                # Headerless format
                df = (
                    pl.scan_csv(
                        pvar_path,
                        separator='\t',
                        has_header=False,
                        new_columns=['chrom', 'pos', 'id', 'a1', 'a2', 'rest'],
                        dtypes={'chrom': pl.Utf8, 'pos': pl.Int64}
                    )
                    .select(['chrom', 'pos', 'id', 'a1', 'a2'])
                )
            
            # Common processing for both header and headerless formats
            df = (df.with_columns([
                # Standardize and uppercase alleles
                pl.col('chrom').cast(pl.Utf8).str.strip_chars(),
                pl.col('id').cast(pl.Utf8).str.strip_chars(),
                pl.col('a1').cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                pl.col('a2').cast(pl.Utf8).str.strip_chars().str.to_uppercase()
            ])
            .with_columns([
                # Create all 4 variant arrangements for harmonization
                (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
                 pl.col('a1') + ':' + pl.col('a2')).alias('variant_exact'),
                    
                    # Swapped variant (a1 and a2 reversed)
                    (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
                     pl.col('a2') + ':' + pl.col('a1')).alias('variant_swap')
                ])
            )
            
            # Add flipped variants
            df = self._add_flipped_variants(df)
            df = df.collect()
            
            build_time = time.time() - start_time
            logger.info(f"Built index for {len(df)} variants in {build_time:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to build variant index for {pvar_path}: {e}")
            # Try fallback with header detection
            return self._build_variant_index_fallback(pvar_path)
    
    def _build_variant_index_fallback(self, pvar_path: str) -> pl.DataFrame:
        """Fallback method with header detection"""
        logger.info(f"Using fallback method for {pvar_path}")
        
        # Read first few lines to detect format
        with open(pvar_path, 'r') as f:
            first_line = f.readline().strip()
            
        if first_line.startswith('#CHROM') or first_line.startswith('CHROM'):
            # Header format - don't use comment_prefix to preserve header
            # First read to check which column name exists
            temp_df = pl.scan_csv(pvar_path, separator='\t').collect_schema()
            chrom_col = '#CHROM' if '#CHROM' in temp_df else 'CHROM'
            
            df = (
                pl.scan_csv(pvar_path, separator='\t', 
                           dtypes={chrom_col: pl.Utf8, 'POS': pl.Int64, 'ID': pl.Utf8, 'REF': pl.Utf8, 'ALT': pl.Utf8})
                .select([
                    pl.col(chrom_col).alias('chrom'),
                    'POS',
                    'ID', 
                    'REF',
                    'ALT'
                ])
                .rename({'POS': 'pos', 'ID': 'id', 'REF': 'a1', 'ALT': 'a2'})
                .with_columns([
                    pl.col('chrom').cast(pl.Utf8).str.strip_chars(),
                    pl.col('id').cast(pl.Utf8).str.strip_chars(),
                    pl.col('a1').cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                    pl.col('a2').cast(pl.Utf8).str.strip_chars().str.to_uppercase()
                ])
                .with_columns([
                    # Create all 4 variant arrangements for harmonization (same as main method)
                    (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
                     pl.col('a1') + ':' + pl.col('a2')).alias('variant_exact'),
                    
                    # Swapped variant (a1 and a2 reversed)
                    (pl.col('chrom') + ':' + pl.col('pos').cast(pl.Utf8) + ':' + 
                     pl.col('a2') + ':' + pl.col('a1')).alias('variant_swap')
                ])
            )
            
            # Add flipped variants (same as main method)
            df = self._add_flipped_variants(df)
            df = df.collect()
        else:
            # Headerless format - use pandas as fallback
            pandas_df = pd.read_csv(
                pvar_path, 
                sep='\t', 
                comment='#', 
                header=None,
                names=['chrom', 'pos', 'id', 'a1', 'a2'],
                usecols=[0, 1, 2, 3, 4],
                dtype={'chrom': str}
            )
            
            # Clean and standardize
            pandas_df['chrom'] = pandas_df['chrom'].astype(str).str.strip()
            pandas_df['id'] = pandas_df['id'].astype(str).str.strip()
            pandas_df['a1'] = pandas_df['a1'].astype(str).str.strip().str.upper()
            pandas_df['a2'] = pandas_df['a2'].astype(str).str.strip().str.upper()
            pandas_df['variant_key'] = (pandas_df['chrom'] + ':' + 
                                       pandas_df['pos'].astype(str) + ':' + 
                                       pandas_df['a1'] + ':' + 
                                       pandas_df['a2'])
            
            # Convert to polars
            df = pl.from_pandas(pandas_df)
        
        return df
    
    def _add_flipped_variants(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Add flipped and flip-swapped variant keys using polars string operations"""
        # Create complement columns using polars string replacements
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
    
    def get_or_build_index(self, pvar_path: str, ancestry: str = None, 
                          chromosome: str = None, release: str = None, 
                          force_rebuild: bool = False) -> pl.DataFrame:
        """
        Get variant index from permanent database or build if not exists
        
        Args:
            pvar_path: Path to .pvar file
            ancestry: Ancestry label for cache key
            chromosome: Chromosome for cache key  
            release: Release version for cache key
            force_rebuild: Force rebuild even if cache exists
            
        Returns:
            pl.DataFrame: Variant index
        """
        cache_key = self._generate_cache_key(pvar_path, ancestry, chromosome, release)
        cache_path = self.cache_dir / cache_key
        
        operation_name = f"variant_cache_{ancestry}_{chromosome}" if ancestry and chromosome else "variant_cache"
        
        # Check if database entry exists
        if not force_rebuild and cache_path.exists():
            # Just load it - no validation needed since variant content never changes
            with track_performance(f"{operation_name}_load", cache_key=cache_key):
                logger.info(f"Loading variant index from database: {cache_key}")
                df = pl.read_parquet(cache_path)
                record_cache_hit(operation_name)
                record_items_processed(operation_name, len(df))
                logger.info(f"Loaded {len(df)} variants from database")
                return df
        
        # Build new index
        with track_performance(f"{operation_name}_build", cache_key=cache_key):
            logger.info(f"Building new variant database entry: {cache_key}")
            record_cache_miss(operation_name)
            df = self._build_variant_index(pvar_path)
            record_items_processed(operation_name, len(df))
            
            # Save to database
            try:
                df.write_parquet(cache_path)
                logger.info(f"Saved variant index to database: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to save database entry {cache_key}: {e}")
            
            return df
    
    def find_matching_variants(self, pvar_path: str, reference_variants: Set[str], 
                              ancestry: str = None, chromosome: str = None, 
                              release: str = None) -> pl.DataFrame:
        """
        Find variants that match reference set using all 4 possible arrangements
        
        Args:
            pvar_path: Path to .pvar file
            reference_variants: Set of variant keys to match against
            ancestry: Ancestry label for cache key
            chromosome: Chromosome for cache key
            release: Release version for cache key
            
        Returns:
            pl.DataFrame: Matching variants with original columns plus match_type
        """
        # Get cached index
        variant_index = self.get_or_build_index(pvar_path, ancestry, chromosome, release)
        
        # Check all 4 variant arrangements
        start_time = time.time()
        
        matching_variants = variant_index.filter(
            pl.col('variant_exact').is_in(reference_variants) |
            pl.col('variant_swap').is_in(reference_variants) |
            pl.col('variant_flip').is_in(reference_variants) |
            pl.col('variant_flip_swap').is_in(reference_variants)
        )
        
        # Add column indicating which match type was found
        matching_variants = matching_variants.with_columns([
            pl.when(pl.col('variant_exact').is_in(reference_variants))
              .then(pl.lit('exact'))
              .when(pl.col('variant_swap').is_in(reference_variants))
              .then(pl.lit('swap'))
              .when(pl.col('variant_flip').is_in(reference_variants))
              .then(pl.lit('flip'))
              .when(pl.col('variant_flip_swap').is_in(reference_variants))
              .then(pl.lit('flip_swap'))
              .otherwise(pl.lit('unknown'))
              .alias('match_type'),
            
            # Also add the actual matched variant key for downstream use
            pl.when(pl.col('variant_exact').is_in(reference_variants))
              .then(pl.col('variant_exact'))
              .when(pl.col('variant_swap').is_in(reference_variants))
              .then(pl.col('variant_swap'))
              .when(pl.col('variant_flip').is_in(reference_variants))
              .then(pl.col('variant_flip'))
              .when(pl.col('variant_flip_swap').is_in(reference_variants))
              .then(pl.col('variant_flip_swap'))
              .otherwise(pl.lit(''))
              .alias('matched_variant_key')
        ])
        
        filter_time = time.time() - start_time
        logger.info(f"Found {len(matching_variants)} matching variants in {filter_time:.3f}s")
        
        return matching_variants
    
    def get_matching_variant_ids(self, pvar_path: str, reference_df: pd.DataFrame,
                                ancestry: str = None, chromosome: str = None,
                                release: str = None) -> pd.DataFrame:
        """
        Get matching variant IDs and their match types for harmonization.
        This replaces the slow _find_common_snps logic in harmonizer.py
        
        Args:
            pvar_path: Path to .pvar file
            reference_df: DataFrame with hg38 coordinates
            ancestry: Ancestry label for cache key
            chromosome: Chromosome for cache key
            release: Release version for cache key
            
        Returns:
            DataFrame with columns: id_geno, variant_id_ref, match_type, a1_ref, a2_ref
        """
        # Create reference variant set from all possible arrangements
        ref_variants = set()
        ref_mapping = {}
        
        for _, row in reference_df.iterrows():
            # Parse coordinates from hg38
            hg38_coord = str(row['hg38']).strip()
            if ':' not in hg38_coord:
                continue
                
            parts = hg38_coord.split(':')
            if len(parts) < 4:
                continue
                
            chrom = parts[0]
            pos = parts[1] 
            a1 = parts[2].upper()
            a2 = parts[3].upper()
            
            # Create all 4 arrangements
            exact = f"{chrom}:{pos}:{a1}:{a2}"
            swap = f"{chrom}:{pos}:{a2}:{a1}"
            
            # Create complement mapping
            complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
            a1_flip = complement.get(a1, a1)
            a2_flip = complement.get(a2, a2)
            flip = f"{chrom}:{pos}:{a1_flip}:{a2_flip}"
            flip_swap = f"{chrom}:{pos}:{a2_flip}:{a1_flip}"
            
            # Add to set and mapping
            for variant, match_type in [(exact, 'exact'), (swap, 'swap'), 
                                        (flip, 'flip'), (flip_swap, 'flip_swap')]:
                ref_variants.add(variant)
                ref_mapping[variant] = {
                    'variant_id_ref': hg38_coord.upper(),
                    'match_type': match_type,
                    'a1_ref': a1,
                    'a2_ref': a2
                }
                # Add snp_name if available
                if 'snp_name' in row and pd.notna(row['snp_name']):
                    ref_mapping[variant]['snp_name_ref'] = row['snp_name']
        
        logger.info(f"Created {len(ref_variants)} reference variant arrangements from {len(reference_df)} reference variants")
        
        # Find matches using the cache
        matches = self.find_matching_variants(pvar_path, ref_variants, 
                                             ancestry, chromosome, release)
        
        # Convert to format expected by harmonizer
        result = []
        if len(matches) > 0:
            matches_pandas = matches.to_pandas()
            
            for _, row in matches_pandas.iterrows():
                matched_key = row['matched_variant_key']
                if matched_key in ref_mapping:
                    match_info = ref_mapping[matched_key]
                    result_row = {
                        'id_geno': row['id'],
                        'variant_id_ref': match_info['variant_id_ref'],
                        'match_type': match_info['match_type'],
                        'a1_ref': match_info['a1_ref'],
                        'a2_ref': match_info['a2_ref']
                    }
                    # Add snp_name_ref if available
                    if 'snp_name_ref' in match_info:
                        result_row['snp_name_ref'] = match_info['snp_name_ref']
                    result.append(result_row)
        
        logger.info(f"Found {len(result)} harmonized variant matches")
        return pd.DataFrame(result)
    
    def clear_cache(self, pattern: str = None) -> int:
        """
        Clear database entries
        
        Args:
            pattern: Glob pattern to match files (e.g., "AAC_*"). If None, clears all.
            
        Returns:
            int: Number of files removed
        """
        if pattern:
            files = list(self.cache_dir.glob(pattern))
        else:
            files = list(self.cache_dir.glob("*.parquet"))
        
        count = 0
        for file_path in files:
            try:
                file_path.unlink()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
        
        logger.info(f"Cleared {count} database entries")
        return count
    
    def get_cache_stats(self) -> Dict:
        """Get database statistics"""
        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'num_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'files': [f.name for f in cache_files]
        }