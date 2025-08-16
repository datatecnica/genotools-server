import pandas as pd
import polars as pl
import os
import tempfile
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ChromosomeSnpSplitter:
    """
    Utility to pre-split SNP lists by chromosome for faster processing.
    This avoids repeatedly filtering the same SNP list for each chromosome.
    """
    
    def __init__(self, use_polars: bool = True):
        """
        Initialize SNP splitter
        
        Args:
            use_polars: Whether to use polars for faster processing (default: True)
        """
        self.use_polars = use_polars
        
    def split_snp_list_by_chromosome(self, snplist_path: str, output_dir: str = None, 
                                   cleanup_existing: bool = True) -> Dict[str, str]:
        """
        Split a master SNP list into chromosome-specific files
        
        Args:
            snplist_path: Path to master SNP list file
            output_dir: Directory to save chromosome files (None = temp directory)
            cleanup_existing: Whether to remove existing chromosome files first
            
        Returns:
            Dict[str, str]: Mapping of chromosome -> file path
        """
        logger.info(f"Splitting SNP list {snplist_path} by chromosome")
        
        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="chromosome_snps_")
        else:
            os.makedirs(output_dir, exist_ok=True)
        
        # Clean up existing files if requested
        if cleanup_existing:
            self._cleanup_existing_files(output_dir)
        
        # Read SNP list
        if self.use_polars:
            chromosome_files = self._split_with_polars(snplist_path, output_dir)
        else:
            chromosome_files = self._split_with_pandas(snplist_path, output_dir)
        
        logger.info(f"Created {len(chromosome_files)} chromosome-specific SNP files in {output_dir}")
        
        return chromosome_files
    
    def _split_with_polars(self, snplist_path: str, output_dir: str) -> Dict[str, str]:
        """Split SNP list using polars for speed"""
        try:
            # Read the SNP list
            df = pl.read_csv(snplist_path)
            
            # Ensure we have chromosome information
            if 'chrom' not in df.columns and 'hg38' in df.columns:
                # Parse chromosome from hg38 coordinates
                df = df.with_columns([
                    pl.col('hg38').str.split(':').list.get(0).alias('chrom')
                ])
            
            if 'chrom' not in df.columns:
                raise ValueError("Cannot determine chromosome information. Need 'chrom' column or 'hg38' coordinates.")
            
            # Get unique chromosomes
            chromosomes = df.select('chrom').unique().to_pandas()['chrom'].tolist()
            chromosomes = [str(c) for c in sorted(chromosomes) if str(c) not in ['', 'nan', 'None']]
            
            chromosome_files = {}
            
            for chrom in chromosomes:
                # Filter for this chromosome
                chrom_df = df.filter(pl.col('chrom') == chrom)
                
                if len(chrom_df) == 0:
                    continue
                
                # Save chromosome-specific file
                output_path = os.path.join(output_dir, f"chr{chrom}_snps.csv")
                chrom_df.write_csv(output_path)
                
                chromosome_files[chrom] = output_path
                logger.debug(f"Created {output_path} with {len(chrom_df)} SNPs")
            
            return chromosome_files
            
        except Exception as e:
            logger.warning(f"Polars splitting failed: {e}, falling back to pandas")
            return self._split_with_pandas(snplist_path, output_dir)
    
    def _split_with_pandas(self, snplist_path: str, output_dir: str) -> Dict[str, str]:
        """Split SNP list using pandas (fallback)"""
        # Read the SNP list
        df = pd.read_csv(snplist_path)
        
        # Ensure we have chromosome information
        if 'chrom' not in df.columns and 'hg38' in df.columns:
            # Parse chromosome from hg38 coordinates
            hg38_parts = df['hg38'].str.split(':')
            df['chrom'] = hg38_parts.str[0]
        
        if 'chrom' not in df.columns:
            raise ValueError("Cannot determine chromosome information. Need 'chrom' column or 'hg38' coordinates.")
        
        # Get unique chromosomes
        chromosomes = df['chrom'].dropna().unique()
        chromosomes = [str(c) for c in sorted(chromosomes) if str(c) not in ['', 'nan', 'None']]
        
        chromosome_files = {}
        
        for chrom in chromosomes:
            # Filter for this chromosome
            chrom_df = df[df['chrom'] == chrom]
            
            if chrom_df.empty:
                continue
            
            # Save chromosome-specific file
            output_path = os.path.join(output_dir, f"chr{chrom}_snps.csv")
            chrom_df.to_csv(output_path, index=False)
            
            chromosome_files[chrom] = output_path
            logger.debug(f"Created {output_path} with {len(chrom_df)} SNPs")
        
        return chromosome_files
    
    def _cleanup_existing_files(self, output_dir: str) -> None:
        """Remove existing chromosome SNP files"""
        if os.path.exists(output_dir):
            for file_path in Path(output_dir).glob("chr*_snps.csv"):
                try:
                    file_path.unlink()
                    logger.debug(f"Removed existing file: {file_path}")
                except OSError as e:
                    logger.warning(f"Could not remove {file_path}: {e}")
    
    def create_chromosome_iterator(self, snplist_path: str, chromosomes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create an iterator that yields chromosome-specific SNP DataFrames
        More memory efficient than splitting to files for small datasets
        
        Args:
            snplist_path: Path to master SNP list
            chromosomes: List of chromosomes to process (None = all)
            
        Yields:
            Dict[str, pd.DataFrame]: chromosome -> DataFrame pairs
        """
        # Read the SNP list
        if self.use_polars:
            try:
                df = pl.read_csv(snplist_path)
                
                # Ensure chromosome column
                if 'chrom' not in df.columns and 'hg38' in df.columns:
                    df = df.with_columns([
                        pl.col('hg38').str.split(':').list.get(0).alias('chrom')
                    ])
                
                # Convert to pandas for downstream compatibility
                df = df.to_pandas()
            except:
                df = pd.read_csv(snplist_path)
        else:
            df = pd.read_csv(snplist_path)
        
        # Ensure chromosome column
        if 'chrom' not in df.columns and 'hg38' in df.columns:
            hg38_parts = df['hg38'].str.split(':')
            df['chrom'] = hg38_parts.str[0]
        
        if 'chrom' not in df.columns:
            raise ValueError("Cannot determine chromosome information")
        
        # Determine chromosomes to process
        if chromosomes is None:
            chromosomes = df['chrom'].dropna().unique()
            chromosomes = [str(c) for c in sorted(chromosomes) if str(c) not in ['', 'nan', 'None']]
        
        # Create chromosome dictionary
        chromosome_data = {}
        for chrom in chromosomes:
            chrom_df = df[df['chrom'] == chrom]
            if not chrom_df.empty:
                chromosome_data[chrom] = chrom_df
        
        return chromosome_data
    
    @staticmethod
    def get_standard_chromosomes(include_x: bool = True, include_y: bool = False, include_mt: bool = False) -> List[str]:
        """
        Get list of standard human chromosomes
        
        Args:
            include_x: Include X chromosome
            include_y: Include Y chromosome  
            include_mt: Include mitochondrial chromosome
            
        Returns:
            List[str]: List of chromosome names
        """
        chromosomes = [str(i) for i in range(1, 23)]  # 1-22
        
        if include_x:
            chromosomes.append('X')
        if include_y:
            chromosomes.append('Y')
        if include_mt:
            chromosomes.extend(['MT', 'M'])
        
        return chromosomes