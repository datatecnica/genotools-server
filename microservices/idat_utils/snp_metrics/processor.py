"""SNP Metrics Processor

Core processor for converting IDAT files to VCF and extracting SNP metrics to parquet format.
Uses DRAGEN for IDAT->GTC->VCF conversion and pandas for data processing.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

import pandas as pd
import gzip

from .vcf_parser import VCFParser
from .config import ProcessorConfig


class SNPProcessor:
    """Processes genetic data from IDAT to parquet via DRAGEN pipeline.
    
    This class encapsulates the complete workflow:
    1. IDAT -> GTC conversion using DRAGEN
    2. GTC -> VCF conversion using DRAGEN  
    3. VCF parsing and data extraction
    4. Output to parquet format
    """

    def __init__(self, config: ProcessorConfig, num_threads: int = 1):
        """Initialize processor with configuration.
        
        Args:
            config: ProcessorConfig instance with all required paths and settings
            num_threads: Number of threads to use for DRAGEN processing
        """
        self.config = config
        self.num_threads = num_threads
        self.logger = self._setup_logging()
        
    def process_barcode(self, barcode: str, output_path: Optional[str] = None) -> str:
        """Process a single barcode through the complete pipeline.
        
        Args:
            barcode: The barcode identifier (used for output naming and sample ID)
            output_path: Optional custom output path for the parquet file
            
        Returns:
            Path to the generated parquet file
            
        Raises:
            ProcessingError: If any step in the pipeline fails
        """
        self.logger.info(f"Starting processing for barcode: {barcode}")
        
        try:
            # Step 1: Convert IDAT to GTC
            gtc_dir = self._convert_idat_to_gtc(barcode)
            
            # Step 2: Convert GTC to VCF
            vcf_paths = self._convert_gtc_to_vcf(barcode, gtc_dir)
            
            
            vcf_dir = self.config.vcf_path / barcode
            
            # TEMPORARY: Hardcoded VCF paths for testing
            # vcf_paths = [str(vcf_file) for vcf_file in vcf_dir.glob("*.vcf.gz")]

            if not vcf_paths:
                raise ProcessingError(f"No VCF files found in {vcf_dir}")
            self.logger.info(f"Using existing VCF files: {vcf_paths}")
            
            # Step 3: Parse VCF and convert to dataframe
            df = self._parse_vcf_to_dataframe(vcf_paths, barcode)
            
            # Step 4: Write to parquet
            parquet_path = self._write_to_parquet(df, barcode, output_path)
            
            self.logger.info(f"Successfully processed {barcode} -> {parquet_path}")
            return parquet_path
            
        except Exception as e:
            self.logger.error(f"Failed to process barcode {barcode}: {str(e)}")
            raise ProcessingError(f"Processing failed for {barcode}") from e

    def create_variant_reference(self, vcf_path: str, output_path: Optional[str] = None) -> str:
        """Create a variant reference file from a single VCF file.
        
        This extracts variant-specific information (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO)
        and saves it as a chromosome-partitioned parquet file. This should be run once to create
        a central reference since variant information is the same across all samples.
        
        Args:
            vcf_path: Path to any VCF file from the dataset (variant info is the same across samples)
            output_path: Optional custom output path for the variant reference parquet
            
        Returns:
            Path to the generated variant reference parquet
            
        Raises:
            ProcessingError: If variant reference creation fails
        """
        self.logger.info(f"Creating variant reference from: {vcf_path}")
        
        try:
            # Parse variant information only
            parser = VCFParser()
            df = parser.parse_variants(vcf_path)
            
            # Write variant reference to parquet
            variant_ref_path = self._write_variant_reference(df, output_path)
            
            self.logger.info(f"Successfully created variant reference -> {variant_ref_path}")
            return variant_ref_path
            
        except Exception as e:
            self.logger.error(f"Failed to create variant reference: {str(e)}")
            raise ProcessingError(f"Variant reference creation failed") from e

    def _convert_idat_to_gtc(self, barcode: str) -> str:
        """Convert IDAT files to GTC format using DRAGEN."""
        gtc_output_dir = self.config.gtc_path / barcode
        gtc_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(self.config.dragen_path),
            "genotype", "call",
            "--bpm-manifest", str(self.config.bpm_path),
            "--cluster-file", str(self.config.egt_path),
            "--idat-folder", str(self.config.barcode_path),
            "--output-folder", str(gtc_output_dir),
            "--num-threads", str(self.num_threads)
        ]
        
        self.logger.info(f"Converting IDAT to GTC for {barcode} (threads: {self.num_threads})")
        self._run_command(cmd, f"IDAT to GTC conversion failed for {barcode}")
        
        return str(gtc_output_dir)
    
    def _convert_gtc_to_vcf(self, barcode: str, gtc_dir: str) -> List[str]:
        """Convert GTC files to VCF format using DRAGEN."""
        vcf_output_dir = self.config.vcf_path / barcode
        vcf_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(self.config.dragen_path),
            "genotype", "gtc-to-vcf",
            "--bpm-manifest", str(self.config.bpm_path),
            "--csv-manifest", str(self.config.bpm_csv_path),
            "--genome-fasta-file", str(self.config.ref_fasta_path),
            "--gtc-folder", gtc_dir,
            "--unsquash-duplicates",
            "--output-folder", str(vcf_output_dir)
        ]
        
        self.logger.info(f"Converting GTC to VCF for {barcode}")
        self._run_command(cmd, f"GTC to VCF conversion failed for {barcode}")
        
        # Find all generated VCF files
        vcf_files = list(vcf_output_dir.glob("*.vcf.gz"))
        if not vcf_files:
            raise ProcessingError(f"No VCF files found in {vcf_output_dir}")
        
        self.logger.info(f"Found {len(vcf_files)} VCF files for {barcode}")
        return [str(vcf_file) for vcf_file in vcf_files]
    
    def _parse_vcf_to_dataframe(self, vcf_paths: List[str], barcode: str) -> pd.DataFrame:
        """Parse all VCF files and combine them into a single dataframe."""
        parser = VCFParser()
        all_dfs = []
        
        for vcf_path in vcf_paths:
            # Extract sample ID from VCF filename (e.g., "205746280003_R01C01.snv.vcf.gz" -> "205746280003_R01C01")
            vcf_filename = Path(vcf_path).name
            sample_id = vcf_filename.replace('.snv.vcf.gz', '').replace('.vcf.gz', '')
            
            df = parser.parse_vcf(vcf_path, sample_id=sample_id)
            all_dfs.append(df)
            self.logger.info(f"Extracted {len(df)} SNPs from VCF for sample {sample_id}")
        
        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"Combined {len(all_dfs)} samples with {len(combined_df)} total SNPs for {barcode}")
        
        return combined_df
    
    def _write_to_parquet(self, df: pd.DataFrame, barcode: str, custom_path: Optional[str] = None) -> str:
        """Write dataframe to parquet format with partitioning by IID and chromosome."""
        if custom_path:
            output_path = Path(custom_path)
        else:
            output_path = self.config.metrics_path / f"{barcode}"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if output already exists and throw informative error
        if output_path.exists():
            raise ProcessingError(
                f"Output already exists at: {output_path}\n"
                f"Please remove the existing file/directory before rerunning"
            )
        
        # Write with partitioning by IID and chromosome
        df.to_parquet(
            output_path, 
            compression='brotli', 
            index=False,
            partition_cols=['IID', 'chromosome']
        )
        
        unique_samples = df['IID'].nunique()
        unique_chromosomes = df['chromosome'].nunique()
        self.logger.info(f"Wrote {len(df)} records to {output_path} "
                        f"({unique_samples} samples, {unique_chromosomes} chromosomes)")
        
        return str(output_path)

    def _write_variant_reference(self, df: pd.DataFrame, custom_path: Optional[str] = None) -> str:
        """Write variant reference dataframe to parquet format partitioned by chromosome."""
        if custom_path:
            output_path = Path(custom_path)
        else:
            output_path = self.config.metrics_path / "variant_reference"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if output already exists and throw informative error
        if output_path.exists():
            raise ProcessingError(
                f"Variant reference already exists at: {output_path}\n"
                f"Please remove the existing file/directory before rerunning"
            )
        
        # Write with partitioning by chromosome only
        df.to_parquet(
            output_path, 
            compression='brotli', 
            index=False,
            partition_cols=['chromosome']
        )
        
        unique_chromosomes = df['chromosome'].nunique()
        total_variants = len(df)
        self.logger.info(f"Wrote {total_variants} variants to {output_path} "
                        f"({unique_chromosomes} chromosomes)")
        
        return str(output_path)
    
    def _run_command(self, cmd: list, error_message: str) -> None:
        """Execute a shell command with proper error handling."""
        try:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True
            )
            self.logger.debug(f"Command succeeded: {' '.join(cmd)}")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {' '.join(cmd)}")
            self.logger.error(f"Return code: {e.returncode}")
            self.logger.error(f"STDERR: {e.stderr}")
            raise ProcessingError(error_message) from e
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processor."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger


class ProcessingError(Exception):
    """Custom exception for SNP processing errors."""
    pass 