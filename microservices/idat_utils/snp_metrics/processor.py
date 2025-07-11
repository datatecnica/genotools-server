"""SNP Metrics Processor

Core processor for converting IDAT files to VCF and extracting SNP metrics to parquet format.
Uses DRAGEN for IDAT->GTC->VCF conversion and pandas for data processing.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
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
            gtc_path = self._convert_idat_to_gtc(barcode)
            
            # Step 2: Convert GTC to VCF  
            vcf_path = self._convert_gtc_to_vcf(barcode, gtc_path)
            
            # Step 3: Parse VCF and convert to dataframe
            df = self._parse_vcf_to_dataframe(vcf_path, barcode)
            
            # Step 4: Write to parquet
            parquet_path = self._write_to_parquet(df, barcode, output_path)
            
            self.logger.info(f"Successfully processed {barcode} -> {parquet_path}")
            return parquet_path
            
        except Exception as e:
            self.logger.error(f"Failed to process barcode {barcode}: {str(e)}")
            raise ProcessingError(f"Processing failed for {barcode}") from e

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
    
    def _convert_gtc_to_vcf(self, barcode: str, gtc_path: str) -> str:
        """Convert GTC files to VCF format using DRAGEN."""
        vcf_output_dir = self.config.vcf_path / barcode
        vcf_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(self.config.dragen_path),
            "genotype", "gtc-to-vcf",
            "--bpm-manifest", str(self.config.bpm_path),
            "--csv-manifest", str(self.config.bpm_csv_path),
            "--genome-fasta-file", str(self.config.ref_fasta_path),
            "--gtc-folder", gtc_path,
            "--output-folder", str(vcf_output_dir)
        ]
        
        self.logger.info(f"Converting GTC to VCF for {barcode}")
        self._run_command(cmd, f"GTC to VCF conversion failed for {barcode}")
        
        # Find the generated VCF file
        vcf_files = list(vcf_output_dir.glob("*.vcf.gz"))
        if not vcf_files:
            raise ProcessingError(f"No VCF files found in {vcf_output_dir}")
        
        return str(vcf_files[0])
    
    def _parse_vcf_to_dataframe(self, vcf_path: str, barcode: str) -> pd.DataFrame:
        """Parse VCF file and extract relevant data into a dataframe."""
        parser = VCFParser()
        df = parser.parse_vcf(vcf_path, sample_id=barcode)
        
        self.logger.info(f"Extracted {len(df)} SNPs from VCF for {barcode}")
        return df
    
    def _write_to_parquet(self, df: pd.DataFrame, barcode: str, custom_path: Optional[str] = None) -> str:
        """Write dataframe to parquet format."""
        if custom_path:
            output_path = Path(custom_path)
        else:
            output_path = self.config.metrics_path / f"{barcode}_snp_metrics.parquet"
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, compression='snappy', index=False)
        self.logger.info(f"Wrote {len(df)} records to {output_path}")
        
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