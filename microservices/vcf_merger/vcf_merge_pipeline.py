#!/usr/bin/env python3
"""
VCF Merge Pipeline with Scatter-Gather Pattern
Merges multiple VCF files by processing genomic regions in parallel
"""

import os
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineParams:
    """Parameters for the VCF merge pipeline"""
    vcf_files: List[str]
    regions_file: str
    output_file: str = "final_merged.vcf.gz"
    max_workers: int = 10
    work_dir: Optional[str] = None


@dataclass
class ProcessConfig:
    """Configuration for each process"""
    cpu: int
    memory_gb: int


class VCFMergePipeline:
    """
    Scatter-Gather pipeline for merging VCF files
    
    1. Scatter: Process each genomic region independently
    2. Gather: Concatenate all processed shards into final output
    """
    
    # Process configurations
    MERGE_CONFIG = ProcessConfig(
        cpu=2,
        memory_gb=8
    )
    
    CONCAT_CONFIG = ProcessConfig(
        cpu=4,
        memory_gb=16
    )
    
    def __init__(self, params: PipelineParams):
        """
        Initialize the pipeline with parameters
        
        Args:
            params: Pipeline parameters including input VCFs and regions file
        """
        self.params = params
        self.work_dir = Path(params.work_dir) if params.work_dir else Path(tempfile.mkdtemp(prefix="vcf_merge_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.work_dir / "shards"
        self.shard_dir.mkdir(exist_ok=True)
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate input files exist and are accessible"""
        for vcf_file in self.params.vcf_files:
            if not Path(vcf_file).exists():
                raise FileNotFoundError(f"VCF file not found: {vcf_file}")
        
        if not Path(self.params.regions_file).exists():
            raise FileNotFoundError(f"Regions file not found: {self.params.regions_file}")
    
    def _read_regions(self) -> List[str]:
        """
        Read genomic regions from the regions file
        
        Returns:
            List of region strings (e.g., ["chr1:1-10000000", "chr1:10000001-20000000"])
        """
        regions = []
        with open(self.params.regions_file, 'r') as f:
            for line in f:
                region = line.strip()
                if region and not region.startswith('#'):
                    regions.append(region)
        
        logger.info(f"Loaded {len(regions)} regions from {self.params.regions_file}")
        return regions
    
    def _run_command(self, 
                     cmd: List[str], 
                     description: str) -> Tuple[int, str, str]:
        """
        Run a command and capture output
        
        Args:
            cmd: Command to run as list of strings
            description: Description of the command for logging
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        logger.info(f"Running {description}")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode == 0:
                logger.debug(f"Command successful: {description}")
            else:
                logger.error(f"Command failed: {description}")
                logger.error(f"Exit code: {result.returncode}")
                logger.error(f"Stderr: {result.stderr}")
            
            return result.returncode, result.stdout, result.stderr
            
        except Exception as e:
            logger.error(f"Exception running command: {e}")
            raise
    
    def merge_shard(self, region: str, shard_index: int) -> Optional[Path]:
        """
        Process 1: MERGE_SHARDS (Scatter)
        Merge VCF files for a specific genomic region
        
        Args:
            region: Genomic region string (e.g., "chr1:1-10000000")
            shard_index: Index of this shard for naming
            
        Returns:
            Path to the output BCF file if successful, None otherwise
        """
        # Clean region string for filename
        region_clean = region.replace(':', '_').replace('-', '_')
        output_file = self.shard_dir / f"shard_{shard_index:04d}_{region_clean}.bcf"
        
        # Build bcftools merge command
        cmd = [
            "bcftools", "merge",
            "--threads", str(self.MERGE_CONFIG.cpu),
            "-r", region,
            "-Ou",  # Uncompressed BCF output
            "-o", str(output_file)
        ] + self.params.vcf_files
        
        # Run command
        returncode, stdout, stderr = self._run_command(
            cmd,
            f"merge shard {shard_index} (region: {region})"
        )
        
        if returncode == 0:
            logger.info(f"Successfully created shard: {output_file}")
            return output_file
        else:
            logger.error(f"Failed to create shard for region {region}")
            return None
    
    def gather_concat(self, shard_files: List[Path]) -> bool:
        """
        Process 2: GATHER_CONCAT (Gather)
        Concatenate all BCF shards into final VCF.gz file
        
        Args:
            shard_files: List of paths to BCF shard files
            
        Returns:
            True if successful, False otherwise
        """
        if not shard_files:
            logger.error("No shard files to concatenate")
            return False
        
        # Sort shards numerically by extracting shard index
        def extract_shard_index(filepath: Path) -> int:
            match = re.search(r'shard_(\d+)_', filepath.name)
            return int(match.group(1)) if match else 0
        
        sorted_shards = sorted(shard_files, key=extract_shard_index)
        logger.info(f"Concatenating {len(sorted_shards)} shards in numerical order")
        
        # Create file list for bcftools concat
        file_list = self.work_dir / "shard_list.txt"
        with open(file_list, 'w') as f:
            for shard in sorted_shards:
                f.write(f"{shard}\n")
        
        output_path = Path(self.params.output_file)
        
        # Build bcftools concat command
        concat_cmd = [
            "bcftools", "concat",
            "-f", str(file_list),
            "-Oz",  # Compressed VCF output
            "-o", str(output_path),
            "--threads", str(self.CONCAT_CONFIG.cpu)
        ]
        
        # Run concatenation
        returncode, stdout, stderr = self._run_command(
            concat_cmd,
            "concatenate shards"
        )
        
        if returncode != 0:
            logger.error("Failed to concatenate shards")
            return False
        
        logger.info(f"Successfully concatenated shards to {output_path}")
        
        # Index the final VCF.gz file
        index_cmd = [
            "bcftools", "index",
            "-t",  # Create TBI index
            str(output_path),
            "--threads", str(self.CONCAT_CONFIG.cpu)
        ]
        
        returncode, stdout, stderr = self._run_command(
            index_cmd,
            "index final VCF"
        )
        
        if returncode != 0:
            logger.error("Failed to index final VCF")
            return False
        
        logger.info(f"Successfully indexed {output_path}")
        return True
    
    def run(self) -> bool:
        """
        Execute the complete scatter-gather pipeline
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        logger.info("Starting VCF merge pipeline")
        logger.info(f"Working directory: {self.work_dir}")
        logger.info(f"Input VCFs: {len(self.params.vcf_files)}")
        
        # Read regions for scatter phase
        regions = self._read_regions()
        if not regions:
            logger.error("No regions found in regions file")
            return False
        
        # Phase 1: Scatter - Process regions in parallel
        logger.info(f"Phase 1: Scatter - Processing {len(regions)} regions in parallel")
        successful_shards = []
        failed_regions = []
        
        with ProcessPoolExecutor(max_workers=self.params.max_workers) as executor:
            # Submit all merge tasks
            future_to_region = {
                executor.submit(self.merge_shard, region, idx): (region, idx)
                for idx, region in enumerate(regions, 1)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_region):
                region, idx = future_to_region[future]
                try:
                    shard_file = future.result()
                    if shard_file and shard_file.exists():
                        successful_shards.append(shard_file)
                        logger.info(f"Completed shard {idx}/{len(regions)}: {region}")
                    else:
                        failed_regions.append(region)
                        logger.error(f"Failed shard {idx}/{len(regions)}: {region}")
                except Exception as e:
                    failed_regions.append(region)
                    logger.error(f"Exception processing region {region}: {e}")
        
        logger.info(f"Scatter phase complete: {len(successful_shards)}/{len(regions)} shards successful")
        
        if failed_regions:
            logger.warning(f"Failed regions: {failed_regions}")
        
        if not successful_shards:
            logger.error("No shards were successfully created")
            return False
        
        # Phase 2: Gather - Concatenate all shards
        logger.info("Phase 2: Gather - Concatenating shards")
        success = self.gather_concat(successful_shards)
        
        if success:
            logger.info(f"Pipeline completed successfully. Output: {self.params.output_file}")
            
            # Optionally clean up temporary files
            if self.params.work_dir is None:  # Only clean up if we created temp dir
                logger.info("Cleaning up temporary files")
                for shard in successful_shards:
                    try:
                        shard.unlink()
                    except Exception as e:
                        logger.warning(f"Could not remove shard file {shard}: {e}")
        else:
            logger.error("Pipeline failed during gather phase")
        
        return success


def merge_vcf_files(vcf_files: List[str], 
                   regions_file: str,
                   output_file: str = "final_merged.vcf.gz",
                   max_workers: int = 10,
                   work_dir: Optional[str] = None) -> bool:
    """
    Main method to execute the VCF merge pipeline with scatter-gather pattern
    
    Args:
        vcf_files: List of input VCF file paths
        regions_file: Path to file containing genomic regions (one per line)
        output_file: Path for the final merged VCF.gz output
        max_workers: Maximum number of parallel workers for scatter phase
        work_dir: Working directory for temporary files (auto-created if None)
    
    Returns:
        True if pipeline completed successfully, False otherwise
        
    Example:
        >>> vcf_files = ["sample1.vcf.gz", "sample2.vcf.gz", "sample3.vcf.gz"]
        >>> regions_file = "regions.txt"  # Contains: chr1:1-10000000, chr1:10000001-20000000, etc.
        >>> success = merge_vcf_files(vcf_files, regions_file, "merged_output.vcf.gz")
    """
    params = PipelineParams(
        vcf_files=vcf_files,
        regions_file=regions_file,
        output_file=output_file,
        max_workers=max_workers,
        work_dir=work_dir
    )
    
    pipeline = VCFMergePipeline(params)
    return pipeline.run()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge VCF files using scatter-gather pattern")
    parser.add_argument("--vcf-files", nargs="+", required=True, help="Input VCF files")
    parser.add_argument("--regions-file", required=True, help="File containing genomic regions")
    parser.add_argument("--output", default="final_merged.vcf.gz", help="Output file path")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")
    parser.add_argument("--work-dir", help="Working directory for temporary files")
    
    args = parser.parse_args()
    
    success = merge_vcf_files(
        vcf_files=args.vcf_files,
        regions_file=args.regions_file,
        output_file=args.output,
        max_workers=args.max_workers,
        work_dir=args.work_dir
    )
    
    exit(0 if success else 1)