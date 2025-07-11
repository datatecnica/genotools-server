"""Configuration management for SNP Metrics Processor

Handles all paths, parameters, and settings required for the processing pipeline.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProcessorConfig:
    """Configuration container for SNP processing pipeline.
    
    All paths are converted to Path objects for consistent handling.
    Required paths are validated on initialization.
    """
    
    # Input paths
    barcode_path: Path
    dragen_path: Path
    bpm_path: Path
    bpm_csv_path: Path
    egt_path: Path
    ref_fasta_path: Path
    
    # Output paths
    gtc_path: Path
    vcf_path: Path
    metrics_path: Path
    
    def __post_init__(self):
        """Convert string paths to Path objects and validate required files."""
        # Convert all paths to Path objects
        self.barcode_path = Path(self.barcode_path)
        self.dragen_path = Path(self.dragen_path)
        self.bpm_path = Path(self.bpm_path)
        self.bpm_csv_path = Path(self.bpm_csv_path)
        self.egt_path = Path(self.egt_path)
        self.ref_fasta_path = Path(self.ref_fasta_path)
        self.gtc_path = Path(self.gtc_path)
        self.vcf_path = Path(self.vcf_path)
        self.metrics_path = Path(self.metrics_path)
        
        # Validate required input files exist
        self._validate_required_files()
        
        # Create output directories
        self._create_output_directories()
    
    def _validate_required_files(self):
        """Validate that required input files exist."""
        required_files = [
            self.dragen_path,
            self.bpm_path,
            self.bpm_csv_path,
            self.egt_path,
            self.ref_fasta_path
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            raise ConfigurationError(
                f"Required files not found: {', '.join(str(f) for f in missing_files)}"
            )
            
        # Validate barcode directory exists
        if not self.barcode_path.exists():
            raise ConfigurationError(f"Barcode directory not found: {self.barcode_path}")
    
    def _create_output_directories(self):
        """Create output directories if they don't exist."""
        for path in [self.gtc_path, self.vcf_path, self.metrics_path]:
            path.mkdir(parents=True, exist_ok=True)


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass 