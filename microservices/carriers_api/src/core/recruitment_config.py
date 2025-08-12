"""
Configuration for precision medicine recruitment analysis.

This module provides configuration classes following the carriers API patterns
defined in .cursorrules, with proper validation and type safety.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class RecruitmentAnalysisConfig:
    """
    Configuration for precision medicine recruitment analysis.
    
    Attributes:
        release: GP2 release version (e.g., "10")
        mnt_path: Base mount path for data files
        output_dir: Output directory for results files (optional)
    """
    release: str
    mnt_path: str = "~/gcs_mounts"
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.release.isdigit():
            raise ValueError(f"Release must be a numeric string, got: {self.release}")
        
        # Expand user path
        self.mnt_path = os.path.expanduser(self.mnt_path)
        
        # Set default output directory if not provided
        if self.output_dir is None:
            self.output_dir = os.path.join(
                self.mnt_path, 
                "clinical_trial_output", 
                f"release{self.release}"
            )
        else:
            self.output_dir = os.path.expanduser(self.output_dir)
    
    @property
    def release_version(self) -> int:
        """Get release version as integer."""
        return int(self.release)
    
    @property
    def carriers_path(self) -> str:
        """Get carriers data path."""
        return os.path.join(self.mnt_path, "genotools_server", "carriers")
    
    @property
    def clinical_path(self) -> str:
        """Get clinical data path."""
        return os.path.join(
            self.mnt_path, 
            "gp2tier2_vwb", 
            f"release{self.release}",
            "clinical_data"
        )