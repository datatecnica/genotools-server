"""
Configuration management for precision medicine data access.
Provides release-based path configuration for different data types.
"""
from typing import Dict, List
from pathlib import Path


class DataConfig:
    """Configuration class for managing data paths across releases."""
    
    # Base GCS mount paths
    GCS_BASE = Path("~/gcs_mounts/genotools_server").expanduser()
    
    # Available releases (starting with release10 as specified)
    AVAILABLE_RELEASES = ["release10"]
    DEFAULT_RELEASE = "release10"
    
    def __init__(self, release: str = DEFAULT_RELEASE):
        """Initialize configuration for a specific release.
        
        Args:
            release: Release version (e.g., "release10")
        """
        if release not in self.AVAILABLE_RELEASES:
            raise ValueError(f"Release {release} not available. Available: {self.AVAILABLE_RELEASES}")
        
        self.release = release
        self._setup_paths()
    
    def _setup_paths(self) -> None:
        """Setup all data paths for the current release."""
        # Base path for carriers data
        self.carriers_base = self.GCS_BASE / "carriers"
        
        # NBA (Next-generation Biomarker Analysis) paths
        self.nba_base = self.carriers_base / "nba" / self.release / "combined"
        
        # NBA combined data files
        self.nba_files = {
            "info": self.nba_base / f"nba_{self.release}_combined_info.csv",
            "int": self.nba_base / f"nba_{self.release}_combined_int.csv", 
            "string": self.nba_base / f"nba_{self.release}_combined_string.csv"
        }
        
        # WGS (Whole Genome Sequencing) paths
        self.wgs_base = self.carriers_base / "wgs" / self.release
        
        # WGS data files
        self.wgs_files = {
            "info": self.wgs_base / f"{self.release}_var_info.csv",
            "int": self.wgs_base / f"{self.release}_carriers_int.csv",
            "string": self.wgs_base / f"{self.release}_carriers_string.csv"
        }
        
        # Validate paths exist
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that configured paths exist."""
        if not self.carriers_base.exists():
            raise FileNotFoundError(f"Carriers base path not found: {self.carriers_base}")
        
        # Validate NBA paths
        if not self.nba_base.exists():
            raise FileNotFoundError(f"NBA base path not found: {self.nba_base}")
        
        for file_type, path in self.nba_files.items():
            if not path.exists():
                raise FileNotFoundError(f"NBA {file_type} file not found: {path}")
        
        # Validate WGS paths
        if not self.wgs_base.exists():
            raise FileNotFoundError(f"WGS base path not found: {self.wgs_base}")
        
        for file_type, path in self.wgs_files.items():
            if not path.exists():
                raise FileNotFoundError(f"WGS {file_type} file not found: {path}")
    
    def get_nba_file_path(self, file_type: str) -> Path:
        """Get path for specific NBA file type.
        
        Args:
            file_type: Type of file ('info', 'int', 'string')
            
        Returns:
            Path to the requested file
            
        Raises:
            ValueError: If file_type is not valid
        """
        if file_type not in self.nba_files:
            raise ValueError(f"Invalid file type: {file_type}. Available: {list(self.nba_files.keys())}")
        
        return self.nba_files[file_type]
    
    def get_all_nba_paths(self) -> Dict[str, Path]:
        """Get all NBA file paths as a dictionary."""
        return self.nba_files.copy()
    
    def get_wgs_file_path(self, file_type: str) -> Path:
        """Get path for specific WGS file type.
        
        Args:
            file_type: Type of file ('info', 'int', 'string')
            
        Returns:
            Path to the requested file
            
        Raises:
            ValueError: If file_type is not valid
        """
        if file_type not in self.wgs_files:
            raise ValueError(f"Invalid file type: {file_type}. Available: {list(self.wgs_files.keys())}")
        
        return self.wgs_files[file_type]
    
    def get_all_wgs_paths(self) -> Dict[str, Path]:
        """Get all WGS file paths as a dictionary."""
        return self.wgs_files.copy()
    
    def switch_release(self, new_release: str) -> None:
        """Switch to a different release and update all paths.
        
        Args:
            new_release: New release version to switch to
        """
        if new_release not in self.AVAILABLE_RELEASES:
            raise ValueError(f"Release {new_release} not available. Available: {self.AVAILABLE_RELEASES}")
        
        self.release = new_release
        self._setup_paths()
    
    @classmethod
    def get_available_releases(cls) -> List[str]:
        """Get list of available releases."""
        return cls.AVAILABLE_RELEASES.copy()


# Global configuration instance (can be reconfigured)
config = DataConfig()
