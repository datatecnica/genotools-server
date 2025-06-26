"""
Configuration management for GP2 Precision Medicine Data Browser.
"""
from typing import Optional
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Data paths
    data_root: Path = Field(default=Path("./data"), description="Root directory for data files")
    
    # Clinical data
    clinical_data_dir: Optional[Path] = Field(default=None, description="Clinical data directory")
    master_key_file: str = Field(default="master_key_release10_final_vwb.csv", description="Master key filename")
    
    # WGS data
    wgs_data_dir: Optional[Path] = Field(default=None, description="WGS data directory")
    wgs_var_info_file: str = Field(default="release10_var_info.csv", description="WGS variant info filename")
    wgs_carriers_int_file: str = Field(default="release10_carriers_int.csv", description="WGS carriers int filename")
    wgs_carriers_string_file: str = Field(default="release10_carriers_string.csv", description="WGS carriers string filename")
    
    # NBA data
    nba_data_dir: Optional[Path] = Field(default=None, description="NBA data directory")
    nba_info_file: str = Field(default="nba_release10_combined_info.csv", description="NBA variant info filename")
    nba_carriers_int_file: str = Field(default="nba_release10_combined_int.csv", description="NBA carriers int filename")
    nba_carriers_string_file: str = Field(default="nba_release10_combined_string.csv", description="NBA carriers string filename")
    
    # Simple settings for small datasets
    enable_cache: bool = Field(default=True, description="Enable data caching")
    
    def model_post_init(self, __context) -> None:
        """Set default paths if not provided."""
        if self.clinical_data_dir is None:
            self.clinical_data_dir = self.data_root / "clinical_data"
        if self.wgs_data_dir is None:
            self.wgs_data_dir = self.data_root / "wgs"
        if self.nba_data_dir is None:
            self.nba_data_dir = self.data_root / "nba" / "combined"
    
    @property
    def clinical_master_key_path(self) -> Path:
        """Get full path to clinical master key file."""
        return self.clinical_data_dir / self.master_key_file
    
    @property
    def wgs_var_info_path(self) -> Path:
        """Get full path to WGS variant info file."""
        return self.wgs_data_dir / self.wgs_var_info_file
    
    @property
    def wgs_carriers_int_path(self) -> Path:
        """Get full path to WGS carriers int file."""
        return self.wgs_data_dir / self.wgs_carriers_int_file
    
    @property
    def wgs_carriers_string_path(self) -> Path:
        """Get full path to WGS carriers string file."""
        return self.wgs_data_dir / self.wgs_carriers_string_file
    
    @property
    def nba_info_path(self) -> Path:
        """Get full path to NBA variant info file."""
        return self.nba_data_dir / self.nba_info_file
    
    @property
    def nba_carriers_int_path(self) -> Path:
        """Get full path to NBA carriers int file."""
        return self.nba_data_dir / self.nba_carriers_int_file
    
    @property
    def nba_carriers_string_path(self) -> Path:
        """Get full path to NBA carriers string file."""
        return self.nba_data_dir / self.nba_carriers_string_file
    
    model_config = {
        "env_prefix": "GP2_",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Global settings instance
settings = Settings() 