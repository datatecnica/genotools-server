"""
Application settings and configuration.
"""

import os
from typing import Dict, Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Settings can be configured via:
    1. Environment variables (CARRIERS_API_ prefix)
    2. .env file
    3. Direct instantiation
    """
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="CARRIERS_API_HOST")
    api_port: int = Field(default=8000, env="CARRIERS_API_PORT")
    api_workers: int = Field(default=1, env="CARRIERS_API_WORKERS")
    api_reload: bool = Field(default=False, env="CARRIERS_API_RELOAD")
    
    # Storage Configuration (gcsfuse mounted)
    base_path: str = Field(default="~/gcs_mounts", env="CARRIERS_API_BASE_PATH")
    
    # Dataset Paths (can use {mount}, {release}, {version} placeholders)
    dataset_paths: Dict[str, str] = Field(
        default={
            "nba": "{mount}/gp2tier2_vwb/release{release}/raw_genotypes",
            "wgs": "{mount}/gp2tier2_vwb/release{release}/wgs_genotypes",
            "imputed": "{mount}/gp2tier2_vwb/release{release}/imputed_genotypes"
        },
        env="CARRIERS_API_DATASET_PATHS"
    )
    
    # Output Configuration
    output_base_pattern: str = Field(
        default="{mount}/genotools_server/carriers",
        env="CARRIERS_API_OUTPUT_BASE"
    )
    harmonization_cache_pattern: str = Field(
        default="{mount}/genotools_server/harmonization",
        env="CARRIERS_API_HARMONIZATION_CACHE"
    )
    
    # Processing Configuration
    max_parallel_workers: int = Field(default=4, env="CARRIERS_API_MAX_WORKERS")
    chunk_size: int = Field(default=500000, env="CARRIERS_API_CHUNK_SIZE")
    use_harmonization_cache: bool = Field(default=True, env="CARRIERS_API_USE_CACHE")
    
    # Clinical Data Paths (for recruitment analysis)
    clinical_data_pattern: str = Field(
        default="{mount}/gp2tier2_vwb/release{release}/clinical_data",
        env="CARRIERS_API_CLINICAL_DATA"
    )
    
    # Security
    require_api_key: bool = Field(default=False, env="CARRIERS_API_REQUIRE_KEY")
    api_key: Optional[str] = Field(default=None, env="CARRIERS_API_KEY")
    
    # Logging
    log_level: str = Field(default="INFO", env="CARRIERS_API_LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="CARRIERS_API_LOG_FORMAT"
    )
    
    class Config:
        env_file = ".env"
        env_prefix = "CARRIERS_API_"
        case_sensitive = False
    
    def get_dataset_path(self, dataset_type: str, release: str) -> str:
        """Get resolved dataset path."""
        pattern = self.dataset_paths.get(dataset_type, "")
        mount = Path(self.base_path).expanduser()
        return pattern.format(mount=mount, release=release, version=release)
    
    def get_output_base(self, release: str = None) -> str:
        """Get resolved output base path."""
        mount = Path(self.base_path).expanduser()
        path = self.output_base_pattern.format(mount=mount)
        if release:
            path = os.path.join(path, f"release{release}")
        return path
    
    def get_harmonization_cache_dir(self, release: str = None) -> str:
        """Get resolved harmonization cache directory."""
        mount = Path(self.base_path).expanduser()
        path = self.harmonization_cache_pattern.format(mount=mount)
        if release:
            path = os.path.join(path, f"release{release}")
        return path
    
    def get_clinical_data_path(self, release: str) -> str:
        """Get resolved clinical data path."""
        mount = Path(self.base_path).expanduser()
        return self.clinical_data_pattern.format(mount=mount, release=release)
    
    def to_dict(self) -> Dict:
        """Convert settings to dictionary."""
        return {
            "api": {
                "host": self.api_host,
                "port": self.api_port,
                "workers": self.api_workers,
                "reload": self.api_reload
            },
            "storage": {
                "base_path": self.base_path
            },
            "processing": {
                "max_workers": self.max_parallel_workers,
                "chunk_size": self.chunk_size,
                "use_cache": self.use_harmonization_cache
            },
            "security": {
                "require_api_key": self.require_api_key
            },
            "logging": {
                "level": self.log_level,
                "format": self.log_format
            }
        }


# Global settings instance
settings = Settings()
