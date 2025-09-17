"""
Frontend configuration with dependency injection pattern.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List
from app.core.config import Settings


@dataclass
class FrontendConfig:
    """Frontend-specific configuration with dependency injection."""

    backend_settings: Settings
    debug_mode: bool
    results_base_path: str
    data_types: List[str] = field(default_factory=lambda: ["NBA", "WGS", "IMPUTED"])

    @classmethod
    def create(cls, debug_mode: bool = None) -> 'FrontendConfig':
        """Create frontend configuration with auto-detected settings."""
        backend_settings = Settings.create_optimized()

        # Auto-detect debug mode if not specified
        if debug_mode is None:
            debug_mode = "--debug" in sys.argv or os.getenv("STREAMLIT_DEBUG", "").lower() in ["true", "1", "yes"]

        return cls(
            backend_settings=backend_settings,
            debug_mode=debug_mode,
            results_base_path=backend_settings.results_path
        )

    @property
    def release(self) -> str:
        """Get current release version."""
        return self.backend_settings.release

    @property
    def ancestries(self) -> List[str]:
        """Get available ancestries."""
        return self.backend_settings.ANCESTRIES

    @property
    def gcs_results_path(self) -> str:
        """Convert local mount path to GCS bucket path."""
        # Convert ~/gcs_mounts/genotools_server/... to gs://genotools-server/...
        local_path = self.results_base_path
        if "/gcs_mounts/genotools_server/" in local_path:
            # Extract path after the mount point
            bucket_path = local_path.split("/gcs_mounts/genotools_server/", 1)[1]
            return f"gs://genotools-server/{bucket_path}"
        elif "/gcs_mounts/gp2tier2_vwb/" in local_path:
            # Handle gp2tier2_vwb bucket if needed
            bucket_path = local_path.split("/gcs_mounts/gp2tier2_vwb/", 1)[1]
            return f"gs://gp2tier2_vwb/{bucket_path}"
        else:
            # Fallback to local path if not a recognized mount
            return local_path