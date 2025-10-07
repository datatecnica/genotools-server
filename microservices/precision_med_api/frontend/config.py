"""
Frontend configuration (simplified).
"""

import os
import sys
from dataclasses import dataclass
from typing import List
from app.core.config import Settings


@dataclass
class FrontendConfig:
    """Frontend configuration."""

    backend_settings: Settings
    debug_mode: bool
    results_base_path: str

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
