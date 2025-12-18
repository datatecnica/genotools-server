"""
Frontend configuration.
"""

import os
import sys
from dataclasses import dataclass


@dataclass
class FrontendConfig:
    """Frontend configuration for results viewer."""

    debug_mode: bool
    results_base_path: str

    @classmethod
    def create(cls, debug_mode: bool = None, results_base_path: str = None) -> 'FrontendConfig':
        """Create frontend configuration with auto-detected settings.

        Args:
            debug_mode: Enable debug mode. Auto-detected if None.
            results_base_path: Path to results directory. Uses default if None.

        Returns:
            FrontendConfig instance
        """
        # Auto-detect debug mode if not specified
        if debug_mode is None:
            debug_mode = "--debug" in sys.argv or os.getenv("STREAMLIT_DEBUG", "").lower() in ["true", "1", "yes"]

        # Auto-detect results path if not specified
        if results_base_path is None:
            # Check environment variable first
            results_base_path = os.getenv("RESULTS_PATH")

            # Fall back to default path
            if not results_base_path:
                results_base_path = os.path.expanduser(
                    # "~/gcs_mounts/genotools_server/precision_med/results/release10"
                    "/app/data/apps-data/precision_med/results/release10"
                    # "/app/data/precision_med/results/release10"
                )

        return cls(
            debug_mode=debug_mode,
            results_base_path=results_base_path
        )
