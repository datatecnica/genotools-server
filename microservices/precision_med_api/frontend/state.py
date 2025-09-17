"""
Centralized state management using Streamlit session state.
"""

import streamlit as st
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class AppState:
    """Centralized application state container."""

    selected_release: Optional[str] = None
    selected_job: Optional[str] = None
    cached_data: Dict[str, Any] = field(default_factory=dict)

    def clear_cache(self) -> None:
        """Clear cached data."""
        self.cached_data.clear()

    def get_cached(self, key: str) -> Any:
        """Get cached data by key."""
        return self.cached_data.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        """Set cached data by key."""
        self.cached_data[key] = value


def get_app_state() -> AppState:
    """Get or create app state from session state."""
    if 'app_state' not in st.session_state:
        st.session_state.app_state = AppState()
    return st.session_state.app_state


def reset_app_state() -> None:
    """Reset application state."""
    if 'app_state' in st.session_state:
        del st.session_state.app_state