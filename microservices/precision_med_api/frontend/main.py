"""
Main Streamlit application orchestrator.
"""

import streamlit as st
import os
import sys
import pandas as pd
from typing import Tuple

# Add parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.config import FrontendConfig
from frontend.state import get_app_state
from frontend.utils.data_facade import DataFacade
from frontend.pages import overview


def setup_app_config() -> FrontendConfig:
    """Set up application configuration and page settings."""
    st.set_page_config(
        page_title="Carriers Pipeline Results Viewer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Create frontend configuration (auto-detects debug mode)
    return FrontendConfig.create()


def setup_sidebar(config: FrontendConfig) -> Tuple[str, str]:
    """
    Set up sidebar navigation with release and job selection.

    Args:
        config: Frontend configuration

    Returns:
        Tuple of (selected_release, selected_job)
    """
    st.sidebar.header("📁 Data Selection")
    app_state = get_app_state()
    data_facade = DataFacade(config)

    # Release selection
    releases = data_facade.discover_releases()
    if not releases:
        st.error(f"No releases found in {config.results_base_path}")
        st.info(
            "**Troubleshooting:**\n"
            "- Ensure GCS mounts are accessible\n"
            "- Check that pipeline results exist\n"
            "- Verify file permissions"
        )
        st.stop()

    # Use session state for persistence
    if app_state.selected_release not in releases:
        app_state.selected_release = releases[0]

    selected_release = st.sidebar.selectbox(
        "Select Release",
        releases,
        index=releases.index(app_state.selected_release) if app_state.selected_release in releases else 0
    )
    app_state.selected_release = selected_release

    # Job selection (debug mode only)
    if config.debug_mode:
        jobs = data_facade.discover_jobs(selected_release)

        if len(jobs) > 1:
            if app_state.selected_job not in jobs:
                app_state.selected_job = jobs[0]

            selected_job = st.sidebar.selectbox(
                "Select Job",
                jobs,
                index=jobs.index(app_state.selected_job) if app_state.selected_job in jobs else 0,
                help="Choose pipeline run to view. Main release results are typically named after the release."
            )
        else:
            selected_job = jobs[0] if jobs else selected_release

        app_state.selected_job = selected_job

        # Debug mode indicator
        st.sidebar.info("🔧 **Debug Mode Active**\nJob selection enabled")

        # Debug info (simplified)
        with st.sidebar.expander("🛠️ Debug Info", expanded=False):
            st.text(f"Available jobs: {len(jobs)}")
            available_data_types = data_facade.get_available_data_types(selected_release, selected_job)
            st.text(f"Data types: {', '.join(available_data_types)}")

            if st.button("Clear Cache", help="Clear all cached data"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()

    else:
        # Production mode: always use release name as job name
        selected_job = selected_release
        app_state.selected_job = selected_job

    return selected_release, selected_job


def render_main_header(release: str, job_name: str, config: FrontendConfig) -> None:
    """
    Render main application header.

    Args:
        release: Selected release
        job_name: Selected job name
        config: Frontend configuration
    """
    st.title("🧬 Carriers Pipeline Results Viewer")
    st.markdown("Browse and analyze results from the precision medicine carriers pipeline.")

    # Show configuration info in debug mode
    if config.debug_mode:
        with st.expander("🔧 Configuration Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"GCS Results Bucket: {config.gcs_results_path}")
                st.text(f"Backend Release: {config.backend_settings.release}")
            with col2:
                st.text(f"Debug Mode: {config.debug_mode}")
                st.text(f"Available Data Types: {', '.join(config.data_types)}")


def render_main_content(release: str, job_name: str, config: FrontendConfig) -> None:
    """
    Render main content area with overview section.

    Args:
        release: Selected release
        job_name: Selected job name
        config: Frontend configuration
    """
    # Overview section
    overview.render_overview_with_validation(release, job_name, config)


def handle_errors() -> None:
    """Handle application-level errors and show user-friendly messages."""
    try:
        # This is where we'd catch any unhandled exceptions
        # and show user-friendly error messages
        pass
    except Exception as e:
        st.error("Application Error")
        st.error(f"An unexpected error occurred: {e}")

        if st.button("Reset Application"):
            st.cache_data.clear()
            st.rerun()


def main():
    """Main application entry point."""
    try:
        # Setup configuration
        config = setup_app_config()

        # Render main header
        render_main_header("", "", config)  # Will be updated after sidebar setup

        # Setup sidebar navigation and get selections
        release, job_name = setup_sidebar(config)

        # Validate selections before proceeding
        data_facade = DataFacade(config)
        if not data_facade.validate_release_and_job(release, job_name):
            st.error(f"Invalid release '{release}' or job '{job_name}' combination")
            st.info("Please select a valid release and job from the sidebar.")
            return

        # Render main content
        render_main_content(release, job_name, config)

        # Footer information
        if config.debug_mode:
            st.caption(f"Release Overview | Debug Mode | {release} | {job_name}")
        else:
            st.caption("Release Overview | Precision Medicine Carriers Pipeline")

    except Exception as e:
        st.error("Application Initialization Error")
        st.error(f"Failed to initialize application: {e}")

        st.info(
            "**Troubleshooting Steps:**\n"
            "1. Check GCS mount accessibility\n"
            "2. Verify backend configuration\n"
            "3. Ensure required files exist\n"
            "4. Try refreshing the page"
        )

        if st.button("Retry Initialization"):
            st.rerun()


if __name__ == "__main__":
    main()