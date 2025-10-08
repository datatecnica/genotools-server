"""
Main Streamlit application - simplified frontend.
"""

import streamlit as st
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.config import FrontendConfig
from frontend.utils.data_loaders import discover_releases, discover_jobs, check_data_availability


def setup_page_config():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Carriers Pipeline Results Viewer",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def setup_sidebar(config: FrontendConfig):
    """
    Setup sidebar with navigation.

    Returns:
        Tuple of (release, job_name, selected_page)
    """
    st.sidebar.header("📁 Data Selection")

    # Release selection
    releases = discover_releases(config.results_base_path)
    if not releases:
        st.error(f"No releases found in {config.results_base_path}")
        st.info("Ensure GCS mounts are accessible and pipeline results exist.")
        st.stop()

    # Initialize session state
    if 'selected_release' not in st.session_state:
        st.session_state.selected_release = releases[0]
    if 'selected_job' not in st.session_state:
        st.session_state.selected_job = releases[0]
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "Release Overview"

    # Release selector
    selected_release = st.sidebar.selectbox(
        "Select Release",
        releases,
        index=releases.index(st.session_state.selected_release) if st.session_state.selected_release in releases else 0
    )
    st.session_state.selected_release = selected_release

    # Job selection (debug mode only)
    if config.debug_mode:
        jobs = discover_jobs(selected_release, config.results_base_path)

        if len(jobs) > 1:
            selected_job = st.sidebar.selectbox(
                "Select Job",
                jobs,
                index=jobs.index(st.session_state.selected_job) if st.session_state.selected_job in jobs else 0,
                help="Choose pipeline run to view"
            )
        else:
            selected_job = jobs[0] if jobs else selected_release

        st.session_state.selected_job = selected_job
        st.sidebar.info("🔧 **Debug Mode Active**")

        # Cache clear button
        with st.sidebar.expander("🛠️ Debug Tools", expanded=False):
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
    else:
        # Production mode: use release name as job name
        selected_job = selected_release
        st.session_state.selected_job = selected_job

    # Check data availability
    data_available = check_data_availability(selected_release, selected_job, config.results_base_path)

    # Page navigation
    st.sidebar.header("📊 Navigation")
    page_options = ["Release Overview"]

    # Add pages based on data availability
    if data_available['locus_reports_nba'] or data_available['locus_reports_imputed']:
        page_options.append("Locus Reports")
    if data_available['probe_validation']:
        page_options.append("Probe Validation")

    # Ensure selected page is valid
    if st.session_state.selected_page not in page_options:
        st.session_state.selected_page = "Release Overview"

    selected_page = st.sidebar.radio(
        "Select Page",
        page_options,
        index=page_options.index(st.session_state.selected_page) if st.session_state.selected_page in page_options else 0,
        key="page_selector"
    )

    # Update session state if changed
    if selected_page != st.session_state.selected_page:
        st.session_state.selected_page = selected_page
        st.rerun()

    # Show info for unavailable pages
    if not data_available['locus_reports_nba'] and not data_available['locus_reports_imputed']:
        st.sidebar.info("📊 **Locus Reports**\nRequires locus report data")
    if not data_available['probe_validation']:
        st.sidebar.info("🔬 **Probe Validation**\nRequires probe selection data")

    return selected_release, selected_job, selected_page


def render_main_content(release: str, job_name: str, page: str, config: FrontendConfig):
    """Render the main content area based on selected page."""

    # Page header
    st.title("🧬 Carriers Pipeline Results Viewer")
    st.markdown(f"**Release:** {release} | **Job:** {job_name}")

    # Lazy load pages
    if page == "Release Overview":
        from frontend.page_modules.overview import render_overview
        render_overview(release, job_name, config)

    elif page == "Locus Reports":
        from frontend.page_modules.locus_reports import render_locus_reports
        render_locus_reports(release, job_name, config)

    elif page == "Probe Validation":
        from frontend.page_modules.probe_validation import render_probe_validation
        render_probe_validation(release, job_name, config)


def main():
    """Main application entry point."""
    try:
        # Setup
        setup_page_config()
        config = FrontendConfig.create()

        # Sidebar navigation
        release, job_name, selected_page = setup_sidebar(config)

        # Main content
        render_main_content(release, job_name, selected_page, config)

        # Footer
        if config.debug_mode:
            st.caption(f"{selected_page} | Debug Mode | {release} | {job_name}")
        else:
            st.caption(f"{selected_page} | Precision Medicine Carriers Pipeline")

    except Exception as e:
        st.error("Application Error")
        st.error(f"Failed to initialize: {e}")
        st.info("Try refreshing the page or check GCS mount accessibility")

        if st.button("Retry"):
            st.rerun()


if __name__ == "__main__":
    main()
