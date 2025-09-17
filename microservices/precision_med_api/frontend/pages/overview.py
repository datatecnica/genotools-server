"""
Overview page implementation using builder pattern.
"""

import streamlit as st
from typing import List, Tuple
from frontend.config import FrontendConfig
from frontend.utils.data_facade import DataFacade
from frontend.utils.ui_components import UIComponentFactory, render_expandable_section
from frontend.models.frontend_models import OverviewData


class OverviewBuilder:
    """Builder for constructing overview section components."""

    def __init__(self, config: FrontendConfig):
        """
        Initialize builder with configuration.

        Args:
            config: Frontend configuration
        """
        self.config = config
        self.data_facade = DataFacade(config)
        self.ui_factory = UIComponentFactory()
        self.components = []

    def add_metrics(self, release: str, job_name: str) -> 'OverviewBuilder':
        """
        Add metrics component to overview.

        Args:
            release: Release identifier
            job_name: Job name

        Returns:
            OverviewBuilder: Self for chaining
        """
        try:
            overview_data = self.data_facade.get_overview_data(release, job_name)
            metrics_data = {
                'release': release,
                'total_variants': overview_data.total_variants,
                'success': overview_data.pipeline_success
            }
            self.components.append(('metrics', metrics_data))
        except Exception as e:
            st.error(f"Error loading metrics: {e}")

        return self


    def add_sample_breakdown(self, release: str, job_name: str) -> 'OverviewBuilder':
        """
        Add data type breakdown component to overview.

        Args:
            release: Release identifier
            job_name: Job name

        Returns:
            OverviewBuilder: Self for chaining
        """
        try:
            overview_data = self.data_facade.get_overview_data(release, job_name)
            breakdown_data = {
                'sample_counts': overview_data.sample_counts,
                'variant_counts': overview_data.variants_by_data_type
            }
            self.components.append(('sample_breakdown', breakdown_data))
        except Exception as e:
            st.error(f"Error loading data breakdown: {e}")

        return self

    def add_pipeline_summary(self, release: str, job_name: str) -> 'OverviewBuilder':
        """
        Add pipeline summary component to overview.

        Args:
            release: Release identifier
            job_name: Job name

        Returns:
            OverviewBuilder: Self for chaining
        """
        try:
            overview_data = self.data_facade.get_overview_data(release, job_name)
            self.components.append(('pipeline_summary', overview_data))
        except Exception as e:
            st.error(f"Error loading pipeline summary: {e}")

        return self

    def build(self) -> None:
        """Render all added components in order."""
        for component_type, data in self.components:
            self._render_component(component_type, data)

    def _render_component(self, component_type: str, data) -> None:
        """
        Render a single component.

        Args:
            component_type: Type of component to render
            data: Data for the component
        """
        try:
            if component_type == 'pipeline_summary':
                self._render_pipeline_summary_section(data)
            else:
                renderer = self.ui_factory.get_renderer(component_type)
                renderer.render(data)
        except Exception as e:
            st.error(f"Error rendering {component_type}: {e}")

    def _render_pipeline_summary_section(self, overview_data: OverviewData) -> None:
        """
        Render the pipeline summary in an expandable section.

        Args:
            overview_data: Overview data containing pipeline results
        """
        def render_summary_content():
            # Pipeline execution details
            st.subheader("Execution Details")
            pipeline_renderer = self.ui_factory.get_renderer('pipeline_details')
            pipeline_renderer.render(overview_data)

            # File information
            file_renderer = self.ui_factory.get_renderer('file_info')
            file_renderer.render(overview_data.file_info)

        render_expandable_section(
            "📋 Pipeline Summary",
            render_summary_content,
            expanded=False
        )


def render_overview(release: str, job_name: str, config: FrontendConfig) -> None:
    """
    Render the complete overview section.

    Args:
        release: Release identifier
        job_name: Job name
        config: Frontend configuration
    """
    # Show debug info for non-standard jobs
    if config.debug_mode and job_name != release:
        st.info(f"📊 **Viewing Job Results: {job_name}**")

    # Build and render overview components
    builder = OverviewBuilder(config)
    builder.add_metrics(release, job_name)\
           .add_sample_breakdown(release, job_name)\
           .add_pipeline_summary(release, job_name)\
           .build()


def validate_overview_data(release: str, job_name: str, config: FrontendConfig) -> Tuple[bool, List[str]]:
    """
    Validate that overview data can be loaded.

    Args:
        release: Release identifier
        job_name: Job name
        config: Frontend configuration

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    try:
        data_facade = DataFacade(config)
        available_data_types = data_facade.get_available_data_types(release, job_name)
        if not available_data_types:
            errors.append("No data files found")
    except Exception as e:
        errors.append(f"Error validating data: {e}")

    return len(errors) == 0, errors


def render_overview_with_validation(release: str, job_name: str, config: FrontendConfig) -> None:
    """
    Render overview with data validation.

    Args:
        release: Release identifier
        job_name: Job name
        config: Frontend configuration
    """
    is_valid, errors = validate_overview_data(release, job_name, config)

    if not is_valid:
        st.error("No pipeline results found")
        for error in errors:
            st.error(f"• {error}")
        return

    # Render overview
    render_overview(release, job_name, config)