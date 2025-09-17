"""
UI components using strategy pattern for flexible rendering.
"""

import streamlit as st
import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from frontend.models.frontend_models import OverviewData


class ComponentRenderer(ABC):
    """Abstract strategy for rendering UI components."""

    @abstractmethod
    def render(self, data: Any) -> None:
        """Render the component with given data."""
        pass


class MetricsRenderer(ComponentRenderer):
    """Strategy for rendering metrics in a 3-column layout."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render metrics in 3 columns.

        Args:
            data: Dictionary with keys: release, total_variants, success
        """
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Release", data.get('release', 'N/A'))

        with col2:
            total_variants = data.get('total_variants', 0)
            st.metric("Total Variants", f"{total_variants:,}")

        with col3:
            success = data.get('success', False)
            status = "✅ Success" if success else "❌ Failed"
            st.metric("Pipeline Status", status)



class SampleBreakdownRenderer(ComponentRenderer):
    """Strategy for rendering data type breakdown table."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render data type breakdown as a clean table with variants and samples.

        Args:
            data: Dictionary with sample_counts and variant_counts
        """
        sample_counts = data.get('sample_counts', {})
        variant_counts = data.get('variant_counts', {})

        breakdown_data = []
        data_types = ["NBA", "WGS", "IMPUTED"]

        for data_type in data_types:
            if data_type in sample_counts:
                breakdown_data.append({
                    "Data Type": data_type,
                    "Variants": f"{variant_counts.get(data_type, 0):,}",
                    "Samples": f"{sample_counts[data_type]:,}"
                })

        if breakdown_data:
            df = pd.DataFrame(breakdown_data)
            st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.warning("No data available")


class FileInfoRenderer(ComponentRenderer):
    """Strategy for rendering file information table."""

    def render(self, data: Dict[str, Dict[str, Any]]) -> None:
        """
        Render file information as a table.

        Args:
            data: Nested dict with data_type -> file_type -> file_info structure
        """
        st.subheader("Available Files")

        file_rows = []
        for data_type, files in data.items():
            for file_desc, info in files.items():
                # Use the formatted size display if available, otherwise fall back to MB
                size_display = info.get('size_display', f"{info['size_mb']} MB")

                file_rows.append({
                    "Data Type": data_type,
                    "File Type": file_desc,
                    "Size": size_display,
                    "Path": info['path'].split('/')[-1]  # Just filename
                })

        if file_rows:
            files_df = pd.DataFrame(file_rows)
            st.dataframe(files_df, width='stretch', hide_index=True)
        else:
            st.warning("No files found")


class PipelineDetailsRenderer(ComponentRenderer):
    """Strategy for rendering pipeline execution details."""

    def render(self, data: OverviewData) -> None:
        """
        Render pipeline execution details.

        Args:
            data: OverviewData containing pipeline results
        """
        if not data.pipeline_results:
            st.info("No pipeline execution details available")
            return

        col1, col2 = st.columns(2)

        with col1:
            if data.start_time:
                st.text(f"Start Time: {data.start_time}")

            if data.execution_time:
                st.text(f"Execution Time: {data.execution_time:.1f} seconds")

        with col2:
            if data.error_count > 0:
                st.error(f"Errors: {data.error_count} found")
            else:
                st.success("No errors reported")




class UIComponentFactory:
    """Factory for creating UI component renderers."""

    _renderers = {
        'metrics': MetricsRenderer,
        'sample_breakdown': SampleBreakdownRenderer,
        'file_info': FileInfoRenderer,
        'pipeline_details': PipelineDetailsRenderer
    }

    @classmethod
    def get_renderer(cls, component_type: str) -> ComponentRenderer:
        """
        Get a component renderer by type.

        Args:
            component_type: Type of renderer to create

        Returns:
            ComponentRenderer: Instance of the requested renderer

        Raises:
            ValueError: If component_type is not recognized
        """
        if component_type not in cls._renderers:
            raise ValueError(f"Unknown component type: {component_type}")
        return cls._renderers[component_type]()

    @classmethod
    def list_available_renderers(cls) -> List[str]:
        """Get list of available renderer types."""
        return list(cls._renderers.keys())


# Helper functions for common UI patterns
def render_expandable_section(title: str, content_func, expanded: bool = False) -> None:
    """
    Render an expandable section with consistent styling.

    Args:
        title: Section title
        content_func: Function to call for rendering content
        expanded: Whether section should be expanded by default
    """
    with st.expander(title, expanded=expanded):
        content_func()