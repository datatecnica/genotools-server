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




class ProbeValidationSummaryRenderer(ComponentRenderer):
    """Strategy for rendering probe validation summary metrics."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render probe validation summary metrics.

        Args:
            data: Probe validation report data
        """
        if not data:
            st.warning("No probe validation data available")
            return

        summary = data.get('summary', {})
        methodology = data.get('methodology_comparison', {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_mutations = summary.get('total_mutations_analyzed', 0)
            st.metric("Mutations Analyzed", total_mutations)

        with col2:
            multiple_probes = summary.get('mutations_with_multiple_probes', 0)
            st.metric("Multiple Probe Mutations", multiple_probes)

        with col3:
            total_comparisons = summary.get('total_probe_comparisons', 0)
            st.metric("Probe Comparisons", total_comparisons)

        with col4:
            agreement_rate = methodology.get('agreement_rate', 0.0)
            st.metric("Method Agreement", f"{agreement_rate:.1%}")


class ProbeValidationTableRenderer(ComponentRenderer):
    """Strategy for rendering probe validation analysis table."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render probe validation analysis table.

        Args:
            data: Probe validation report data
        """
        if not data:
            st.warning("No probe validation data available")
            return

        probe_comparisons = data.get('probe_comparisons', [])
        if not probe_comparisons:
            st.info("No probe comparisons available")
            return

        # Build table data
        table_rows = []
        for mutation_analysis in probe_comparisons:
            mutation = mutation_analysis.get('mutation', 'Unknown')
            probes = mutation_analysis.get('probes', [])

            for probe in probes:
                diagnostic = probe.get('diagnostic_metrics', {})
                concordance = probe.get('concordance_metrics', {})

                table_rows.append({
                    "Mutation": mutation,
                    "Probe ID": probe.get('variant_id', 'Unknown'),
                    "Probe Type": probe.get('probe_type', 'Unknown'),
                    "Sensitivity": f"{diagnostic.get('sensitivity', 0):.3f}",
                    "Specificity": f"{diagnostic.get('specificity', 0):.3f}",
                    "Overall Concordance": f"{concordance.get('overall_concordance', 0):.3f}",
                    "Carrier Sensitivity": f"{concordance.get('carrier_sensitivity', 0):.3f}",
                    "Quality Score": f"{concordance.get('quality_score', 0):.3f}"
                })

        if table_rows:
            df = pd.DataFrame(table_rows)

            # Add filtering
            st.subheader("Probe Performance Analysis")

            col1, col2 = st.columns(2)
            with col1:
                mutation_filter = st.selectbox(
                    "Filter by Mutation",
                    ["All"] + list(df["Mutation"].unique()),
                    key="probe_mutation_filter"
                )
            with col2:
                probe_type_filter = st.selectbox(
                    "Filter by Probe Type",
                    ["All"] + list(df["Probe Type"].unique()),
                    key="probe_type_filter"
                )

            # Apply filters
            filtered_df = df.copy()
            if mutation_filter != "All":
                filtered_df = filtered_df[filtered_df["Mutation"] == mutation_filter]
            if probe_type_filter != "All":
                filtered_df = filtered_df[filtered_df["Probe Type"] == probe_type_filter]

            st.dataframe(filtered_df, width='stretch', hide_index=True)
        else:
            st.warning("No probe data available for table")


class ProbeValidationVisualizationRenderer(ComponentRenderer):
    """Strategy for rendering probe validation visualizations."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render probe validation visualizations.

        Args:
            data: Probe validation report data
        """
        if not data:
            st.warning("No probe validation data available")
            return

        probe_comparisons = data.get('probe_comparisons', [])
        methodology = data.get('methodology_comparison', {})

        # Performance scatter plot
        st.subheader("Probe Performance Overview")

        if probe_comparisons:
            plot_data = []
            for mutation_analysis in probe_comparisons:
                mutation = mutation_analysis.get('mutation', 'Unknown')
                probes = mutation_analysis.get('probes', [])

                for probe in probes:
                    diagnostic = probe.get('diagnostic_metrics', {})
                    concordance = probe.get('concordance_metrics', {})

                    plot_data.append({
                        'Mutation': mutation,
                        'Probe ID': probe.get('variant_id', 'Unknown'),
                        'Sensitivity': diagnostic.get('sensitivity', 0),
                        'Specificity': diagnostic.get('specificity', 0),
                        'Overall Concordance': concordance.get('overall_concordance', 0),
                        'Quality Score': concordance.get('quality_score', 0)
                    })

            if plot_data:
                plot_df = pd.DataFrame(plot_data)

                # Sensitivity vs Specificity scatter plot
                st.subheader("Performance Scatter Plot")

                # Show summary table instead of complex scatter plot for now
                summary_cols = ['Mutation', 'Probe ID', 'Sensitivity', 'Specificity', 'Quality Score']
                display_df = plot_df[summary_cols].copy()
                display_df['Sensitivity'] = display_df['Sensitivity'].apply(lambda x: f"{x:.3f}")
                display_df['Specificity'] = display_df['Specificity'].apply(lambda x: f"{x:.3f}")
                display_df['Quality Score'] = display_df['Quality Score'].apply(lambda x: f"{x:.3f}")

                st.dataframe(display_df, width='stretch', hide_index=True)

        # Method disagreement analysis
        if methodology.get('disagreement_analysis'):
            st.subheader("Method Disagreement Analysis")

            disagreements = methodology['disagreement_analysis']
            if disagreements:
                disagreement_df = pd.DataFrame(disagreements)

                st.write(f"**Total Disagreements:** {len(disagreements)}")
                st.dataframe(disagreement_df, width='stretch', hide_index=True)
            else:
                st.success("No disagreements found between diagnostic and concordance methods")


class GenotypeMatrixRenderer(ComponentRenderer):
    """Strategy for rendering genotype matrices."""

    def render(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Render genotype matrices for each data type.

        Args:
            data: Dictionary of DataFrames with genotype data per data type
        """
        if 'error' in data:
            st.error(f"Error loading genotype data: {data['error']}")
            return

        if not data:
            st.warning("No genotype data available")
            return

        for data_type, genotype_df in data.items():
            if genotype_df.empty:
                continue

            st.subheader(f"{data_type} Genotype Matrix")

            # Add info about data size
            total_variants, total_samples = genotype_df.shape
            st.caption(f"Showing {total_variants} variants × {total_samples} samples (limited for display)")

            # Create styled dataframe
            styled_df = genotype_df.copy()

            # Color-code genotypes: 0=gray, 1=yellow, 2=red
            def style_genotypes(val):
                if pd.isna(val):
                    return 'background-color: lightgray'
                elif val == 0:
                    return 'background-color: #f0f0f0'
                elif val == 1:
                    return 'background-color: #fff2cc'
                elif val == 2:
                    return 'background-color: #ffcccc'
                else:
                    return ''

            # Apply styling and display
            try:
                styled = styled_df.style.map(style_genotypes)
                st.dataframe(styled, width='stretch')
            except:
                # Fallback to regular dataframe if styling fails
                st.dataframe(styled_df, width='stretch')


class CarrierSummaryRenderer(ComponentRenderer):
    """Strategy for rendering carrier summary statistics."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render carrier summary statistics.

        Args:
            data: Dictionary with carrier analysis results
        """
        if 'error' in data:
            st.error(f"Error in carrier analysis: {data['error']}")
            return

        # Overall metrics
        st.subheader("Carrier Summary")

        by_data_type = data.get('by_data_type', {})
        summary = data.get('summary', {})

        if not by_data_type:
            st.warning("No carrier data available")
            return

        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_carriers = summary.get('total_unique_carriers', 0)
            st.metric("Unique Carriers", f"{total_carriers:,}")

        with col2:
            # Calculate overall frequency from first data type
            first_dt = list(by_data_type.keys())[0]
            overall_freq = by_data_type[first_dt].get('carrier_frequency', 0)
            st.metric("Carrier Frequency", f"{overall_freq:.2%}")

        with col3:
            data_types_count = len(by_data_type)
            st.metric("Data Types", data_types_count)

        with col4:
            overlap_count = summary.get('carriers_in_multiple_types', 0)
            st.metric("Multi-Type Carriers", f"{overlap_count:,}")

        # Per data type breakdown
        st.subheader("Breakdown by Data Type")

        breakdown_rows = []
        for data_type, stats in by_data_type.items():
            breakdown_rows.append({
                "Data Type": data_type,
                "Carriers": f"{stats['carrier_count']:,}",
                "Total Samples": f"{stats['total_samples']:,}",
                "Frequency": f"{stats['carrier_frequency']:.2%}",
                "Variants with Carriers": f"{stats['variants_with_carriers']:,}",
                "Total Variants": f"{stats['total_variants']:,}"
            })

        if breakdown_rows:
            breakdown_df = pd.DataFrame(breakdown_rows)
            st.dataframe(breakdown_df, width='stretch', hide_index=True)


class GeneFilterRenderer(ComponentRenderer):
    """Strategy for rendering gene/locus selection interface."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render gene selection interface.

        Args:
            data: Dictionary with available genes and current selection
        """
        available_genes = data.get('available_genes', [])
        current_selection = data.get('current_selection', [])

        if not available_genes:
            st.warning("No genes available for filtering")
            return

        # Gene selection
        selected_genes = st.multiselect(
            "Select Genes/Loci to Analyze",
            options=available_genes,
            default=current_selection,
            help="Leave empty to analyze all genes"
        )

        # Display selection info
        if selected_genes:
            st.caption(f"Selected {len(selected_genes)} of {len(available_genes)} available genes")
        else:
            st.caption(f"All {len(available_genes)} genes will be analyzed")

        return selected_genes


class DataTypeFilterRenderer(ComponentRenderer):
    """Strategy for rendering data type selection interface."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render data type selection interface.

        Args:
            data: Dictionary with available data types and current selection
        """
        available_types = data.get('available_types', ['NBA', 'WGS', 'IMPUTED'])
        current_selection = data.get('current_selection', available_types)

        selected_types = st.multiselect(
            "Select Data Types",
            options=available_types,
            default=current_selection,
            help="Choose which data types to include in analysis"
        )

        if not selected_types:
            st.warning("Please select at least one data type")

        return selected_types


class CarrierFilterRenderer(ComponentRenderer):
    """Strategy for rendering carrier-only filter interface."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render carrier-only filter interface.

        Args:
            data: Dictionary with current filter state
        """
        current_state = data.get('carrier_only', False)

        carrier_only = st.checkbox(
            "Show Carriers Only",
            value=current_state,
            help="Filter to only show samples and variants with carrier genotypes (>0)"
        )

        if carrier_only:
            st.caption("Filtering to carriers only (genotype > 0)")
        else:
            st.caption("Showing all samples and variants")

        return carrier_only


class GenotypeControlsRenderer(ComponentRenderer):
    """Strategy for rendering genotype analysis controls."""

    def render(self, data: Dict[str, Any]) -> None:
        """
        Render complete genotype analysis controls.

        Args:
            data: Dictionary with control options and current state
        """
        st.subheader("Analysis Options")

        # Create columns for controls
        col1, col2 = st.columns(2)

        with col1:
            # Data type selection
            data_type_renderer = DataTypeFilterRenderer()
            selected_types = data_type_renderer.render({
                'available_types': data.get('available_types', ['NBA', 'WGS', 'IMPUTED']),
                'current_selection': data.get('selected_types', ['NBA', 'WGS', 'IMPUTED'])
            })

            # Carrier filter
            carrier_filter_renderer = CarrierFilterRenderer()
            carrier_only = carrier_filter_renderer.render({
                'carrier_only': data.get('carrier_only', False)
            })

        with col2:
            # Gene selection
            gene_filter_renderer = GeneFilterRenderer()
            selected_genes = gene_filter_renderer.render({
                'available_genes': data.get('available_genes', []),
                'current_selection': data.get('selected_genes', [])
            })

        return {
            'selected_types': selected_types,
            'selected_genes': selected_genes,
            'carrier_only': carrier_only
        }


class UIComponentFactory:
    """Factory for creating UI component renderers."""

    _renderers = {
        'metrics': MetricsRenderer,
        'sample_breakdown': SampleBreakdownRenderer,
        'file_info': FileInfoRenderer,
        'pipeline_details': PipelineDetailsRenderer,
        'probe_validation_summary': ProbeValidationSummaryRenderer,
        'probe_validation_table': ProbeValidationTableRenderer,
        'probe_validation_visualization': ProbeValidationVisualizationRenderer,
        'genotype_matrix': GenotypeMatrixRenderer,
        'carrier_summary': CarrierSummaryRenderer,
        'gene_filter': GeneFilterRenderer,
        'data_type_filter': DataTypeFilterRenderer,
        'carrier_filter': CarrierFilterRenderer,
        'genotype_controls': GenotypeControlsRenderer
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