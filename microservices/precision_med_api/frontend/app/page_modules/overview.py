"""
Release overview page - displays pipeline summary and metrics.
"""

import streamlit as st
from typing import Dict, Any
from app.config import FrontendConfig
from app.utils.data_loaders import load_pipeline_results, get_selected_probe_ids
from app.utils.methods_descriptions import MethodsDescriptions


def render_overview(release: str, job_name: str, config: FrontendConfig):
    """
    Render the release overview page.

    Args:
        release: Release identifier
        job_name: Job name
        config: Frontend configuration
    """
    st.header("📊 Release Overview")

    # Methods section
    with st.expander("📖 Pipeline Methods Overview", expanded=False):
        st.markdown(MethodsDescriptions.get_pipeline_overview_methods())

    # Load pipeline results
    pipeline_data = load_pipeline_results(release, job_name, config.results_base_path)

    if not pipeline_data:
        st.warning("No pipeline results found for this release.")
        st.info("Run the pipeline to generate results:\n```bash\npython run_carriers_pipeline.py --job-name " + job_name + "\n```")
        return

    # Pipeline status
    render_pipeline_status(pipeline_data)

    # Summary metrics
    if 'summary' in pipeline_data:
        render_summary_metrics(pipeline_data['summary'])

    # Data type breakdown
    if 'summary' in pipeline_data and 'by_data_type' in pipeline_data['summary']:
        render_data_type_breakdown(pipeline_data['summary']['by_data_type'], release, job_name, config)

    # Output files
    if 'output_files' in pipeline_data:
        render_output_files(pipeline_data['output_files'])


def render_pipeline_status(data: Dict[str, Any]):
    """Render pipeline execution status."""
    st.subheader("Pipeline Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        status = "✅ Success" if data.get('success', False) else "❌ Failed"
        st.metric("Status", status)

    with col2:
        job_id = data.get('job_id', 'N/A')
        st.metric("Job ID", job_id)

    with col3:
        timestamp = data.get('start_time', 'N/A')
        if timestamp != 'N/A':
            # Show just the date part
            timestamp = timestamp.split('T')[0]
        st.metric("Execution Date", timestamp)


def render_summary_metrics(summary: Dict[str, Any]):
    """Render summary metrics."""
    st.subheader("Summary Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        total_variants = summary.get('total_variants', 0)
        st.metric("Total Variants", f"{total_variants:,}")

    with col2:
        unique_variants = summary.get('unique_variants', 0)
        st.metric("Unique Variants", f"{unique_variants:,}")

    with col3:
        extraction_time = summary.get('extraction_timestamp', 'N/A')
        if extraction_time != 'N/A':
            extraction_time = extraction_time.split('T')[0]
        st.metric("Extraction Date", extraction_time)


def render_data_type_breakdown(by_data_type: Dict[str, Any], release: str, job_name: str, config: FrontendConfig):
    """Render data type breakdown table."""
    st.subheader("Data Type Breakdown")

    # Get probe selection info for NBA
    probe_info = get_selected_probe_ids(release, job_name, config.results_base_path)

    # Create table data
    table_data = []
    for data_type, stats in by_data_type.items():
        if isinstance(stats, dict):
            variants = stats.get('variants', 0)
            samples = stats.get('samples', 0)

            # Calculate unique variants for NBA if probe selection is available
            variant_str = f"{variants:,}"
            if data_type == 'NBA' and probe_info.get('has_probe_selection'):
                # Calculate unique variants: total_probes - rejected_multi_probe_variants
                mutations_with_multiple_probes = probe_info.get('mutations_with_multiple_probes', 0)
                mutations_with_selection = probe_info.get('total_mutations', 0)
                rejected_variants = mutations_with_multiple_probes - mutations_with_selection
                unique_variants = variants - rejected_variants
                variant_str = f"{unique_variants:,}"

            table_data.append({
                'Data Type': data_type,
                'Variants': variant_str,
                'Samples': f"{samples:,}"
            })

    if table_data:
        st.table(table_data)
    else:
        st.info("No data type breakdown available")


def render_output_files(output_files: Dict[str, str]):
    """Render output files section."""
    with st.expander("📁 Output Files", expanded=False):
        # Group by data type
        grouped_files = {}
        for key, path in output_files.items():
            # Extract data type from key
            if '_' in key:
                data_type = key.split('_')[0]
            else:
                data_type = 'Other'

            if data_type not in grouped_files:
                grouped_files[data_type] = []

            # Get filename from path
            filename = path.split('/')[-1]
            grouped_files[data_type].append(filename)

        # Display grouped files
        for data_type, files in sorted(grouped_files.items()):
            st.markdown(f"**{data_type}:**")
            for filename in files:
                st.text(f"  • {filename}")
