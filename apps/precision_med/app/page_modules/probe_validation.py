"""
Probe validation page - displays NBA probe quality metrics.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any
from app.config import FrontendConfig
from app.utils.data_loaders import load_probe_validation


def render_probe_validation(release: str, job_name: str, config: FrontendConfig):
    """
    Render the probe validation page.

    Args:
        release: Release identifier
        job_name: Job name
        config: Frontend configuration
    """
    st.header("🔬 Probe Validation Analysis")
    st.markdown("Quality assessment of NBA probes against WGS ground truth data.")

    # Load probe validation data
    probe_data = load_probe_validation(release, job_name, config.results_base_path)

    if not probe_data:
        st.info(
            "**No probe validation data available.**\n\n"
            "Probe validation analysis requires:\n"
            "- Both NBA and WGS data types\n"
            "- Multiple NBA probes per genomic position\n"
            "- Overlapping samples between datasets\n\n"
            "Run pipeline without `--skip-probe-selection` to generate this data."
        )
        return

    # Render sections
    render_validation_summary(probe_data)
    render_probe_comparisons(probe_data)
    render_recommendations(probe_data)


def render_validation_summary(data: Dict[str, Any]):
    """Render validation summary metrics."""
    st.subheader("📊 Validation Summary")

    summary = data.get('summary', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_mutations = summary.get('total_mutations_analyzed', 0)
        st.metric("Mutations Analyzed", f"{total_mutations:,}")

    with col2:
        multiple_probes = summary.get('mutations_with_multiple_probes', 0)
        st.metric("Multiple Probes", f"{multiple_probes:,}")

    with col3:
        probe_comparisons = summary.get('total_probe_comparisons', 0)
        st.metric("Probe Comparisons", f"{probe_comparisons:,}")

    with col4:
        samples_compared = summary.get('samples_compared', 0)
        st.metric("Samples Compared", f"{samples_compared:,}")


def render_probe_comparisons(data: Dict[str, Any]):
    """Render probe comparison table."""
    st.subheader("🔍 Probe Comparisons")

    probe_comparisons = data.get('probe_comparisons', [])

    if not probe_comparisons:
        st.info("No probe comparisons available")
        return

    # Create table data
    comparison_rows = []
    for comp in probe_comparisons:
        mutation = comp.get('mutation', comp.get('snp_list_id', 'N/A'))
        probes = comp.get('probes', [])
        consensus = comp.get('consensus', {})
        recommended_probe = consensus.get('recommended_probe', '')

        # Add row for each probe
        for probe in probes:
            variant_id = probe.get('variant_id', 'N/A')

            # Extract metrics from nested structures
            concordance_metrics = probe.get('concordance_metrics', {})
            diagnostic_metrics = probe.get('diagnostic_metrics', {})

            is_selected = (variant_id == recommended_probe)

            comparison_rows.append({
                'Mutation': mutation,
                'Probe/Variant ID': variant_id,
                'Concordance': f"{concordance_metrics.get('overall_concordance', 0):.3f}",
                'Sensitivity': f"{diagnostic_metrics.get('sensitivity', 0):.3f}",
                'Specificity': f"{diagnostic_metrics.get('specificity', 0):.3f}",
                'Quality Score': f"{concordance_metrics.get('quality_score', 0):.3f}",
                'Selected': '✅' if is_selected else ''
            })

    if comparison_rows:
        df = pd.DataFrame(comparison_rows)

        # Add search filter
        search = st.text_input("🔍 Search by Mutation or Probe/Variant ID:", "")
        if search:
            df = df[
                df['Mutation'].str.contains(search, case=False, na=False) |
                df['Probe/Variant ID'].str.contains(search, case=False, na=False)
            ]

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(df):,} probe comparisons")


def render_recommendations(data: Dict[str, Any]):
    """Render probe selection recommendations and methodology."""

    # Methodology section
    methodology = data.get('methodology', {})
    if methodology:
        with st.expander("📋 Selection Methodology", expanded=False):
            st.markdown("**Probe Selection Approach:**")
            st.json(methodology)

    # Methodology comparison
    methodology_comparison = data.get('methodology_comparison', {})
    if methodology_comparison:
        with st.expander("🔬 Methodology Comparison", expanded=True):
            st.markdown("**Comparison between diagnostic and concordance-based selection:**")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Total Mutations",
                    methodology_comparison.get('total_mutations', 0)
                )
                st.metric(
                    "Both Methods Agree",
                    methodology_comparison.get('both_methods_agree', 0)
                )

            with col2:
                agreement_rate = methodology_comparison.get('agreement_rate', 0)
                st.metric(
                    "Agreement Rate",
                    f"{agreement_rate:.1%}"
                )
                st.metric(
                    "Disagreements",
                    methodology_comparison.get('disagreements', 0)
                )

            # Show disagreement details if any
            disagreement_details = methodology_comparison.get('disagreement_details', [])
            if disagreement_details:
                st.markdown("**Disagreements:**")
                disagreement_df = pd.DataFrame(disagreement_details)
                st.dataframe(disagreement_df, use_container_width=True, hide_index=True)
