"""
Probe validation analysis page for visualizing probe quality metrics.

Displays probe validation results including diagnostic metrics, concordance analysis,
and methodology comparisons for NBA probe quality assessment.
"""

import streamlit as st
import sys
import os
from typing import Dict, Any
import pandas as pd

# Add parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontend.config import FrontendConfig
from frontend.utils.data_facade import DataFacade
from frontend.utils.ui_components import UIComponentFactory


def render_probe_validation_page(release: str, job_name: str, config: FrontendConfig) -> None:
    """
    Render the probe validation analysis page.

    Args:
        release: Selected release
        job_name: Selected job name
        config: Frontend configuration
    """
    try:
        st.header("🔬 Probe Validation Analysis")
        st.markdown("Quality assessment of NBA probes against WGS ground truth data.")

        # Initialize data facade
        data_facade = DataFacade(config)

        # Load probe validation data
        probe_data = data_facade.get_probe_validation_data(release, job_name)

        if not probe_data:
            st.info(
                "**No probe validation data available for this release.**\n\n"
                "Probe validation analysis requires:\n"
                "- Both NBA and WGS data types\n"
                "- Multiple NBA probes per genomic position\n"
                "- Overlapping samples between datasets\n\n"
                "Run the pipeline with `--enable-probe-selection` to generate this data."
            )
            return

        # Debug info
        st.write(f"**Debug:** Loaded probe data with {len(probe_data.get('probe_comparisons', []))} mutations")

        # Render probe validation sections
        render_probe_validation_summary(probe_data)
        render_probe_validation_analysis(probe_data)
        render_probe_validation_visualizations(probe_data)
        render_methodology_comparison(probe_data)
        render_probe_recommendations(probe_data)

    except Exception as e:
        st.error(f"Error rendering probe validation page: {e}")
        import traceback
        st.code(traceback.format_exc())


def render_probe_validation_summary(probe_data: Dict[str, Any]) -> None:
    """
    Render probe validation summary metrics.

    Args:
        probe_data: Probe validation report data
    """
    st.subheader("📊 Validation Summary")

    # Use UI component for summary rendering
    summary_renderer = UIComponentFactory.get_renderer('probe_validation_summary')
    summary_renderer.render(probe_data)

    # Additional context information
    with st.expander("ℹ️ About Probe Validation", expanded=False):
        st.markdown("""
        **Probe Validation Analysis** assesses the quality of NBA (array-based) probes by comparing
        their results against WGS (whole genome sequencing) ground truth data.

        **Key Metrics:**
        - **Sensitivity**: Ability to detect true carriers (avoid false negatives)
        - **Specificity**: Ability to avoid false alarms (avoid false positives)
        - **Overall Concordance**: Genotype-level agreement between NBA and WGS
        - **Quality Score**: Weighted combination of concordance and carrier detection

        **Analysis Methods:**
        - **Diagnostic**: Traditional medical test evaluation (carrier vs non-carrier)
        - **Concordance**: Detailed genotype transition analysis (0/0, 0/1, 1/1)

        Only mutations with multiple NBA probes are analyzed to enable quality comparisons.
        """)


def render_probe_validation_analysis(probe_data: Dict[str, Any]) -> None:
    """
    Render detailed probe validation analysis table.

    Args:
        probe_data: Probe validation report data
    """
    # Use UI component for table rendering
    table_renderer = UIComponentFactory.get_renderer('probe_validation_table')
    table_renderer.render(probe_data)


def render_probe_validation_visualizations(probe_data: Dict[str, Any]) -> None:
    """
    Render probe validation visualizations.

    Args:
        probe_data: Probe validation report data
    """
    # Use UI component for visualization rendering
    viz_renderer = UIComponentFactory.get_renderer('probe_validation_visualization')
    viz_renderer.render(probe_data)


def render_methodology_comparison(probe_data: Dict[str, Any]) -> None:
    """
    Render methodology comparison analysis.

    Args:
        probe_data: Probe validation report data
    """
    methodology = probe_data.get('methodology_comparison', {})
    if not methodology:
        return

    st.subheader("🔄 Methodology Comparison")

    col1, col2 = st.columns(2)

    with col1:
        total_agreements = methodology.get('total_agreements', 0)
        total_disagreements = methodology.get('total_disagreements', 0)
        total_comparisons = total_agreements + total_disagreements

        if total_comparisons > 0:
            agreement_rate = total_agreements / total_comparisons
            st.metric("Agreement Rate", f"{agreement_rate:.1%}")

            # Agreement breakdown chart
            agreement_data = {
                'Method Agreement': [total_agreements, total_disagreements],
                'Count': [total_agreements, total_disagreements]
            }
            agreement_df = pd.DataFrame({
                'Outcome': ['Agreement', 'Disagreement'],
                'Count': [total_agreements, total_disagreements]
            })

            st.bar_chart(agreement_df.set_index('Outcome'))

    with col2:
        st.write("**Method Comparison Summary:**")
        st.write(f"- Total mutations compared: {total_comparisons}")
        st.write(f"- Methods agree: {total_agreements}")
        st.write(f"- Methods disagree: {total_disagreements}")

        if total_disagreements > 0:
            st.warning(f"Review {total_disagreements} disagreements for manual curation")

    # Show disagreement details if any exist
    disagreements = methodology.get('disagreement_analysis', [])
    if disagreements:
        with st.expander(f"🔍 View {len(disagreements)} Method Disagreements", expanded=False):
            disagreement_df = pd.DataFrame(disagreements)
            st.dataframe(disagreement_df, width='stretch', hide_index=True)

            st.info(
                "**Disagreement Analysis:** When diagnostic and concordance methods recommend "
                "different probes, manual review is recommended to determine the best choice "
                "based on study-specific requirements."
            )


def render_probe_recommendations(probe_data: Dict[str, Any]) -> None:
    """
    Render probe recommendations summary.

    Args:
        probe_data: Probe validation report data
    """
    probe_comparisons = probe_data.get('probe_comparisons', [])
    if not probe_comparisons:
        return

    st.subheader("🎯 Probe Recommendations")

    recommendations = []
    for mutation_analysis in probe_comparisons:
        mutation = mutation_analysis.get('mutation', 'Unknown')
        consensus = mutation_analysis.get('consensus', {})

        if consensus.get('both_methods_agree', False):
            recommended_probe = consensus.get('recommended_probe', 'N/A')
            confidence = consensus.get('combined_confidence', 0)
            recommendations.append({
                'Mutation': mutation,
                'Recommended Probe': recommended_probe,
                'Consensus': 'Yes',
                'Confidence': f"{confidence:.2f}"
            })
        else:
            diagnostic_choice = consensus.get('diagnostic_choice', 'N/A')
            concordance_choice = consensus.get('concordance_choice', 'N/A')
            recommendations.append({
                'Mutation': mutation,
                'Recommended Probe': f"Manual Review Required",
                'Consensus': 'No',
                'Diagnostic Choice': diagnostic_choice,
                'Concordance Choice': concordance_choice
            })

    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        st.dataframe(rec_df, width='stretch', hide_index=True)

        # Summary statistics
        consensus_count = sum(1 for r in recommendations if r['Consensus'] == 'Yes')
        manual_review_count = len(recommendations) - consensus_count

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Consensus Recommendations", consensus_count)
        with col2:
            st.metric("Manual Review Required", manual_review_count)