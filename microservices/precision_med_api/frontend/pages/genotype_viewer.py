"""
Genotype viewer page for exploring raw genotype data and carrier analysis.
"""

import streamlit as st
import sys
import os
from typing import Dict, Any, List, Optional

# Add parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from frontend.config import FrontendConfig
from frontend.utils.data_loaders import DataLoaderFactory
from frontend.utils.ui_components import UIComponentFactory
from frontend.utils.genotype_analysis import (
    identify_carriers, calculate_carrier_frequencies,
    compare_data_types, generate_carrier_summary,
    create_genotype_matrix_display
)


def render_genotype_viewer_page(release: str, job_name: str, config: FrontendConfig) -> None:
    """
    Render the genotype viewer page with carrier analysis capabilities.

    Args:
        release: Selected release
        job_name: Selected job name
        config: Frontend configuration
    """
    st.header("🧬 Genotype Viewer")
    st.markdown("Explore raw genotype data and identify carriers across data types.")

    # Initialize data loaders
    genotype_loader = DataLoaderFactory.get_loader('genotype_data')

    # Get available genes for filtering
    available_genes = genotype_loader.get_available_genes(release, job_name, config)

    # Render analysis controls
    controls_renderer = UIComponentFactory.get_renderer('genotype_controls')

    # Set up default state in session if not exists
    if 'genotype_controls' not in st.session_state:
        st.session_state.genotype_controls = {
            'selected_types': config.data_types,
            'selected_genes': [],
            'carrier_only': False
        }

    # Render controls
    control_data = {
        'available_types': config.data_types,
        'available_genes': available_genes,
        'selected_types': st.session_state.genotype_controls.get('selected_types', config.data_types),
        'selected_genes': st.session_state.genotype_controls.get('selected_genes', []),
        'carrier_only': st.session_state.genotype_controls.get('carrier_only', False)
    }

    updated_controls = controls_renderer.render(control_data)

    # Update session state
    st.session_state.genotype_controls.update(updated_controls)

    # Add refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        refresh_data = st.button("🔄 Load Data", type="primary")

    # Load and display data if controls are valid
    if updated_controls['selected_types'] and (refresh_data or 'genotype_data' not in st.session_state):
        with st.spinner("Loading genotype data..."):
            # Load genotype data with current filters
            genotype_data = genotype_loader.load(
                release=release,
                job_name=job_name,
                _config=config,
                data_types=updated_controls['selected_types'],
                genes=updated_controls['selected_genes'] if updated_controls['selected_genes'] else None,
                carrier_only=updated_controls['carrier_only']
            )

            st.session_state.genotype_data = genotype_data

    # Display results if data is available
    if 'genotype_data' in st.session_state:
        genotype_data = st.session_state.genotype_data

        if 'error' in genotype_data:
            st.error(f"Error loading data: {genotype_data['error']}")
            return

        # Display data summary
        _render_data_summary(genotype_data)

        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Carrier Analysis", "🧬 Genotype Matrix", "📈 Statistics", "🔍 Comparisons"])

        with tab1:
            _render_carrier_analysis_tab(genotype_data)

        with tab2:
            _render_genotype_matrix_tab(genotype_data)

        with tab3:
            _render_statistics_tab(genotype_data)

        with tab4:
            _render_comparisons_tab(genotype_data)

    else:
        st.info("👆 Configure your analysis options above and click 'Load Data' to begin.")

        # Show example of what's available
        if available_genes:
            st.subheader("Available Genes/Loci")
            st.write(f"**{len(available_genes)} genes available:** {', '.join(available_genes[:10])}" +
                    (f" and {len(available_genes)-10} more..." if len(available_genes) > 10 else ""))


def _render_data_summary(genotype_data: Dict[str, Any]) -> None:
    """Render high-level summary of loaded data."""
    summary = genotype_data.get('summary', {})
    filters = genotype_data.get('filters_applied', {})

    st.subheader("📋 Data Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_variants = summary.get('total_variants', 0)
        st.metric("Total Variants", f"{total_variants:,}")

    with col2:
        total_samples = summary.get('total_samples', 0)
        st.metric("Total Samples", f"{total_samples:,}")

    with col3:
        total_carriers = summary.get('total_carriers', 0)
        st.metric("Total Carriers", f"{total_carriers:,}")

    with col4:
        carrier_freq = summary.get('carrier_frequency', 0)
        st.metric("Carrier Frequency", f"{carrier_freq:.3%}")

    # Show applied filters
    if any(filters.values()):
        st.subheader("🔍 Applied Filters")
        filter_info = []

        if filters.get('data_types'):
            filter_info.append(f"**Data Types:** {', '.join(filters['data_types'])}")

        if filters.get('genes'):
            filter_info.append(f"**Genes:** {', '.join(filters['genes'])}")

        if filters.get('carrier_only'):
            filter_info.append("**Filter:** Carriers only")

        st.markdown(" • ".join(filter_info))


def _render_carrier_analysis_tab(genotype_data: Dict[str, Any]) -> None:
    """Render carrier analysis tab."""
    st.subheader("🔬 Carrier Identification")

    # Perform carrier analysis
    with st.spinner("Analyzing carriers..."):
        carrier_info = identify_carriers(genotype_data)

    # Render carrier summary
    carrier_renderer = UIComponentFactory.get_renderer('carrier_summary')
    carrier_renderer.render(carrier_info)

    # Show carrier overlap between data types if multiple types
    if len(genotype_data.get('data', {})) > 1:
        st.subheader("🔄 Data Type Overlaps")

        overlap_data = carrier_info.get('summary', {}).get('data_type_overlap', {})

        if overlap_data:
            overlap_rows = []
            for comparison, stats in overlap_data.items():
                dt1, dt2 = comparison.split('_vs_')
                overlap_rows.append({
                    "Comparison": f"{dt1} vs {dt2}",
                    "Shared Carriers": f"{stats['overlap_count']:,}",
                    f"{dt1} Only": f"{len(stats['dt1_only']):,}",
                    f"{dt2} Only": f"{len(stats['dt2_only']):,}",
                    "Overlap %": f"{stats['overlap_count'] / max(1, stats['overlap_count'] + len(stats['dt1_only']) + len(stats['dt2_only'])) * 100:.1f}%"
                })

            if overlap_rows:
                import pandas as pd
                overlap_df = pd.DataFrame(overlap_rows)
                st.dataframe(overlap_df, width='stretch', hide_index=True)
        else:
            st.info("No overlaps to display with current data.")


def _render_genotype_matrix_tab(genotype_data: Dict[str, Any]) -> None:
    """Render genotype matrix tab."""
    st.subheader("🧬 Raw Genotype Data")

    st.markdown("""
    **Genotype Legend:**
    - `0` = No pathogenic alleles (homozygous reference)
    - `1` = One pathogenic allele (heterozygous carrier)
    - `2` = Two pathogenic alleles (homozygous carrier)
    """)

    # Create display matrices
    with st.spinner("Preparing genotype matrices..."):
        display_matrices = create_genotype_matrix_display(genotype_data, max_variants=20, max_samples=20)

    # Render matrices
    matrix_renderer = UIComponentFactory.get_renderer('genotype_matrix')
    matrix_renderer.render(display_matrices)

    # Add download options
    st.subheader("📥 Download Options")

    data_dict = genotype_data.get('data', {})
    for data_type, data in data_dict.items():
        if not data['genotypes'].empty:
            col1, col2 = st.columns(2)

            with col1:
                # Download genotype matrix as CSV
                csv_data = data['genotypes'].to_csv()
                st.download_button(
                    label=f"Download {data_type} Genotypes (CSV)",
                    data=csv_data,
                    file_name=f"{data_type}_genotypes.csv",
                    mime="text/csv"
                )

            with col2:
                # Download metadata as CSV
                if not data['metadata'].empty:
                    metadata_csv = data['metadata'].to_csv()
                    st.download_button(
                        label=f"Download {data_type} Metadata (CSV)",
                        data=metadata_csv,
                        file_name=f"{data_type}_metadata.csv",
                        mime="text/csv"
                    )


def _render_statistics_tab(genotype_data: Dict[str, Any]) -> None:
    """Render statistics tab."""
    st.subheader("📈 Carrier Frequency Analysis")

    # Calculate frequencies
    with st.spinner("Calculating frequencies..."):
        frequencies = calculate_carrier_frequencies(genotype_data, by_variant=True, by_gene=True)

    # Overall frequencies
    st.subheader("Overall Frequencies by Data Type")

    overall_data = frequencies.get('overall', {})
    if overall_data:
        freq_rows = []
        for data_type, stats in overall_data.items():
            freq_rows.append({
                "Data Type": data_type,
                "Carrier Frequency": f"{stats['carrier_frequency']:.3%}",
                "Total Carriers": f"{stats['total_carriers']:,}",
                "Total Genotypes": f"{stats['total_genotypes']:,}",
                "Samples": f"{stats['sample_count']:,}",
                "Variants": f"{stats['variant_count']:,}"
            })

        if freq_rows:
            import pandas as pd
            freq_df = pd.DataFrame(freq_rows)
            st.dataframe(freq_df, width='stretch', hide_index=True)

    # Gene-level frequencies
    gene_data = frequencies.get('by_gene', {})
    if gene_data:
        st.subheader("Carrier Frequencies by Gene")

        for data_type, gene_freqs in gene_data.items():
            if gene_freqs:
                st.write(f"**{data_type} Data:**")

                gene_rows = []
                for gene, stats in gene_freqs.items():
                    gene_rows.append({
                        "Gene": gene,
                        "Carrier Frequency": f"{stats['carrier_frequency']:.3%}",
                        "Carrier Count": f"{stats['carrier_count']:,}",
                        "Sample Count": f"{stats['sample_count']:,}",
                        "Variants": f"{stats['variant_count']:,}"
                    })

                if gene_rows:
                    gene_df = pd.DataFrame(gene_rows)
                    gene_df = gene_df.sort_values('Carrier Count', ascending=False)
                    st.dataframe(gene_df, width='stretch', hide_index=True)


def _render_comparisons_tab(genotype_data: Dict[str, Any]) -> None:
    """Render comparisons tab."""
    st.subheader("🔍 Cross-Data Type Comparisons")

    data_types = list(genotype_data.get('data', {}).keys())

    if len(data_types) < 2:
        st.info("Need at least 2 data types loaded to perform comparisons.")
        return

    # Perform comparison analysis
    with st.spinner("Comparing data types..."):
        comparison = compare_data_types(genotype_data)

    if 'error' in comparison:
        st.error(f"Comparison error: {comparison['error']}")
        return

    # Sample overlap
    st.subheader("Sample Overlap Analysis")

    overlap_data = comparison.get('sample_overlap', {})
    if overlap_data:
        overlap_rows = []
        for comp_key, stats in overlap_data.items():
            overlap_rows.append({
                "Comparison": comp_key.replace('_vs_', ' vs '),
                "Shared Samples": f"{stats['shared_samples']:,}",
                "Overlap %": f"{stats['overlap_percentage']:.1f}%",
                "Total Type 1": f"{stats['dt1_total']:,}",
                "Total Type 2": f"{stats['dt2_total']:,}"
            })

        if overlap_rows:
            import pandas as pd
            overlap_df = pd.DataFrame(overlap_rows)
            st.dataframe(overlap_df, width='stretch', hide_index=True)

    # Carrier concordance
    st.subheader("Carrier Detection Concordance")

    concordance_data = comparison.get('carrier_concordance', {})
    if concordance_data:
        concordance_rows = []
        for comp_key, stats in concordance_data.items():
            concordance_rows.append({
                "Comparison": comp_key.replace('_vs_', ' vs '),
                "Both Carriers": f"{stats['both_carriers']:,}",
                "Type 1 Only": f"{stats['dt1_only_carriers']:,}",
                "Type 2 Only": f"{stats['dt2_only_carriers']:,}",
                "Neither": f"{stats['neither_carrier']:,}",
                "Concordance Rate": f"{stats['concordance_rate']:.1%}",
                "Samples Analyzed": f"{stats['shared_samples_analyzed']:,}"
            })

        if concordance_rows:
            concordance_df = pd.DataFrame(concordance_rows)
            st.dataframe(concordance_df, width='stretch', hide_index=True)

            # Add interpretation
            st.info("""
            **Concordance Rate**: Percentage of samples where both data types agree on carrier status.
            High concordance suggests good data quality and consistency between platforms.
            """)
    else:
        st.info("No shared samples available for concordance analysis.")