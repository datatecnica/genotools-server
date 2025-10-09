"""
Locus reports page - displays per-gene clinical phenotype statistics.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from app.config import FrontendConfig
from app.utils.data_loaders import load_locus_report, get_selected_probe_ids, load_variant_to_mutation_map


def render_locus_reports(release: str, job_name: str, config: FrontendConfig):
    """
    Render the locus reports page.

    Args:
        release: Release identifier
        job_name: Job name
        config: Frontend configuration
    """
    st.header("📊 Per-Locus Clinical Reports")
    st.markdown("Ancestry-stratified clinical phenotype statistics for each gene/locus.")

    # Load all available data types
    wgs_data = load_locus_report(release, job_name, "WGS", config.results_base_path)
    nba_data = load_locus_report(release, job_name, "NBA", config.results_base_path)
    imputed_data = load_locus_report(release, job_name, "IMPUTED", config.results_base_path)

    if not wgs_data and not nba_data and not imputed_data:
        st.info(
            "**No locus report data available.**\n\n"
            "Locus reports require:\n"
            "- Genotype data (WGS, NBA, or IMPUTED)\n"
            "- Clinical data from phenotype files\n"
            "- Carriers identified in the dataset\n\n"
            "Run pipeline without `--skip-locus-reports` to generate this data."
        )
        return

    # Data type selector
    available_data_types = []
    data_map = {}
    if wgs_data:
        available_data_types.append("WGS")
        data_map["WGS"] = wgs_data
    if nba_data:
        available_data_types.append("NBA")
        data_map["NBA"] = nba_data
    if imputed_data:
        available_data_types.append("IMPUTED")
        data_map["IMPUTED"] = imputed_data

    data_type = st.selectbox(
        "Select data type:",
        available_data_types,
        help="Choose which genotype data source to view (WGS = whole genome sequencing, NBA = genotyping array, IMPUTED = imputed genotypes)"
    )

    # Select appropriate data
    current_data = data_map.get(data_type)

    if not current_data:
        st.error("Selected data type not available")
        return

    # Get probe selection info for NBA data
    probe_info = None
    if data_type == "NBA":
        probe_info = get_selected_probe_ids(release, job_name, config.results_base_path)

    # Load variant-to-mutation mapping for better display
    variant_map = load_variant_to_mutation_map(release, job_name, config.results_base_path)

    # Render sections
    render_summary(current_data, probe_info)
    render_loci_table(current_data, variant_map)


def render_summary(data: Dict[str, Any], probe_info: Dict[str, Any] = None):
    """Render summary section."""
    st.subheader("Summary")

    summary = data.get('summary', {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_loci = summary.get('total_loci_analyzed', 0)
        st.metric("Total Loci", f"{total_loci:,}")

    with col2:
        total_carriers = summary.get('total_carriers_identified', 0)
        st.metric("Total Carriers", f"{total_carriers:,}")

    with col3:
        samples_with_clinical = summary.get('total_samples_with_clinical_data', 0)
        st.metric("Samples with Clinical Data", f"{samples_with_clinical:,}")

    with col4:
        ancestries = len(summary.get('ancestries_represented', []))
        st.metric("Ancestries", ancestries)


def render_loci_table(data: Dict[str, Any], variant_map: Dict[str, str] = None):
    """Render loci table with expandable details."""
    st.subheader("Locus Details")

    # locus_reports is a list, not a dict
    loci_list = data.get('locus_reports', [])

    if not loci_list:
        st.info("No locus data available")
        return

    # Create summary table
    summary_rows = []
    for locus_entry in loci_list:
        gene_name = locus_entry.get('locus', 'Unknown')
        total_metrics = locus_entry.get('total_metrics', {})
        summary_rows.append({
            'Gene/Locus': gene_name,
            'Total Variants': locus_entry.get('n_variants', 0),
            'Total Carriers': locus_entry.get('total_carriers_all_ancestries', 0),
            'Carriers with Clinical Data': total_metrics.get('carriers_with_clinical_data', 0),
            'Ancestries Analyzed': len(locus_entry.get('ancestries_represented', []))
        })

    # Display summary table
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Locus selector for detailed view
        locus_names = [entry.get('locus', 'Unknown') for entry in loci_list]
        selected_locus = st.selectbox(
            "Select locus for detailed ancestry breakdown:",
            locus_names
        )

        if selected_locus:
            # Find the selected locus entry
            locus_data = next((entry for entry in loci_list if entry.get('locus') == selected_locus), None)
            if locus_data:
                render_locus_details(locus_data, selected_locus, variant_map)


def render_locus_details(locus_data: Dict[str, Any], locus_name: str, variant_map: Dict[str, str] = None):
    """Render detailed ancestry breakdown for a specific locus."""
    st.subheader(f"🧬 {locus_name} - Ancestry Breakdown")

    # by_ancestry is a list of ancestry statistics
    by_ancestry = locus_data.get('by_ancestry', [])

    if not by_ancestry:
        st.info("No ancestry data available for this locus")
        return

    # Create ancestry table
    ancestry_rows = []
    for ancestry_stats in by_ancestry:
        ancestry = ancestry_stats.get('ancestry', 'Unknown')
        if ancestry == 'TOTAL':  # Skip total row
            continue
        ancestry_rows.append({
            'Ancestry': ancestry,
            'Total Carriers': ancestry_stats.get('total_carriers', 0),
            'Carriers with Clinical Data': ancestry_stats.get('carriers_with_clinical_data', 0),
            'Clinical Data %': f"{ancestry_stats.get('clinical_data_availability_pct', 0):.1f}%",
            'H&Y < 2': ancestry_stats.get('hy_less_than_2', 0),
            'MoCA ≥ 24': ancestry_stats.get('moca_gte_24', 0)
        })

    if ancestry_rows:
        df = pd.DataFrame(ancestry_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Show total metrics
    total_metrics = locus_data.get('total_metrics', {})
    if total_metrics:
        with st.expander("📊 Total Metrics", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Carriers", f"{total_metrics.get('total_carriers', 0):,}")
                st.metric("Disease Duration ≤ 5 years", f"{total_metrics.get('disease_duration_lte_5_years', 0):,}")
            with col2:
                st.metric("Carriers with Clinical Data", f"{total_metrics.get('carriers_with_clinical_data', 0):,}")
                st.metric("Disease Duration ≤ 7 years", f"{total_metrics.get('disease_duration_lte_7_years', 0):,}")
            with col3:
                st.metric("H&Y < 2", f"{total_metrics.get('hy_less_than_2', 0):,}")
                st.metric("MoCA ≥ 24", f"{total_metrics.get('moca_gte_24', 0):,}")

    # Show variants for this locus with carrier counts
    variant_details = locus_data.get('variant_details', [])
    if variant_details:
        with st.expander(f"📋 Variants in {locus_name} ({len(variant_details)} total)", expanded=False):
            # Build table from variant_details
            variant_rows = []

            for variant in variant_details:
                variant_rows.append({
                    'Mutation': variant.get('mutation_name', '-'),
                    'Variant ID': variant.get('variant_id', ''),
                    'Chromosome': variant.get('chromosome', ''),
                    'Position': f"{variant.get('position', 0):,}",
                    'Ref': variant.get('ref_allele', ''),
                    'Alt': variant.get('alt_allele', ''),
                    'Total Carriers': variant.get('carrier_count', 0),
                    'Heterozygous': variant.get('heterozygous_count', 0),
                    'Homozygous': variant.get('homozygous_count', 0)
                })

            if variant_rows:
                df_variants = pd.DataFrame(variant_rows)
                st.dataframe(df_variants, use_container_width=True, hide_index=True)
