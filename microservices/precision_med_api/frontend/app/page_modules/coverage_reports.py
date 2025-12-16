"""
Coverage Reports page module.

Displays SNP coverage profiling data showing which variants from the SNP list
exist in each data type's source files (WGS, IMPUTED, NBA, EXOMES).
"""

import streamlit as st
import pandas as pd
from typing import Optional

from app.config import FrontendConfig
from app.utils.data_loaders import (
    load_coverage_by_locus,
    load_coverage_by_variant,
    load_coverage_summary
)
from app.utils.methods_descriptions import MethodsDescriptions


def render_coverage_reports(release: str, job_name: str, config: FrontendConfig):
    """Main render function for Coverage Reports page."""
    st.header("📈 Coverage Reports")
    st.markdown(
        "Analysis of which SNP list variants exist in each data type's source files. "
        "This helps identify data availability before interpreting extraction results."
    )

    # Methods section
    with st.expander("📖 Methods", expanded=False):
        st.markdown(MethodsDescriptions.get_coverage_reports_methods())

    # Load data
    locus_df = load_coverage_by_locus(release, job_name, config.results_base_path)
    variant_df = load_coverage_by_variant(release, job_name, config.results_base_path)
    summary = load_coverage_summary(release, job_name, config.results_base_path)

    if locus_df is None:
        st.info("No coverage report data available for this release/job.")
        st.markdown(
            "Coverage reports are generated when running the pipeline with coverage profiling enabled. "
            "Use `python run_carriers_pipeline.py` (coverage profiling is enabled by default)."
        )
        return

    # Render sections
    render_summary_metrics(locus_df, summary)
    render_locus_table(locus_df)
    if variant_df is not None:
        render_variant_table(variant_df)


def render_summary_metrics(locus_df: pd.DataFrame, summary: Optional[dict]):
    """Render summary metrics section."""
    st.subheader("Summary by Data Type")

    # Calculate totals from locus data
    total_variants = locus_df['total_variants'].sum()

    # Calculate coverage stats per data type
    data_types = ['WGS', 'IMPUTED', 'NBA', 'EXOMES']
    stats = {}

    for dt in data_types:
        dt_lower = dt.lower()
        exact_col = f'{dt_lower}_exact'
        position_col = f'{dt_lower}_position'

        if exact_col in locus_df.columns:
            exact = locus_df[exact_col].sum()
            position = locus_df[position_col].sum()
            exact_pct = (exact / total_variants * 100) if total_variants > 0 else 0
            position_pct = (position / total_variants * 100) if total_variants > 0 else 0
            stats[dt] = {
                'exact': exact,
                'position': position,
                'exact_pct': exact_pct,
                'position_pct': position_pct
            }

    # Display in columns
    cols = st.columns(4)
    for i, dt in enumerate(data_types):
        with cols[i]:
            if dt in stats:
                s = stats[dt]
                # Color code based on coverage
                if s['exact_pct'] >= 50:
                    color = "green"
                elif s['exact_pct'] >= 20:
                    color = "orange"
                else:
                    color = "red"

                st.metric(
                    label=dt,
                    value=f"{s['exact_pct']:.1f}%",
                    delta=f"{int(s['exact'])}/{total_variants} exact"
                )
                st.caption(f"Position: {s['position_pct']:.1f}% ({int(s['position'])})")
            else:
                st.metric(label=dt, value="N/A")

    # Total variants info
    st.info(f"**Total SNP List Variants:** {total_variants} across {len(locus_df)} loci")


def render_locus_table(locus_df: pd.DataFrame):
    """Render locus coverage table."""
    st.subheader("Coverage by Locus")

    # Search filter
    search = st.text_input("🔍 Search locus:", "", key="locus_search")

    # Prepare display dataframe
    display_df = locus_df.copy()

    # Filter by search
    if search:
        display_df = display_df[
            display_df['locus'].str.contains(search, case=False, na=False)
        ]

    # Format percentage columns for display
    pct_cols = [col for col in display_df.columns if col.endswith('_pct')]
    for col in pct_cols:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%")

    # Rename columns for clarity
    column_rename = {
        'locus': 'Locus',
        'chromosome': 'Chr',
        'total_variants': 'Total',
        'wgs_exact': 'WGS',
        'wgs_exact_pct': 'WGS %',
        'imputed_exact': 'IMP',
        'imputed_exact_pct': 'IMP %',
        'nba_exact': 'NBA',
        'nba_exact_pct': 'NBA %',
        'exomes_exact': 'EXO',
        'exomes_exact_pct': 'EXO %'
    }

    # Select columns to display (exact matches only for cleaner view)
    display_cols = ['locus', 'chromosome', 'total_variants',
                    'wgs_exact', 'wgs_exact_pct',
                    'imputed_exact', 'imputed_exact_pct',
                    'nba_exact', 'nba_exact_pct',
                    'exomes_exact', 'exomes_exact_pct']

    display_cols = [c for c in display_cols if c in display_df.columns]
    display_df = display_df[display_cols].rename(columns=column_rename)

    # Sort by total variants descending
    display_df = display_df.sort_values('Total', ascending=False)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Show position matches in expander
    with st.expander("📊 Position Matches (includes allele mismatches)", expanded=False):
        st.markdown(
            "Position matches find variants at the same genomic position but potentially "
            "with different allele representations (strand flips, indel normalization)."
        )

        pos_cols = ['locus', 'total_variants',
                    'wgs_position', 'imputed_position', 'nba_position', 'exomes_position']
        pos_cols = [c for c in pos_cols if c in locus_df.columns]

        if pos_cols:
            pos_df = locus_df[pos_cols].copy()
            pos_df = pos_df.sort_values('total_variants', ascending=False)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)


def render_variant_table(variant_df: pd.DataFrame):
    """Render variant-level coverage table."""
    st.subheader("Coverage by Variant")

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        # Locus filter
        loci = ['All'] + sorted(variant_df['locus'].unique().tolist())
        selected_locus = st.selectbox("Filter by Locus:", loci, key="variant_locus_filter")

    with col2:
        # Search filter
        search = st.text_input("🔍 Search variant/rsid:", "", key="variant_search")

    # Apply filters
    display_df = variant_df.copy()

    if selected_locus != 'All':
        display_df = display_df[display_df['locus'] == selected_locus]

    if search:
        mask = (
            display_df['variant_id'].str.contains(search, case=False, na=False) |
            display_df['snp_name'].str.contains(search, case=False, na=False) |
            display_df['rsid'].astype(str).str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    # Convert boolean columns to checkmarks for display
    bool_cols = [col for col in display_df.columns if col.endswith('_exact') or col.endswith('_position')]
    for col in bool_cols:
        display_df[col] = display_df[col].apply(lambda x: '✓' if x else '✗')

    # Select and rename columns
    display_cols = ['snp_name', 'locus', 'variant_id', 'rsid',
                    'WGS_exact', 'IMPUTED_exact', 'NBA_exact', 'EXOMES_exact']
    display_cols = [c for c in display_cols if c in display_df.columns]

    column_rename = {
        'snp_name': 'Mutation',
        'locus': 'Locus',
        'variant_id': 'Variant ID',
        'rsid': 'rsID',
        'WGS_exact': 'WGS',
        'IMPUTED_exact': 'IMP',
        'NBA_exact': 'NBA',
        'EXOMES_exact': 'EXO'
    }

    display_df = display_df[display_cols].rename(columns=column_rename)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    # Stats
    st.caption(f"Showing {len(display_df)} of {len(variant_df)} variants")

    # Position matches in expander
    with st.expander("📊 View Position Match Details", expanded=False):
        pos_cols = ['snp_name', 'locus', 'variant_id',
                    'WGS_position', 'IMPUTED_position', 'NBA_position', 'EXOMES_position']
        pos_cols = [c for c in pos_cols if c in variant_df.columns]

        if pos_cols:
            # Apply same filters
            pos_df = variant_df.copy()
            if selected_locus != 'All':
                pos_df = pos_df[pos_df['locus'] == selected_locus]
            if search:
                mask = (
                    pos_df['variant_id'].str.contains(search, case=False, na=False) |
                    pos_df['snp_name'].str.contains(search, case=False, na=False) |
                    pos_df['rsid'].astype(str).str.contains(search, case=False, na=False)
                )
                pos_df = pos_df[mask]

            pos_df = pos_df[pos_cols].copy()
            for col in pos_cols:
                if col.endswith('_position'):
                    pos_df[col] = pos_df[col].apply(lambda x: '✓' if x else '✗')

            st.dataframe(pos_df, use_container_width=True, hide_index=True)
