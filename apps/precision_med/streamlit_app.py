#!/usr/bin/env python3
"""
GP2 Carrier Data Viewer
======================

Simple viewer for WGS and NBA carrier data with variant filtering.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.repositories.variant_repo import CarrierDataRepository
from src.repositories import ClinicalRepository

# Page config
st.set_page_config(
    page_title="GP2 Carrier Data Viewer",
    page_icon="🧬", 
    layout="wide"
)

def main():
    st.title("🧬 GP2 Carrier Data Viewer")
    st.markdown("View WGS and NBA carrier data with variant filtering")
    
    # Initialize repositories
    @st.cache_resource
    def load_repositories():
        """Load and cache repositories."""
        wgs_repo = CarrierDataRepository("WGS", use_string_format=True)
        nba_repo = CarrierDataRepository("NBA", use_string_format=True)
        clinical_repo = ClinicalRepository()
        
        # Load data if available
        if wgs_repo.data_path.exists():
            wgs_repo.load()
        if nba_repo.data_path.exists():
            nba_repo.load()
        if clinical_repo.data_path.exists():
            clinical_repo.load()
            
        return wgs_repo, nba_repo, clinical_repo
    
    try:
        wgs_repo, nba_repo, clinical_repo = load_repositories()
        
        # Data overview
        st.subheader("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if wgs_repo.is_loaded:
                wgs_samples = len(wgs_repo.get_sample_ids())
                st.metric("🧪 WGS Samples", f"{wgs_samples:,}")
            else:
                st.metric("🧪 WGS Samples", "Not available")
        
        with col2:
            if wgs_repo.is_loaded:
                wgs_variants = len(wgs_repo.get_variant_ids())
                st.metric("🧬 WGS Variants", f"{wgs_variants:,}")
            else:
                st.metric("🧬 WGS Variants", "Not available")
        
        with col3:
            if nba_repo.is_loaded:
                nba_samples = len(nba_repo.get_sample_ids())
                st.metric("🔬 NBA Samples", f"{nba_samples:,}")
            else:
                st.metric("🔬 NBA Samples", "Not available")
        
        with col4:
            if nba_repo.is_loaded:
                nba_variants = len(nba_repo.get_variant_ids())
                st.metric("🧬 NBA Variants", f"{nba_variants:,}")
            else:
                st.metric("🧬 NBA Variants", "Not available")
        
        # Sample filters using clinical data
        if clinical_repo.is_loaded:
            st.subheader("👥 Sample Filters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # NBA Label filter
                available_nba_labels = sorted(list(set([
                    record.nba_label for record in clinical_repo.get_all() 
                    if record.nba_label and record.nba_label.strip()
                ])))
                selected_nba_labels = st.multiselect(
                    "NBA Labels:",
                    options=available_nba_labels,
                    default=[],
                    help="Filter by NBA sample labels"
                )
            
            with col2:
                # Baseline GP2 phenotype filter
                available_phenotypes = sorted(list(set([
                    record.baseline_gp2_phenotype for record in clinical_repo.get_all() 
                    if record.baseline_gp2_phenotype and record.baseline_gp2_phenotype.strip()
                ])))
                selected_phenotypes = st.multiselect(
                    "GP2 Phenotypes:",
                    options=available_phenotypes,
                    default=[],
                    help="Filter by baseline GP2 phenotype"
                )
            
            with col3:
                # Biological sex filter
                available_sex = sorted(list(set([
                    record.biological_sex_for_qc for record in clinical_repo.get_all() 
                    if record.biological_sex_for_qc and record.biological_sex_for_qc.strip()
                ])))
                selected_sex = st.multiselect(
                    "Biological Sex:",
                    options=available_sex,
                    default=[],
                    help="Filter by biological sex for QC"
                )
        else:
            selected_nba_labels = []
            selected_phenotypes = []
            selected_sex = []
        
        # Variant filter (shared between both datasets)
        st.subheader("🧬 Variant Filter")
        
        # Get all available variants from both datasets
        all_variants = set()
        if wgs_repo.is_loaded:
            all_variants.update(wgs_repo.get_variant_ids())
        if nba_repo.is_loaded:
            nba_variants = [v for v in nba_repo.get_variant_ids() if v != 'ancestry']
            all_variants.update(nba_variants)
        
        all_variants = sorted(list(all_variants))
        
        if all_variants:
            selected_variants = st.multiselect(
                "Select variants to display:",
                options=all_variants,
                default=[],
                help="Select specific variants to filter the data. Leave empty to show all data."
            )
        else:
            st.error("No variant data available")
            return
        
        # Function to filter samples based on clinical criteria
        def get_filtered_sample_ids():
            """Get GP2IDs that match the clinical filter criteria."""
            if not clinical_repo.is_loaded:
                return None
            
            filtered_samples = []
            for record in clinical_repo.get_all():
                # Apply NBA label filter
                if selected_nba_labels and (not record.nba_label or record.nba_label not in selected_nba_labels):
                    continue
                
                # Apply baseline GP2 phenotype filter
                if selected_phenotypes and (not record.baseline_gp2_phenotype or record.baseline_gp2_phenotype not in selected_phenotypes):
                    continue
                
                # Apply biological sex filter
                if selected_sex and (not record.biological_sex_for_qc or record.biological_sex_for_qc not in selected_sex):
                    continue
                
                # If all filters pass, add the GP2ID
                filtered_samples.append(record.gp2_id)
            
            return filtered_samples if any([selected_nba_labels, selected_phenotypes, selected_sex]) else None
        
        # WGS Data Section
        if wgs_repo.is_loaded:
            st.subheader("🧪 WGS Carrier Data")
            
            try:
                # Get filtered sample IDs
                filtered_sample_ids = get_filtered_sample_ids()
                
                # Get WGS data
                if selected_variants:
                    # Filter by selected variants
                    with st.spinner("Loading WGS data..."):
                        wgs_data = {}
                        for variant_id in selected_variants:
                            if variant_id in wgs_repo.get_variant_ids():
                                variant_data = wgs_repo.get_carrier_data_for_variant(variant_id)
                                
                                # Apply sample filtering if clinical filters are active
                                if filtered_sample_ids is not None:
                                    variant_data = {k: v for k, v in variant_data.items() if k in filtered_sample_ids}
                                
                                wgs_data[variant_id] = variant_data
                        
                        if wgs_data:
                            # Create wide format dataframe
                            wgs_df = pd.DataFrame(wgs_data)
                            wgs_df.index.name = 'Sample_ID'
                            wgs_df = wgs_df.reset_index()
                            
                            # Remove rows that are all NaN (except Sample_ID)
                            wgs_df = wgs_df.dropna(how='all', subset=wgs_df.columns[1:])
                            
                            st.write(f"**Showing {len(wgs_df)} samples × {len(selected_variants)} variants**")
                            st.dataframe(wgs_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No WGS data found for selected variants")
                
                else:
                    # Show preview with first few variants
                    with st.spinner("Loading WGS data preview..."):
                        variant_subset = wgs_repo.get_variant_ids()[:5]  # First 5 variants
                        
                        wgs_data = {}
                        for variant_id in variant_subset:
                            variant_data = wgs_repo.get_carrier_data_for_variant(variant_id)
                            
                            # Apply sample filtering if clinical filters are active
                            if filtered_sample_ids is not None:
                                variant_data = {k: v for k, v in variant_data.items() if k in filtered_sample_ids}
                            
                            # Limit to first 200 samples for each variant
                            sample_subset = list(variant_data.keys())[:200]
                            limited_data = {k: v for k, v in variant_data.items() if k in sample_subset}
                            wgs_data[variant_id] = limited_data
                        
                        if wgs_data:
                            # Create wide format dataframe
                            wgs_df = pd.DataFrame(wgs_data)
                            wgs_df.index.name = 'Sample_ID'
                            wgs_df = wgs_df.reset_index()
                            
                            # Remove rows that are all NaN (except Sample_ID)
                            wgs_df = wgs_df.dropna(how='all', subset=wgs_df.columns[1:])
                            
                            st.write(f"**Preview: Showing {len(wgs_df)} samples × {len(variant_subset)} variants**")
                            st.dataframe(wgs_df, use_container_width=True, hide_index=True)
                            st.info("💡 Select specific variants above to see filtered results")
            
            except Exception as e:
                st.error(f"Error loading WGS data: {e}")
        
        else:
            st.info("WGS carrier data not available")
        
        # NBA Data Section
        if nba_repo.is_loaded:
            st.subheader("🔬 NBA Carrier Data")
            
            try:
                # Get filtered sample IDs (reuse from WGS section)
                filtered_sample_ids = get_filtered_sample_ids()
                
                # Get NBA data
                if selected_variants:
                    # Filter by selected variants
                    with st.spinner("Loading NBA data..."):
                        nba_data = {}
                        for variant_id in selected_variants:
                            if variant_id in nba_repo.get_variant_ids():
                                variant_data = nba_repo.get_carrier_data_for_variant(variant_id)
                                
                                # Apply sample filtering if clinical filters are active
                                if filtered_sample_ids is not None:
                                    variant_data = {k: v for k, v in variant_data.items() if k in filtered_sample_ids}
                                
                                nba_data[variant_id] = variant_data
                        
                        if nba_data:
                            # Create wide format dataframe
                            nba_df = pd.DataFrame(nba_data)
                            nba_df.index.name = 'Sample_ID'
                            nba_df = nba_df.reset_index()
                            
                            # Remove ancestry column if it exists
                            if 'ancestry' in nba_df.columns:
                                nba_df = nba_df.drop(columns=['ancestry'])
                            
                            # Remove rows that are all NaN (except Sample_ID)
                            if len(nba_df.columns) > 1:  # Check if we have any variant columns left
                                nba_df = nba_df.dropna(how='all', subset=nba_df.columns[1:])
                            
                            variant_count = len(nba_df.columns) - 1  # Subtract 1 for Sample_ID column
                            st.write(f"**Showing {len(nba_df)} samples × {variant_count} variants**")
                            st.dataframe(nba_df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No NBA data found for selected variants")
                
                else:
                    # Show preview with first few variants
                    with st.spinner("Loading NBA data preview..."):
                        variant_subset = [v for v in nba_repo.get_variant_ids()[:5] if v != 'ancestry']  # First 5 variants, excluding ancestry
                        
                        nba_data = {}
                        for variant_id in variant_subset:
                            variant_data = nba_repo.get_carrier_data_for_variant(variant_id)
                            
                            # Apply sample filtering if clinical filters are active
                            if filtered_sample_ids is not None:
                                variant_data = {k: v for k, v in variant_data.items() if k in filtered_sample_ids}
                            
                            # Limit to first 200 samples for each variant
                            sample_subset = list(variant_data.keys())[:200]
                            limited_data = {k: v for k, v in variant_data.items() if k in sample_subset}
                            nba_data[variant_id] = limited_data
                        
                        if nba_data:
                            # Create wide format dataframe
                            nba_df = pd.DataFrame(nba_data)
                            nba_df.index.name = 'Sample_ID'
                            nba_df = nba_df.reset_index()
                            
                            # Remove ancestry column if it exists
                            if 'ancestry' in nba_df.columns:
                                nba_df = nba_df.drop(columns=['ancestry'])
                            
                            # Remove rows that are all NaN (except Sample_ID)
                            if len(nba_df.columns) > 1:  # Check if we have any variant columns left
                                nba_df = nba_df.dropna(how='all', subset=nba_df.columns[1:])
                            
                            variant_count = len(nba_df.columns) - 1  # Subtract 1 for Sample_ID column
                            st.write(f"**Preview: Showing {len(nba_df)} samples × {variant_count} variants**")
                            st.dataframe(nba_df, use_container_width=True, hide_index=True)
                            st.info("💡 Select specific variants above to see filtered results")
            
            except Exception as e:
                st.error(f"Error loading NBA data: {e}")
        
        else:
            st.info("NBA carrier data not available")
    
    except Exception as e:
        st.error(f"Error loading application: {e}")
        with st.expander("🔧 Debug Information"):
            st.exception(e)

if __name__ == "__main__":
    main() 