"""
Precision Medicine Data Access App
Bare-bones Streamlit app for viewing NBA carriers data.
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import zipfile
import io
import os
import hashlib

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.core.config import DataConfig
from src.data_service.repositories import create_nba_repository, create_wgs_repository
from src.core.exceptions import DataAccessError


def generate_zip_filename(release: str, snp_filter: list = None) -> str:
    """Generate consistent filename for ZIP exports."""
    if snp_filter:
        # Create a hash of the sorted SNP filter for consistent naming
        filter_hash = hashlib.md5(str(sorted(snp_filter)).encode()).hexdigest()[:8]
        return f"precision_med_data_{release}_filtered_{filter_hash}.zip"
    else:
        return f"precision_med_data_{release}_all.zip"


def create_zip_file(release: str, snp_filter: list = None) -> Path:
    """Create ZIP file with all datasets and save to shared directory."""
    # Initialize configuration first
    config = DataConfig(release=release)
    export_dir = config.get_export_directory()
    filename = generate_zip_filename(release, snp_filter)
    filepath = export_dir / filename
    
    # If file already exists, return the path
    if filepath.exists():
        return filepath
    
    # Initialize repositories
    nba_repo = create_nba_repository(config)
    wgs_repo = create_wgs_repository(config)
    
    # Create ZIP file
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Load and add all 6 DataFrames to the zip
        datasets = [
            ("nba_info", nba_repo.load_filtered_info_df(snp_names=snp_filter)),
            ("nba_int", nba_repo.load_filtered_int_df(snp_names=snp_filter)),
            ("nba_string", nba_repo.load_filtered_string_df(snp_names=snp_filter)),
            ("wgs_info", wgs_repo.load_filtered_info_df(snp_names=snp_filter)),
            ("wgs_int", wgs_repo.load_filtered_int_df(snp_names=snp_filter)),
            ("wgs_string", wgs_repo.load_filtered_string_df(snp_names=snp_filter))
        ]
        
        for name, df in datasets:
            csv_data = df.to_csv(index=False)
            zip_file.writestr(f"{name}_{release}.csv", csv_data)
    
    return filepath


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Precision Medicine Data Access",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Precision Medicine Data Access")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Release selection
        available_releases = DataConfig.get_available_releases()
        selected_release = st.selectbox(
            "Select Release:",
            available_releases,
            index=0
        )
        
        st.markdown("---")
        st.info(f"Current Release: **{selected_release}**")
    
    # Initialize configuration and repositories
    try:
        config = DataConfig(release=selected_release)
        nba_repo = create_nba_repository(config)
        wgs_repo = create_wgs_repository(config)
        
        # Global SNP Filter - get intersection of SNPs from both datasets
        st.header("🎯 Global SNP Filter")
        with st.spinner("Loading available SNPs from both datasets..."):
            nba_snps = set(nba_repo.get_available_snp_names())
            wgs_snps = set(wgs_repo.get_available_snp_names())
            common_snps = sorted(list(nba_snps.intersection(wgs_snps)))
            all_snps = sorted(list(nba_snps.union(wgs_snps)))
        
        # Global filter dropdown
        global_snp_filter = st.multiselect(
            "Select SNPs to view across all data:",
            all_snps,
            default=None,
            key="global_snp_filter",
            help=f"Filter all data by these SNPs. {len(common_snps)} SNPs are common to both datasets."
        )
        
        # Apply global filter if selected
        active_snp_filter = global_snp_filter if global_snp_filter else None
        
        # Download All Data button
        st.subheader("📦 Bulk Data Export")
        
        # Check if file already exists
        filename = generate_zip_filename(selected_release, active_snp_filter)
        export_dir = config.get_export_directory()
        filepath = export_dir / filename
        file_exists = filepath.exists()
        
        if file_exists:
            file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB
            st.info(f"📁 **File already exists**: `{filename}` ({file_size:.1f} MB)")
        
        if st.button("🗂️ Download All Data (ZIP)", type="primary"):
            try:
                if file_exists:
                    status_msg = "📁 Using existing file..."
                else:
                    status_msg = "🔧 Creating new ZIP file..."
                
                with st.spinner(status_msg):
                    zip_filepath = create_zip_file(selected_release, active_snp_filter)
                    
                    # Read the file for download
                    with open(zip_filepath, 'rb') as f:
                        zip_data = f.read()
                    
                    # Create download button for the zip file
                    st.download_button(
                        label="📥 Download ZIP File",
                        data=zip_data,
                        file_name=filename,
                        mime="application/zip",
                        key="download_all_zip"
                    )
                    
                    file_size = len(zip_data) / (1024 * 1024)  # Size in MB
                    status = "📁 Retrieved" if file_exists else "✅ Created"
                    st.success(f"{status} ZIP file with all 6 datasets ({file_size:.1f} MB){' (filtered)' if active_snp_filter else ''}")
                    
            except Exception as e:
                st.error(f"❌ Error preparing download: {e}")
        
        # Show directory info
        export_files = list(export_dir.glob("*.zip")) if export_dir.exists() else []
        if export_files:
            with st.expander(f"📂 Available Export Files ({len(export_files)} total)"):
                for file_path in sorted(export_files):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    st.write(f"• `{file_path.name}` ({size_mb:.1f} MB)")
        
        st.info("💡 **Shared Storage**: Files are saved to GCS mount (`~/gcs_mounts/genotools_server/carriers/exports/`) and reused for identical filter combinations.")
        
        st.markdown("---")
        
        # Display combined summary statistics
        st.header("📊 Dataset Overview")
        
        with st.spinner("Loading summary statistics..."):
            nba_stats = nba_repo.get_summary_stats()
            wgs_stats = wgs_repo.get_summary_stats()
        
        # NBA Overview
        st.subheader("🏀 NBA (NeuroBooster Array)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variants", nba_stats['num_variants'])
        with col2:
            st.metric("Samples", nba_stats['num_samples'])
        with col3:
            st.metric("Loci", len(nba_stats['loci']))
        
        # NBA Available Loci
        st.write("**Available NBA Loci:**")
        st.write(", ".join(nba_stats['loci']))
        
        # NBA Variant Info directly below NBA metrics
        with st.expander("📋 NBA Variant Info", expanded=False):
            with st.spinner("Loading NBA variant info..."):
                nba_info_df = nba_repo.load_filtered_info_df(snp_names=active_snp_filter)
            
            st.metric("NBA Variants", len(nba_info_df))
            if len(nba_info_df) > 0:
                st.dataframe(nba_info_df, use_container_width=True)
                
                # Download option
                nba_info_csv = nba_info_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download NBA Variant Info",
                    data=nba_info_csv,
                    file_name=f"nba_{selected_release}_variant_info.csv",
                    mime="text/csv",
                    key="nba_info_download"
                )
            else:
                st.info("No variants match the current filter.")
        
        # WGS Overview
        st.subheader("🧬 WGS (Whole Genome Sequencing)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Variants", wgs_stats['num_variants'])
        with col2:
            st.metric("Samples", wgs_stats['num_samples'])
        with col3:
            st.metric("Loci", len(wgs_stats['loci']))
        
        # WGS Available Loci
        st.write("**Available WGS Loci:**")
        st.write(", ".join(wgs_stats['loci']))
        
        # WGS Variant Info directly below WGS metrics
        with st.expander("📋 WGS Variant Info", expanded=False):
            with st.spinner("Loading WGS variant info..."):
                wgs_info_df = wgs_repo.load_filtered_info_df(snp_names=active_snp_filter)
            
            st.metric("WGS Variants", len(wgs_info_df))
            if len(wgs_info_df) > 0:
                st.dataframe(wgs_info_df, use_container_width=True)
                
                # Download option
                wgs_info_csv = wgs_info_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download WGS Variant Info",
                    data=wgs_info_csv,
                    file_name=f"wgs_{selected_release}_variant_info.csv",
                    mime="text/csv",
                    key="wgs_info_download"
                )
            else:
                st.info("No variants match the current filter.")
        
        st.markdown("---")
        
        # NBA Data viewing section
        st.header("🏀 NBA Genotype Data Viewer")
        
        # NBA Data type selection (only genotype data since variant info is in overview)
        nba_data_type = st.selectbox(
            "Select NBA Genotype Data Type:",
            ["int", "string"],
            format_func=lambda x: {
                "int": "🔢 Integer Genotypes", 
                "string": "📝 String Genotypes"
            }[x],
            key="nba_data_type"
        )
        
        # Sample filtering (SNP filtering is global)
        with st.expander("👥 NBA Sample Filter"):
            with st.spinner("Loading available NBA samples..."):
                available_samples = nba_repo.get_available_samples()
            
            nba_selected_samples = st.multiselect(
                "Filter by Sample IDs:",
                available_samples,
                default=None,
                key="nba_sample_filter",
                help="Select specific samples to view, or leave empty for all samples"
            )
            
            nba_sample_filter = nba_selected_samples if nba_selected_samples else None
        
        # Load and display selected NBA data
        with st.spinner(f"Loading NBA {nba_data_type} data..."):
            try:
                if nba_data_type == "int":
                    nba_df = nba_repo.load_filtered_int_df(
                        sample_ids=nba_sample_filter, 
                        snp_names=active_snp_filter
                    )
                else:  # string
                    nba_df = nba_repo.load_filtered_string_df(
                        sample_ids=nba_sample_filter, 
                        snp_names=active_snp_filter
                    )
                
                # Display DataFrame info
                st.subheader(f"📊 NBA {nba_data_type.title()} Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(nba_df))
                with col2:
                    st.metric("Columns", len(nba_df.columns))
                with col3:
                    filter_status = "🔍 Filtered" if (nba_sample_filter or active_snp_filter) else "📋 All Data"
                    st.metric("Status", filter_status)
                
                # Display data with smart sampling for large datasets
                st.subheader("🔍 NBA Data")
                
                # Calculate DataFrame size in memory
                df_size_mb = nba_df.memory_usage(deep=True).sum() / (1024 * 1024)
                
                # If DataFrame is large, show only the head
                if df_size_mb > 150 or len(nba_df) > 10000:
                    st.info(f"📊 **Large Dataset ({df_size_mb:.1f} MB, {len(nba_df):,} rows)**: Showing first 10 rows. Use filters or download for full data.")
                    st.dataframe(nba_df.head(10), use_container_width=True)
                else:
                    st.dataframe(nba_df, use_container_width=True)
                
                # Download option (always includes full dataset)
                st.subheader("💾 NBA Download")
                nba_csv = nba_df.to_csv(index=False)
                st.download_button(
                    label=f"Download NBA {nba_data_type} data as CSV (Full Dataset: {len(nba_df):,} rows)",
                    data=nba_csv,
                    file_name=f"nba_{selected_release}_{nba_data_type}.csv",
                    mime="text/csv",
                    key="nba_download"
                )
                
            except Exception as e:
                st.error(f"Error loading NBA {nba_data_type} data: {e}")
        
        st.markdown("---")
        
        # WGS Data viewing section
        st.header("🧬 WGS Genotype Data Viewer")
        
        # WGS Data type selection (only genotype data since variant info is in overview)
        wgs_data_type = st.selectbox(
            "Select WGS Genotype Data Type:",
            ["int", "string"],
            format_func=lambda x: {
                "int": "🔢 Integer Genotypes", 
                "string": "📝 String Genotypes"
            }[x],
            key="wgs_data_type"
        )
        
        # Sample filtering (SNP filtering is global)
        with st.expander("👥 WGS Sample Filter"):
            with st.spinner("Loading available WGS samples..."):
                available_samples = wgs_repo.get_available_samples()
            
            wgs_selected_samples = st.multiselect(
                "Filter by Sample IDs:",
                available_samples,
                default=None,
                key="wgs_sample_filter",
                help="Select specific samples to view, or leave empty for all samples"
            )
            
            wgs_sample_filter = wgs_selected_samples if wgs_selected_samples else None
        
        # Load and display selected WGS data
        with st.spinner(f"Loading WGS {wgs_data_type} data..."):
            try:
                if wgs_data_type == "int":
                    wgs_df = wgs_repo.load_filtered_int_df(
                        sample_ids=wgs_sample_filter, 
                        snp_names=active_snp_filter
                    )
                else:  # string
                    wgs_df = wgs_repo.load_filtered_string_df(
                        sample_ids=wgs_sample_filter, 
                        snp_names=active_snp_filter
                    )
                
                # Display DataFrame info
                st.subheader(f"📊 WGS {wgs_data_type.title()} Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(wgs_df))
                with col2:
                    st.metric("Columns", len(wgs_df.columns))
                with col3:
                    filter_status = "🔍 Filtered" if (wgs_sample_filter or active_snp_filter) else "📋 All Data"
                    st.metric("Status", filter_status)
                
                # Display data with smart sampling for large datasets
                st.subheader("🔍 WGS Data")
                
                # Calculate DataFrame size in memory
                df_size_mb = wgs_df.memory_usage(deep=True).sum() / (1024 * 1024)
                
                # If DataFrame is large, show only the head
                if df_size_mb > 150 or len(wgs_df) > 10000:
                    st.info(f"📊 **Large Dataset ({df_size_mb:.1f} MB, {len(wgs_df):,} rows)**: Showing first 10 rows. Use filters or download for full data.")
                    st.dataframe(wgs_df.head(10), use_container_width=True)
                else:
                    st.dataframe(wgs_df, use_container_width=True)
                
                # Download option (always includes full dataset)
                st.subheader("💾 WGS Download")
                wgs_csv = wgs_df.to_csv(index=False)
                st.download_button(
                    label=f"Download WGS {wgs_data_type} data as CSV (Full Dataset: {len(wgs_df):,} rows)",
                    data=wgs_csv,
                    file_name=f"wgs_{selected_release}_{wgs_data_type}.csv",
                    mime="text/csv",
                    key="wgs_download"
                )
                
            except Exception as e:
                st.error(f"Error loading WGS {wgs_data_type} data: {e}")
        
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.info("Please check that the GCS mounts are available and the data files exist.")


if __name__ == "__main__":
    main() 