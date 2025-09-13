#!/usr/bin/env python3
"""
Streamlit app for viewing carriers pipeline results from GCS mounts.
Direct access to ~/gcs_mounts/genotools_server/precision_med/results/
Views release-level aggregated results (e.g., release10_NBA.parquet)
"""

import streamlit as st
import pandas as pd
import os
import glob
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
import json

# Page configuration
st.set_page_config(
    page_title="Carriers Pipeline Results Viewer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
RESULTS_BASE_PATH = os.path.expanduser("~/gcs_mounts/genotools_server/precision_med/results")
DATA_TYPES = ["NBA", "WGS", "IMPUTED"]

# Debug mode detection
DEBUG_MODE = "--debug" in sys.argv or os.getenv("STREAMLIT_DEBUG", "").lower() in ["true", "1", "yes"]

@st.cache_data
def discover_releases() -> List[str]:
    """Discover available releases in results directory."""
    if not os.path.exists(RESULTS_BASE_PATH):
        return []
    
    releases = []
    for item in os.listdir(RESULTS_BASE_PATH):
        if os.path.isdir(os.path.join(RESULTS_BASE_PATH, item)) and item.startswith('release'):
            releases.append(item)
    
    return sorted(releases, reverse=True)  # Most recent first

@st.cache_data
def discover_jobs(release: str) -> List[str]:
    """Discover available job names in a release directory."""
    release_path = os.path.join(RESULTS_BASE_PATH, release)
    if not os.path.exists(release_path):
        return []
    
    jobs = set()
    # Look for parquet files and extract job names
    for file in os.listdir(release_path):
        if file.endswith('.parquet'):
            # Format: {job_name}_{data_type}.parquet
            parts = file.replace('.parquet', '').split('_')
            if len(parts) >= 2 and parts[-1] in DATA_TYPES:
                job_name = '_'.join(parts[:-1])
                jobs.add(job_name)
        elif file.endswith('_pipeline_results.json'):
            # Format: {job_name}_pipeline_results.json
            job_name = file.replace('_pipeline_results.json', '')
            jobs.add(job_name)
    
    # Add release name as default if no specific jobs found
    if not jobs:
        jobs.add(release)
    elif release not in jobs:
        jobs.add(release)
    
    return sorted(list(jobs))

@st.cache_data
def load_pipeline_results(release: str, job_name: str) -> Optional[Dict]:
    """Load pipeline results JSON file."""
    file_path = os.path.join(RESULTS_BASE_PATH, release, f"{job_name}_pipeline_results.json")
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading pipeline results: {e}")
    return None

@st.cache_data
def load_variant_summary(release: str, job_name: str, data_type: str) -> Optional[pd.DataFrame]:
    """Load variant summary CSV for a specific data type."""
    file_path = os.path.join(RESULTS_BASE_PATH, release, f"{job_name}_{data_type}_variant_summary.csv")
    
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            st.error(f"Error loading {data_type} variant summary: {e}")
    return None

@st.cache_data 
def get_sample_counts(release: str, job_name: str) -> Dict[str, int]:
    """Get sample counts from parquet files."""
    sample_counts = {}
    total_samples = 0
    
    for data_type in DATA_TYPES:
        file_path = os.path.join(RESULTS_BASE_PATH, release, f"{job_name}_{data_type}.parquet")
        if os.path.exists(file_path):
            try:
                df = pd.read_parquet(file_path)
                # Get sample columns (exclude metadata columns)
                metadata_cols = {'variant_id', 'snp_list_id', 'chromosome', 'position', 
                                'counted_allele', 'alt_allele', 'harmonization_action', 
                                'data_type', 'ancestry', 'pgen_variant_id', 'snp_list_a1', 
                                'snp_list_a2', 'pgen_a1', 'pgen_a2', 'file_path', 'source_file',
                                'genotype_transform', 'original_a1', 'original_a2', '(C)M', 'COUNTED', 'ALT'}
                
                sample_cols = [col for col in df.columns if col not in metadata_cols]
                sample_count = len(sample_cols)
                sample_counts[data_type] = sample_count
                total_samples += sample_count
            except Exception as e:
                st.error(f"Error counting samples in {data_type}: {e}")
    
    sample_counts['TOTAL'] = total_samples
    return sample_counts

@st.cache_data
def load_genotype_data(release: str, job_name: str, data_type: str, sample_size: int = 100) -> Optional[pd.DataFrame]:
    """Load sample of genotype data from parquet file."""
    file_path = os.path.join(RESULTS_BASE_PATH, release, f"{job_name}_{data_type}.parquet")
    
    if os.path.exists(file_path):
        try:
            # Load just a sample for preview
            df = pd.read_parquet(file_path)
            if len(df) > sample_size:
                df = df.head(sample_size)
            return df
        except Exception as e:
            st.error(f"Error loading {data_type} genotype data: {e}")
    return None

def get_file_info(release: str, job_name: str) -> Dict[str, Dict[str, any]]:
    """Get information about available files for a release."""
    base_path = os.path.join(RESULTS_BASE_PATH, release)
    file_info = {}
    
    for data_type in DATA_TYPES:
        files = {}
        
        # Check for parquet and variant summary files only
        for ext, desc in [('.parquet', 'Genotype Data'), ('_variant_summary.csv', 'Variant Summary')]:
            if ext == '_variant_summary.csv':
                file_path = os.path.join(base_path, f"{job_name}_{data_type}{ext}")
            else:
                file_path = os.path.join(base_path, f"{job_name}_{data_type}{ext}")
            
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                files[desc] = {
                    'path': file_path,
                    'size_mb': round(stat.st_size / 1024 / 1024, 1),
                    'exists': True
                }
        
        if files:  # Only add data type if files exist
            file_info[data_type] = files
    
    return file_info

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🧬 Carriers Pipeline Results Viewer")
    st.markdown("Browse and analyze results from the precision medicine carriers pipeline.")
    
    # Sidebar navigation
    st.sidebar.header("📁 Data Selection")
    
    # Release selection
    releases = discover_releases()
    if not releases:
        st.error(f"No releases found in {RESULTS_BASE_PATH}")
        st.info("Make sure the GCS mount is accessible and contains pipeline results.")
        return
    
    selected_release = st.sidebar.selectbox("Select Release", releases)
    
    # Job selection (only show in debug mode)
    if DEBUG_MODE:
        jobs = discover_jobs(selected_release)
        if len(jobs) > 1:
            selected_job = st.sidebar.selectbox("Select Job", jobs, 
                                              help="Choose pipeline run to view. 'release10' is the main aggregated results.")
        else:
            selected_job = jobs[0] if jobs else selected_release
        
        # Show debug indicator
        st.sidebar.info("🔧 Debug Mode: Job selection enabled")
    else:
        # Production mode: always use release name as job name
        selected_job = selected_release
    
    # Get file information
    file_info = get_file_info(selected_release, selected_job)
    available_data_types = list(file_info.keys())
    
    if not available_data_types:
        st.error(f"No data files found for {selected_job} in {selected_release}")
        return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧬 Variant Browser", "📈 Statistics", "💾 File Downloads"])
    
    with tab1:
        st.header("Release Overview")
        
        # Show info if viewing test results
        if selected_job != selected_release:
            if "test_multiple_probes" in selected_job:
                st.info("🔬 **Viewing Test Results: Multiple Probe Detection Fix**\n\nThis shows results from testing the fix for NBA variants with multiple probes at the same genomic position. You should see multiple `variant_id` entries for the same `snp_list_id` when multiple probes exist.")
            else:
                st.info(f"📊 **Viewing Job Results: {selected_job}**")
        
        # Load pipeline results and sample counts
        pipeline_results = load_pipeline_results(selected_release, selected_job)
        sample_counts = get_sample_counts(selected_release, selected_job)
        
        # Display key metrics
        if pipeline_results and 'summary' in pipeline_results:
            summary = pipeline_results['summary']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Release", selected_release)
            with col2:
                st.metric("Total Variants", f"{summary.get('total_variants', 'N/A'):,}")
            with col3:
                # Show actual sample count from data files
                total_samples = sample_counts.get('TOTAL', 0)
                st.metric("Total Samples", f"{total_samples:,}")
            with col4:
                success = pipeline_results.get('success', False)
                st.metric("Pipeline Status", "✅ Success" if success else "❌ Failed")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Release", selected_release)
            with col2:
                st.metric("Data Types Available", f"{len(available_data_types)}/{len(DATA_TYPES)}")
        
        # File information table
        st.subheader("Available Files")
        file_rows = []
        for data_type, files in file_info.items():
            for file_desc, info in files.items():
                file_rows.append({
                    "Data Type": data_type,
                    "File Type": file_desc,
                    "Size (MB)": info['size_mb'],
                    "Path": os.path.basename(info['path'])
                })
        
        if file_rows:
            files_df = pd.DataFrame(file_rows)
            st.dataframe(files_df, width='stretch')
        
        # Show sample count breakdown by data type
        if sample_counts:
            st.subheader("Sample Counts by Data Type")
            sample_breakdown = []
            for data_type in DATA_TYPES:
                if data_type in sample_counts:
                    sample_breakdown.append({
                        "Data Type": data_type,
                        "Sample Count": f"{sample_counts[data_type]:,}"
                    })
            
            if sample_breakdown:
                breakdown_df = pd.DataFrame(sample_breakdown)
                st.dataframe(breakdown_df, width='stretch')
        
        # Show pipeline execution info if available
        if pipeline_results:
            st.subheader("Pipeline Execution Details")
            col1, col2 = st.columns(2)
            with col1:
                if 'start_time' in pipeline_results:
                    start_time = pipeline_results['start_time'].replace('T', ' ').split('.')[0]
                    st.text(f"Start Time: {start_time}")
                if 'execution_time_seconds' in pipeline_results:
                    exec_time = pipeline_results['execution_time_seconds']
                    st.text(f"Execution Time: {exec_time:.1f} seconds")
            with col2:
                if 'summary' in pipeline_results and 'export_method' in pipeline_results['summary']:
                    method = pipeline_results['summary']['export_method'].replace('_', ' ').title()
                    st.text(f"Export Method: {method}")
                if 'errors' in pipeline_results and pipeline_results['errors']:
                    st.error(f"Errors: {len(pipeline_results['errors'])} found")
    
    with tab2:
        st.header("Variant Browser")
        
        # Multiple probes analysis for test results
        if "test_multiple_probes" in selected_job and "NBA" in available_data_types:
            st.subheader("🔬 Multiple Probes Analysis")
            
            # Load NBA genotype data to check for multiple probes
            nba_genotype_data = load_genotype_data(selected_release, selected_job, "NBA", sample_size=1000)
            if nba_genotype_data is not None and not nba_genotype_data.empty:
                # Count SNPs with multiple variant_ids
                snp_counts = nba_genotype_data['snp_list_id'].value_counts()
                multi_probe_snps = snp_counts[snp_counts > 1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total NBA Variants", len(nba_genotype_data))
                with col2:
                    st.metric("Unique SNPs", len(snp_counts))
                with col3:
                    st.metric("SNPs with Multiple Probes", len(multi_probe_snps))
                
                if len(multi_probe_snps) > 0:
                    st.success(f"✅ **Multiple probe detection working!** Found {len(multi_probe_snps)} SNPs with multiple probes.")
                    
                    # Show examples
                    st.write("**Examples of SNPs with multiple probes:**")
                    examples_data = []
                    for snp_id, count in multi_probe_snps.head(10).items():
                        snp_variants = nba_genotype_data[nba_genotype_data['snp_list_id'] == snp_id]
                        variant_ids = snp_variants['variant_id'].tolist()
                        examples_data.append({
                            'SNP Name': snp_id,
                            'Probe Count': count,
                            'PVAR Variant IDs': ', '.join(variant_ids)
                        })
                    
                    examples_df = pd.DataFrame(examples_data)
                    st.dataframe(examples_df, width='stretch')
                else:
                    st.warning("⚠️ No multiple probes detected. This could indicate:")
                    st.write("- No variants with multiple probes in this dataset")
                    st.write("- Allele mismatches preventing harmonization")
                    st.write("- Other filtering issues")
            
            st.divider()
        
        # Data type selection for variant browser
        selected_data_type = st.selectbox("Select Data Type", available_data_types)
        
        # Create sub-tabs for variant summary and genotype data
        variant_tab1, variant_tab2 = st.tabs(["📋 Variant Summary", "🧬 Genotype Data"])
        
        with variant_tab1:
            # Load variant summary
            variant_summary = load_variant_summary(selected_release, selected_job, selected_data_type)
            
            if variant_summary is not None:
                st.subheader(f"{selected_data_type} Variant Summary")
                
                # Add filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'chromosome' in variant_summary.columns:
                        chroms = ['All'] + sorted(variant_summary['chromosome'].unique().astype(str))
                        selected_chrom = st.selectbox("Chromosome", chroms)
                        if selected_chrom != 'All':
                            variant_summary = variant_summary[variant_summary['chromosome'].astype(str) == selected_chrom]
                
                with col2:
                    if 'harmonization_action' in variant_summary.columns:
                        actions = ['All'] + list(variant_summary['harmonization_action'].unique())
                        selected_action = st.selectbox("Harmonization Action", actions)
                        if selected_action != 'All':
                            variant_summary = variant_summary[variant_summary['harmonization_action'] == selected_action]
                
                with col3:
                    # Filter by source file (ancestry)
                    if 'source_file' in variant_summary.columns:
                        # Extract ancestry from source file path
                        ancestries = variant_summary['source_file'].apply(
                            lambda x: x.split('/')[-1].split('_')[0] if pd.notna(x) else 'Unknown'
                        ).unique()
                        ancestries = ['All'] + sorted([a for a in ancestries if a != 'Unknown'])
                        if len(ancestries) > 1:
                            selected_ancestry = st.selectbox("Ancestry", ancestries)
                            if selected_ancestry != 'All':
                                variant_summary = variant_summary[variant_summary['source_file'].str.contains(f'/{selected_ancestry}_', na=False)]
                
                # Display filtered data
                st.dataframe(variant_summary, width='stretch')
                st.caption(f"Showing {len(variant_summary)} variants")
                
                # Download filtered data
                csv = variant_summary.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name=f"{selected_job}_{selected_data_type}_filtered.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No variant summary found for {selected_data_type}")
        
        with variant_tab2:
            # Load genotype data from parquet
            genotype_data = load_genotype_data(selected_release, selected_job, selected_data_type)
            
            if genotype_data is not None:
                st.subheader(f"{selected_data_type} Genotype Data (Sample)")
                
                # Add explanation of genotype data
                st.info("📊 **Genotype values represent counts of the pathogenic allele (counted_allele).** "
                       "0 = no pathogenic alleles, 1 = heterozygous carrier, 2 = homozygous carrier.")
                
                # Controls for data display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Number of variants to show
                    max_variants = min(len(genotype_data), 1000)
                    num_variants = st.slider("Number of variants to display", 
                                           min_value=1, max_value=max_variants, value=min(100, max_variants))
                
                with col2:
                    # Number of samples to show
                    # Get sample columns (exclude metadata columns)
                    # Note: Using only new format columns (counted_allele, alt_allele) since they're now consistent with COUNTED/ALT
                    metadata_cols = {'variant_id', 'snp_list_id', 'chromosome', 'position', 
                                   'counted_allele', 'alt_allele', 'harmonization_action', 
                                   'data_type', 'ancestry', 'pgen_variant_id', 'snp_list_a1', 
                                   'snp_list_a2', 'pgen_a1', 'pgen_a2', 'file_path', 'source_file',
                                   'genotype_transform', 'original_a1', 'original_a2', '(C)M', 
                                   'COUNTED', 'ALT'}  # Hide redundant old format columns
                    
                    sample_cols = [col for col in genotype_data.columns if col not in metadata_cols]
                    max_samples = min(len(sample_cols), 100)
                    if max_samples > 0:
                        num_samples = st.slider("Number of samples to display", 
                                               min_value=1, max_value=max_samples, value=min(20, max_samples))
                    else:
                        num_samples = 0
                
                with col3:
                    # Show metadata columns checkbox
                    show_metadata = st.checkbox("Show metadata columns", value=True)
                
                # Prepare display data
                display_data = genotype_data.head(num_variants).copy()
                
                if not show_metadata:
                    # Show only essential variant info + genotype columns
                    if sample_cols and num_samples > 0:
                        cols_to_show = ['variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele'] + sample_cols[:num_samples]
                        display_data = display_data[[col for col in cols_to_show if col in display_data.columns]]
                else:
                    # Show curated metadata + sample subset (avoid redundant columns)
                    if sample_cols and num_samples > 0:
                        # Show only the most useful metadata columns 
                        useful_metadata = ['variant_id', 'chromosome', 'position', 'counted_allele', 'alt_allele', 
                                         'harmonization_action', 'snp_list_a1', 'snp_list_a2', 'pgen_a1', 'pgen_a2']
                        metadata_to_show = [col for col in useful_metadata if col in display_data.columns]
                        cols_to_show = metadata_to_show + sample_cols[:num_samples]
                        display_data = display_data[cols_to_show]
                
                # Display the data
                st.dataframe(display_data, width='stretch')
                st.caption(f"Showing {len(display_data)} variants × {len(display_data.columns)} columns "
                          f"(Total in file: {len(genotype_data)} variants × {len(sample_cols)} samples)")
                
                # Summary statistics
                if sample_cols:
                    st.subheader("Genotype Summary Statistics")
                    
                    # Calculate summary stats for displayed samples
                    sample_data = genotype_data[sample_cols[:num_samples]]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # Count genotype frequencies across all displayed variants/samples
                        all_genotypes = sample_data.values.flatten()
                        unique_vals, counts = pd.Series(all_genotypes).value_counts().items(), pd.Series(all_genotypes).value_counts().values
                        
                        freq_data = []
                        for val, count in zip(unique_vals, counts):
                            if pd.notna(val):
                                freq_data.append({"Genotype": val, "Count": count, "Frequency": f"{count/len(all_genotypes)*100:.1f}%"})
                        
                        if freq_data:
                            st.dataframe(pd.DataFrame(freq_data), width='stretch')
                    
                    with col2:
                        # Show variant with highest carrier frequency
                        carrier_rates = []
                        for idx, row in display_data.iterrows():
                            variant_id = row.get('variant_id', f"Row_{idx}")
                            genotypes = row[sample_cols[:num_samples]]
                            carrier_count = (genotypes > 0).sum() if len(genotypes) > 0 else 0
                            carrier_rate = carrier_count / len(genotypes) * 100 if len(genotypes) > 0 else 0
                            carrier_rates.append({"Variant": variant_id, "Carrier Rate": f"{carrier_rate:.1f}%", "Carriers": carrier_count})
                        
                        if carrier_rates:
                            carrier_df = pd.DataFrame(carrier_rates).sort_values("Carriers", ascending=False).head(10)
                            st.write("**Top Variants by Carrier Count:**")
                            st.dataframe(carrier_df, width='stretch')
                    
                    with col3:
                        # Show samples with highest carrier burden
                        sample_burdens = []
                        for sample in sample_cols[:num_samples]:
                            genotypes = display_data[sample]
                            carrier_count = (genotypes > 0).sum()
                            sample_burdens.append({"Sample": sample, "Carrier Count": carrier_count})
                        
                        if sample_burdens:
                            burden_df = pd.DataFrame(sample_burdens).sort_values("Carrier Count", ascending=False).head(10)
                            st.write("**Top Samples by Carrier Burden:**")
                            st.dataframe(burden_df, width='stretch')
                
                # Download options
                st.subheader("Download Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download displayed data
                    csv_data = display_data.to_csv(index=False)
                    st.download_button(
                        label="Download Displayed Data as CSV",
                        data=csv_data,
                        file_name=f"{selected_job}_{selected_data_type}_genotype_sample.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Info about full file download
                    full_size_mb = file_info[selected_data_type]['Genotype Data']['size_mb']
                    st.info(f"Full parquet file: {full_size_mb} MB\nUse File Downloads tab for complete data")
            
            else:
                st.warning(f"No genotype data found for {selected_data_type}")
                st.info("Genotype data should be available as parquet files in the results directory.")
    
    with tab3:
        st.header("Statistics & Visualizations")
        
        # Combine all variant summaries for statistics
        all_variants = []
        for data_type in available_data_types:
            df = load_variant_summary(selected_release, selected_job, data_type)
            if df is not None:
                df['data_type'] = data_type
                all_variants.append(df)
        
        if all_variants:
            combined_df = pd.concat(all_variants, ignore_index=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Variants by data type
                st.subheader("Variants by Data Type")
                type_counts = combined_df['data_type'].value_counts()
                fig = px.bar(x=type_counts.index, y=type_counts.values, 
                           labels={'x': 'Data Type', 'y': 'Number of Variants'})
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                # Harmonization actions
                if 'harmonization_action' in combined_df.columns:
                    st.subheader("Harmonization Actions")
                    action_counts = combined_df['harmonization_action'].value_counts()
                    fig = px.pie(values=action_counts.values, names=action_counts.index)
                    st.plotly_chart(fig, width='stretch')
            
            # Chromosome distribution
            if 'chromosome' in combined_df.columns:
                st.subheader("Variants by Chromosome")
                chrom_counts = combined_df['chromosome'].value_counts().sort_index()
                fig = px.bar(x=chrom_counts.index.astype(str), y=chrom_counts.values,
                           labels={'x': 'Chromosome', 'y': 'Number of Variants'})
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No data available for statistics")
    
    with tab4:
        st.header("File Downloads")
        
        st.markdown(f"Direct links to download {selected_release} pipeline output files:")
        
        for data_type, files in file_info.items():
            st.subheader(f"{data_type} Files")
            
            for file_desc, info in files.items():
                col1, col2, col3 = st.columns([3, 1, 2])
                with col1:
                    st.text(file_desc)
                with col2:
                    st.text(f"{info['size_mb']} MB")
                with col3:
                    # Create download button for each file
                    try:
                        with open(info['path'], 'rb') as f:
                            st.download_button(
                                label="Download",
                                data=f.read(),
                                file_name=os.path.basename(info['path']),
                                key=f"download_{data_type}_{file_desc}".replace(' ', '_')
                            )
                    except Exception as e:
                        st.error(f"Cannot access file: {e}")

if __name__ == "__main__":
    main()