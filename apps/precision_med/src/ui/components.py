"""
Reusable UI components for Streamlit app.
Functional components following the project plan.
"""
import streamlit as st
import pandas as pd
from typing import List, Dict, Any


def render_release_selector(available_releases: List[str], default_index: int = 0) -> str:
    """Render release selection dropdown.
    
    Args:
        available_releases: List of available releases
        default_index: Default selection index
        
    Returns:
        Selected release string
    """
    return st.selectbox(
        "Select Release:",
        available_releases,
        index=default_index,
        help="Choose the data release version to view"
    )


def render_data_type_selector() -> str:
    """Render data type selection dropdown.
    
    Returns:
        Selected data type string
    """
    return st.selectbox(
        "Select Data Type:",
        ["info", "int", "string"],
        format_func=lambda x: {
            "info": "📄 Variant Info (Metadata)",
            "int": "🔢 Integer Genotypes", 
            "string": "📝 String Genotypes"
        }[x],
        help="Choose the type of NBA data to view"
    )


def render_summary_metrics(stats: Dict[str, Any]) -> None:
    """Render summary statistics as metrics.
    
    Args:
        stats: Dictionary containing summary statistics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Variants", stats['num_variants'])
    with col2:
        st.metric("Samples", stats['num_samples'])
    with col3:
        st.metric("Ancestries", len(stats['ancestries']))
    with col4:
        st.metric("Loci", len(stats['loci']))


def render_dataframe_viewer(df: pd.DataFrame, data_type: str, release: str) -> None:
    """Render DataFrame viewer with controls.
    
    Args:
        df: DataFrame to display
        data_type: Type of data being displayed
        release: Current release version
    """
    # Display DataFrame info
    st.subheader(f"📊 {data_type.title()} Data")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    
    # Display sample of data
    st.subheader("🔍 Data Sample")
    
    # Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        num_rows = st.slider("Number of rows to display:", 5, 100, 10)
    with col2:
        show_all_cols = st.checkbox("Show all columns", value=False)
    
    # Display data (DataFrame should already be cleaned in repository)
    display_df = df.head(num_rows)
    if not show_all_cols and len(df.columns) > 10:
        st.info(f"Showing first 10 of {len(df.columns)} columns. Check 'Show all columns' to see more.")
        display_df = display_df.iloc[:, :10]
    
    st.dataframe(display_df, use_container_width=True)


def render_column_info(df: pd.DataFrame) -> None:
    """Render column information in an expandable section.
    
    Args:
        df: DataFrame to analyze
    """
    # Column information section removed per user request
    pass


def render_download_section(df: pd.DataFrame, data_type: str, release: str) -> None:
    """Render download section with CSV export.
    
    Args:
        df: DataFrame to export
        data_type: Type of data
        release: Current release version
    """
    st.subheader("💾 Download")
    csv = df.to_csv(index=False)
    st.download_button(
        label=f"Download {data_type} data as CSV",
        data=csv,
        file_name=f"nba_{release}_{data_type}.csv",
        mime="text/csv",
        help=f"Download the current {data_type} dataset as CSV file"
    )


def render_ancestry_breakdown(ancestries: List[str]) -> None:
    """Render ancestry breakdown display.
    
    Args:
        ancestries: List of ancestry codes
    """
    st.subheader("🌍 Ancestries")
    
    # Create a more readable display
    ancestry_mapping = {
        'AAC': 'African American Controls',
        'AFR': 'African',
        'AJ': 'Ashkenazi Jewish',
        'AMR': 'American',
        'CAH': 'Central Asian/Himalayan',
        'CAS': 'Central Asian Steppe',
        'EAS': 'East Asian',
        'EUR': 'European',
        'FIN': 'Finnish',
        'MDE': 'Middle Eastern',
        'SAS': 'South Asian'
    }
    
    cols = st.columns(min(len(ancestries), 4))
    for i, ancestry in enumerate(ancestries):
        with cols[i % 4]:
            full_name = ancestry_mapping.get(ancestry, ancestry)
            st.info(f"**{ancestry}**\n{full_name}")


def render_error_message(error: Exception, context: str = "") -> None:
    """Render standardized error message.
    
    Args:
        error: Exception that occurred
        context: Additional context for the error
    """
    st.error(f"Error {context}: {error}")
    
    with st.expander("🔍 Error Details"):
        st.code(str(error))
        if hasattr(error, '__traceback__'):
            import traceback
            st.code(''.join(traceback.format_tb(error.__traceback__))) 