import streamlit as st
import streamlit.components.v1 as components
from utils.hold_data import (
    release_select, 
    config_page
)
from utils.quality_control_utils import load_qc_data
from utils.config import AppConfig

config = AppConfig()

def main():
    
    config_page('Quality Control')
    release_select()

    funnel_plot, related_df, variant_plot = load_qc_data()

    st.title(f"Release {st.session_state['release_choice']} Metrics")

    st.header('QC Step 1: Sample-Level Filtering')
    with st.expander("Description", expanded=False):
        st.markdown(config.DESCRIPTIONS['qc'])

    left_col1, right_col1 = st.columns([1.5,2])

    with left_col1:
        st.header("**All Sample Filtering Counts**")
        components.html(funnel_plot, height=440)

    with right_col1:
        if not related_df.empty:
            st.header("**Relatedness per Ancestry**")
            related_df.set_index('Ancestry Category', inplace=True)
            st.dataframe(related_df, use_container_width = True, height = 423)

    st.header('QC Step 2: Variant-Level Filtering')
    with st.expander("Description", expanded=False):
        st.markdown(config.DESCRIPTIONS['variant'])

    components.html(variant_plot, height=600)

if __name__ == "__main__":
    main()
