import streamlit as st
from utils.hold_data import (
    blob_as_csv, 
    get_gcloud_bucket
)
from utils.config import AppConfig
config = AppConfig()

def load_rare_variant_data(bucket_name, file_path):
    """Load rare variant data from the specified bucket and file."""
    bucket = config.FRONTEND_BUCKET_NAME+"/" #get_gcloud_bucket(bucket_name)
    return blob_as_csv(bucket, file_path, sep=',')

def filter_rare_variant_data(rv_data):
    """Filter the rare variant data based on user selections in session state."""
    rv_data_filtered = rv_data

    if st.session_state.get('rv_cohort_choice'):
        rv_data_filtered = rv_data_filtered[rv_data_filtered['Study code'].isin(st.session_state['rv_cohort_choice'])]

    if st.session_state.get('method_choice'):
        rv_data_filtered = rv_data_filtered[rv_data_filtered['Methods'].isin(st.session_state['method_choice'])]

    if st.session_state.get('rv_gene_choice'):
        rv_data_filtered = rv_data_filtered[rv_data_filtered['Gene'].isin(st.session_state['rv_gene_choice'])]

    return rv_data_filtered.reset_index(drop=True)