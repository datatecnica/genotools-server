import streamlit as st
from utils.hold_data import (
    get_gcloud_bucket, 
    chr_ancestry_select, 
    config_page
)
from utils.snp_metrics_utils import (
    load_metrics_data,
    display_snp_metrics
)
from utils.config import AppConfig

config = AppConfig()

def main():
    config_page("SNP Metrics")
    st.title("GP2 SNP Metrics Browser")

    with st.expander("Description"):
        st.markdown(config.DESCRIPTIONS['snp_metrics'])

    snp_metrics_bucket = config.FRONTEND_BUCKET_NAME+"/" #get_gcloud_bucket("gt_app_utils")
    chr_ancestry_select()
    chr_choice = st.session_state['chr_choice']
    ancestry_choice = st.session_state['ancestry_choice']

    metrics, maf, full_maf = load_metrics_data(snp_metrics_bucket, ancestry_choice, chr_choice)

    num_snps = metrics['snpID'].nunique()
    num_samples = metrics['Sample_ID'].nunique()
    metric1, metric2 = st.columns([1, 1])
    metric1.metric(f"Number of SNPs on Chromosome {chr_choice} for {ancestry_choice}", f"{num_snps}")
    metric2.metric(f"Number of {ancestry_choice} samples with SNP metrics available", f"{num_samples}")

    if num_samples > 0:
        metrics['snp_label'] = metrics['snpID'] + ' (' + metrics['chromosome'].astype(str) + ':' + metrics['position'].astype(str) + ')'
        snp_options = ['Select SNP!'] + metrics['snp_label'].unique().tolist()

        snp_choice = st.selectbox("Select SNP", snp_options, key="snp_choice")

        if snp_choice != 'Select SNP!':
            display_snp_metrics(metrics, maf, full_maf, ancestry_choice, snp_choice)

if __name__ == "__main__":
    main()
