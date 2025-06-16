import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from utils.hold_data import (
    get_gcloud_bucket,  
    release_select, 
    config_page,
    get_master_key,
    filter_by_cohort,
    filter_by_ancestry,
    update_sex_labels
)
from utils.metadata_utils import (
    display_ancestry, 
    ancestry_pca,
    plot_age_distribution, 
    display_phenotype_counts, 
    display_pruned_samples, 
    display_related_samples,
    display_carriers
)
from utils.config import AppConfig

config = AppConfig()

def main():
    config_page('GP2 Release')
    release_select()

    gp2_data_bucket = config.FRONTEND_BUCKET_NAME+"/"

    # gp2_data_bucket = get_gcloud_bucket('gt_app_utils')
    master_key = get_master_key()
    master_key_cohort = filter_by_cohort(master_key)
    
    pruned_key = st.session_state['master_key']
    st.session_state['master_key'] = master_key_cohort
    st.title(f"{st.session_state['cohort_choice']} Metadata")

    master_key = filter_by_ancestry(master_key_cohort)
    master_key = update_sex_labels(master_key)

    tab_ancestry, tab_age, tab_qc, tab_carriers = st.tabs([
        "Ancestry",
        "Age",
        "Quality Control",
        "Carriers"
    ])

    with tab_ancestry:
        display_ancestry(master_key_cohort)
        st.markdown('----') 
        
        plot_title = f'{st.session_state["cohort_choice"]} PCA for {st.session_state["meta_ancestry_choice"]} Samples'
        if plot_title not in st.session_state:
            ancestry_pca(master_key, plot_title, gp2_data_bucket)
        
        st.markdown(f'#### {plot_title}')
        st.plotly_chart(st.session_state[plot_title])

    with tab_age:
        st.markdown('#### Stratify Age by:')
        stratify = st.selectbox("Stratify Age by:", options = ['None', 'Sex', 'Phenotype'], label_visibility="collapsed")

        plot1, plot2 = st.columns([1, 1.75], vertical_alignment = 'center')

        plot_age_distribution(master_key, stratify, plot2)
        display_phenotype_counts(master_key, plot1)

    with tab_qc:
        pruned1, pruned2 = st.columns([1, 1.75])
        pruned_key['prune_reason'] = pruned_key['prune_reason'].map(config.PRUNE_MAP)
        display_pruned_samples(pruned_key, pruned1)
        display_related_samples(pruned_key, pruned2)

    with tab_carriers:
        st.markdown('#### Genetic Variant Carrier Status Viewer')
        display_carriers(master_key, gp2_data_bucket)

if __name__ == "__main__":
    main()
