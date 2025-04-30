import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.hold_data import (
    get_gcloud_bucket,
    release_select,
    config_page
)

from utils.ancestry_utils import (
    render_tab_pca,
    render_tab_admix,
    render_tab_pie,
    render_tab_pred_stats
)
from utils.config import AppConfig

config = AppConfig()

def main():
    """
    Main function to set up the Streamlit page and render all Ancestry-related tabs.
    """
    config_page('Ancestry')
    release_select()

    gp2_data_bucket = config.FRONTEND_BUCKET_NAME+"/" #get_gcloud_bucket('gt_app_utils') # used to be gp2tier2
    plot_folder = f"qc_metrics/release{st.session_state['release_choice']}"

    tab_pca, tab_pred_stats, tab_pie, tab_admix, tab_methods = st.tabs([
        "Ancestry Prediction",
        "Model Performance",
        "Ancestry Distribution",
        "Admixture Populations",
        "Method Description"
    ])

    with tab_pca:
        render_tab_pca(plot_folder, gp2_data_bucket)

    with tab_pred_stats:
        render_tab_pred_stats(plot_folder, gp2_data_bucket)

    with tab_pie:
        render_tab_pie(plot_folder, gp2_data_bucket)

    with tab_admix:
        render_tab_admix(plot_folder, gp2_data_bucket)

    with tab_methods:
        st.markdown(config.DESCRIPTIONS['ancestry_methods'])


if __name__ == "__main__":
    main()
