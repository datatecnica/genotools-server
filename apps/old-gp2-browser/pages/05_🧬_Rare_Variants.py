import pandas as pd
import streamlit as st
from utils.hold_data import (
    config_page, 
    rv_select
)
from utils.rare_variants_utils import (
    load_rare_variant_data,
    filter_rare_variant_data
)
from utils.config import AppConfig

config = AppConfig()

def main():
    """Main function for the GP2 Rare Variant Browser."""
    config_page("GP2 Rare Variant Browser")
    st.title("GP2 Rare Variant Browser")

    # Load data
    bucket_name = "gt_app_utils"
    file_path = "gp2_RV_browser_input.csv"
    rv_data = load_rare_variant_data(config.FRONTEND_BUCKET_NAME+"/", file_path)

    # User selections
    rv_select(rv_data)

    # Filter data based on selections
    rv_data_filtered = filter_rare_variant_data(rv_data)

    # Display the filtered data
    st.dataframe(rv_data_filtered, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
