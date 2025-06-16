import streamlit as st
import streamlit.components.v1 as components
from google.cloud import storage
import pandas as pd
from io import StringIO

def load_csv_from_gcs(bucket_name, blob_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    csv_data = blob.download_as_text()
    return pd.read_csv(StringIO(csv_data))

def main():
    st.title("Pre-generated SNP Plots Viewer")

    bucket_name = "gp2_working_eu"
    coords_path = "dan/gba1_snp_metrics/metrics_coords.csv"

    coords_df = load_csv_from_gcs(bucket_name, coords_path)
    variant_list = coords_df["variant_id"].unique().tolist()

    chosen_variant_id = st.selectbox(
        "Choose or search for a Variant ID",
        options=variant_list
    )

    if chosen_variant_id:
        gcs_file_path = f"dan/gba1_snp_metrics/plots/{chosen_variant_id}.html"

        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(gcs_file_path)
            html_content = blob.download_as_text()
            components.html(html_content, height=600, scrolling=True)
        except Exception as e:
            st.warning(f"Could not load HTML from GCS. Error: {e}")

if __name__ == "__main__":
    main()