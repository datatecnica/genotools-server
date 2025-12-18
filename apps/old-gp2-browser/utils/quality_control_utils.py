import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from utils.hold_data import (
    blob_as_csv, 
    blob_as_html, 
    get_gcloud_bucket
)

def load_qc_data():
    gp2_data_bucket = get_gcloud_bucket('genotools-server')

    qc_metrics_path = f"cohort_browser/nba/release{st.session_state['release_choice']}"
    related_df = blob_as_csv(gp2_data_bucket, f'{qc_metrics_path}/related_plot.csv', sep=',')
    funnel_plot =  blob_as_html(gp2_data_bucket, f'{qc_metrics_path}/funnel_plot.html')
    variant_plot =  blob_as_html(gp2_data_bucket, f'{qc_metrics_path}/variant_plot.html')
 
    return funnel_plot, related_df, variant_plot

def relatedness_plot(relatedness_df):
    relatedness_plot = go.Figure(
        data=[
            go.Bar(
                y=relatedness_df.label, 
                x=-relatedness_df['duplicated_count'], 
                orientation='h', 
                name="Duplicated", 
                marker_color="#D55E00"
            ),
            go.Bar(
                y=relatedness_df.label, 
                x=relatedness_df['related_count'], 
                orientation='h', 
                name="Related", 
                marker_color="#0072B2"
            )
        ]
    )

    relatedness_plot.update_layout(
        barmode='relative', 
        height=450, 
        width=750, 
        autosize=False,
        margin=dict(l=0, r=0, t=10, b=70)
    )

    relatedness_plot.update_yaxes(
        ticktext=relatedness_df.label,
        tickvals=relatedness_df.label
    )

    return relatedness_plot