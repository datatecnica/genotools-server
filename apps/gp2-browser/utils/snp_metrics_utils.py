import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from utils.hold_data import (
    blob_as_csv
)

def load_metrics_data(bucket, ancestry_choice, chr_choice):
    metrics_blob_name = f"gp2_snp_metrics/{ancestry_choice}/chr{chr_choice}_metrics.csv"
    maf_blob_name = f"gp2_snp_metrics/{ancestry_choice}/{ancestry_choice}_maf.afreq"
    full_maf_blob_name = "gp2_snp_metrics/full_maf.afreq"

    if f"{ancestry_choice}_{chr_choice}" not in st.session_state:
        metrics = blob_as_csv(bucket, metrics_blob_name, sep=',')
        st.session_state[f"{ancestry_choice}_{chr_choice}"] = metrics
    else:
        metrics = st.session_state[f"{ancestry_choice}_{chr_choice}"]

    if f"{ancestry_choice}_maf" not in st.session_state:
        maf = blob_as_csv(bucket, maf_blob_name, sep='\t')
        st.session_state[f"{ancestry_choice}_maf"] = maf
    else:
        maf = st.session_state[f"{ancestry_choice}_maf"]

    if "full_maf" not in st.session_state:
        full_maf = blob_as_csv(bucket, full_maf_blob_name, sep='\t')
        st.session_state["full_maf"] = full_maf
    else:
        full_maf = st.session_state["full_maf"]

    return metrics, maf, full_maf

def plot_clusters(df, x_col='theta', y_col='r', gtype_col='gt', title='SNP Plot'):
    d3 = px.colors.qualitative.D3
    cmap = {'AA': d3[0], 'AB': d3[1], 'BB': d3[2], 'NC': d3[3]}
    smap = {'Control': 'circle', 'PD': 'diamond-open-dot'}

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=gtype_col,
        color_discrete_map=cmap,
        symbol='phenotype',
        symbol_map=smap,
        title=title,
        width=650,
        height=497,
        labels={'r': 'R', 'theta': 'Theta'}
    )
    fig.update_layout(margin=dict(r=76, t=63, b=75), legend_title_text='Genotype')
    return fig

def display_snp_metrics(metrics, maf, full_maf, ancestry_choice, snp_label):
    snp_df = metrics[metrics['snp_label'] == snp_label].reset_index(drop=True)

    cluster_plot = plot_clusters(snp_df, x_col='Theta', y_col='R', gtype_col='GT', title=snp_label)
    col1, col2 = st.columns([2.5, 1])

    with col1:
        st.plotly_chart(cluster_plot, use_container_width=True)

    with col2:
        st.metric("GenTrain Score", f"{snp_df['GenTrain_Score'].iloc[0]:.3f}")
        within_ancestry_maf = maf[maf['ID'] == snp_df['snpID'].iloc[0]]
        across_ancestry_maf = full_maf[full_maf['ID'] == snp_df['snpID'].iloc[0]]
        st.metric(f"Minor Allele Frequency within {ancestry_choice}", f"{within_ancestry_maf['ALT_FREQS'].iloc[0]:.3f}")
        st.metric("Minor Allele Frequency across ancestries", f"{across_ancestry_maf['ALT_FREQS'].iloc[0]:.3f}")

        for phenotype in ['Control', 'PD']:
            with st.expander(f"**{phenotype} Genotype Distribution**"):
                gt_counts = snp_df[snp_df['phenotype'] == phenotype]['GT'].value_counts().rename_axis('Genotype').reset_index(name='Counts')
                gt_rel_counts = snp_df[snp_df['phenotype'] == phenotype]['GT'].value_counts(normalize=True).rename_axis('Genotype').reset_index(name='Frequency')
                gt_counts = pd.concat([gt_counts, gt_rel_counts['Frequency']], axis=1)
                st.table(gt_counts)