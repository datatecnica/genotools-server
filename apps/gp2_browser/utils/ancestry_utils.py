import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.hold_data import (
    blob_as_csv,
    get_gcloud_bucket,
    admix_ancestry_select
)
from utils.config import AppConfig

config = AppConfig()


def plot_3d(labeled_df, color, symbol=None, x='PC1', y='PC2', z='PC3', title=None, x_range=None, y_range=None, z_range=None):
    """
    Create a 3D scatter plot using Plotly.

    Parameters:
        labeled_df (pd.DataFrame): Input dataframe containing PCA components and labels.
        color (str): Column name containing labels for ancestry.
        symbol (str, optional): Secondary label (e.g., predicted vs reference ancestry).
        x (str, optional): Column name of the x-dimension. Defaults to 'PC1'.
        y (str, optional): Column name of the y-dimension. Defaults to 'PC2'.
        z (str, optional): Column name of the z-dimension. Defaults to 'PC3'.
        title (str, optional): Plot title.
        x_range (list of float, optional): Range for x-axis [min, max].
        y_range (list of float, optional): Range for y-axis [min, max].
        z_range (list of float, optional): Range for z-axis [min, max].
    """
    fig = px.scatter_3d(
        labeled_df,
        x=x,
        y=y,
        z=z,
        color=color,
        symbol=symbol,
        title=title,
        color_discrete_map=config.ANCESTRY_COLOR_MAP,
        color_discrete_sequence=px.colors.qualitative.Bold,
        range_x=x_range,
        range_y=y_range,
        range_z=z_range,
        hover_name="IID",
        height=700
    )
    fig.update_traces(marker={'size': 3})
    return fig


def plot_pie(df, proportion_label='Proportion'):
    """
    Create an interactive pie chart using Plotly.

    Parameters:
        df (pd.DataFrame): Dataframe with columns ['Ancestry Category', 'Proportion'].
    """
    pie_chart = px.pie(
        df,
        names='Ancestry Category',
        values=proportion_label,
        color='Ancestry Category',
        color_discrete_map=config.ANCESTRY_COLOR_MAP
    )
    pie_chart.update_layout(showlegend=True, width=500, height=500)

    return pie_chart


def render_tab_pca(pca_folder, gp2_data_bucket):
    """
    Render the PCA tab in the Streamlit interface.

    Parameters:
        pca_folder (str): Path to the folder containing PCA data.
        gp2_data_bucket (google.cloud.storage.bucket.Bucket): GCloud bucket object.
        master_key (pd.DataFrame): Master key dataframe.
    """
    ref_pca = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/ref_pca_plot.csv', sep=',')
    proj_pca = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/proj_pca_plot.csv', sep=',')
    proj_labels = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/anc_summary.csv', sep=',')
    total_pca = pd.concat([ref_pca, proj_pca], axis=0)

    pca_col1, pca_col2 = st.columns([1.75, 3], vertical_alignment='center')

    with pca_col1:
        st.markdown(
            f'### Reference Panel vs. Release {st.session_state["release_choice"]} PCA')
        with st.expander("Description"):
            st.write(config.DESCRIPTIONS['pca1'])

        proj_labels['Select'] = False
        select_ancestry = st.data_editor(
            proj_labels, hide_index=True, use_container_width=True, height=423)
        selection_list = select_ancestry.loc[select_ancestry['Select']
                                             == True]['Predicted Ancestry']

    with pca_col2:
        if not selection_list.empty:
            selected_pca = proj_pca.copy()
            selected_pca.drop(
                selected_pca[~selected_pca['Predicted Ancestry'].isin(
                    selection_list)].index,
                inplace=True
            )
            total_pca_selected = pd.concat([ref_pca, selected_pca], axis=0)
            fig = plot_3d(total_pca_selected, 'label')
            st.plotly_chart(fig)
        else:
            fig = plot_3d(total_pca, 'label')
            st.plotly_chart(fig)


def plot_confusion_matrix(confusion_matrix):
    """
    Plot the given confusion matrix as percentages rather than raw counts,
    using a color scale that looks good in both light and dark mode.

    Parameters:
        confusion_matrix (pd.DataFrame): Confusion matrix with reference ancestry as rows
                                         and predicted ancestry as columns.

    Returns:
        fig (plotly.graph_objs._figure.Figure): The Plotly figure object.
    """
    # Convert raw counts to row-based percentages
    confusion_matrix_percent = confusion_matrix.div(
        confusion_matrix.sum(axis=1), axis=0) * 100

    fig = px.imshow(
        # Round to one decimal place for cleaner display
        confusion_matrix_percent.round(1),
        labels=dict(x="Predicted Ancestry",
                    y="Reference Panel Ancestry", color="Percentage"),
        text_auto=".1f",  # Format text with one decimal place
        color_continuous_scale='Mint'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    fig.update_yaxes(title_font_color="black", tickfont=dict(color='black'))
    fig.update_xaxes(title_font_color="black", tickfont=dict(color='black'))

    return fig


def render_tab_pred_stats(pca_folder, gp2_data_bucket):
    """
    Render the Model Performance tab containing confusion matrix and performance metrics.

    Parameters:
        pca_folder (str): Path to the folder containing PCA data.
        gp2_data_bucket (google.cloud.storage.bucket.Bucket): GCloud bucket object.
    """
    st.markdown('## **Model Accuracy**')
    confusion_matrix = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/confusion_matrix.csv', sep=',')
    model_metrics = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/model_metrics.csv', sep=',')

    confusion_matrix.set_index(confusion_matrix.columns, inplace=True)
    metrics = model_metrics.columns.to_list()

    heatmap1, heatmap2 = st.columns([2, 1])
    with heatmap1:
        st.markdown('### Confusion Matrix')
        fig = plot_confusion_matrix(confusion_matrix)
        st.plotly_chart(fig)

    with heatmap2:
        st.markdown('### Test Set Performance')
        st.metric(
            metrics[0], f"{model_metrics.iloc[0, 0]} \u00B1 {model_metrics.iloc[0, 1]}")
        st.metric(metrics[2], f"{model_metrics.iloc[0, 2]}")
        st.metric(metrics[3], f"{model_metrics.iloc[0, 3]}")


def render_tab_pie(pca_folder, gp2_data_bucket):
    """
    Render the Ancestry Distribution tab with reference and predicted pie charts.

    Parameters:
        pca_folder (str): Path to the folder containing PCA data.
        gp2_data_bucket (google.cloud.storage.bucket.Bucket): GCloud bucket object.
    """
    pie1, _, pie3 = st.columns([2, 1, 2])
    pie_table = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/pie_table.csv', sep=',')

    with pie1:
        st.markdown('### **Reference Panel Ancestry**')
        ref_pie = plot_pie(pie_table, proportion_label='Ref Panel Proportion')
        st.plotly_chart(ref_pie)

    with pie3:
        st.markdown(
            f'### Release {st.session_state["release_choice"]} Predicted Ancestry')
        pred_pie = plot_pie(pie_table, proportion_label='Predicted Proportion')
        st.plotly_chart(pred_pie)

    st.dataframe(
        pie_table[['Ancestry Category', 'Ref Panel Counts', 'Predicted Counts']],
        hide_index=True,
        use_container_width=True
    )


def render_tab_admix(pca_folder, gp2_data_bucket):
    """
    Render the Admixture Populations tab.

    Pulls admixture data from a known GCS location and displays
    the reference panel admixture table and plots.
    """
    frontend_bucket_name = 'gt_app_utils'
    frontend_bucket = config.FRONTEND_BUCKET_NAME+"/" #get_gcloud_bucket(frontend_bucket_name)

    st.markdown('## **Reference Panel Admixture Populations**')
    with st.expander("Description"):
        st.write(config.DESCRIPTIONS['admixture'])

    ref_admix = blob_as_csv(frontend_bucket, 'ref_panel_admixture.txt')
    admix_plot_blob = frontend_bucket+'refpanel_admix.png' #.get_blob('refpanel_admix.png')
    admix_plot = admix_plot_blob #admix_plot_blob.download_as_bytes()
    st.image(admix_plot)

    proj_labels = blob_as_csv(
        gp2_data_bucket, f'{pca_folder}/anc_summary.csv', sep=',')
    admix_ancestry_select(proj_labels)
    admix_ancestry_choice = st.session_state['admix_ancestry_choice']

    if admix_ancestry_choice != 'All':
        ref_admix = ref_admix[ref_admix['ancestry'] == admix_ancestry_choice]

    st.dataframe(ref_admix, hide_index=True, use_container_width=True)
