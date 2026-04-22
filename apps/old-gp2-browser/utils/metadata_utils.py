import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass

from utils.hold_data import blob_as_csv
from utils.ancestry_utils import plot_pie, plot_3d
from utils.quality_control_utils import relatedness_plot

from utils.carriers_utils import (
    CarriersConfig,
    CarrierDataProcessor
)


def plot_age_distribution(master_key, stratify, plot2):
    master_key_age = master_key[master_key['age'].notnull()]
    if master_key_age.empty:
        plot2.info('No age values available for the selected cohort.')
        return

    if stratify == 'None':
        fig = px.histogram(master_key_age, x='age', nbins=25,
                           color_discrete_sequence=["#332288"])
        fig.update_layout(title_text=f'<b>Age Distribution<b>')
    elif stratify == 'Sex':
        fig = px.histogram(
            master_key_age,
            x='age',
            color='sex',
            nbins=25,
            color_discrete_map={'Male': "#332288", 'Female': "#CC6677"})
        fig.update_layout(title_text=f'<b>Age Distribution by Sex<b>')
    elif stratify == 'Phenotype':
        fig = px.histogram(
            master_key_age,
            x='age',
            color='pheno',
            nbins=25,
            color_discrete_map={
                'Control': "#332288",
                'PD': "#CC6677",
                'Other': "#117733",
                'Not Reported': "#D55E00"
            }
        )
        fig.update_layout(title_text=f'<b>Age Distribution by Phenotype<b>')

    plot2.plotly_chart(fig)


def display_phenotype_counts(master_key, plot1):
    master_key.rename(columns={'pheno': 'Phenotype'}, inplace=True)
    male_pheno = master_key.loc[master_key['sex'] == 'Male', 'Phenotype']
    female_pheno = master_key.loc[master_key['sex'] == 'Female', 'Phenotype']

    combined_counts = pd.DataFrame({
        'Male': male_pheno.value_counts(),
        'Female': female_pheno.value_counts()
    })

    combined_counts['Total'] = combined_counts.sum(axis=1)
    combined_counts.fillna(0, inplace=True)
    combined_counts = combined_counts.astype(int)
    combined_counts.sort_values(by='Total', ascending=False, inplace=True)

    plot1.dataframe(combined_counts, use_container_width=True)


def display_ancestry(full_cohort):
    anc1, anc2 = st.columns(2, vertical_alignment='center')
    anc_choice = st.session_state["meta_ancestry_choice"]

    anc_df = full_cohort.label.value_counts().reset_index()
    anc_df['Proportion'] = anc_df['count'] / anc_df['count'].sum()

    if anc_choice != 'All':
        percent_anc = anc_df[anc_df.label ==
                             anc_choice]['Proportion'].iloc[0] * 100
        anc1.metric(f"Count of {anc_choice} Samples in this Cohort",
                    anc_df[anc_df.label == anc_choice]['count'].iloc[0])
        anc2.metric(
            f"Percent of {anc_choice} Samples in this Cohort", f"{percent_anc:.2f}%")
    else:
        anc_df.rename(
            columns={'label': 'Ancestry Category', 'count': 'Count'}, inplace=True)
        release_pie = plot_pie(anc_df)
        anc2.plotly_chart(release_pie)
        anc_df.set_index('Ancestry Category', inplace=True)
        anc1.markdown(
            f'#### {st.session_state["cohort_choice"]} Ancestry Breakdown')
        anc1.dataframe(anc_df['Count'], use_container_width=True)


def ancestry_pca(master_key, plot_title, gp2_data_bucket):
    proj_samples = blob_as_csv(
        gp2_data_bucket, f"qc_metrics/release{st.session_state['release_choice']}/proj_pca_plot.csv", sep=',')
    display_samples = proj_samples[proj_samples.IID.isin(
        master_key.IID2)]  # eventually update with new dataframe
    st.session_state[plot_title] = plot_3d(
        display_samples, 'Predicted Ancestry')


def display_pruned_samples(pruned_key, pruned1):
    anc_choice = st.session_state["meta_ancestry_choice"]
    if anc_choice != "All":
        pruned_key = pruned_key[pruned_key["label"] == anc_choice]

    pruned_steps = pruned_key.prune_reason.value_counts().reset_index()
    pruned_steps.rename(
        columns={'prune_reason': 'Pruned Reason', 'count': 'Count'}, inplace=True)
    pruned_steps.set_index('Pruned Reason', inplace=True)

    pruned1.markdown("#####")
    pruned1.markdown("##### Sample-Level Release Prep")
    pruned1.dataframe(pruned_steps, use_container_width=True)


def display_related_samples(pruned_key, pruned2):
    related_samples = pruned_key[pruned_key.related == 1][['label', 'related']]
    related_samples['related_count'] = related_samples.groupby(['label'])[
        'related'].transform('sum')

    duplicated_samples = pruned_key[pruned_key.prune_reason ==
                                    'Duplicated Prune']
    duplicated_samples['duplicated'] = 1
    duplicated_samples['duplicated_count'] = duplicated_samples.groupby(['label'])[
        'duplicated'].transform('sum')

    relatedness_df = related_samples[['label', 'related_count']].merge(
        duplicated_samples[['label', 'duplicated_count']], on='label', how='left')
    relatedness_df.drop_duplicates(inplace=True)

    if len(relatedness_df) == 0:
        pruned2.markdown("#####")
        pruned2.markdown("##### Related Samples per Ancestry")
        pruned2.metric(f'Total Related Samples', 0)
    elif len(relatedness_df.label) > 3:
        pruned2.markdown("##### Relatedness per Ancestry")
        related_plot = relatedness_plot(relatedness_df)
        pruned2.plotly_chart(related_plot, use_container_width=True)
    else:
        pruned2.markdown("#####")
        pruned2.markdown("##### Related Samples per Ancestry")
        for i in range(len(relatedness_df)):
            pruned2.metric(f'{relatedness_df.iloc[i, 0]} Related Samples', int(
                relatedness_df.iloc[i, 1]))


def display_carriers(master_key, gp2_data_bucket):
    # load carriers data - subset by samples in master key
    carriers_path = f'carriers_data/release{st.session_state["release_choice"]}_carriers/release{st.session_state["release_choice"]}_carriers_string.csv'
    carriers_df_in = blob_as_csv(
        gp2_data_bucket, carriers_path, sep=',')
    carriers_df = carriers_df_in.loc[carriers_df_in.IID.isin(master_key.IID)]
    snp_cols = [
        col for col in carriers_df.columns if col not in CarriersConfig.NON_VARIANT_COLUMNS]
    filtered_carriers_df = carriers_df[carriers_df[snp_cols].apply(
        lambda row: any(val not in ["WT/WT", "./."] for val in row),
        axis=1
    )]

    processor = CarrierDataProcessor(filtered_carriers_df)

    # ui controls
    selected_ancestry = st.session_state['meta_ancestry_choice']

    show_all_carriers = st.checkbox("Show All Carriers")
    zygosity_filter = st.radio("Filter by Zygosity", [
                               'All', 'Homozygous', 'Heterozygous'])

    selected_variants = processor.variants if show_all_carriers else st.multiselect(
        "Choose variants to display",
        processor.variants
    )

    # display results
    if selected_variants:
        status_df = processor.process_carriers(
            selected_variants, selected_ancestry, zygosity_filter)

        if status_df is not None:
            st.header(f"Carriers Found: {len(status_df)}")
            st.dataframe(status_df)

            csv_filtered = status_df.to_csv(index=False)
            st.download_button(
                label="Download filtered dataset",
                data=csv_filtered,
                file_name=f"carriers_{selected_ancestry.lower()}_{zygosity_filter.lower()}.csv",
                mime="text/csv",
                key="download_filtered"
            )
        else:
            st.info(
                f"No {zygosity_filter.lower()} carriers found for selected variants.")
    else:
        st.info("Please select variants to view carrier status.")

    # download button
    csv_full = carriers_df_in.to_csv(index=False)
    st.download_button(
        label="Download complete dataset",
        data=csv_full,
        file_name="complete_dataset.csv",
        mime="text/csv"
    )
