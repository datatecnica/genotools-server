import streamlit as st
import yaml
import os

# CONFIG_FILE = 'deployments/workflow/idat-ped-bed-merge-values.yaml'

# Function to load YAML data
def load_config(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return yaml.safe_load(file)
    return {}

# Function to save YAML data
def save_config(filename, data):
    with open(filename, 'w') as file:
        yaml.dump(data, file, sort_keys=False)

# Main Streamlit application
# st.title("YAML File Editor")

# config_data = load_config(CONFIG_FILE)
def edit_yaml_config(config_data):
    st.title("YAML Configuration Editor")

    if config_data:
        st.header("Current Configuration")
        st.json(config_data) # Display current config in sidebar

        st.header("Edit Configuration")

        # Use Streamlit widgets to edit specific fields
        st.subheader("Job Settings")
        config_data['job']['log_file_path'] = st.text_input("log_file_path", value=config_data['job']['log_file_path'])

        config_data['job']['study_id']['id1'] = st.text_input("id1", value=config_data['job']['study_id']['id1'])
        config_data['job']['study_id']['id2'] = st.text_input("id2", value=config_data['job']['study_id']['id2'])
        config_data['job']['study_id']['id3'] = st.text_input("id3", value=config_data['job']['study_id']['id3'])


        config_data['job']['key_path'] = st.text_input("key_path", value=config_data['job']['key_path'])
        config_data['job']['fam_path'] = st.text_input("fam_path", value=config_data['job']['fam_path'])

        config_data['job']['raw_plink_path'] = st.text_input("raw_plink_path", value=config_data['job']['raw_plink_path'])
        config_data['job']['batch_folder_path'] = st.text_input("batch_folder_path", value=config_data['job']['batch_folder_path'])
        config_data['job']['exec_folder_path'] = st.text_input("exec_folder_path", value=config_data['job']['exec_folder_path'])


        config_data['job']['num_threads'] = st.text_input("num_threads", value=config_data['job']['num_threads'])
        config_data['job']['path_idat_ped_jobs'] = st.text_input("path_idat_ped_jobs", value=config_data['job']['path_idat_ped_jobs'])
        config_data['job']['idat_path'] = st.text_input("idat_path", value=config_data['job']['idat_path'])
        config_data['job']['barcodes_per_job'] = st.text_input("barcodes_per_job", value=config_data['job']['barcodes_per_job']) 


        config_data['job']['path_ped_bed_jobs'] = st.text_input("path_ped_bed_jobs", value=config_data['job']['path_ped_bed_jobs'])
        config_data['job']['codes_per_job'] = st.text_input("codes_per_job", value=config_data['job']['codes_per_job'])
        config_data['job']['clinical_key_dir'] = st.text_input("clinical_key_dir", value=config_data['job']['clinical_key_dir'])
        config_data['job']['path_beds_merge_jobs'] = st.text_input("path_beds_merge_jobs", value=config_data['job']['path_beds_merge_jobs']) 
        config_data['user_email'] = st.text_input("user_email", value=config_data['user_email']) 


        # # Save button
        # if st.button("Save Changes"):
        #     save_config(CONFIG_FILE, config_data)
        #     st.success("Configuration updated and saved to config.yaml!")
        #     st.experimental_rerun() # Rerun the app to show updated sidebar
    # else:
    #     st.error(f"Configuration file '{CONFIG_FILE}' not found or is empty.")
