import streamlit as st
import yaml
import os
import glob

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
        # st.header("Current Configuration")
        # st.json(config_data) # Display current config in sidebar

        st.header("Edit Configuration")

        # Use Streamlit widgets to edit specific fields
        st.subheader("Job Settings")
        config_data['user_email'] = st.text_input(":red[**User Email**]", value=config_data['user_email']) 
        # config_data['job']['idat_path'] = st.text_input(":red[**IDATs Path**]", value=config_data['job']['idat_path'])
        config_data['job']['key_name'] = st.text_input(":red[**Key Files Name**]", value=config_data['job']['key_name'])

        config_data['job']['output_folder'] = st.text_input(":red[**Output Folder Name (Must Exist)**]", value=config_data['job']['output_folder'])
        config_data['study_id'] = st.text_input(":red[**Please Enter Comma Separated Study List (no ""quotes):**]", value=config_data['study_id'])
        
        config_data['job']['num_threads'] = st.text_input(":red[**Num of Threads Per Job**]", value=config_data['job']['num_threads'])
        
        config_data['job']['barcodes_per_job'] = st.text_input(":red[**Barcodes Per Job**]", value=config_data['job']['barcodes_per_job']) 

        config_data['job']['codes_per_job'] = st.text_input(":red[**Codes Per Job**]", value=config_data['job']['codes_per_job'])

def generate_study_id_yaml(filename, data_dict):
    # filename = 'deployments/workflow/studies.yaml'
    try:
        with open(filename, 'w') as file:
            # The 'sort_keys=False' argument maintains the order of keys from the dictionary
            # 'default_flow_style=False' (or just omitting it in newer PyYAML versions)
            # ensures a readable block style (indented) output rather than inline style
            yaml.dump(data_dict, file, sort_keys=False)
        print(f"Successfully generated '{filename}'")
    except IOError as e:
        print(f"Error writing to file: {e}")


def write_indented_yaml(data, filename):
    class IndentedDumper(yaml.SafeDumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)
    with open(filename, 'w') as f:
        yaml.dump(data, f, Dumper=IndentedDumper, default_flow_style=False, indent=2)
    # return yaml.dump(data, Dumper=IndentedDumper, default_flow_style=False, indent=2)

def merge_yaml_files(output_filename, *input_filenames):
    """
    Loads multiple YAML files, merges them, and saves the result.
    Later files in the input list will override keys in earlier files.
    """
    merged_data = {}

    for filename in input_filenames:
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                merged_data.update(data) # Merge the dictionaries
            else:
                print(f"Warning: {filename} did not contain a top-level dictionary, skipping merge.")

    # Write the combined data to a new YAML file
    with open(output_filename, 'w') as f:
        yaml.dump(merged_data, f, sort_keys=False) # sort_keys=False preserves original order where possible
