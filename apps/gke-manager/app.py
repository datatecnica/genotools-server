# streamlit_app.py
import streamlit as st
import os, time, sys, io
from utils import files, gcp_helpers, utils, deployments, infrastructure_deploy, python_helper, yaml_helpers
from manage.login import user_login
# from watchdog.observers import Observer
import requests
import json


st.set_page_config(page_title='GKE Manager', layout='wide')
st.markdown(
    """
<style>
.streamlit-expanderHeader {
    font-size: x-large;
}
</style>
""",
    unsafe_allow_html=True,
)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.markdown(
    '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
    unsafe_allow_html=True)

# Title of the app
st.title(":green[GKE Job Scheduler and Log Stream Manager]")

control_list = ['Zih-Hua.Fang@dzne.de', 'dan@datatecnica.com','mike@datatecnica.com','syed@datatecnica.com','kristin@datatecnica.com']
cw = os.getcwd()

user_login()

if st.session_state["authentication_status"] and st.session_state["user_email"] in control_list:    
    
    #get bash and yaml files for genotools_api
    bash_files_genotools_api = files.get_bash_files('scripts/api')
    dep_files_genotools_api = files.get_deployment_files('deployments/api')
    #get bash and yaml files
    bash_files_wf = files.get_bash_files('scripts/workflow')
    dep_files_wf = files.get_deployment_files('deployments/workflow')


    card_removebg = "static/card-removebg.png" #"static/card-removebg.png"

    if "card_removebg" not in st.session_state:
        st.session_state["card_removebg"] = card_removebg
    # Initialize session state variables
    if 'processing' not in st.session_state:
        st.session_state.processing = False  # Tracks whether processing is ongoing
    if 'cred' not in st.session_state:
        st.session_state.cred = False
    if 'job_submitted' not in st.session_state:
        st.session_state.job_submitted = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Home"        
    st.sidebar.image(card_removebg, use_container_width=True)
    
    tabs = ["Home", "PreGenotools", "GenotoolsAPI", "Help"]

    selected_tab = st.sidebar.selectbox("Navigate", tabs, index=tabs.index(st.session_state.active_tab))
    st.session_state.active_tab = selected_tab

    if st.session_state.active_tab == "Home":
        st.header("Welcome to GKE Manager App")
        expander1 =  st.expander(":blue[**Need Help, Expand Me**]") 
        expander1.markdown("**This app helps provision/destroy GKE node pools for genotools-api deploy and workloads.**")
        expander1.markdown("+ To **Add/Delete** node-pools to the existing GKE cluster, Select appropriate bash script from dropdown list in the sidebar and **Press Run Bash Script** Button.")  
        expander1.markdown("+ To **List currently running Cluster/Node pools** please **Press List CLuster/Node pool** Button.")  
        expander1.markdown("+ To **Create a deployment** on the created node-pool (step above), Please select appropriate deployment file from the list in sidebar and **Press Run YAML Deployment** Button. :red[**Please make sure that :blue[**Ancesstory Node Pool**] is running in the cluster before creating deployment**]")  
        expander1.markdown("+ To check all existing **deployments/services**, Press Deployment Status Button")  
        expander1.markdown("+ To **delete deployment**, Press Delete Deployment Button")  
        expander1.markdown("+ To **view live logs** of the running job, Please provide location of the output folder :red[**It is string in the 'out' flag for API Call**] in the provided text box and **Press View Live Logs** Button.")  
        expander1.markdown("+ In the event of **Errors** in the output logs, **You can delete the node pole to save compute cost**.")  
        expander1.markdown("+ To submit Job, Please select **Submit Job** check box. :red[**Please make sure that deployment.apps/gtcluster-pod is running (Press Deployment Status to verify)**]")  

    elif st.session_state.active_tab == "PreGenotools":
        st.header("Pre-Genotools Workflow")
        st.write(":red[**IDAT-PED-BED-MERGER Workflow**]")
        with st.sidebar:
            select_bash_script = st.selectbox(
            ":green[**Select Bash Script To Run**]",
            [" "]+bash_files_wf,
            )

            run_bash_script = st.button(":blue[**Run Bash Script**]")
            list_cluster = st.button(":blue[**List Cluster/Node Pool**]")

            select_dep_script = st.selectbox(
            ":green[**Select Deployment Script To Run**]",
            [" "]+dep_files_wf,
            )
            run_dep_script = st.button(":blue[**Install Helm Chart**]")
            check_deployment = st.button(":red[**Workflow Status**]")  
            delete_deployment = st.button(":red[**Delete Workflow**]")  
            call_workflow = st.checkbox(":red[**Setup Workflow Config**]", value=False)
        if run_bash_script and select_bash_script:
            # st.session_state.processing = True  # Start processing
            infrastructure_deploy.configure_infrastructure(cw+"/scripts/workflow/"+select_bash_script)
            # st.session_state.processing = False  # End processing
        if list_cluster:
            # st.session_state.processing = True  # Start processing
            gcp_helpers.check_cluster()
            # st.session_state.processing = False  # End processing
        if run_dep_script and select_dep_script is not None:
            # st.session_state.processing = True  # Start processing
            gcp_helpers.get_gcp_cluster_credentials()
            deployments.argo_workflow_deployment_yaml(cw+"/deployments/workflow/"+select_dep_script, cw+"/deployments/workflow/helm/dep")

        if check_deployment:
            if not st.session_state['cred']:
                try:
                    gcp_helpers.get_gcp_cluster_credentials()
                    st.session_state['cred'] = True
                except:
                    st.warning('Failed to Retrieve Deafult Credentials')
                # st.session_state.processing = True  # Start processing
                deployments.check_workflow_jobs()
            else:
                deployments.check_workflow_jobs()
            # st.session_state.processing = False  # End processing
        if delete_deployment:
            if not st.session_state['cred']:
                try:
                    gcp_helpers.get_gcp_cluster_credentials()
                    st.session_state['cred'] = True
                except:
                    st.warning('Failed to Retrieve Deafult Credentials')

                deployments.argo_workflow_delete()
            else:
                deployments.argo_workflow_delete()
        if call_workflow:
            form = st.form("checkboxes", clear_on_submit = True)
            with form:
                expander2 =  st.expander(":blue[**Please adjust your workflow parameters below and :red[**Press Save Workflow**]**]") 
                expander2.markdown("""
                # Important
                1. GCS bucket: Bucket is mounted in /aa/input/ folder in the container file system.
                2. Make sure that all paths in the gcs bucket are properly provided in the below variables.
                3. EXAMPLE: In the below variables, /app/input/new_test_data/pipe-line-results
                    corresponds to new_test_data/pipe-line-results folder in the GCS bucket. THIS PATH MUST BE A VALID FOLDER IN THE GCS BUCKET.
                4. logs, batch_files, ped_bed, clinical folders and folders (inside thses for each step) are automatically created by the workflow.
                5. User must ensure that the input files (idats, fam, key_file etc.) are already present in the GCS bucket before starting the workflow.
                6. User provided folder are below:
                    a. /app/input/new_test_data/idats  --> Input IDAT files folder
                    b. /app/input/new_test_data/syed_test_idats_n313.txt  --> Key file path
                    c. /app/input/new_test_data/exec  --> Execution folder path in GCS bucket where iaap-cli and plink executables are present.
                    exec folder contains:
                      i. NeuroBooster_20042459_A2.bpm
                      ii. recluster_09272022.egt
                     iii. iaap-cli/ (folder with iaap-cli executable and all dependencies)
                     iv. plink_modules// (folder with plink installer files plink2_module.sh and plink_module.sh)
                     PLEASE DO NOT CHANGE PATHS IN plink2_module.sh and plink_module.sh FILES AS THESE ARE BEING REFERRED IN VARIOUS STEPS OF THE WORKFLOW.""")
                config_data = yaml_helpers.load_config('deployments/workflow/idat-ped-bed-merge-values.yaml')
                yaml_helpers.edit_yaml_config(config_data)
                #update_settings = st.checkbox(":red[**Update Settings**]") #, value=st.session_state.job_submitted)
            submit = form.form_submit_button(":red[**Save Workflow**]")   
            if submit: # and update_settings: 
                yaml_helpers.save_config('deployments/workflow/idat-ped-bed-merge-values.yaml', config_data)
                st.rerun()


    elif st.session_state.active_tab == "GenotoolsAPI":
        st.header("# GKE Manager and GenotoolsAPI Deployment")
        with st.sidebar:
            select_bash_script = st.selectbox(
            ":green[**Select Bash Script To Run**]",
            [" "]+bash_files_genotools_api,
            )
            # if not st.session_state.processing:
            run_bash_script = st.button(":blue[**Run Bash Script**]")
            list_cluster = st.button(":blue[**List Cluster/Node Pool**]")

            select_dep_script = st.selectbox(
            ":green[**Select Deployment Script To Run**]",
            [" "]+ dep_files_genotools_api,
            )
            # if not st.session_state.processing:        
            run_dep_script = st.button(":blue[**Create Deployment**]")
            check_deployment = st.button(":red[**Deployment Status**]")  
            delete_deployment = st.button(":red[**Delete Deployment**]")  
            view_logs = st.button(":blue[**View Live Logs**]")
            logs_stop = st.button(":red[**Stop Live Logs**]")
            call_api = st.checkbox(":red[**Submit Job**]", value=False)
            # call_api = st.button(":red[**Submit Job**]")
                


        if run_bash_script and select_bash_script:
            # st.session_state.processing = True  # Start processing
            infrastructure_deploy.configure_infrastructure(cw+"/scripts/"+select_bash_script)
            # st.session_state.processing = False  # End processing
        if list_cluster:
            # st.session_state.processing = True  # Start processing
            gcp_helpers.check_cluster()
            # st.session_state.processing = False  # End processing
        if run_dep_script and select_dep_script is not None:
            # st.session_state.processing = True  # Start processing
            gcp_helpers.get_gcp_cluster_credentials()
            deployments.deployment_yaml(cw+"/deployments/api/"+select_dep_script)
            # st.session_state.processing = False  # End processing
        if call_api:
            form = st.form("checkboxes", clear_on_submit = True)
            with form:

                st.write(":blue[**Please provide following Parameters for API Call**]")
                # Input fields for API endpoint and payload
                api_url = st.text_input("API URL", value="https://genotools-api.genotoolsserver.com/run-genotools/")
                payload = st.text_area("Request Payload (JSON)", value='{"email":"syed@datatecnica.com", "storage_type": "local", "pfile": "syed-test/input/GP2_merge_AAPDGC", "out": "syed-test/output/test_1", "skip_fails":"True", "ref_panel":"ref/new_panel/ref_panel_gp2_prune_rm_underperform_pos_update","ref_labels":"ref/new_panel/ref_panel_ancestry_updated.txt","model":"ref/models/python3_11/GP2_merge_release6_NOVEMBER_ready_genotools_qc_umap_linearsvc_ancestry_model.pkl", "ancestry":"True", "all_sample":"True", "all_variant":"True", "amr_het":"True", "prune_duplicated": "False", "full_output":"True"}')
                api_key = st.text_area("X-API-KEY", value='YOUR-API-KEY')


                # python_code = st.text_area(":red[**Please adjust your parameters in the code here and Indent Properly:**]", height=200)  

                submit_workload = st.checkbox(":red[**Perform API Request**]", value=st.session_state.job_submitted)
            submit = form.form_submit_button("Submit Button")    
            # Button to execute the Python code
            # if st.button("Submit API Request"):
            if submit and submit_workload and api_url and payload and api_key:
                st.session_state.job_submitted = False
                st.write(f"api_url: {api_url}\n payload: {payload}\n api_key: {api_key}")
                try:
                    # Parse the payload as JSON
                    payload_dict = json.loads(payload)  # Use `json.loads` in production for safety

                    # Make the API request
                    response = requests.post(api_url, json=payload_dict, headers={"X-API-KEY": api_key, "Content-Type": "application/json"})

                    # Display the response
                    st.subheader(":green[**API Response**]")
                    st.json(response.json())
                except Exception as e:
                    st.error(f":red[**An error occurred: {e}**]")
                                
            else:
                st.info("**Please enter valid Parameters for API Call and :red[**Press Submit Button**] after checking all parameters**")

        # python_helper.api_call()
    #get log folder location
        log_folders = st.text_input(f":blue[**Please Enter Log Files Location and :red[**Press View Live Logs**] for logs streaming**]", "")
        if view_logs and log_folders and not logs_stop:
            # if log_folders:
            # Path to the log file
            LOG_FILE = "/app/data/"+log_folders+"_all_logs.log"
            st.write(f"got log file path: {LOG_FILE}")
            # st.session_state.processing = True  # Start processing
            with st.spinner(f":red[**Now viewing {LOG_FILE} logs, Please wait...**]"):
                text_placeholder = st.empty()

                # Stream the data
                stream = utils.stream_data(LOG_FILE)
                

                # Display the streamed data
                buffer = []
                for line in stream:
                    buffer.append(line)  # Add the new line to the buffer
                    # Join all lines into a single string with newlines
                    text_content = "\n".join(buffer)
                    # Display the content in a scrollable text box
                    text_placeholder.text_area("", value=text_content, height=400)    
            # st.session_state.processing = False  # End processing
        else:
            # st.write(f"log folder provided is: {log_folders}")
            # st.session_state.processing = False  # End processing
            st.warning(f":red[**Provided Path {log_folders} is not valid/or api has not been called yet :blue[**Hint: It is the string provided in the 'out' flag during API Call**], Please provide valid location**]")
        if check_deployment:
            if not st.session_state['cred']:
                try:
                    gcp_helpers.get_gcp_cluster_credentials()
                    st.session_state['cred'] = True
                except:
                    st.warning('Failed to Retrieve Deafult Credentials')
                # st.session_state.processing = True  # Start processing
                deployments.check_dep()
            else:
                deployments.check_dep()
            # st.session_state.processing = False  # End processing
        if delete_deployment:
            if not st.session_state['cred']:
                try:
                    gcp_helpers.get_gcp_cluster_credentials()
                    st.session_state['cred'] = True
                except:
                    st.warning('Failed to Retrieve Deafult Credentials')

                deployments.delete_dep('deployment.apps')
            else:
                deployments.delete_dep('deployment.apps')
    elif st.session_state.active_tab == "Help":
        st.header("Help and Documentation")
        st.markdown("**For any issues or questions, please contact the support team at [syed@datatecnica.com]**")