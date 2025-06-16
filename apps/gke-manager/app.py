# streamlit_app.py
import streamlit as st
import os, time, sys, io
from utils import files, gcp_helpers, utils, deployments, infrastructure_deploy, python_helper
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

control_list = ['dan@datatecnica.com','mike@datatecnica.com','syed@datatecnica.com','kristin@datatecnica.com']
cw = os.getcwd()

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

user_login()

if st.session_state["authentication_status"] and st.session_state["user_email"] in control_list:    
    
    #get bash and yaml files
    bash_files = files.get_bash_files('scripts')
    dep_files = files.get_deployment_files('deployments')

    sidebar1, sidebar2 = st.sidebar.columns(2)

    card_removebg = "data/gt_app_utils/card-removebg.png" #"static/card-removebg.png"

    if "card_removebg" not in st.session_state:
        st.session_state["card_removebg"] = card_removebg
    # Initialize session state variables
    if 'processing' not in st.session_state:
        st.session_state.processing = False  # Tracks whether processing is ongoing
    if 'cred' not in st.session_state:
        st.session_state.cred = False
    if 'job_submitted' not in st.session_state:
        st.session_state.job_submitted = False

    sidebar1.image(card_removebg, use_container_width=True)


    with st.sidebar:
        select_bash_script = st.selectbox(
        ":green[**Select Bash Script To Run**]",
        bash_files,
        )
        # if not st.session_state.processing:
        run_bash_script = st.button(":blue[**Run Bash Script**]")
        list_cluster = st.button(":blue[**List CLuster/Node Pool**]")
        select_dep_script = st.selectbox(
        ":green[**Select Deployment Script To Run**]",
        dep_files,
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
        deployments.deployment_yaml(cw+"/deployments/"+select_dep_script)
        # st.session_state.processing = False  # End processing
    if call_api:
        form = st.form("checkboxes", clear_on_submit = True)
        with form:

            st.write(":blue[**Please provide following Parameters for API Call**]")
            # Input fields for API endpoint and payload
            api_url = st.text_input("API URL", value="http://genotools-api.genotoolsserver.com/run-genotools/")
            payload = st.text_area("Request Payload (JSON)", value='{"email":"syed@datatecnica.com", "storage_type": "local", "pfile": "syed-test/input/GP2_merge_AAPDGC", "out": "syed-test/output/test_1", "skip_fails":"True", "ref_panel":"ref/new_panel/ref_panel_gp2_prune_rm_underperform_pos_update","ref_labels":"ref/new_panel/ref_panel_ancestry_updated.txt","model":"ref/models/python3_11/GP2_merge_release6_NOVEMBER_ready_genotools_qc_umap_linearsvc_ancestry_model.pkl", "ancestry":"True", "all_sample":"True", "all_variant":"True", "amr_het":"True", "prune_duplicated": "False", "full_output":"True"}')
            api_key = st.text_area("X-API-KEY", value='3hHAx2FG9U5WS0yjjHbq6MMlMHc9LIQnQfLHX0edwGvidA-wtV')


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
    log_folders = st.text_input(f":red[**Please Enter Log Files Location and :green[**Press View Live Logs**] for logs streaming**]", "")
    if view_logs and log_folders and not logs_stop:
        # if log_folders:
        # Path to the log file
        LOG_FILE = "data/"+log_folders+"_all_logs.log"
        # st.write(f"got log file path: {LOG_FILE}")
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
