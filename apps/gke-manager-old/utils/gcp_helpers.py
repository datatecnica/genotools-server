import streamlit as st
import subprocess

from . import GCP_PROJECT, CLUSTER_NAME, ZONE, K8S_NAMESPACE

def get_gcp_cluster_credentials():
    command = [
        "gcloud", "config", "set", "project", GCP_PROJECT
    ]
    gcloud_related(command,'Set GCP Project')

    # Build the gcloud command
    command = [
        "gcloud", "container", "clusters", "get-credentials",
        CLUSTER_NAME,
        "--region" if "-" in ZONE else "--zone",  # Auto-detect region vs zone
        ZONE,
        "--project", GCP_PROJECT
    ]
    gcloud_related(command,"Retrieving Cluster Credentials")
def check_cluster():
    command = [
        "gcloud", "container", "clusters", "list"
    ]
    gcloud_related(command,f"CLusters List in Current Namespace: {K8S_NAMESPACE}")

    # Build the gcloud command
    command = [
        "gcloud", "container", "node-pools", "list", 
        "--cluster", CLUSTER_NAME,
        "--region" if "-" in ZONE else "--zone",  # Auto-detect region vs zone
        ZONE
        # "--project", GCP_PROJECT
    ]
    gcloud_related(command,f"List of Node Pools in Cluster: {CLUSTER_NAME}")

def gcloud_related(command, title):
    # Run the command
    st.subheader(title)
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            st.success("Credentials configured successfully!")
            st.code(result.stdout)
        else:
            st.error("Failed to configure credentials:")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while running the command: {e}")    
