import streamlit as st
import yaml
import subprocess
from . import K8S_NAMESPACE, GENOTOOLS_API_POD

def deployment_yaml(yaml_file):
    try:
        st.subheader("Deployment Output")
        try:
            # Run kubectl apply command
            result = subprocess.run(
                ["kubectl", "apply", "-f", yaml_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Display the output
            if result.returncode == 0:
                st.success("Deployment successful!")
                st.code(result.stdout)
            else:
                st.error("Deployment failed:")
                st.code(result.stderr)

        except Exception as e:
            st.error(f"An error occurred while deploying: {e}")

    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    st.subheader("Retrieving all Deployments/Services")
    try:
        result = subprocess.run(
            ["kubectl", "get", "all", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            st.success(f"All Deployment/Services Found in namespace: {K8S_NAMESPACE}")
            st.code(result.stdout)
        else:
            st.error("Deployment failed:")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")

def check_dep():
    st.subheader("Retrieving All Deployments, Services and Ingress")
    try:
        result = subprocess.run(
            ["kubectl", "get", "all", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                st.success("Current Deployments/Services")
                st.code(result.stdout)
            else:
                st.info("No Deployments found, :red[**Please Select a deployment script and hit *Run YAML Deployment* Button**]")
        else:
            st.error("No Deployments found, Please check the error below")
            st.code(result.stderr)
    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")

    try:
        result = subprocess.run(
            ["kubectl", "get", "ingress", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                st.success("Ingress")
                st.code(result.stdout)
            else:
                st.info("No Ingress found")
        else:
            st.error("No Ingress found, Please check the error below")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")


def delete_dep(dep):
    st.subheader(f"Deleting {dep+'/'+GENOTOOLS_API_POD} deployment")
    try:
        result = subprocess.run(
            ["kubectl", "delete", dep+"/"+GENOTOOLS_API_POD, "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                # st.success(f"Deployment: {dep} ")
                st.code(result.stdout)
            else:
                st.info(f"No {dep} deployment found")
        else:
            st.error("No Deployments found, Please check the error below")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")

