import streamlit as st
import yaml
import subprocess
from . import K8S_NAMESPACE, GENOTOOLS_API_POD, IDAT_PED_BED_MERGE_WF, IDAT_PED_BED_MERGE_CHART

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
            ["kubectl", "get", "gateway", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                st.success("GKE Gateway API")
                st.code(result.stdout)
            else:
                st.info("No Gateway API found, Please make sure that It is installed and Routing Traffic Properly")
        else:
            st.error("No Gateway API found, Please check the error below")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")

def check_workflow_jobs():
    st.subheader("Retrieving All Jobs in the Workflow")
    try:
        result = subprocess.run(
            # ["kubectl", "get", "pods", "--selector=job-name", "-n", K8S_NAMESPACE],
            ["kubectl", "get", "pods", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                st.success("Current Jobs in the Workflow")
                st.code(result.stdout)
            else:
                st.info(f"No Jobs found (Output From Server): {result.stdout}, :red[**Please Select a Workflow script and hit **Run YAML Deployment* Button**]")
        else:
            st.error("No Jobs found, Please check the error below")
            st.code(result.stderr)
    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")
    st.subheader("Retrieving Workflow Status")
    try:
        result = subprocess.run(
            ["kubectl", "get", "wf", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                st.success("Workflow Status")
                st.code(result.stdout)
            else:
                st.info("No Workflow found, Please make sure that It is installed correctly")
        else:
            st.error("No Workflow found, Please check the error below")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while Installing Workflow: {e}")
    #also check helm charts
    st.subheader("Retrieving Helm Charts Status")
    try:
        result = subprocess.run(
            ["helm", "list", "-n", K8S_NAMESPACE],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Display the output
        if result.returncode == 0:
            if result.stdout:
                st.success("Helm Charts Status")
                st.code(result.stdout)
            else:
                st.info("No Helm Chart found, Please make sure that It is installed correctly")
        else:
            st.error("No Helm CHart found, Please check the error below")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while Installing Workflow: {e}")

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

def argo_workflow_deployment_yaml(values_file, chart_path):
    try:
        st.subheader("Workflow Deployment Output")
        try:
           
            # helm install <RELEASE_NAME> <CHART_REFERENCE> --values <PATH_TO_VALUES_FILE.yaml>
            # Run kubectl apply command
            result = subprocess.run(
                ["helm", "install", IDAT_PED_BED_MERGE_CHART, chart_path, "--values", values_file, "--namespace", K8S_NAMESPACE],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Display the output
            if result.returncode == 0:
                st.success("Workflow Deployment successful!")
                st.code(result.stdout)
            else:
                st.error("Workflow Deployment failed:")
                st.code(result.stderr)

        except Exception as e:
            st.error(f"An error occurred while deploying: {e}")

    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    st.subheader("Retrieving all Pods, workflow and helm charts")
    check_workflow_jobs()
    # try:
    #     result = subprocess.run(
    #         ["kubectl", "get", "all", "-n", K8S_NAMESPACE],
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.PIPE,
    #         text=True
    #     )
    #     # Display the output
    #     if result.returncode == 0:
    #         st.success(f"All Deployment/Services Found in namespace: {K8S_NAMESPACE}")
    #         st.code(result.stdout)
    #     else:
    #         st.error("Workflow Deployment failed:")
    #         st.code(result.stderr)

    # except Exception as e:
    #     st.error(f"An error occurred while deploying: {e}")

def argo_workflow_delete():
    st.subheader(f"Deleting {IDAT_PED_BED_MERGE_WF} Workflow and all related jobs")
    try:
        result = subprocess.run(
            ["kubectl", "delete", 'wf', IDAT_PED_BED_MERGE_WF, "-n", K8S_NAMESPACE],
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
                st.info(f"No {IDAT_PED_BED_MERGE_WF} Workflow found")
        else:
            st.error("No Workflow is running, Please check the error below")
            st.code(result.stderr)

    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")
    #also delete all related jobs
    st.subheader(f"Deleting all Jobs related to {IDAT_PED_BED_MERGE_WF} Workflow")
    try:
        result = subprocess.run(
            ["kubectl", "delete", "jobs", "--all", "-n", K8S_NAMESPACE],
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
                st.info(f"No Jobs related to {IDAT_PED_BED_MERGE_WF} Workflow found")
        else:
            st.error("No Jobs found, Please check the error below")
            st.code(result.stderr)
    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")

    #also delete helm charts
    st.subheader(f"Uninstalling helm chart {IDAT_PED_BED_MERGE_CHART}")
    try:
        result = subprocess.run(
            ["helm", "uninstall", IDAT_PED_BED_MERGE_CHART, "-n", K8S_NAMESPACE],
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
                st.info(f"No Jobs related to {IDAT_PED_BED_MERGE_WF} Workflow found")
        else:
            st.error("No Jobs found, Please check the error below")
            st.code(result.stderr)
    except Exception as e:
        st.error(f"An error occurred while deploying: {e}")
