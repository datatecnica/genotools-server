# Genotools-Server
## Overview
GenoTools-Server is a set of apps and services to facilitate various genetics processing pipeline and visualization apps.
## apps
+ gp2-browser
+ genotracker
## microservices
+ genotools-api
+ genotracker
+ carriers
## Management apps
+ gke-manager
+ Workflow Manager
## Setup
All apps and microservices are deployed in GKE cluster. To deploy, you need to have GKE cluster, gcloud SDK and kubectl configured. 
<span style="color:red">Please note that all infrastructe provisioning and deployment scripts are for gp2-code-test-env project. For gp2-release-terra deploy we need replace PROJECT_ID to gp2-release-terra in gke-cluster.sh and all othere scripts</span>.
### Server Setup
+ Genotools-Server runs on a GKE cluster that manages gp2-browser and genotracker app, genotracker, carriers and genotools-api microservices in a scalabale and costeffective manner.
  + GKE-Cluster: A single node (e2-standard-4) genotools-server configured with gbucket and workload identity federation access is deployed on gcp.
  + Various apps/micro-service in the cluster are exposed via Ingress.
  + Each app/service is deployed on a dedicated node pool and exposed via kubernetes service which is exposed via Ingress.
  + Apps and services are configured with workload identity federation to access gbucket.
  + Airflow is used to orchestrate the workflow of various microservices.
#### Apps folder
##### gke-manager
  + A simple streamlit app (GKE-Manager) to manage and deploy node-pools to an existing GKE cluster for __running genotools-api workflow__.
    + docker image: gcloud builds submit --tag us-east1-docker.pkg.dev/gp2-code-test-env/syed-test/genotools-server/apps/gke-management/gke-manager .
    + https://syed-gke-manager-664722061460.us-central1.run.app
    + Has access to gtserver-eu-west4-gp2-code-test-env bucket via app/data/ folder
      + Currently all infrastructure provisioning and deployment scripts are read by scripts and deployments folder __But we can use gbucket if we want to make the app more flexible.__
    + App has access to limited users: Dan, Mike, Kristin and Syed
    + 
  + It facilitates server monitoring, work load submission, live log streaming etc.

##### gp2-browser
  + A simple streamlit app to query gp2 data.
  + Deployment scripts for gp2-browser is in deploy/gp2-browser folder.
### deploy
+ gp2-browser
    + This folder contains deployment scripts for gp2-browser.
    + Add/delete node-pool script
    + Deployment script.
