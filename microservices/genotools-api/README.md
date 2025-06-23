# GenoTools API

This project provides a RESTful API interface to the [GenoTools](https://github.com/dvitale199/GenoTools) package, allowing users to perform genomic data quality control and analysis through HTTP requests. The API handles data transfer from Google Cloud Storage (GCS), executes GenoTools commands, and returns the results, facilitating integration with other services and automation of genomic workflows.

## Table of Contents

- [GenoTools API](#genotools-api)
  - [Table of Contents](#table-of-contents)
  - [features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration and GKE setup](#configuration-and-gke-setup)
  - [Destroying k8s](#destroying-k8s)
  - [Coming Soon](#coming-soon)
  - [Usage](#usage)
    - [API Endpoints](#api-endpoints)
      - [`GET /`](#get-)
      - [`POST /run-genotools/`](#post-run-genotools)
    - [Example Request](#example-request)
    - [Parameters](#parameters)
    - [Response](#response)
  - [Testing](#testing)
  - [docker usage](#docker-usage)

## features

- **run genotools commands**: Execute GenoTools commands remotely via API calls.
- **data handling with gcs**: Use your gcs bucket to directly read input and write output via mount.
- **API Key Authentication**: Secure endpoints using API key authentication.
- **Flexible Parameters**: Support for various GenoTools parameters and options.
- **Dockerized Deployment**: Easy deployment using Docker.

## Prerequisites

- Python 3.11 or higher
- Google Cloud SDK (if interacting with GCS)
- GenoTools package (`the-real-genotools`)
- Docker (optional, for containerized deployment)
- Poetry (for dependency management)

## Installation
Image: gcloud builds submit --tag us-east1-docker.pkg.dev/gp2-code-test-env/syed-test/testing/genotools-server/microservices/genotools-api .
1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/genotools-api.git
   cd genotools-api
   ```

2. **Install Dependencies**
Ensure you have Poetry installed.

```
poetry install
```

3. **set up environment**
create .env file in the project root direcotry and define the variables
```
API_KEY_NAME=your_api_key_name
API_KEY=your_api_key_value
```

## Configuration and GKE setup

- **API Key Authentication**

  - The API uses API key authentication via a custom header.
  - Set `API_KEY_NAME` to the header name you want to use (e.g., `X-API-KEY`).
  - Set `API_KEY` to the secret key clients must provide.

- **Google Cloud Credentials**

  - Ensure the application has access to Google Cloud credentials if interacting with GCS.
  - Set up authentication by configuring the `GOOGLE_APPLICATION_CREDENTIALS` environment variable or using default application credentials.
- **GKE deployment:** Requires gcloud sdk and minikube (or similar) installed on your system.
	- Deploy on gke cluster via yaml manifest files and issue API calls to run pipeline.
	- **TL;DR**: From terminal, issue following commands to authenticate to your gcloud account: 
		- gcloud init
			- Select your gcp account to use
		- gcloud auth login
			- To authenticate to gcp account
		- To Provision k8s cluster, run following command from terminal, Please make sure that you have all required bash and yaml files with all variables set to proper values (See next section for details) :
			- bash gtclusterV2.sh    
			- kubectl apply -f gtcluster-deployment-secretV1.yml
- **GKE Setup Configurations** 
	- In order to prepare k8s cluster for genotools work loads, we need to perform following steps via terminal
	- Create a k8s cluster using gtclusterV1.sh (with comments for each step) file. This file has variious parameters (such as gcp zone, machine type, service account name etc) that can be changed based on needs. Please note that this script has various steps to provision k8s cluster and configure various options and services for it to work properly. Namely:
		- crate cluster
		- configure/update cluster with various add-ons
		- create a gcp bucket (if one does not exist) to use with the cluster
		- provision GCP Service Account
		- provision k8s service account and namespace
		- configure bucket access via GCP service account
		- deploy k8s ServiceAccount service to impersonate for gcp resource access (like gcp bucket)
		- bind k8s ServiceAccount service to use gcp resources
		- **Once k8s cluster is up and running, we can check it with:** gcloud container clusters list
		- create Persistent Volume (PV):
			- From Terminal run: kubectl apply -f pv.yaml
		- Persistent Volume Claim (PVC)
			- From Terminal run: kubectl apply -f pvc.yaml
		- __Test PV and PVC:__ 
			- kubectl get pv 
			- kubectl get pvc
	- Create Deployment (pod and related services)
		- From terminal run:
			- kubectl apply -f gtcluster-deployment-secret.yaml
			- This will create following services:
				- Service: Secret, Name: gt-api-sec, Purpose: API Key
				- Service: Deployment, Name: gtcluster-pod, Purpose: Creates 1 Pod, 1 Sidecar Container, 1 genotools_api container to run on pod, consume pv via pvc and API Key secret
				- Service: NodePort Service, Name: gtcluster-svc, Purpose: Provides access to genotoolsapi endpoionts
				- Service: Ingress, Name: gtcluster-ingress, Purpose: Provide access to pod and hence to genotoolsapi via web url (here IP address) 
		- In order to test if pod is running and ready to serve worloads, from terminal, use following command: kubectl get all 
		- Also to get the IP address to use for API calls, Please use following command from terminal: kubectk get ingress
			- This will provide the IP address for API calls
	- Making API Calls:

```python
import requests
import json
import pandas as pd
from requests.exceptions import HTTPError

d = {"email":"syed@datatecnica.com", "storage_type": "local", "pfile": "syed-test/input/GP2_merge_AAPDGC", "out": "syed-test/output/test6", "skip_fails":True, "ref_panel":"ref/ref_panel/1kg_30x_hgdp_ashk_ref_panel","ref_labels":"ref/ref_panel/1kg_30x_hgdp_ashk_ref_panel_labels.txt","model":"ref/models/python3_11/GP2_merge_release6_NOVEMBER_ready_genotools_qc_umap_linearsvc_ancestry_model.pkl", "callrate":.01}

headers = {
    "X-API-KEY": "YOUR_API-KEY",
    "Content-Type": "application/json"
}

link="<IP Address From Ingress>/run-genotools/" 
try:
    
    r = requests.post(f"{link}", data=json.dumps(d), headers=headers)

    print(f'r: {r}')
    r.raise_for_status()
    res=r.json()
    print(f"res: {res}")
except HTTPError as http_err:
    print(f'HTTP error occurred: {http_err}')
except Exception as err:
    print(f'Other error occurred: {err}')            
```

-	Once pipeline is submitted, an automated job submission confirmation email will be sent at the email address provided in the API call.
- At job completion, a confirmation email will be sent to the submitter.

## Destroying k8s

Please use following steps to destroy the cluster.

- gcloud container clusters delete $CLUSTER_NAME --zone $ZONE
- This will delete the k8s cluster and all related resources.
- You can also check if cluster is still running or not by using following command: kubectl get all


## Coming Soon

We are also working on an automated single step k8s cluster deployment workflow and will share once it is done. 


## Usage

### API Endpoints

#### `GET /`

- **Description**: Health check endpoint.
- **Response**: `"Welcome to GenoTools"`

#### `POST /run-genotools/`

- **Description**: Execute a GenoTools command with specified parameters.
- **Authentication**: Requires API key in the header.
- **Request Body**: JSON object with parameters as defined in the `GenoToolsParams` model.
- **Response**: JSON object containing the execution result.

### Example Request

Here's how to use the API to run a GenoTools command:

**Python Script (`test_main.py`):**

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = 'http://0.0.0.0:8080/run-genotools/'

payload = {
    "pfile": "gs://your_bucket/your_pfile_prefix",
    "out": "gs://your_bucket/output_prefix",
    "callrate": 0.5,
    "sex": True,
    "storage_type": "gcs"
}

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("API_KEY_NAME")

headers = {
    API_KEY_NAME: API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.status_code)
print(response.json())
```

**Environment Variables (`.env`):**

```env
API_KEY_NAME=X-API-KEY
API_KEY=your_api_key_value
```

### Parameters

The `GenoToolsParams` model supports the following fields:

- **Input Files:**
  - `bfile`: Path to PLINK binary fileset.
  - `pfile`: Path to PLINK 2 binary fileset.
  - `vcf`: Path to VCF file.
- **Output:**
  - `out`: Output file prefix.
- **Options:**
  - `full_output`: `bool` (optional)
  - `skip_fails`: `bool` (optional)
  - `warn`: `bool` (optional)
  - `callrate`: `float` or `bool` (optional); call rate threshold.
  - `sex`: `bool` (optional); perform sex checks.
  - `related`: `bool` (optional); check for relatedness.
  - `related_cutoff`: `float` (optional)
  - `duplicated_cutoff`: `float` (optional)
  - `prune_related`: `bool` (optional)
  - `prune_duplicated`: `bool` (optional)
  - `het`: `bool` (optional)
  - `all_sample`: `bool` (optional)
  - `all_variant`: `bool` (optional)
  - `maf`: `float` (optional)
  - `ancestry`: `bool` (optional); perform ancestry inference.
  - `ref_panel`: `str` (optional); path to reference panel.
  - `ref_labels`: `str` (optional); path to reference labels.
  - `model`: `str` (optional); path to ML model for ancestry prediction.
  - `storage_type`: `'local'` or `'gcs'`; storage location of files.

### Response

A successful response will include:

- `"message"`: Status message (e.g., `"Job submitted"`).
- `"command"`: The GenoTools command executed.
- `"result"`: Output from the command execution.

## Testing

To run the provided test script:

1. **Ensure the API is Running**

   ```bash
   uvicorn genotools_api.main:app --host 0.0.0.0 --port 8080

2. **Run the Test Script**

   ```bash
   python test_main.py
   ```

3. **check the output**
The script will print the HTTP status code and response JSON.

## docker usage

** build image and run docker container **
```
docker build -t genotools-api .

docker run -d -p 8080:8080 \
  -e API_KEY_NAME="X-API-KEY" \
  -e API_KEY="your_api_key_value" \
  --name genotools-api \
  genotools-api
```


Maintainer: Dan Vitale (dan@datatecnica.com)
