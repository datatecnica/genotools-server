# GenoTools Server

Monorepo containing genomic data processing services and applications for the **Global Parkinson's Genetics Program (GP2)**. The platform processes, analyzes, and visualizes large-scale genomic datasets including quality control, ancestry analysis, carrier screening, and variant exploration.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Applications](#applications)
- [Microservices](#microservices)
- [Workflows](#workflows)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [CI/CD](#cicd)
- [Environment Variables](#environment-variables)

## Architecture Overview

```
GCS Buckets (gp2tier2_vwb, genotools-server)
        |
  Microservices (FastAPI) ── process PLINK/VCF/IDAT files
        |
  Output (Parquet, JSON reports)
        |
  Streamlit Apps ── visualize results
        |
  GKE Cluster ── served via HTTPS (Gateway API + cert-manager)
```

Services are containerized with Docker, orchestrated on Google Kubernetes Engine (GKE), and deployed via Helm charts with ArgoCD for GitOps-based continuous delivery.

## Project Structure

```
genotools-server/
├── apps/                        # Streamlit web applications
│   ├── gp2-browser/             # GP2 cohort data browser
│   ├── genotracker/             # Cohort data tracking & visualization
│   └── gke-manager/             # GKE cluster management dashboard
│
├── microservices/               # FastAPI backend services
│   ├── genotools-api/           # GenoTools QC command execution
│   ├── precision_med_api/       # Carrier screening pipeline (primary)
│   ├── genotracker/             # Cohort data API
│   ├── browser_api/             # Browser file preparation
│   ├── idat_utils/              # IDAT file processing
│   ├── snp_checksums/           # Duplicate detection via SNP hashing
│   ├── gt-precheck/             # Preprocessing validation
│   └── carriers_api/            # (DEPRECATED) Legacy carrier analysis
│
├── workflows/                   # Argo Workflows for batch processing
│   └── idat-ped-bed-merge/      # IDAT → PED → BED conversion pipeline
│
├── deployments/                 # Infrastructure & deployment
│   ├── gke-setup/               # GKE cluster & nodepool provisioning
│   └── helm-charts/             # Helm charts (dev/staging/prod)
│
├── cloudbuild.yaml              # Google Cloud Build CI/CD pipeline
├── apps-argo-cd.yaml            # ArgoCD app deployment manifest
└── workflow-argo-cd.yaml        # ArgoCD workflow deployment manifest
```

## Applications

### GP2 Browser
Interactive web application for browsing GP2 cohort data releases.
- **Location**: `apps/gp2-browser/`
- **Framework**: Streamlit (port 8501)
- **Features**: Data release browsing, QC metrics, ancestry analysis, SNP metrics, rare variant exploration

### GenoTracker
Data tracking and visualization application for genomic cohort management.
- **Location**: `apps/genotracker/`
- **Framework**: Streamlit (port 8501)
- **Features**: Cohort data visualization, interactive exploration, local/GCS data support

### GKE Manager
Administrative dashboard for managing the GKE cluster infrastructure.
- **Location**: `apps/gke-manager/`
- **Framework**: Streamlit

## Microservices

### GenoTools API
RESTful API for executing genomic data quality control commands.
- **Location**: `microservices/genotools-api/`
- **Framework**: FastAPI (port 8080)
- **Features**: GenoTools QC command execution, GCS integration, API key authentication, email notifications
- **Package Manager**: Poetry

### Precision Med API
High-performance genomic carrier screening pipeline for pathogenic variant analysis. This is the primary carrier analysis service.
- **Location**: `microservices/precision_med_api/`
- **Framework**: FastAPI (port 8000) + Streamlit frontend
- **Features**:
  - 760+ pathogenic SNP extraction across 4 data types (NBA, WGS, IMPUTED, EXOMES)
  - Real-time allele harmonization with MAF correction
  - Parallel processing with auto-optimization based on system resources
  - Probe validation (NBA vs. WGS concordance)
  - Locus reports with clinical phenotype integration
  - Ancestry-stratified analysis (11 ancestries)
- **Performance**: ~10 min extraction for 760 variants across 254+ PLINK files

### GenoTracker API
API service for cohort data management.
- **Location**: `microservices/genotracker/`
- **Framework**: FastAPI (port 8080)
- **Features**: Cohort data retrieval, API key authentication

### Browser API
File preparation service for the GP2 Browser application.
- **Location**: `microservices/browser_api/`
- **Framework**: FastAPI (port 8000)
- **Features**: Cohort browser file generation from clinical data, GCS export

### IDAT Utils
Illumina IDAT file processing toolkit.
- **Location**: `microservices/idat_utils/`
- **Features**: IDAT to VCF conversion, SNP metrics extraction, Parquet output, partitioned storage

### SNP Checksums
Duplicate sample detection service using genotype-based hashing.
- **Location**: `microservices/snp_checksums/`
- **Features**: High-callrate SNP selection, genotype hashing, duplicate identification

### GT-Precheck
Preprocessing validation service.
- **Location**: `microservices/gt-precheck/`

### Carriers API (DEPRECATED)
Legacy carrier analysis API, superseded by the Precision Med API.
- **Location**: `microservices/carriers_api/`

## Workflows

### IDAT-PED-BED Merge
Argo Workflow for batch processing IDAT files through the PED/BED conversion pipeline.
- **Location**: `workflows/idat-ped-bed-merge/`

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Languages** | Python 3.8+ (3.11 for genotools-api) |
| **Web Frameworks** | FastAPI, Streamlit |
| **ASGI Server** | uvicorn |
| **Data Processing** | Pandas, NumPy, PyArrow, pgenlib, SciPy |
| **ML / Stats** | scikit-learn, XGBoost, UMAP, numba |
| **Visualization** | Plotly, Matplotlib, Seaborn, geneview |
| **Cloud** | Google Cloud Storage, GCP Secret Manager |
| **Orchestration** | GKE, Helm, Docker, Argo Workflows, ArgoCD |
| **CI/CD** | Google Cloud Build |
| **External Tools** | PLINK 2.0 |

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Google Cloud SDK (`gcloud`)
- Access to GP2 GCS buckets

### Running a Service Locally

Each service can be run independently. For example:

**GenoTools API:**
```bash
cd microservices/genotools-api
poetry install
export API_KEY_NAME="X-API-KEY"
export API_KEY="your_api_key"
uvicorn genotools_api.main:app --host 0.0.0.0 --port 8080
```

**Precision Med API:**
```bash
cd microservices/precision_med_api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Mount GCS buckets
gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server

# Run via CLI (fastest)
python run_carriers_pipeline.py --ancestries AAC AFR

# Or run the API server
python start_api.py
```

**GP2 Browser:**
```bash
cd apps/gp2-browser
pip install -r requirements.txt
streamlit run Home.py
```

### Building Docker Images

```bash
# Example: GenoTools API
docker build -t genotools-api -f microservices/genotools-api/Dockerfile microservices/genotools-api/
docker run -d -p 8080:8080 -e API_KEY_NAME="X-API-KEY" -e API_KEY="your_key" genotools-api
```

## API Reference

### GenoTools API (port 8080)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/run-genotools/` | Execute GenoTools QC command (requires `X-API-Key` header) |

### Precision Med API (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root info |
| `GET` | `/api/v1/carriers/health` | Health check |
| `POST` | `/api/v1/carriers/pipeline` | Submit pipeline job |
| `GET` | `/api/v1/carriers/pipeline/{job_id}` | Check job status |
| `GET` | `/api/v1/carriers/pipeline/{job_id}/results` | Get job results |

### GenoTracker API (port 8080)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/data` | Get cohort data (requires `X-API-Key` header) |

### Browser API (port 8000)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/process_browsers` | Prepare browser files (background task) |

## Deployment

### GKE Cluster Setup

```bash
# 1. Provision cluster
cd deployments/gke-setup/cluster
bash cluster-setup.sh

# 2. Create node pools
cd ../nodepools
bash nodepools-setup.sh

# 3. Install third-party packages (cert-manager, external-secrets)
cd ../../helm-charts/packages
bash dep-packages.sh
```

### Helm Deployment

The deployment follows a 3-stage TLS certificate progression:

```bash
# Stage 1: Deploy with HTTP (dev)
cd deployments/helm-charts/dev
helm install genotools-server .

# Stage 2: Enable staging TLS certificates (~10 min wait)
helm upgrade genotools-server ../staging

# Stage 3: Switch to production TLS certificates
helm upgrade genotools-server ../prod
```

### Deployed Services

**Apps** (NodePool: gp2-browser-apps):
- GP2 Browser (Streamlit, port 8501)
- GenoTracker App (Streamlit, port 8501)
- Precision Med Viewer (Streamlit, port 8501)

**APIs** (Dedicated NodePools):
- GenoTools API (FastAPI, port 8080)
- GenoTracker API (FastAPI, port 8080)
- Precision Med API (FastAPI, port 8000)
- Browser API (FastAPI, port 8000)
- GT-Precheck API (FastAPI)

**Supporting Services**:
- cert-manager (TLS via Let's Encrypt)
- External Secrets Operator (GCP Secret Manager integration)
- Argo Workflows (batch job orchestration)

## CI/CD

The project uses **Google Cloud Build** for CI and **ArgoCD** for CD:

1. Cloud Build triggers on push:
   - Builds Docker images for all services
   - Pushes to Google Artifact Registry (`europe-west4-docker.pkg.dev`)
   - Updates Helm `values.yaml` with new image tags (commit SHA)
   - Commits updated manifests back to the repository
2. ArgoCD detects manifest changes and syncs deployments to GKE

Configuration files:
- `cloudbuild.yaml` - Build pipeline definition
- `apps-argo-cd.yaml` - ArgoCD application manifest
- `workflow-argo-cd.yaml` - ArgoCD workflow manifest

## Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `API_KEY_NAME` | GenoTools API, GenoTracker | API key header name (default: `X-API-KEY`) |
| `API_KEY` / `API_TOKEN` | GenoTools API | Secret API key value |
| `GTRACKER_API` | GenoTracker | GenoTracker API key |
| `PAT_TOKEN` | GenoTools API | Gmail PAT for email notifications |
| `AUTO_OPTIMIZE` | Precision Med API | Enable performance auto-optimization |
| `CHUNK_SIZE` | Precision Med API | Processing chunk size |
| `MAX_WORKERS` | Precision Med API | Max parallel workers |
| `GOOGLE_APPLICATION_CREDENTIALS` | All | Path to GCP service account key |

In production, secrets are stored in **GCP Secret Manager** and injected into pods via the **External Secrets Operator**.

---

Each service contains its own README with more detailed setup and usage instructions.
