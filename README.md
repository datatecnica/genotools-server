# GenoTools Server

Monorepo containing genomic data processing services and applications for the Global Parkinson's Genetics Program (GP2).

## Applications

### GP2 Browser
Streamlit web application for browsing GP2 cohort data releases.
- **Location**: `apps/gp2_browser/`
- **Features**: Data releases, quality control, ancestry analysis, SNP metrics, rare variants

### Precision Medicine Viewer
Interactive Streamlit application for exploring precision medicine carriers pipeline results.
- **Location**: `apps/precision_med/`
- **Features**: Release overview, genotype viewer, locus reports, probe validation
- **Integration**: Visualizes output from Precision Med API pipeline

## Microservices

### GenoTools API
RESTful API interface for genomic data quality control and analysis.
- **Location**: `microservices/genotools_api/`
- **Features**: GenoTools command execution, GCS integration, API key authentication
- **Deployment**: Docker, Kubernetes (GKE)

### GenoTracker
FastAPI + Streamlit application for managing genomic cohort data.
- **Location**: `microservices/genotracker/`
- **Features**: Data visualization, local/GCS data support, interactive exploration

### Precision Med API
High-performance genomic carrier screening pipeline for pathogenic variant analysis.
- **Location**: `microservices/precision_med_api/`
- **Features**: Real-time allele harmonization, parallel processing, probe validation, clinical phenotype integration
- **Performance**: ~10 min extraction for 400 variants across 254 PLINK files
- **Data Sources**: NBA, WGS, and IMPUTED genotypes with ancestry stratification

### Carriers API (DEPRECATED)
API service for genetic carrier analysis.
- **Location**: `microservices/carriers_api/`
- **Features**: Carrier status processing, PLINK integration
- **Status**: Deprecated in favor of Precision Med API

### IDAT Utils
Illumina IDAT file processing toolkit for SNP metrics generation.
- **Location**: `microservices/idat_utils/`
- **Features**: IDAT to VCF conversion, SNP metrics extraction, parquet output

### SNP Checksums
Duplicate detection service using SNP-based hashing.
- **Location**: `microservices/snp_checksums/`
- **Features**: Genotype hashing, duplicate identification, PLINK file processing

## Infrastructure

- **Batch Services**: `batch_services/`
- **Deployment**: `deploy/`

## Getting Started

Each service contains its own README with specific setup instructions. Services are containerized with Docker and can be deployed to Google Cloud Platform.
