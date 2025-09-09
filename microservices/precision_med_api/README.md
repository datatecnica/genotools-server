# Precision Med API

A microservice for processing genetic carrier information. This API allows you to extract carrier data from genotype files based on specified SNP lists.

## Overview

This service is part of the Genotools Server framework, designed to analyze genetic variant data across different ancestry populations. It provides an easy-to-use interface for extracting carrier information from PLINK2 formatted genotype files.

## Data Types Supported

The API supports three main types of genetic data:

### 1. NBA (NeuroBooster Array) Data
- **Format**: Single consolidated PLINK2 files per ancestry
- **Structure**: `{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- **Example**: `AAC_release10_vwb.pgen`, `AAC_release10_vwb.pvar`, `AAC_release10_vwb.psam`
- **Processing**: Direct single-file processing

### 2. WGS (Whole Genome Sequencing) Data  
- **Format**: Single consolidated PLINK2 files across all ancestries
- **Structure**: `R{version}_wgs_carrier_vars.{pgen|pvar|psam}`
- **Example**: `R10_wgs_carrier_vars.pgen`, `R10_wgs_carrier_vars.pvar`, `R10_wgs_carrier_vars.psam`
- **Processing**: Direct single-file processing

### 3. Imputed Data
- **Format**: Chromosome-split PLINK2 files per ancestry
- **Structure**: `chr{chrom}_{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- **Example**: `chr1_AAC_release10_vwb.pgen`, `chr1_AAC_release10_vwb.pvar`, `chr1_AAC_release10_vwb.psam`
- **Processing**: Multi-chromosome processing with automatic combination

## Installation

### Prerequisites

- Python 3.8+
- FastAPI
- Pandas
- PLINK2 (must be installed and available on PATH)

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd carriers_api
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. If using GCS storage:
   ```
   gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
   gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
   ```

## Usage

### Starting the API Server

Start the FastAPI server with:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
