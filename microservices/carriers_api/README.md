# Carriers API

A microservice for processing genetic carrier information. This API allows you to extract carrier data from genotype files based on specified SNP lists.

## Overview

This service is part of the Genotools Server framework, designed to analyze genetic variant data across different ancestry populations. It provides an easy-to-use interface for extracting carrier information from PLINK2 formatted genotype files.

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
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### Health Check
```
GET /health
```
Returns the health status of the API.

#### Process Carriers
```
POST /process_carriers
```
Processes carrier information from genotype files and returns paths to the generated output files.

Request Body:
```json
{
  "geno_path": "/path/to/plink/files/prefix",
  "key_file_path": "/path/to/key/file.csv",
  "snplist_path": "/path/to/snplist.csv",
  "out_path": "/path/to/output/prefix",
  "release_version": "10"
}
```

Response:
```json
{
  "status": "success",
  "outputs": {
    "var_info": "/path/to/output/prefix_var_info.csv",
    "carriers_string": "/path/to/output/prefix_carriers_string.csv",
    "carriers_int": "/path/to/output/prefix_carriers_int.csv"
  }
}
```

### Example Script Usage

The repository includes a sample script `run_carriers.py` that demonstrates how to use the API:

```python
import pandas as pd
import os
import requests
from src.core.api_utils import format_error_response, post_to_api

# Define paths
mnt_dir = '/home/user/gcs_mounts'
carriers_dir = f'{mnt_dir}/genotools_server/carriers'
summary_dir = f'{carriers_dir}/summary_data'

release = '10'
release_dir = '/path/to/genotypes/'
key_file = '/path/to/master_key.csv'
output_dir = f'/path/to/output/release{release}'

# Population labels
labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN','MDE', 'SAS']

# Check API health
response = requests.get("http://localhost:8000/health")
print(f"Health check: {response.json()}")

# Process each population
for label in labels:
    os.makedirs(f'{output_dir}/{label}', exist_ok=True)
    
    payload = {
        "geno_path": f"{release_dir}/{label}/{label}_release{release}_prefix",
        "key_file_path": key_file,
        "snplist_path": f'{summary_dir}/carriers_report_snps_full.csv',
        "out_path": f"{output_dir}/{label}/{label}_release{release}",
        "release_version": "10"
    }

    api_url = "http://localhost:8000/process_carriers"
    post_to_api(api_url, payload)
```

## File Structure

- `main.py` - FastAPI server definition
- `run_carriers.py` - Example script to process multiple populations
- `src/core/` - Core implementation modules
  - `api_utils.py` - Utility functions for API interaction
  - `carrier_processor.py` - Processing logic for carrier extraction
  - `harmonizer.py` - Allele harmonization utilities
  - `manager.py` - Orchestration of carrier analysis workflow
  - `carrier_validator.py` - Validation logic for carrier data
  - `data_repository.py` - Data access layer
  - `genotype_converter.py` - Utilities for genotype format conversion

## Output Files

Processing carriers generates three output files:

1. `*_var_info.csv` - Comprehensive variant information including:
   - Original input metadata (snp_name, locus, rsid, hg38, hg19, etc.)
   - Harmonization results (genotype ID mapping)
   - PLINK statistics (frequencies, missingness rates)
   - PLINK metadata (chromosome format, positions, alleles)
2. `*_carriers_string.csv` - Carriers in string format (e.g., A/G)
3. `*_carriers_int.csv` - Carriers in integer format

## Error Handling

The API provides detailed error responses including:
- Error message
- Stack trace (for debugging)

The `api_utils` module contains utilities for processing and displaying these errors in a readable format.