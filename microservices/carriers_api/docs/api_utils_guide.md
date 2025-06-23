# API Utilities Guide

The `src/core/api_utils.py` module provides utilities for interacting with the Carriers API in a clean and consistent way. This guide explains how to use these utilities effectively.

## Available Functions

### `post_to_api(url, payload)`

Sends a POST request to the API and processes the response in a formatted way.

**Parameters:**
- `url` (str): The API endpoint URL
- `payload` (dict): The request payload

**Example:**
```python
from src.core.api_utils import post_to_api

payload = {
    "geno_path": "/path/to/genotype/files_prefix",
    "key_file_path": "/path/to/key_file.csv",
    "snplist_path": "/path/to/snplist.csv",
    "out_path": "/path/to/output_prefix",
    "release_version": "10"
}

api_url = "http://localhost:8000/process_carriers"
post_to_api(api_url, payload)
```

**Output:**
```
Status code: 200 - Success!
Output Files:
- var_info: /path/to/output_prefix_var_info.csv
- carriers_string: /path/to/output_prefix_carriers_string.csv
- carriers_int: /path/to/output_prefix_carriers_int.csv
```

### `format_error_response(response_text)`

Formats error responses from the API in a readable way.

**Parameters:**
- `response_text` (str): The raw response text from the API

**Example:**
```python
import requests
from src.core.api_utils import format_error_response

response = requests.post("http://localhost:8000/process_carriers", json=payload)
if response.status_code != 200:
    format_error_response(response.text)
```

**Output for error case:**
```
Error Message:
  Processing failed: Could not find genotype file at specified path

Traceback:
File "/path/to/app/main.py", line 45, in process_carriers
    results = manager.extract_carriers(
...
```

## Batch Processing Example

Here's a complete example of batch processing carriers data for multiple populations:

```python
import pandas as pd
import os
import requests
from src.core.api_utils import post_to_api

# Define paths
mnt_dir = '/home/user/data'
carriers_dir = f'{mnt_dir}/carriers'
summary_dir = f'{carriers_dir}/summary'

release = '10'
release_dir = '/path/to/release_data/'
key_file = '/path/to/master_key.csv'
output_dir = f'/path/to/output/release{release}'

# Population labels
labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN','MDE', 'SAS']

# Check API health
response = requests.get("http://localhost:8000/health")
print(f"Health check: {response.json()}")

# Process each population
for label in labels:
    print(f"\nProcessing {label} population...")
    
    # Create output directory if it doesn't exist
    os.makedirs(f'{output_dir}/{label}', exist_ok=True)
    
    # Prepare API request payload
    payload = {
        "geno_path": f"{release_dir}/{label}/{label}_release{release}_prefix",
        "key_file_path": key_file,
        "snplist_path": f'{summary_dir}/carriers_report_snps_full.csv',
        "out_path": f"{output_dir}/{label}/{label}_release{release}",
        "release_version": release
    }

    # Send request to API
    api_url = "http://localhost:8000/process_carriers"
    post_to_api(api_url, payload)
```

## Notes

1. The `post_to_api` function handles both successful and error responses automatically.
2. For successful responses, the function displays the status code and the paths to all output files.
3. For error responses, the function formats the error message and traceback in a readable format.
4. All output is formatted as plain text for compatibility with both terminal and Jupyter environments.