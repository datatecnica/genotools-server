# Precision Medicine Carriers Pipeline API

RESTful API for executing the genomic carrier screening pipeline with the same functionality as the CLI script.

## Quick Start

### 1. Install Dependencies
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start API Server
```bash
./run_api.sh
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/carriers/health

### 3. Test the API
```bash
python test_api.py
```

## API Endpoints

### POST /api/v1/carriers/pipeline
Start a new pipeline execution job.

**Request Body:**
```json
{
  "job_name": "my_analysis",
  "ancestries": ["AAC", "AFR", "EUR"],
  "data_types": ["NBA", "WGS", "IMPUTED"],
  "parallel": true,
  "max_workers": null,
  "optimize": true,
  "skip_extraction": false,
  "skip_probe_selection": false,
  "skip_locus_reports": false,
  "output_dir": null
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Pipeline job created. Use /pipeline/{job_id} to check status."
}
```

### GET /api/v1/carriers/pipeline/{job_id}
Get job status and results (if completed).

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "running",
  "progress": "Processing NBA data...",
  "started_at": "2025-10-09T10:00:00",
  "completed_at": null,
  "execution_time_seconds": null,
  "result": null
}
```

### GET /api/v1/carriers/pipeline/{job_id}/results
Get completed job results (only works when job is completed).

**Response:**
```json
{
  "success": true,
  "job_id": "carriers_analysis",
  "status": "completed",
  "message": "Pipeline completed successfully",
  "execution_time_seconds": 450.2,
  "output_files": {
    "NBA_parquet": "/path/to/carriers_analysis_NBA.parquet",
    "WGS_parquet": "/path/to/carriers_analysis_WGS.parquet",
    "IMPUTED_parquet": "/path/to/carriers_analysis_IMPUTED.parquet"
  },
  "summary": {
    "total_variants": 400,
    "total_samples": 15000,
    "by_data_type": {
      "NBA": 380,
      "WGS": 395,
      "IMPUTED": 400
    }
  },
  "errors": null
}
```

### GET /api/v1/carriers/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-09T10:00:00",
  "version": "1.0.0"
}
```

## Parameters

### job_name
Human-readable name for the job. Used for output file naming.
- **Type**: string
- **Default**: "carriers_analysis"

### ancestries
List of ancestries to process.
- **Type**: list of strings
- **Default**: All 11 ancestries from config (AAC, AFR, AJ, AMR, CAH, CAS, EAS, EUR, FIN, MDE, SAS)
- **Example**: ["AAC", "AFR", "EUR"]

### data_types
Data types to process.
- **Type**: list of strings
- **Choices**: "NBA", "WGS", "IMPUTED"
- **Default**: ["NBA", "WGS", "IMPUTED"]

### parallel
Enable parallel processing.
- **Type**: boolean
- **Default**: true

### max_workers
Maximum number of parallel workers. Auto-detected if not specified.
- **Type**: integer or null
- **Default**: null (auto-detect)

### optimize
Use performance optimizations (auto-detect chunk_size, process_cap, etc.).
- **Type**: boolean
- **Default**: true

### skip_extraction
Skip extraction phase if results already exist.
- **Type**: boolean
- **Default**: false

### skip_probe_selection
Skip probe selection analysis.
- **Type**: boolean
- **Default**: false

### skip_locus_reports
Skip locus report generation.
- **Type**: boolean
- **Default**: false

### output_dir
Custom output directory. Uses config-based results path if not specified.
- **Type**: string or null
- **Default**: null (use config-based path)

## Usage Examples

### Example 1: Full Pipeline (All Data Types, All Ancestries)
```bash
curl -X POST http://localhost:8000/api/v1/carriers/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "full_analysis",
    "data_types": ["NBA", "WGS", "IMPUTED"],
    "optimize": true
  }'
```

### Example 2: Quick Test (Single Ancestry, NBA Only)
```bash
curl -X POST http://localhost:8000/api/v1/carriers/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "quick_test",
    "ancestries": ["AAC"],
    "data_types": ["NBA"],
    "skip_probe_selection": true,
    "skip_locus_reports": true
  }'
```

### Example 3: Reprocess Existing Data (Skip Extraction)
```bash
curl -X POST http://localhost:8000/api/v1/carriers/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "existing_analysis",
    "skip_extraction": true
  }'
```

### Example 4: Check Job Status
```bash
curl http://localhost:8000/api/v1/carriers/pipeline/550e8400-e29b-41d4-a716-446655440000
```

### Example 5: Get Job Results
```bash
curl http://localhost:8000/api/v1/carriers/pipeline/550e8400-e29b-41d4-a716-446655440000/results
```

## Python Client Example

```python
import requests
import time

# Submit pipeline job
response = requests.post(
    "http://localhost:8000/api/v1/carriers/pipeline",
    json={
        "job_name": "my_analysis",
        "ancestries": ["AAC", "AFR"],
        "data_types": ["NBA", "WGS"],
        "optimize": true
    }
)

job_id = response.json()["job_id"]
print(f"Job submitted: {job_id}")

# Poll for completion
while True:
    status = requests.get(
        f"http://localhost:8000/api/v1/carriers/pipeline/{job_id}"
    ).json()

    print(f"Status: {status['status']} - {status.get('progress')}")

    if status["status"] in ["completed", "failed"]:
        break

    time.sleep(10)

# Get results
if status["status"] == "completed":
    results = requests.get(
        f"http://localhost:8000/api/v1/carriers/pipeline/{job_id}/results"
    ).json()

    print(f"Success! Generated {len(results['output_files'])} files")
    print(f"Execution time: {results['execution_time_seconds']}s")
```

## CLI vs API Comparison

The API provides identical functionality to the CLI script:

| CLI Flag | API Parameter | Description |
|----------|---------------|-------------|
| `--job-name` | `job_name` | Job name for output files |
| `--ancestries` | `ancestries` | Ancestries to process |
| `--data-types` | `data_types` | Data types to process |
| `--parallel` | `parallel` | Enable parallel processing |
| `--max-workers` | `max_workers` | Maximum workers |
| `--optimize` | `optimize` | Use performance optimizations |
| `--skip-extraction` | `skip_extraction` | Skip extraction phase |
| `--skip-probe-selection` | `skip_probe_selection` | Skip probe selection |
| `--skip-locus-reports` | `skip_locus_reports` | Skip locus reports |
| `--output` | `output_dir` | Custom output directory |

## Architecture

```
app/
├── main.py                    # FastAPI application
├── api/
│   ├── routes.py             # API endpoints
│   └── __init__.py
├── services/
│   ├── pipeline_service.py   # Business logic (from CLI)
│   ├── job_manager.py        # Job tracking
│   └── __init__.py
├── models/
│   ├── api.py               # Request/response models
│   └── ...
├── processing/              # Core pipeline (unchanged)
└── core/
    └── config.py           # Settings (unchanged)
```

## Notes

- Jobs run in background tasks (non-blocking)
- Job tracking is in-memory (restart clears job history)
- CLI script remains fully functional alongside API
- Same output files and directory structure as CLI
- Same performance optimizations and auto-detection
