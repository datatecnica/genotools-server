Browser API
===========
FastAPI service that prepares cohort browser files from GP2 release data when the clinical_data folder is populated in the respective release folder of `gs://gp2tier2_vwb`.

## Setup
- Python 3.13+ recommended; install deps:
  - `pip install -r microservices/browser_api/requirements.txt`
- Mount the needed GCS buckets with gcsfuse (read-only for release data, read/write for outputs):
  - `gcsfuse --dir-mode 555 --file-mode 444 --implicit-dirs gp2tier2_vwb gcs_mounts/gp2tier2_vwb`
  - `gcsfuse --dir-mode 777 --file-mode 777 --implicit-dirs genotools-server gcs_mounts/genotools-server`

## Run
- Terminal 1 (API): start the FastAPI app:
  - `uvicorn main:app --host 0.0.0.0 --port 8000`
- Terminal 2 (pipeline driver): from repo root, kick off the prep script `genotools-server/microservices/browser_api/run_browser_prep.py`:
  - `python run_browser_prep.py --mnt-dir gcs_mounts --release 10 --api-url http://localhost:8000`
- Health check: `curl http://localhost:8000/health`

Outputs
- Processed browser files are exported to `gs://genotools-server/cohort_browser/nba/release10/` when using the commands above.
