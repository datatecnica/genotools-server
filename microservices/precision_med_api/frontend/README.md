# Carriers Pipeline Results Viewer

Frontend for viewing precision medicine carriers pipeline results.

## Setup

1. **Clone and install dependencies:**
   ```bash
   git clone <repository-url>
   cd precision_med

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Mount GCS buckets (for production):**
   ```bash
   gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
   gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
   ```

## Configuration

The frontend automatically discovers releases in your results directory.

**Default path:**
```
~/gcs_mounts/genotools_server/precision_med/results/release10
```

**Custom path via environment variable:**
```bash
export RESULTS_PATH=/path/to/your/results
./run_frontend.sh
```

**Custom path via code:**
Edit `app/config.py` line 40 to change the default path.

## Launch

```bash
# Production mode (default port 8501)
./run_frontend.sh

# Debug mode with job selection
./run_frontend.sh --debug

# Custom port
./run_frontend.sh 8502
```

## Expected Data Files

The frontend reads pre-generated pipeline output files:

```
results/
└── release10/
    ├── release10_pipeline_results.json
    ├── release10_WGS.parquet
    ├── release10_NBA.parquet
    ├── release10_IMPUTED.parquet
    ├── release10_locus_reports_WGS.json
    ├── release10_locus_reports_NBA.json
    ├── release10_locus_reports_IMPUTED.json
    └── release10_probe_selection.json
```

## Features

- **Release Overview**: Pipeline execution summary, variant counts, data type breakdown
- **Locus Reports**: Per-gene clinical phenotype statistics with ancestry stratification
- **Probe Validation**: NBA probe quality metrics and selection recommendations
- **Genotype Viewer**: Interactive genotype matrix visualization

## Debug Mode

Debug mode enables:
- Job selection dropdown (for multiple pipeline runs)
- Cache clearing tools
- Extended logging
- Development features

## Troubleshooting

**No releases found:**
- Check that `RESULTS_PATH` points to the correct directory
- Ensure the directory contains `release*` subdirectories
- Verify GCS mounts are accessible if using default path

**Missing data:**
- Ensure pipeline has been run and generated output files
- Check that JSON and parquet files exist in the release directory
- Run pipeline without `--skip-locus-reports` to generate all data

**Import errors:**
- Verify all required packages are installed
- Check Python version (requires Python 3.8+)