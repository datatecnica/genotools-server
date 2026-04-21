# Preflight Input Validation

**Status: Implemented** (`run_carriers_pipeline.py`, commit `6f1ec8b`)

The pipeline runs a preflight check immediately after settings are initialised and
before any extraction starts. If any required file is missing it prints a clear error
report and exits with a non-zero status, avoiding the original failure mode where a
40-minute extraction would complete successfully before a missing clinical data file
caused the postprocessing steps to crash.

---

## What Gets Checked

| Check | Condition | Required by |
|-------|-----------|-------------|
| SNP list | Always | Extraction |
| Master key | Unless `--skip-locus-reports` AND `--skip-variant-report` | Locus reports, variant report |
| Data dictionary | Unless `--skip-locus-reports` | Locus reports |
| NBA base path | When `NBA` in `--data-types` | Extraction: NBA |
| WGS base path | When `WGS` in `--data-types` | Extraction: WGS |

File paths are resolved from `app/core/config.py` (`settings.snp_list_path`,
`settings.get_clinical_paths()`, `settings.nba_base_path`, `settings.wgs_base_path`).
Passing `--nba-path` or `--wgs-path` overrides the default release-convention paths
before the check runs, so non-standard layouts (e.g. R12 joint-calling) are validated
correctly.

---

## CLI Flag

```bash
--skip-preflight    # Disable preflight checks (rarely needed)
```

Use `--skip-preflight` only when a file will be available by the time it is needed but
does not yet exist at check time (e.g. a lazily-mounted GCS path).

---

## Example Output on Failure

```
PREFLIGHT FAILED — missing required files:
  x Master key not found: ~/gcs_mounts/gp2tier2_vwb/release12/clinical_data/master_key_release12_final_vwb.csv

Fix the above before running the pipeline.
```

---

## Running Extraction Only (When Clinical Files Are Missing)

If the master key for a new release is in a non-standard location, run extraction
only and handle postprocessing separately once the file is located:

```bash
# Step 1: extraction only (skips clinical file checks)
python run_carriers_pipeline.py --release 12 ... --skip-locus-reports --skip-variant-report

# Step 2: find the master key
find ~/gcs_mounts -name "master_key*release12*" 2>/dev/null

# Step 3: postprocessing only (reuses extraction results)
python run_carriers_pipeline.py --release 12 ... --skip-extraction
```
