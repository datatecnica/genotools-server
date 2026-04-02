# Preflight Input Validation

## The Problem

The pipeline currently validates inputs lazily — files are only opened when the code
that needs them runs. For a 40-minute extraction run, this means a missing file in a
postprocessing step (locus reports, variant report) is not discovered until after all
the expensive PLINK work has already completed.

**Example (R12, 2026-03-31):** Extraction ran for ~40 minutes successfully. Both the
locus report and variant report then failed immediately because the master key file
did not exist at the expected path:

```
FileNotFoundError: .../gp2tier2_vwb/release12/clinical_data/master_key_release12_final_vwb.csv
```

This file is required by `LocusReportGenerator.__init__()` and is looked up via
`settings.get_clinical_paths()` using a path constructed from the release number. For
non-standard release layouts (e.g. R12 with `--nba-path`/`--wgs-path` overrides), the
clinical data may live in a different location than the default convention expects.

---

## Required Input Files

The following files must exist before the pipeline starts. Their paths are derived from
`app/core/config.py`.

| File | Path (default convention) | Required by |
|------|--------------------------|-------------|
| SNP list | `genotools_server/precision_med/summary_data/precision_med_snp_list.csv` | Extraction (always) |
| Master key | `gp2tier2_vwb/release{N}/clinical_data/master_key_release{N}_final_vwb.csv` | Locus reports, variant report |
| Data dictionary | `gp2tier2_vwb/release{N}/clinical_data/master_key_release{N}_data_dictionary.csv` | Locus reports |
| Extended clinical | `gp2tier2_vwb/release{N}/clinical_data/r{N}_extended_clinical_data_vwb.csv` | Locus reports |
| NBA PGEN files | `--nba-path` (or default release path) | Extraction: NBA |
| WGS PGEN files | `--wgs-path` (or default release path) | Extraction: WGS |

---

## Proposed Solution: Preflight Check

Add a `--dry-run` / preflight validation step that runs before extraction and checks
all required files exist, failing fast with a clear report of what is missing.

### Implementation sketch

New function in `run_carriers_pipeline.py` (or `coordinator.py`):

```python
def preflight_check(settings: Settings, data_types: List[str], skip_flags: dict) -> bool:
    """
    Check all required input files exist before starting extraction.
    Returns True if all checks pass, False (with logged errors) if any fail.
    """
    errors = []

    # Always required
    if not os.path.exists(settings.snp_list_path):
        errors.append(f"SNP list not found: {settings.snp_list_path}")

    # Required unless postprocessing is skipped
    if not skip_flags.get('locus_reports') and not skip_flags.get('variant_report'):
        clinical = settings.get_clinical_paths()
        if not os.path.exists(clinical['master_key']):
            errors.append(f"Master key not found: {clinical['master_key']}")
        if not os.path.exists(clinical['data_dictionary']):
            errors.append(f"Data dictionary not found: {clinical['data_dictionary']}")

    # Spot-check PGEN input directories exist (not every file — just the base paths)
    if 'NBA' in data_types:
        nba_base = settings.nba_base_path or settings.release_path
        if not os.path.exists(nba_base):
            errors.append(f"NBA base path not found: {nba_base}")

    if 'WGS' in data_types:
        wgs_base = settings.wgs_base_path or settings.release_path
        if not os.path.exists(wgs_base):
            errors.append(f"WGS base path not found: {wgs_base}")

    if errors:
        print("\nPREFLIGHT FAILED — missing required files:")
        for e in errors:
            print(f"  ✗ {e}")
        print("\nFix the above before running the pipeline.\n")
        return False

    return True
```

Call it immediately after settings are initialised, before extraction starts:

```python
if not preflight_check(settings, args.data_types, skip_flags):
    sys.exit(1)
```

### CLI flag (optional)

Add `--skip-preflight` for cases where the user knows a file will be available by the
time it is needed (e.g. mounted lazily) or wants to run extraction only and handle
postprocessing separately.

---

## Workaround Until Implemented

If the master key for a new release is in a non-standard location, use
`--skip-locus-reports` and `--skip-variant-report` to run extraction only, then
locate the file and re-run postprocessing with `--skip-extraction`:

```bash
# Step 1: extraction only
python run_carriers_pipeline.py --release 12 ... --skip-locus-reports --skip-variant-report

# Step 2: find the master key
find ~/gcs_mounts -name "master_key*release12*" 2>/dev/null

# Step 3: postprocessing only (once path is known / file is in place)
python run_carriers_pipeline.py --release 12 ... --skip-extraction
```