# CLAUDE.md - Development Instructions

## Always Run First

```bash
source .venv/bin/activate
# Without this, imports will fail (pydantic, pandas, pgenlib, etc.)
```

## Quick Commands

### Running the Pipeline

```bash
# Direct CLI (fastest for development)
python run_carriers_pipeline.py --ancestries AAC AFR

# Full pipeline
python run_carriers_pipeline.py --job-name release10

# Skip extraction (rapid iteration)
python run_carriers_pipeline.py --job-name release10 --skip-extraction

# Via API
python start_api.py  # Terminal 1
python run_carriers_pipeline_api.py --job-name release10  # Terminal 2
```

### Frontend

```bash
./frontend/run_app.sh              # Production mode
./frontend/run_app.sh --debug      # Debug mode with job selection
```

### Testing

```bash
python -m pytest tests/ -v     # Unit tests
python test_nba_pipeline.py    # NBA ProcessPool test
```

## File Paths

```
Input:  ~/gcs_mounts/gp2tier2_vwb/release10/
Cache:  ~/gcs_mounts/genotools_server/precision_med/cache/
Output: ~/gcs_mounts/genotools_server/precision_med/results/release10/
SNP List: ~/gcs_mounts/genotools_server/precision_med/summary_data/precision_med_snp_list.csv
```

## Critical Fixes (Don't Regress)

### Allele Counting
- Counts **pathogenic alleles**, not reference alleles
- Fixed in `extractor.py` with proper genotype transformation
- Genotype values: 0=none, 1=het carrier, 2=hom carrier

### Minor Allele Frequency (MAF) Correction
- Automatically flips genotypes when ALT AF > 0.5 to count the minor allele
- Critical for rs3115534 (GBA1) where pathogenic G allele is only 3% frequency
- Adds `maf_corrected` (bool) and `original_alt_af` (float) columns to output
- Implemented in `extractor.py::_apply_maf_correction()`
- Without this fix, carriers of the pathogenic G allele would be missed

### Sample IDs
- Normalized without '0_' prefix
- WGS duplicates fixed: `SAMPLE_001234_SAMPLE_001234` → `SAMPLE_001234`
- Applied consistently across all data types

### Multiple Probes
- Fixed deduplication to preserve different variant_id values
- 77 SNPs have multiple probes (different NBA probes for same position)
- Probe selection validates quality against WGS ground truth

### Multi-Ancestry Merge (CRITICAL - Refactored 2025-12-15)
- NBA/IMPUTED merge uses efficient concat+merge approach:
  1. Group DataFrames by ancestry
  2. `pd.concat()` within ancestry (same samples, different variants)
  3. `pd.merge()` across ancestries (different samples)
- `combine_first()` handles any `_dup` columns from cross-ancestry overlap
- See `DEBUGGING_IMPUTED_ISSUE.md` for full history and implementation details

### Probe Selection Integration
- Locus reports now filter to use only selected probes
- Single-probe mutations kept by default
- Multi-probe mutations: keep consensus-recommended probe only
- Implemented in `locus_report_generator.py` + `probe_selector_loader.py`

## Core Files (Test After Changes)

```
app/processing/extractor.py              # Allele counting logic
app/processing/coordinator.py            # ProcessPool orchestration
app/processing/locus_report_generator.py # Clinical phenotype analysis
app/models/harmonization.py              # Data models (many dependencies)
```

## Implementation Rules

- If changing >3 files or adding new module: Create plan first
- If fixing bug or enhancing existing function: Direct implementation
- Never create files unless absolutely necessary
- Always prefer editing existing files to creating new ones
- Never create documentation unless explicitly requested

## Pipeline Execution

```bash
# Full pipeline (~45 minutes)
python run_carriers_pipeline.py --job-name my_analysis

# Rapid development (0.0s - reuses existing results)
python run_carriers_pipeline.py --job-name existing_analysis --skip-extraction

# Quick validation (5-10 minutes)
python run_carriers_pipeline.py --ancestries AAC AFR

# Probe selection (enabled by default)
python run_carriers_pipeline.py                        # probe selection enabled
python run_carriers_pipeline.py --skip-probe-selection # skip probe selection
```

## Architecture Context

- Core pipeline stable with probe selection, allele counting fixes, sample ID normalization
- Frontend: Modular architecture with factory/facade patterns
- Probe selection: Validates NBA probes against WGS ground truth
- Locus reports: Per-gene clinical phenotype statistics stratified by ancestry
- Mature codebase: Prefer enhancing existing modules over creating new files
- Performance: ProcessPool parallelization, <10min for 400 variants

## Development Tips

- Use `--skip-extraction` for rapid iteration on postprocessing logic
- Test changes incrementally - pipeline is in production use
- Run unit tests before committing: `python -m pytest tests/ -v`
- Frontend changes: Test with `./frontend/run_app.sh --debug`
