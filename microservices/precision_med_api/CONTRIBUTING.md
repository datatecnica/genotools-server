# Contributing to the Precision Medicine Carriers Pipeline

This guide is for engineers taking over maintenance or adding new features. Read
[README.md](README.md) first for the full project overview, then come back here.

---

## Environment Setup

**Every session, run this first.** Without it, imports fail.

```bash
cd microservices/precision_med_api
source .venv/bin/activate
```

To create the environment from scratch:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**GCS access is required for production data.** Mount the three buckets before running
the pipeline against real data:

```bash
gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
gcsfuse --implicit-dirs gp2_release12 ~/gcs_mounts/gp2_release12
```

If you don't have GCS access yet, you can still run unit tests and explore the code.
The pipeline itself cannot run without PLINK files on GCS.

---

## Running the Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

There is currently one test module (`tests/test_transformer.py`). Run it before
committing changes to any processing module. If you add new functionality, add a
corresponding test.

---

## Development Workflow

### Iterating on postprocessing (probe selection, locus reports, variant report)

Use `--skip-extraction` to reuse a previous extraction run. This avoids the ~45-minute
PLINK extraction and lets you iterate on postprocessing logic in seconds:

```bash
python run_carriers_pipeline.py --job-name existing_job --skip-extraction
```

### Quick validation run (~10 minutes)

Test with two ancestries rather than all eleven:

```bash
python run_carriers_pipeline.py --release 11 --ancestries AAC AFR --job-name test_run
```

### Full pipeline run (~45 minutes)

```bash
python run_carriers_pipeline.py --release 11 --job-name release11
```

---

## Before You Change Anything

These four files have the most downstream dependencies. Read them before touching
anything nearby:

| File | What it does |
|------|-------------|
| [app/processing/extractor.py](app/processing/extractor.py) | Allele counting, MAF correction — the core extraction logic |
| [app/processing/coordinator.py](app/processing/coordinator.py) | ProcessPool orchestration, multi-ancestry merge |
| [app/processing/locus_report_generator.py](app/processing/locus_report_generator.py) | Clinical phenotype analysis, probe-filtered carrier stats |
| [app/models/harmonization.py](app/models/harmonization.py) | Pydantic models depended on by almost everything |

---

## Critical Behaviours — Do Not Regress

These bugs were hard to find and fix. The logic may look surprising — it is intentional.

**Allele counting**: The pipeline counts *pathogenic* alleles, not reference alleles.
Genotype values are `0=none, 1=het carrier, 2=hom carrier`. This is not the PLINK
default.

**MAF correction** (`extractor.py::_apply_maf_correction`): When a pathogenic allele
is the minor allele (ALT AF > 0.5), genotypes are flipped so carriers are counted
correctly. Without this, rs3115534 (GBA1, ~3% frequency) would show zero carriers.

**Sample ID normalisation**: Sample IDs are stripped of the `0_` prefix. WGS IDs of
the form `SAMPLE_001234_SAMPLE_001234` are deduplicated to `SAMPLE_001234`. Applied
consistently across all data types.

**Multi-ancestry merge**: NBA and IMPUTED use a concat-then-merge strategy:
1. `pd.concat()` within ancestry (same samples, different variants)
2. `pd.merge()` across ancestries (different samples)

Using a flat merge here previously caused entire variants to be silently zeroed out.
`combine_first()` handles any `_dup` columns from cross-ancestry column overlap.

**IMPUTED dosage thresholds**: IMPUTED files contain dosage values (`0.0–2.0`), not
discrete genotypes. Carrier calls use configurable thresholds
(`--dosage-het-min`, `--dosage-het-max`, `--dosage-hom-min`).

---

## Implementation Rules

- **Changing > 3 files or adding a new module**: Write a short plan first and discuss
  before implementing.
- **Bug fix or enhancement to an existing function**: Implement directly.
- **Do not create new files** unless genuinely necessary. Prefer editing existing ones.
- **Do not add speculative features** — implement only what is asked.
- **Run tests before committing**: `python -m pytest tests/ -v`

---

## PR Process

1. Branch from `main` using the naming convention `pmed/<short-description>`.
2. Keep PRs focused — one logical change per PR.
3. Run `python -m pytest tests/ -v` and confirm all tests pass.
4. For pipeline changes, do a quick validation run (`--ancestries AAC AFR`) if you
   have GCS access.
5. Update `README.md` if you add a new CLI flag, output file, or data type.

---

## Key File Paths (Production)

```
SNP list:   ~/gcs_mounts/genotools_server/precision_med/summary_data/precision_med_snp_list.csv
Input:      ~/gcs_mounts/gp2tier2_vwb/release{N}/
Output:     ~/gcs_mounts/genotools_server/precision_med/results/release{N}/
Cache:      ~/gcs_mounts/genotools_server/precision_med/cache/
```

All paths are derived from `app/core/config.py`. Override non-standard layouts with
`--nba-path` and `--wgs-path` at the CLI.

---

## Where to Look for More Context

| Document | What's in it |
|----------|-------------|
| [README.md](README.md) | Full usage guide, all CLI flags, output formats, architecture |
| [OUTPUT_DICTIONARY.md](OUTPUT_DICTIONARY.md) | Complete column-level reference for every output file |
| [CLAUDE.md](CLAUDE.md) | Quick commands and critical implementation notes (terse reference) |
| [docs/PREFLIGHT_CHECKS.md](docs/PREFLIGHT_CHECKS.md) | Preflight validation — what gets checked and when |
| [docs/PARALLEL_PROCESSING_PLAN.md](docs/PARALLEL_PROCESSING_PLAN.md) | Memory-safe parallelisation: what's implemented and what's planned |
