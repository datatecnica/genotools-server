# Precision Medicine Carriers Pipeline

Genomic carrier screening system for identifying carriers of pathogenic variants in large-scale genomic data from the GP2 (Global Parkinson's Genetics Program) cohort.

## Overview

Processes ~760 pathogenic SNPs across 254+ PLINK 2.0 files from four data sources:
- **NBA (NeuroBooster Array)**: 11 files split by ancestry
- **WGS (Whole Genome Sequencing)**: 1 consolidated file
- **IMPUTED**: 242 files (11 ancestries × 22 chromosomes)
- **EXOMES (Clinical Exomes)**: 1 consolidated file (release 8+ only)

### Key Features

- Correct pathogenic allele counting (not reference alleles)
- Minor allele frequency (MAF) correction for variants where pathogenic allele is minor
- Real-time harmonization without pre-processing
- ProcessPool parallelization for concurrent file extraction
- Sample ID normalization across data types
- Probe quality validation against WGS ground truth
- Clinical phenotype integration with ancestry stratification
- Interactive web interface for result exploration

### Performance

- Extraction: <10 minutes for 400 variants across all files
- Parallelization: Auto-detected based on available RAM (not CPU count)
- Memory: Worker count capped at `floor(available_RAM * 0.8 / 20GB)` to prevent OOM
- Auto-optimization: Detects machine specs and tunes performance
- Override with `--max-workers` if OOM errors occur on large joint-calling WGS files

## Installation

### Prerequisites

- Python 3.8+
- PLINK 2.0 (optional, falls back to simulation)
- GCS mounts for production data access

### Setup

```bash
git clone <repository-url>
cd precision_med_api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Mount GCS Buckets (Production)

```bash
gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
gcsfuse --implicit-dirs gp2_release12 ~/gcs_mounts/gp2_release12
```

## Usage

### Three Execution Methods

**1. Direct CLI** - Direct script execution, fastest for development

```bash
source .venv/bin/activate
python run_carriers_pipeline.py --ancestries AAC AFR
```

**2. API Client Script** - CLI-style interface through API

```bash
python start_api.py  # Terminal 1: Start API server
python run_carriers_pipeline_api.py --ancestries AAC AFR  # Terminal 2: Submit job
```

**3. Raw API** - Programmatic HTTP access

```bash
python start_api.py  # Start server
curl -X POST http://localhost:8000/api/v1/carriers/pipeline \
  -H "Content-Type: application/json" \
  -d '{"job_name": "test", "ancestries": ["AAC", "AFR"]}'
```

### Common Commands

```bash
# Quick validation (2 ancestries, 5-10 minutes)
python run_carriers_pipeline.py --release 11 --ancestries AAC AFR

# Full pipeline - standard release layout
python run_carriers_pipeline.py --release 11 --job-name release11

# R12+ with custom paths and QC filter (joint-calling WGS layout)
python run_carriers_pipeline.py \
  --release 12 \
  --data-types NBA WGS \
  --nba-path ~/gcs_mounts/gp2_release12/variant_report_files/nba \
  --wgs-path ~/gcs_mounts/gp2_release12/variant_report_files/wgs/joint-calling/plink \
  --job-name release12 \
  --geno 0.05 \
  --max-workers 20

# Skip extraction (reuse existing results)
python run_carriers_pipeline.py --release 11 --job-name release11 --skip-extraction

# Custom output location
python run_carriers_pipeline.py --release 11 --output /path/to/output
```

### Command-Line Options

```
--release INT            GP2 release version (required, e.g. 11, 12)
--job-name TEXT          Job name for output files (default: carriers_analysis)
--ancestries [list]      Ancestries to process (default: all 11)
--data-types [list]      Data types: NBA, WGS, IMPUTED, EXOMES (default: all four)
--nba-path PATH          Override base path for NBA files (e.g. R12 custom layout)
--wgs-path PATH          Override base path for WGS files (e.g. R12 joint-calling layout)
--geno RATE              Max per-variant missingness for PLINK --geno filter (e.g. 0.05)
--parallel               Enable parallel processing (default: True)
--max-workers INT        Maximum parallel workers (default: auto from RAM)
--optimize               Use performance optimizations (default: True)
--skip-extraction        Skip extraction if results exist
--skip-probe-selection   Skip probe selection phase
--skip-locus-reports     Skip locus report generation
--skip-variant-report    Skip per-sample variant report generation
--dosage-het-min FLOAT   Min dosage to call heterozygous (default: 0.5)
--dosage-het-max FLOAT   Max dosage to call heterozygous (default: 1.5)
--dosage-hom-min FLOAT   Min dosage to call homozygous (default: 1.5)
--retry-failed PATH      Path to failed_files.json from a previous run to retry
--output PATH            Custom output directory
```

### Frontend Viewer

```bash
# Production mode
./frontend/run_app.sh

# Debug mode (with job selection)
./frontend/run_app.sh --debug

# Custom port
./frontend/run_app.sh 8502
```

**Pages**: Release Overview, Genotype Viewer, Locus Reports, Probe Validation

## API Endpoints

### Start API Server

```bash
python start_api.py
```

Server available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### Endpoints

**POST /api/v1/carriers/pipeline** - Submit pipeline job

Request:
```json
{
  "job_name": "my_analysis",
  "ancestries": ["AAC", "AFR"],
  "data_types": ["NBA", "WGS", "IMPUTED"],
  "parallel": true,
  "optimize": true,
  "skip_extraction": false
}
```

Response:
```json
{
  "success": true,
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "Pipeline job created."
}
```

**GET /api/v1/carriers/pipeline/{job_id}** - Check job status

**GET /api/v1/carriers/pipeline/{job_id}/results** - Get job results

**GET /api/v1/carriers/health** - Health check

### API Client Arguments

```
Pipeline arguments (same as CLI):
  --job-name, --ancestries, --data-types, --parallel, --optimize, etc.

API-specific arguments:
  --api-host HOST        API server hostname (default: localhost)
  --api-port PORT        API server port (default: 8000)
  --poll-interval SEC    Status check interval (default: 5)
  --max-wait SEC         Maximum wait time (default: 3600)
  --no-follow            Submit and exit without waiting
```

## Configuration

### Performance Optimization

Auto-optimization detects machine specs and tunes settings:

```python
from app.core.config import Settings

# Auto-optimization (recommended)
settings = Settings.create_optimized()

# Manual override
settings = Settings.create_optimized(max_workers=20, chunk_size=75000)
```

### Environment Variables

```bash
export AUTO_OPTIMIZE=true
export CHUNK_SIZE=75000
export MAX_WORKERS=20
export PROCESS_CAP=30
```

### Performance Tiers

Worker counts are capped by available RAM (`floor(RAM * 0.8 / 20GB)`) in addition to CPU count, since PLINK extraction is memory-bound. Large joint-calling WGS files (R12+) can use ~20 GB per worker.

- **Small** (≤4 CPU, ≤16GB): up to 2 workers, 15K chunk_size
- **Medium** (≤8 CPU, ≤32GB): up to 4 workers, 25K chunk_size
- **Large** (≤16 CPU, ≤64GB): up to 8 workers, 40K chunk_size
- **XLarge** (≤32 CPU, ≤128GB): up to 16 workers, 50K chunk_size
- **XXLarge** (>32 CPU, >128GB): up to 20 workers (RAM-capped), 75K chunk_size

Use `--max-workers` to override. If you see OOM errors, halve the current worker count.

## Output

### Directory Structure

Default: `~/gcs_mounts/genotools_server/precision_med/results/release12/`

```
results/release12/
├── release12_NBA.parquet                  # NBA genotypes
├── release12_WGS.parquet                  # WGS genotypes
├── release12_IMPUTED.parquet              # IMPUTED genotypes
├── release12_probe_selection.json         # Probe quality analysis
├── release12_locus_reports_NBA.json       # Clinical phenotype stats (NBA)
├── release12_locus_reports_NBA.csv
├── release12_locus_reports_WGS.json       # Clinical phenotype stats (WGS)
├── release12_locus_reports_WGS.csv
├── release12_locus_reports_IMPUTED.json   # Clinical phenotype stats (IMPUTED)
├── release12_locus_reports_IMPUTED.csv
├── release12_variant_report.csv           # Per-sample clinician-facing carrier report
└── release12_pipeline_results.json        # Pipeline execution summary
```

### Output Formats

**.parquet** - Optimized columnar storage with:
- Normalized sample IDs (consistent across data types)
- Correct pathogenic allele counting (0=none, 1=het, 2=hom)
- MAF correction columns: `maf_corrected` (bool), `original_alt_af` (float)
- Metadata columns first, then sorted sample columns

**_variant_report.csv** - Per-sample clinician-facing carrier report:
- One row per carrier per variant (NBA and WGS only)
- Columns: GP2ID, Ancestry, Gene, Variant_ID, AA_change, rsID, NBA_probe_name, Zygosity, MOI, Data_type, Pathogenicity, Pathogenicity_source, Variant_interpretation, potential_comp_het, validated_by_wgs
- `Data_type` shows `NBA`, `WGS`, or `NBA, WGS` (cross-validated)
- `validated_by_wgs`: True if variant found in both NBA and WGS for the same sample
- `potential_comp_het`: True if sample has ≥2 het variants in an AR gene
- Only best-performing NBA probe included per variant (via probe selection)

**_probe_selection.json** - Probe quality validation:
- Per-mutation analysis with quality metrics
- Consensus recommendations (diagnostic + concordance approaches)
- Methodology comparison and disagreement analysis

**_locus_reports_*.json/csv** - Clinical phenotype statistics:
- Per-gene carrier frequencies stratified by ancestry
- Clinical metrics: H&Y stage, MoCA scores, DAT imaging
- Variant-level carrier counts (heterozygous/homozygous)

### Data Types

**NBA (NeuroBooster Array)**
- Format: `{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- Example: `AAC_release10_vwb.pgen`
- Processing: Direct single-file per ancestry

**WGS (Whole Genome Sequencing)**
- R10/R11 format: `R{version}_wgs_carrier_vars.{pgen|pvar|psam}` (single consolidated file)
- R12+ format: `{ancestry}/chr{N}.{pgen|pvar|psam}` (joint-calling, split by ancestry and chromosome)
- R12+ requires `--wgs-path` to override the default path

**IMPUTED**
- Format: `chr{chrom}_{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- Example: `chr1_AAC_release10_vwb.pgen`
- Processing: Multi-chromosome with automatic combination
- Note: Contains dosage values (0.0-2.0) instead of discrete genotypes

**EXOMES (Clinical Exomes)**
- Format: `clinical_exomes/deepvariant_joint_calling/plink/all_chrs.{pgen|pvar|psam}`
- Processing: Single consolidated file (like WGS, not split by ancestry)
- Availability: Release 8+ only

## Architecture

```
run_carriers_pipeline.py (CLI)
run_carriers_pipeline_api.py (API Client)
        |
        v
app/services/pipeline_service.py (Business Logic)
        |
        v
app/processing/coordinator.py (Orchestration)
        |
        +---> app/processing/extractor.py (PLINK extraction)
        +---> app/processing/harmonizer.py (Allele harmonization)
        +---> app/processing/transformer.py (Genotype transformation)
        +---> app/processing/probe_selector.py (Probe validation)
        +---> app/processing/locus_report_generator.py (Clinical analysis)
        |
        v
Output: Parquet files + JSON reports
```

### Core Components

**ExtractionCoordinator** - Orchestrates multi-source extraction, ProcessPool management, sample ID normalization

**VariantExtractor** - PLINK 2.0 integration, real-time harmonization, genotype transformation

**HarmonizationEngine** - Allele comparison, strand flip detection, swap detection

**ProbeSelector** - NBA probe validation against WGS, diagnostic metrics (sensitivity/specificity), concordance metrics

**LocusReportGenerator** - Clinical data integration, ancestry-stratified analysis, carrier frequency calculations

## Testing

```bash
# Run unit tests
source .venv/bin/activate
python -m pytest tests/ -v

# Quick validation
python run_carriers_pipeline.py --ancestries AAC

# Full pipeline test
python run_carriers_pipeline.py --ancestries AAC AFR
```

## File Paths

```
Input:  ~/gcs_mounts/gp2tier2_vwb/release{10}/
Cache:  ~/gcs_mounts/genotools_server/precision_med/cache/
Output: ~/gcs_mounts/genotools_server/precision_med/results/
SNP List: ~/gcs_mounts/genotools_server/precision_med/summary_data/precision_med_snp_list.csv
```

## Technology Stack

- **Core**: Python 3.8+, FastAPI, Pydantic v2
- **Genomics**: pgenlib (PLINK file processing)
- **Processing**: NumPy, Pandas, ProcessPoolExecutor
- **Storage**: Parquet files (columnar storage)
- **Frontend**: Streamlit (interactive visualization)
- **API**: uvicorn, FastAPI with background tasks

## Changelog

### March 2026 Update

**Per-sample variant report**: New clinician-facing output (`{job_name}_variant_report.csv`)
- Columns match requested format: GP2ID, Ancestry, Gene, Variant_ID, AA_change, rsID, Zygosity, MOI, Data_type, Pathogenicity, Pathogenicity_source, Variant_interpretation
- Cross-validates NBA and WGS: `validated_by_wgs` flag, combined `Data_type` field
- Flags potential compound heterozygotes (`potential_comp_het`) in AR genes
- Only includes best NBA probe per variant

**R12 path overrides**: `--nba-path` and `--wgs-path` CLI args for non-standard release layouts

**PLINK QC filter**: `--geno RATE` applies PLINK `--geno` during extraction to remove high-missingness variants before they enter the pipeline

**Memory-aware parallelization**: Worker count now capped by available RAM (not CPU count). OOM errors print a clear actionable message with a suggested `--max-workers` value. `--max-workers` now correctly enforced.

### December 2025 Update

**MAF Correction**: Automatic minor allele frequency correction
- Flips genotypes when ALT AF > 0.5 to ensure minor allele counting
- Critical fix for rs3115534 (GBA1) where pathogenic G allele is ~3% frequency
- Adds `maf_corrected` and `original_alt_af` columns to output
- Without this fix, carriers of minor pathogenic alleles would be incorrectly identified

### November 2025 Update

**SNP List v2**: 431 → 762 variants (+77%), 8 → 11 genes
- New genes: ATP13A2, RAB39B, SYNJ1
- Major expansions: GBA1 (+79%), PRKN (+75%), PINK1 (+47%)

**EXOMES Support**: Added clinical exomes as fourth data type (release 8+)

**Clinical Variables**: Added 3-year disease duration filter (alongside 5/7 years)

## Development

See `CLAUDE.md` for development instructions and critical implementation notes.

### Contributing

1. Follow existing code structure and naming conventions
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure all tests pass before submitting

## License

[Add license information here]
