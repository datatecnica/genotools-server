# Precision Medicine Carriers Pipeline

Genomic carrier screening system for identifying carriers of pathogenic variants in large-scale genomic data from the GP2 (Global Parkinson's Genetics Program) cohort.

## Overview

Processes ~400 pathogenic SNPs across 254 PLINK 2.0 files from three data sources:
- **NBA (NeuroBooster Array)**: 11 files split by ancestry
- **WGS (Whole Genome Sequencing)**: 1 consolidated file
- **IMPUTED**: 242 files (11 ancestries × 22 chromosomes)

### Key Features

- Correct pathogenic allele counting (not reference alleles)
- Real-time harmonization without pre-processing
- ProcessPool parallelization for concurrent file extraction
- Sample ID normalization across data types
- Probe quality validation against WGS ground truth
- Clinical phenotype integration with ancestry stratification
- Interactive web interface for result exploration

### Performance

- Extraction: <10 minutes for 400 variants across all files
- Parallelization: 28 concurrent workers on XLarge machines
- Memory: <8GB RAM through stream processing
- Auto-optimization: Detects machine specs and tunes performance

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
python run_carriers_pipeline.py --ancestries AAC AFR

# Full pipeline (all ancestries, ~45 minutes)
python run_carriers_pipeline.py --job-name release10

# Skip extraction (reuse existing results)
python run_carriers_pipeline.py --job-name release10 --skip-extraction

# Custom output location
python run_carriers_pipeline.py --output /path/to/output
```

### Command-Line Options

```
--job-name TEXT          Job name for output files (default: carriers_analysis)
--ancestries [list]      Ancestries to process (default: all 11)
--data-types [list]      Data types: NBA, WGS, IMPUTED (default: all)
--parallel               Enable parallel processing (default: True)
--max-workers INT        Maximum workers (default: auto-detect)
--optimize               Use performance optimizations (default: True)
--skip-extraction        Skip extraction if results exist
--skip-probe-selection   Skip probe selection phase
--skip-locus-reports     Skip locus report generation
--output PATH            Custom output directory
```

### Frontend Viewer

```bash
# Production mode
./run_frontend.sh

# Debug mode (with job selection)
./run_frontend.sh --debug

# Custom port
./run_frontend.sh 8502
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

- **Small** (≤4 CPU, ≤16GB): 2 workers, 15K chunk_size
- **Medium** (≤8 CPU, ≤32GB): 4 workers, 25K chunk_size
- **Large** (≤16 CPU, ≤64GB): 8 workers, 40K chunk_size
- **XLarge** (≤32 CPU, ≤128GB): 16 workers, 50K chunk_size
- **XXLarge** (>32 CPU, >128GB): 24+ workers, 75K chunk_size

## Output

### Directory Structure

Default: `~/gcs_mounts/genotools_server/precision_med/results/release10/`

```
results/release10/
├── release10_NBA.parquet                  # NBA genotypes
├── release10_WGS.parquet                  # WGS genotypes
├── release10_IMPUTED.parquet              # IMPUTED genotypes
├── release10_probe_selection.json         # Probe quality analysis
├── release10_locus_reports_NBA.json       # Clinical phenotype stats (NBA)
├── release10_locus_reports_NBA.csv
├── release10_locus_reports_WGS.json       # Clinical phenotype stats (WGS)
├── release10_locus_reports_WGS.csv
├── release10_locus_reports_IMPUTED.json   # Clinical phenotype stats (IMPUTED)
├── release10_locus_reports_IMPUTED.csv
└── release10_pipeline_results.json        # Pipeline execution summary
```

### Output Formats

**.parquet** - Optimized columnar storage with:
- Normalized sample IDs (consistent across data types)
- Correct pathogenic allele counting (0=none, 1=het, 2=hom)
- Metadata columns first, then sorted sample columns

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
- Format: `R{version}_wgs_carrier_vars.{pgen|pvar|psam}`
- Example: `R10_wgs_carrier_vars.pgen`
- Processing: Single consolidated file

**IMPUTED**
- Format: `chr{chrom}_{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- Example: `chr1_AAC_release10_vwb.pgen`
- Processing: Multi-chromosome with automatic combination
- Note: Contains dosage values (0.0-2.0) instead of discrete genotypes

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

## Development

See `CLAUDE.md` for development instructions and critical implementation notes.

### Contributing

1. Follow existing code structure and naming conventions
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure all tests pass before submitting

## License

[Add license information here]
