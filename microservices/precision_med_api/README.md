# Precision Medicine API

**Genomic Carrier Screening FastAPI Application** - A production-ready system for identifying carriers of pathogenic variants in large-scale genomic data from the GP2 (Global Parkinson's Genetics Program) cohort.

## Overview

This microservice processes ~400 pathogenic SNPs across 242+ PLINK 2.0 files (>1M variants each) from three data sources with different organizational structures:

- **NBA**: 11 files split by ancestry
- **WGS**: 1 consolidated file  
- **Imputed**: 242 files (11 ancestries × 22 chromosomes)

### Technical Solution
- **Merge-Based Harmonization**: Direct merging of PVAR and SNP list data with real-time allele comparison
- **Allele Harmonization**: Handle strand flips and allele swaps to ensure correct genotype extraction
- **Memory-Efficient Processing**: Stream processing and memory mapping to stay under 8GB RAM
- **ProcessPool Parallelization**: True concurrent processing with optimal worker allocation
- **Output**: Harmonized genotypes in PLINK TRAW format for downstream analysis

### Performance Targets
- Reduce variant extraction from **days to <10 minutes** for 400 variants across all files
- Support concurrent analysis of 10+ jobs
- Real-time harmonization without pre-processing delays

## Installation

### Prerequisites

- Python 3.8+
- PLINK 2.0 (optional - falls back to simulation for testing)
- GCS mounts (for production data access)

### Setup

1. **Clone and install dependencies:**
   ```bash
   git clone <repository-url>
   cd precision_med_api
   
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

## Usage

### Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Quick validation run (2 ancestries, ~5-10 minutes)
python run_carriers_pipeline.py --ancestries AAC AFR

# Full production run (all 11 ancestries, ~45 minutes)
python run_carriers_pipeline.py

# Custom output location
python run_carriers_pipeline.py --output /tmp/my_analysis/carriers_test

# Custom job name (uses config-based location)
python run_carriers_pipeline.py --job-name my_carrier_study
```

### Command Line Options

```bash
python run_carriers_pipeline.py [OPTIONS]

Options:
  --output PATH              Custom output path (default: config-based)
  --job-name TEXT           Job name for output files (default: carriers_analysis)
  --ancestries [AAC|AFR|AJ|AMR|CAH|CAS|EAS|EUR|FIN|MDE|SAS]...
                            Ancestries to process (default: all 11)
  --data-types [NBA|WGS|IMPUTED]...
                            Data types to process (default: all 3)
  --parallel / --no-parallel
                            Enable parallel processing (default: True)
  --max-workers INTEGER     Maximum workers (default: auto-detect)
  --optimize / --no-optimize
                            Use performance optimizations (default: True)
```

## Data Types Supported

### 1. NBA (NeuroBooster Array) Data
- **Format**: Single consolidated PLINK2 files per ancestry
- **Structure**: `{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- **Example**: `AAC_release10_vwb.pgen`, `AAC_release10_vwb.pvar`, `AAC_release10_vwb.psam`
- **Processing**: Direct single-file processing

### 2. WGS (Whole Genome Sequencing) Data  
- **Format**: Single consolidated PLINK2 files across all ancestries
- **Structure**: `R{version}_wgs_carrier_vars.{pgen|pvar|psam}`
- **Example**: `R10_wgs_carrier_vars.pgen`, `R10_wgs_carrier_vars.pvar`, `R10_wgs_carrier_vars.psam`
- **Processing**: Direct single-file processing

### 3. Imputed Data
- **Format**: Chromosome-split PLINK2 files per ancestry
- **Structure**: `chr{chrom}_{ancestry}_release{version}_vwb.{pgen|pvar|psam}`
- **Example**: `chr1_AAC_release10_vwb.pgen`, `chr1_AAC_release10_vwb.pvar`, `chr1_AAC_release10_vwb.psam`
- **Processing**: Multi-chromosome processing with automatic combination

## Output

### Directory Structure

**Default Output**: `~/gcs_mounts/genotools_server/precision_med/results/release10/[job_name]/`

```
~/gcs_mounts/genotools_server/precision_med/results/release10/carriers_analysis/
├── carriers_analysis_NBA.traw              # NBA genotypes (PLINK format)
├── carriers_analysis_NBA.parquet           # NBA genotypes (efficient storage)
├── carriers_analysis_NBA.csv               # NBA genotypes (human-readable)
├── carriers_analysis_NBA_qc_report.json    # Quality control metrics
├── carriers_analysis_NBA_harmonization_report.json
├── carriers_analysis_WGS.traw              # WGS genotypes
├── carriers_analysis_WGS.parquet
├── carriers_analysis_WGS.csv
├── carriers_analysis_WGS_qc_report.json
├── carriers_analysis_IMPUTED.traw          # IMPUTED genotypes
├── carriers_analysis_IMPUTED.parquet
├── carriers_analysis_IMPUTED.csv
├── carriers_analysis_IMPUTED_qc_report.json
└── carriers_analysis_pipeline_results.json # Overall pipeline results
```

### Output Formats

Each data type generates:
- **`.traw`**: PLINK format for genomic analysis tools
- **`.parquet`**: Efficient columnar storage for large datasets  
- **`.csv`**: Human-readable format for manual inspection
- **`_qc_report.json`**: Quality control metrics and statistics
- **`_harmonization_report.json`**: Harmonization process statistics

## Configuration

### Performance Optimization

The system automatically detects your machine specifications and optimizes performance settings:

```python
from app.core.config import Settings

# Auto-optimization (recommended)
settings = Settings.create_optimized()

# Manual override
settings = Settings.create_optimized(
    max_workers=20,      # Use fewer workers
    chunk_size=75000     # Larger chunks for more RAM
)
```

### Environment Variables

```bash
# Enable/disable auto-optimization (default: enabled)
export AUTO_OPTIMIZE=true

# Manual performance overrides
export CHUNK_SIZE=75000
export MAX_WORKERS=20
export PROCESS_CAP=30
```

### Performance Tiers

The system automatically detects your machine tier and optimizes accordingly:

- **Small (≤4 CPU, ≤16GB)**: 2 workers, 15K chunk_size
- **Medium (≤8 CPU, ≤32GB)**: 4 workers, 25K chunk_size  
- **Large (≤16 CPU, ≤64GB)**: 8 workers, 40K chunk_size
- **XLarge (≤32 CPU, ≤128GB)**: 16 workers, 50K chunk_size (current system)
- **XXLarge (>32 CPU, >128GB)**: 24+ workers, 75K chunk_size

## Testing

### Test Scenarios

1. **Quick Validation (5-10 minutes)**:
   ```bash
   python run_carriers_pipeline.py --ancestries AAC AFR
   ```

2. **Medium Scale (15-25 minutes)**:
   ```bash
   python run_carriers_pipeline.py --ancestries AAC AFR AJ AMR CAH
   ```

3. **Full Production (45+ minutes)**:
   ```bash
   python run_carriers_pipeline.py
   ```

### Unit Tests

```bash
# Run all tests
source .venv/bin/activate
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_harmonization.py -v
python -m pytest tests/test_transformer.py -v
```

### Pipeline Tests

```bash
# Test individual data types
python test_nba_pipeline.py        # NBA ProcessPool test
python test_imputed_pipeline.py    # IMPUTED ProcessPool test
```

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

### Core Configuration (`app/core/`)

**`app/core/config.py`** - Central configuration management
- `Settings` class with auto-optimization
- Path management for all data sources
- Performance parameter tuning
- File validation and discovery

### Data Models (`app/models/`)

**`app/models/analysis.py`** - Analysis workflow models
- `DataType` enum (NBA, WGS, IMPUTED)
- `AnalysisRequest` and `AnalysisResult` classes

**`app/models/harmonization.py`** - Harmonization models
- `HarmonizationAction` enum (EXACT, SWAP, FLIP, FLIP_SWAP)
- `HarmonizationRecord` for variant mapping
- `ExtractionPlan` for multi-source coordination

**`app/models/carrier.py`** - Carrier analysis models
- `Genotype` and `Carrier` classes
- `CarrierReport` and statistics models

**`app/models/variant.py`** - Variant models
- `Variant` class with genomic coordinates
- `InheritancePattern` enum

### Processing Pipeline (`app/processing/`)

**`app/processing/coordinator.py`** - High-level orchestration
- `ExtractionCoordinator` class
- Multi-source extraction planning
- ProcessPool parallelization
- Pipeline execution and result aggregation

**`app/processing/extractor.py`** - Variant extraction engine
- `VariantExtractor` class
- PLINK 2.0 integration
- Real-time harmonization
- Fallback simulation for testing

**`app/processing/harmonizer.py`** - Allele harmonization
- `HarmonizationEngine` class
- Merge-based harmonization
- Strand flip and allele swap detection
- Real-time allele comparison

**`app/processing/transformer.py`** - Genotype transformation
- `GenotypeTransformer` class
- Batch transformation support
- Validation and QC metrics
- Allele frequency calculations

**`app/processing/output.py`** - Output formatting
- `TrawFormatter` class
- Multiple output formats (TRAW, Parquet, CSV, JSON)
- QC reports and harmonization statistics
- Hardy-Weinberg equilibrium calculations

### Utilities (`app/utils/`)

**`app/utils/parquet_io.py`** - Parquet I/O operations
- Optimized genomic data storage
- Compression and partitioning
- Memory-efficient data types

**`app/utils/paths.py`** - File path utilities
- `PgenFileSet` class for PLINK file validation
- File discovery and validation

## Status

### Current Status
- ✅ **Phase 1 Complete**: Data models, configuration, file discovery
- ✅ **Phase 2 Complete**: Merge-based harmonization, extraction engine, coordination system
- ✅ **Phase 3A Complete**: ProcessPool parallelization for concurrent file extraction
- 🎯 **Phase 3B Ready**: Within-datatype combination, carrier detection, statistical analysis (NEXT FOCUS)
- ⏳ **Phase 4 Planned**: REST API endpoints, background processing, monitoring

### Technology Stack
- **Core**: FastAPI, Pydantic v2, pgenlib (PLINK file processing)
- **Storage**: Parquet files for caching and results
- **Processing**: NumPy, Pandas, ProcessPoolExecutor for true parallelization
- **Future**: Celery + Redis for background jobs, PostgreSQL migration path

## Development

For development guidelines and detailed implementation plans, see:
- `docs/dev_outline.md` - Development roadmap and phase planning
- `CLAUDE.md` - Claude Code development instructions

### File Paths
```
Input:  ~/gcs_mounts/gp2tier2_vwb/release{10}/
Cache:  ~/gcs_mounts/genotools_server/precision_med/cache/
Output: ~/gcs_mounts/genotools_server/precision_med/output/
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive tests for new functionality
3. Update documentation for any API changes
4. Ensure all tests pass before submitting changes

## License

[Add license information here]