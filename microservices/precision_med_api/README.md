# Precision Medicine API

**Genomic Carrier Screening FastAPI Application** - A production-ready system for identifying carriers of pathogenic variants in large-scale genomic data from the GP2 (Global Parkinson's Genetics Program) cohort.

## Overview

This microservice processes ~400 pathogenic SNPs across 242+ PLINK 2.0 files (>1M variants each) from three data sources with different organizational structures:

- **NBA**: 11 files split by ancestry
- **WGS**: 1 consolidated file  
- **Imputed**: 242 files (11 ancestries × 22 chromosomes)

### Technical Solution
- **Correct Allele Counting**: Fixed allele assignment to count pathogenic variants instead of reference alleles
- **Merge-Based Harmonization**: Direct merging of PVAR and SNP list data with real-time allele comparison
- **Advanced Genotype Transformation**: Handles all harmonization scenarios (EXACT, FLIP, SWAP, FLIP_SWAP) with proper genotype flipping
- **Sample ID Normalization**: Consistent sample IDs across data types (removes '0_' prefixes, fixes WGS duplicates)
- **Column Organization**: Metadata columns come first, followed by sorted sample columns
- **Memory-Efficient Processing**: Stream processing and memory mapping to stay under 8GB RAM
- **ProcessPool Parallelization**: True concurrent processing with optimal worker allocation
- **Streamlit Viewer**: Interactive web interface for exploring pipeline results

### Performance Targets
- Reduce variant extraction from **days to <10 minutes** for 400 variants across all files
- Support concurrent analysis of 10+ jobs
- Real-time harmonization without pre-processing delays
- **Achieved**: Correct pathogenic allele counting with proper genotype interpretation

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

### Streamlit Viewer

Launch the interactive web interface to explore pipeline results:

```bash
# Production mode (default - clean interface)
./run_streamlit.sh

# Debug mode (with job selection for development)
./run_streamlit.sh --debug

# Alternative direct command
source .venv/bin/activate
streamlit run streamlit_viewer.py
```

**Streamlit Features:**
- **📊 Overview**: Pipeline summary, sample counts, file information
- **🧬 Variant Browser**: Filter and explore variants with harmonization details
- **🔬 Multiple Probes Analysis**: Shows SNPs with multiple probes (debug mode)
- **📈 Statistics**: Visualizations of harmonization actions and variant distributions  
- **💾 File Downloads**: Access to processed parquet and summary files
- **🔧 Debug Mode**: Job selection for accessing test results and development data

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

**Default Output**: `~/gcs_mounts/genotools_server/precision_med/results/release10/`

```
~/gcs_mounts/genotools_server/precision_med/results/release10/
├── release10_NBA.parquet                   # NBA genotypes with normalized sample IDs
├── release10_NBA_variant_summary.csv       # Per-variant harmonization metadata  
├── release10_WGS.parquet                   # WGS genotypes with normalized sample IDs
├── release10_WGS_variant_summary.csv
├── release10_IMPUTED.parquet               # IMPUTED genotypes with normalized sample IDs
├── release10_IMPUTED_variant_summary.csv
└── release10_pipeline_results.json         # Overall pipeline execution summary
```

### Output Formats

Each data type generates:
- **`.parquet`**: Optimized columnar storage with normalized sample IDs and correct allele counting
- **`_variant_summary.csv`**: Harmonization metadata (Note: redundant with parquet metadata)

### Key Output Features

**✅ Correct Allele Counting**: Genotype values represent counts of **pathogenic alleles**
- `0` = No pathogenic alleles
- `1` = Heterozygous carrier (1 pathogenic allele)
- `2` = Homozygous carrier (2 pathogenic alleles)

**✅ Normalized Sample IDs**: Consistent format across all data types
- NBA: `0_SAMPLE_001234` → `SAMPLE_001234`
- WGS: `SAMPLE_001234_SAMPLE_001234` → `SAMPLE_001234`
- IMPUTED: `0_SAMPLE_001234` → `SAMPLE_001234`

**✅ Organized Columns**: Metadata columns first, then alphabetically sorted samples
- Metadata: `chromosome`, `variant_id`, `position`, `counted_allele`, `alt_allele`, `harmonization_action`, etc.
- Samples: `SAMPLE_001234`, `SAMPLE_001235`, `SAMPLE_001236`, `SAMPLE_001237`, etc.

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
# Run all tests (streamlined test suite)
source .venv/bin/activate
python -m pytest tests/ -v  # 3 tests, zero warnings

# Run specific test modules
python -m pytest tests/test_transformer.py -v
```

**Test Coverage:**
- ✅ Genotype transformation logic (identity, swap, formula-based)
- ✅ Streamlined test suite with only essential tests
- ✅ Zero deprecation warnings

### Pipeline Tests

```bash
# Test individual data types with real data
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
- Sample ID normalization and column organization
- Pipeline execution and result aggregation

**`app/processing/extractor.py`** - Variant extraction engine
- `VariantExtractor` class
- PLINK 2.0 integration
- Corrected allele counting (pathogenic vs reference)
- Advanced genotype transformation for all harmonization scenarios
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
- ✅ **Phase 3B Complete**: Correct allele counting, sample ID normalization, column organization
- ✅ **Streamlit Viewer**: Interactive web interface for pipeline result exploration
- 🎯 **Data Quality**: Carrier detection, statistical analysis, redundancy reduction (CURRENT FOCUS)
- ⏳ **Phase 4 Planned**: REST API endpoints, background processing, monitoring

### Major Recent Achievements

**🔧 Multiple Probe Detection Fix**:
- **CRITICAL BUG FIX**: Resolved issue where NBA variants with multiple probes at the same genomic position were being lost
- Fixed harmonization record creation to use SNP names instead of coordinate-based IDs
- Updated deduplication logic to preserve different probes (different `variant_id` values)
- **Result**: 77 SNPs now correctly show multiple probes (316 total variants vs 218 previously)

**🔧 Allele Counting Fix**: 
- Fixed critical issue where reference alleles were counted instead of pathogenic alleles
- Implemented proper genotype transformation for all harmonization scenarios (EXACT, FLIP, SWAP, FLIP_SWAP)
- Genotype values now correctly represent pathogenic allele counts

**📊 Sample Counting Fix**:
- Fixed pipeline summary reporting "Total samples: 0" instead of actual count
- Added `source_file` to metadata columns list to prevent it being counted as sample
- Implemented proper sample count aggregation across data types
- **Result**: Accurate reporting of 1,215 samples in AAC NBA data

**🏷️ Sample ID Normalization**:
- Consistent sample IDs across all data types  
- Fixed WGS duplicated IDs: `SAMPLE_001234_SAMPLE_001234` → `SAMPLE_001234`
- Removed NBA/IMPUTED prefixes: `0_SAMPLE_001234` → `SAMPLE_001234`

**📊 Data Organization**:
- Column reordering: metadata columns first, then sorted sample columns
- Duplicate handling and conflict resolution
- Streamlined test suite (83% code reduction in transformer.py)

**🖥️ Enhanced Streamlit Viewer**:
- Interactive web interface for exploring pipeline results
- **Multiple Probes Analysis**: Dedicated section showing SNPs with multiple probes
- **Debug Mode**: Optional job selection with `--debug` flag for development
- **Fixed Deprecation**: Updated `use_container_width` to `width='stretch'`
- Variant filtering and statistics visualization
- File downloads and data exploration capabilities

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
Output: ~/gcs_mounts/genotools_server/precision_med/results/
```

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive tests for new functionality
3. Update documentation for any API changes
4. Ensure all tests pass before submitting changes

## License

[Add license information here]