# SNP Metrics Processing Module

A clean, efficient Python module for processing genetic data from IDAT files through GTC to VCF format, with final output as parquet files. Built with senior-level design patterns for maintainability and clarity.

## Features

- **Complete Pipeline**: IDAT → GTC → VCF → Parquet processing
- **DRAGEN Integration**: Uses DRAGEN for efficient genetic data conversion
- **Variant Reference Creation**: Separate variant metadata from sample data for efficiency
- **Partitioned Output**: Chromosome and sample-based partitioning for fast queries
- **Clean Architecture**: Separation of concerns with dedicated classes for configuration, parsing, and processing
- **Type Safety**: Full type hints for better code clarity and IDE support
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Logging**: Built-in logging for process monitoring
- **Explicit Configuration**: All paths and settings are explicitly configured

## Architecture

The module is organized into clean, focused components:

- `processor.py`: Main `SNPProcessor` class that orchestrates the pipeline
- `config.py`: `ProcessorConfig` for managing paths and settings
- `vcf_parser.py`: `VCFParser` for efficient VCF file parsing
- `run_snp_metrics.py`: Command-line interface for processing
- `example.py`: Usage examples and patterns

## Command Line Usage

### Process Sample Data

```bash
python run_snp_metrics.py \
    --barcode-path data/idats/205746280003 \
    --dragen-path exec/dragena \
    --bpm-path data/manifests/array.bpm \
    --bpm-csv-path data/manifests/array.csv \
    --egt-path data/manifests/cluster.egt \
    --ref-fasta-path data/reference/genome.fa \
    --gtc-path data/output/gtc \
    --vcf-path data/output/vcf \
    --metrics-path data/output/metrics \
    --num-threads 4
```

### Create Variant Reference (run once)

```bash
python run_snp_metrics.py \
    --create-variant-ref \
    --vcf-file data/output/vcf/205746280003/205746280003_R01C01.snv.vcf.gz \
    --metrics-path data/output/metrics \
    --output-file data/output/metrics/variant_reference
```

## Python API Usage

### Basic Sample Processing

```python
from pathlib import Path
from snp_metrics.processor import SNPProcessor
from snp_metrics.config import ProcessorConfig

# Configure all paths explicitly - point directly to barcode directory
data_path = Path("/path/to/your/data")
barcode = "205746280003"

config = ProcessorConfig(
    barcode_path=data_path / "idats" / barcode,
    dragen_path=data_path / "bin" / "dragena",
    bpm_path=data_path / "manifests" / "array.bpm",
    bpm_csv_path=data_path / "manifests" / "array.csv",
    egt_path=data_path / "manifests" / "cluster.egt",
    ref_fasta_path=data_path / "reference" / "genome.fa",
    gtc_path=data_path / "output" / "gtcs",
    vcf_path=data_path / "output" / "vcfs",
    metrics_path=data_path / "output" / "metrics"
)

# Create processor and process the barcode
processor = SNPProcessor(config, num_threads=4)
output_file = processor.process_barcode(barcode)

print(f"SNP metrics saved to: {output_file}")
```

### Create Variant Reference

```python
# Create variant reference from any VCF file (run once)
variant_ref_path = processor.create_variant_reference(
    vcf_path="data/output/vcf/205746280003/205746280003_R01C01.snv.vcf.gz",
    output_path="data/output/metrics/variant_reference"
)
```

## Output Format

The module generates two types of parquet datasets for optimal efficiency:

### 1. Variant Reference (created once)

**Path:** `data/output/metrics/variant_reference/`
**Partitioning:** By chromosome only
**Columns:**
- `chromosome`: Chromosome number (cleaned, no 'chr' prefix)
- `POS`: Genomic position
- `ID`: SNP identifier
- `REF`: Reference allele
- `ALT`: Alternative allele
- `QUAL`: Quality score
- `FILTER`: Filter status
- `INFO`: VCF INFO field

**Structure:**
```
variant_reference/
├── chromosome=1/
│   └── part-0.parquet
├── chromosome=2/
│   └── part-0.parquet
└── ...
```

### 2. Sample Data (per barcode)

**Path:** `data/output/metrics/{barcode}/`
**Partitioning:** By IID (sample) and chromosome
**Columns:**
- `chromosome`: Chromosome number
- `ID`: SNP identifier (matches variant reference)
- `GT`: Genotype (0=0/0, 1=0/1, 2=1/1, -9=missing)
- `GS`: GenCall Score (confidence score for genotype call)
- `BAF`: B Allele Frequency
- `LRR`: Log R Ratio
- `IID`: Sample identifier

**Structure:**
```
205746280003/
├── IID=205746280003_R01C01/
│   ├── chromosome=1/
│   │   └── part-0.parquet
│   ├── chromosome=2/
│   │   └── part-0.parquet
│   └── ...
├── IID=205746280003_R02C01/
│   └── ...
└── ...
```

## Data Joining

To get complete variant and sample information:

```python
import pandas as pd

# Load variant reference
variants = pd.read_parquet("data/output/metrics/variant_reference")

# Load sample data
samples = pd.read_parquet("data/output/metrics/205746280003")

# Join on ID and chromosome
combined = samples.merge(variants, on=['ID', 'chromosome'], how='left')
```

## Requirements

- Python 3.8+
- pandas
- pyarrow (for parquet support)
- DRAGEN installed and accessible

## Error Handling

The module provides clear error messages for common issues:

- Missing required files (manifests, reference files, etc.)
- DRAGEN command failures
- VCF parsing errors
- Configuration validation errors
- Existing output conflicts

## Design Principles

This module follows senior-level software engineering practices:

1. **Single Responsibility**: Each class has a focused purpose
2. **Explicit Configuration**: All paths and settings are explicitly defined
3. **Error Propagation**: Clear error handling with custom exceptions
4. **Type Safety**: Comprehensive type hints
5. **Logging**: Built-in observability
6. **Testability**: Clean interfaces that are easy to test
7. **Documentation**: Clear docstrings and examples
8. **Data Efficiency**: Separates variant metadata from sample data

## Performance

- Uses efficient pandas operations for data processing
- Streams VCF files to avoid loading large files into memory
- Leverages parquet's columnar storage for fast downstream analysis
- Partitioned datasets for optimized query performance
- Compressed storage with brotli compression
- Proper path handling with `pathlib.Path`

## Example Workflow

```python
import logging
from pathlib import Path
from snp_metrics.processor import SNPProcessor
from snp_metrics.config import ProcessorConfig

# Enable logging to see progress
logging.basicConfig(level=logging.INFO)

# Step 1: Create variant reference (run once)
data_path = Path("/your/data/path")
config = ProcessorConfig(
    barcode_path=data_path / "idats" / "205746280003",  # Any barcode for config
    dragen_path=data_path / "bin" / "dragena", 
    bpm_path=data_path / "manifests" / "array.bpm",
    bpm_csv_path=data_path / "manifests" / "array.csv",
    egt_path=data_path / "manifests" / "cluster.egt",
    ref_fasta_path=data_path / "reference" / "genome.fa",
    gtc_path=data_path / "output" / "gtcs",
    vcf_path=data_path / "output" / "vcfs",
    metrics_path=data_path / "output" / "metrics"
)

processor = SNPProcessor(config, num_threads=4)

# Create variant reference from any VCF file
variant_ref = processor.create_variant_reference(
    vcf_path=str(data_path / "output" / "vcf" / "205746280003" / "205746280003_R01C01.snv.vcf.gz")
)
print(f"✅ Variant reference created: {variant_ref}")

# Step 2: Process multiple barcodes
barcodes = ["205746280003", "207847320055"]

for barcode in barcodes:
    try:
        # Update config for each barcode
        config.barcode_path = data_path / "idats" / barcode
        processor = SNPProcessor(config, num_threads=4)
        
        output_file = processor.process_barcode(barcode)
        print(f"✅ {barcode} → {output_file}")
    except Exception as e:
        print(f"❌ {barcode} failed: {e}")

# Step 3: Analyze combined data
import pandas as pd

# Load and join data for analysis
variants = pd.read_parquet(variant_ref)
samples_205 = pd.read_parquet(data_path / "output" / "metrics" / "205746280003")
combined = samples_205.merge(variants, on=['ID', 'chromosome'], how='left')

print(f"Combined dataset: {len(combined):,} records with {len(combined.columns)} columns")
```

This module provides a clean, maintainable, and reusable Python package for SNP metrics processing suitable for production use, with optimized data structures for efficient storage and analysis. 