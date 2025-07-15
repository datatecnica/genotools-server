# SNP Metrics Processing Module

A clean, efficient Python module for processing genetic data from IDAT files through GTC to VCF format, with final output as parquet files. Built with senior-level design patterns for maintainability and clarity.

## Features

- **Complete Pipeline**: IDAT → GTC → VCF → Parquet processing
- **DRAGEN Integration**: Uses DRAGEN for efficient genetic data conversion
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
- `example.py`: Usage examples and patterns

## Quick Start

### Basic Usage

```python
from pathlib import Path
from snp_metrics.processor import SNPProcessor
from snp_metrics.config import ProcessorConfig

# Configure all paths explicitly - point directly to barcode directory
data_path = Path("/path/to/your/data")
barcode = "your_barcode_here"

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
processor = SNPProcessor(config)
output_file = processor.process_barcode(barcode)

print(f"SNP metrics saved to: {output_file}")
```

### Custom Configuration

```python
from pathlib import Path
from snp_metrics.config import ProcessorConfig

# Create custom configuration - point directly to barcode directory
config = ProcessorConfig(
    barcode_path=Path("/path/to/idats/your_barcode_here"),
    dragen_path=Path("/path/to/dragen/dragena"),
    bpm_path=Path("/path/to/manifest.bpm"),
    bpm_csv_path=Path("/path/to/manifest.csv"),
    egt_path=Path("/path/to/cluster.egt"),
    ref_fasta_path=Path("/path/to/reference.fa"),
    gtc_path=Path("/output/gtcs"),
    vcf_path=Path("/output/vcfs"),
    metrics_path=Path("/output/metrics")
)

processor = SNPProcessor(config)
output_file = processor.process_barcode("your_barcode")
```

## Output Format

The module generates parquet files with the following columns:

- `snpID`: SNP identifier
- `chromosome`: Chromosome number (cleaned, no 'chr' prefix)
- `position`: Genomic position
- `IID`: Sample identifier
- `GT`: Genotype (0=0/0, 1=0/1, 2=1/1, -9=missing)
- `GQ`: Genotype Quality
- `IGC`: Illumina GenCall Score
- `BAF`: B Allele Frequency
- `LRR`: Log R Ratio
- `NORMX`, `NORMY`: Normalized intensities
- `R`, `THETA`: Raw intensity values
- `X`, `Y`: Raw intensity coordinates

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

## Design Principles

This module follows senior-level software engineering practices:

1. **Single Responsibility**: Each class has a focused purpose
2. **Explicit Configuration**: All paths and settings are explicitly defined
3. **Error Propagation**: Clear error handling with custom exceptions
4. **Type Safety**: Comprehensive type hints
5. **Logging**: Built-in observability
6. **Testability**: Clean interfaces that are easy to test
7. **Documentation**: Clear docstrings and examples

## Performance

- Uses efficient pandas operations for data processing
- Streams VCF files to avoid loading large files into memory
- Leverages parquet's columnar storage for fast downstream analysis
- Proper path handling with `pathlib.Path`

## Example Workflow

```python
import logging
from pathlib import Path
from snp_metrics.processor import SNPProcessor
from snp_metrics.config import ProcessorConfig

# Enable logging to see progress
logging.basicConfig(level=logging.INFO)

# Configure all paths explicitly for each barcode
data_path = Path("/your/data/path")
barcodes = ["205746280003", "207847320055"]

# Process multiple barcodes
processor_configs = {}
for barcode in barcodes:
    processor_configs[barcode] = ProcessorConfig(
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

# Process multiple barcodes
processor = SNPProcessor(config)
barcodes = ["205746280003", "207847320055"]

for barcode in barcodes:
    try:
        output_file = processor.process_barcode(barcode)
        print(f"✅ {barcode} → {output_file}")
    except Exception as e:
        print(f"❌ {barcode} failed: {e}")
```

This module provides a clean, maintainable, and reusable Python package for SNP metrics processing suitable for production use. 