# VCF Merge Pipeline

Scatter-gather pipeline for merging multiple VCF files by genomic region using bcftools.

## Requirements

- Python 3.7+
- bcftools (in PATH)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python vcf_merge_pipeline.py \
    --vcf-files sample1.vcf.gz sample2.vcf.gz sample3.vcf.gz \
    --regions-file regions.txt \
    --output merged.vcf.gz \
    --max-workers 10
```

### Python Script

```python
from vcf_merge_pipeline import merge_vcf_files

vcf_files = ["sample1.vcf.gz", "sample2.vcf.gz", "sample3.vcf.gz"]
regions_file = "regions.txt"

success = merge_vcf_files(
    vcf_files=vcf_files,
    regions_file=regions_file,
    output_file="merged.vcf.gz",
    max_workers=10
)
```

## Input Files

### Regions File Format

Text file with one genomic region per line:
```
chr1:1-10000000
chr1:10000001-20000000
chr2:1-15000000
```

## Output

- `merged.vcf.gz` - Concatenated VCF file
- `merged.vcf.gz.tbi` - Tabix index

## Resource Usage

- Merge process: 2 CPUs, 8GB RAM per region
- Concatenation: 4 CPUs, 16GB RAM