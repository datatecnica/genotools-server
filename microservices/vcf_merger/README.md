# VCF Merge Pipeline

Scatter-gather pipeline for merging multiple VCF files by genomic region using bcftools.

## Requirements

- Python 3.7+ (standard library only)
- bcftools (in PATH)

## Usage

### Basic Command Line

```bash
python vcf_merge_pipeline.py \
    --vcf-files sample1.vcf.gz sample2.vcf.gz sample3.vcf.gz \
    --regions-file regions.txt \
    --output merged.vcf.gz
```

### Advanced Configuration

```bash
python vcf_merge_pipeline.py \
    --vcf-files sample1.vcf.gz sample2.vcf.gz sample3.vcf.gz \
    --regions-file regions.txt \
    --output merged.vcf.gz \
    --max-workers 20 \
    --merge-cpus 4 \
    --concat-cpus 8
```

### Python Script

```python
from vcf_merge_pipeline import merge_vcf_files

# Basic usage
success = merge_vcf_files(
    vcf_files=["sample1.vcf.gz", "sample2.vcf.gz", "sample3.vcf.gz"],
    regions_file="regions.txt",
    output_file="merged.vcf.gz"
)

# Advanced configuration for high-core machine
success = merge_vcf_files(
    vcf_files=vcf_files,
    regions_file=regions_file,
    output_file="merged.vcf.gz",
    max_workers=30,       # Parallel region processes
    merge_cpus=4,         # CPUs per merge process
    merge_memory_gb=16,   # Memory per merge process
    concat_cpus=16,       # CPUs for concatenation
    concat_memory_gb=64   # Memory for concatenation
)
```

## Imputed Genotype Workflow

For merging large imputed datasets (e.g., 100k+ samples in multiple batches):

### Generate Region Files
```bash
# Create 5 Mb regions for imputed data (dense variants)
for chr in {1..22}; do
    for start in $(seq 1 5000000 250000000); do
        end=$((start + 4999999))
        echo "chr${chr}:${start}-${end}"
    done > regions_chr${chr}.txt
done
```

### Recommended Parameters for Imputed Data
```bash
# For 5-6 batches of 20k samples each (100k+ total)
python vcf_merge_pipeline.py \
    --vcf-files batch1_chr1.vcf.gz batch2_chr1.vcf.gz ... batch6_chr1.vcf.gz \
    --regions-file regions_chr1.txt \
    --output merged_chr1.vcf.gz \
    --max-workers 10 \
    --merge-cpus 4 \
    --merge-memory-gb 80 \  # Higher memory for dense imputed data
    --concat-cpus 16
```

### Memory Requirements by Data Type

| Data Type | Samples | Files | Region Size | Memory/Region |
|-----------|---------|-------|-------------|---------------|
| WGS hard calls | 1,000 | 10 | 10 Mb | 8-16 GB |
| Exome | 5,000 | 20 | Whole exome | 16-32 GB |
| Imputed genotypes | 20,000 | 5-6 | 5 Mb | 60-80 GB |
| Imputed (DS only) | 20,000 | 5-6 | 5 Mb | 40-60 GB |

### Processing Multiple Chromosomes
```bash
# Parallel processing of chromosomes
for chr in {1..22}; do
    python vcf_merge_pipeline.py \
        --vcf-files batch*_chr${chr}.vcf.gz \
        --regions-file regions_chr${chr}.txt \
        --output merged_chr${chr}.vcf.gz \
        --max-workers 10 \
        --merge-cpus 4 \
        --merge-memory-gb 80 &
    
    # Limit parallel chromosomes based on available memory
    [ $(jobs -r | wc -l) -ge 2 ] && wait -n
done
wait
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-workers` | 10 | Number of parallel region processes |
| `--merge-cpus` | 2 | CPUs allocated to each merge process |
| `--merge-memory-gb` | 8 | Memory (GB) for each merge process |
| `--concat-cpus` | 4 | CPUs for final concatenation |
| `--concat-memory-gb` | 16 | Memory (GB) for concatenation |

## Resource Calculation Examples

### Example 1: Standard VM (16 cores, 64GB RAM)
```bash
--max-workers 6 --merge-cpus 2 --concat-cpus 4
# Uses: 6 × 2 = 12 cores during merge, 4 cores during concat
```

### Example 2: High-Performance VM (64 cores, 256GB RAM)
```bash
--max-workers 20 --merge-cpus 3 --concat-cpus 8
# Uses: 20 × 3 = 60 cores during merge, 8 cores during concat
```

### Example 3: Memory-Constrained VM (8 cores, 32GB RAM)
```bash
--max-workers 3 --merge-cpus 2 --merge-memory-gb 8
# Uses: 3 × 8GB = 24GB during merge phase
```

## Performance Tuning

**Formula**: `max_workers × merge_cpus` should be 60-80% of total vCPUs

- **CPU-bound workloads**: Increase `merge_cpus` and `concat_cpus`
- **Many small regions**: Increase `max_workers`, decrease `merge_cpus`
- **Few large regions**: Decrease `max_workers`, increase `merge_cpus`

## When to Use This Pipeline

### ✅ Good Use Cases
- Merging 10+ VCF files
- Total data size exceeds available memory
- Imputed genotypes with 50k+ samples
- Running on preemptible/spot instances
- Datasets with uneven variant density

### ❌ Poor Use Cases
- Fewer than 5 VCF files
- Small datasets (< 10GB total)
- Time-critical operations (scatter-gather adds overhead)
- Single sample VCFs

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

## Troubleshooting

### Out of Memory Errors
- Reduce `--max-workers` or `--merge-memory-gb`
- Use smaller regions in regions file
- Remove unnecessary FORMAT fields (e.g., GP if only DS needed)

### Slow Performance
- Increase `--max-workers` if CPU/memory available
- Increase `--merge-cpus` for CPU-bound operations
- Use local SSD for `--work-dir` instead of network storage

### Disk Space Issues
- Temporary BCF files require ~1.5x input size
- Use `--work-dir` on volume with sufficient space
- Clean intermediate files between chromosome runs