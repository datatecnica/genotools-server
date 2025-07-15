# SNP Metrics Processing Toolkit

This toolkit processes Illumina IDAT files to generate SNP metrics and create partitioned parquet files for downstream analysis. It also provides functionality to extract metadata from VCF files.

## Features

- Converts IDAT files to GTC format using Illumina's IAAP CLI
- Processes GTC files to VCF format using bcftools
- Extracts key SNP metrics (BAF, LRR, R, THETA, GT) from VCF files
- Outputs chromosome-partitioned parquet files for efficient querying
- Supports parallel processing for improved performance
- Extracts metadata from VCF files to create sample lists

## Requirements

- Python 3.7+
- bcftools with the gtc2vcf plugin
- IAAP CLI (Illumina Array Analysis Platform)

## Installation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Install bcftools with the gtc2vcf plugin:
```bash
# On Ubuntu/Debian
sudo apt-get install bcftools

# On CentOS/RHEL
sudo yum install bcftools

# On macOS
brew install bcftools
```

Download the Illumina IAAP CLI tool from the Illumina website and make it executable. Add it to the exec/ directory:
```bash
chmod +x exec/path/to/iaap-cli/iaap-cli
```

## Running the Pipeline

The `main.py` script provides two main subcommands:
- `process`: Processes IDAT files to generate SNP metrics
- `extract`: Extracts metadata from VCF files to create sample lists

### Process Subcommand

Process IDAT files to generate SNP metrics:

```bash
python main.py process \
  --idat_path /path/to/idat/directory \
  --output_directory /path/to/output/directory \
  --bpm /path/to/beadpool_manifest.bpm \
  --bpm_csv /path/to/beadpool_manifest.csv \
  --egt /path/to/cluster_file.egt \
  --ref_fasta /path/to/reference_genome.fna \
  --iaap /path/to/iaap-cli/iaap-cli \
  --bcftools_plugins_path /path/to/bcftools/plugins
```

### Extract Subcommand

Extract metadata from VCF files to create sample lists:

```bash
python main.py extract \
  --vcf_path /path/to/vcf/file.vcf.gz \
  --output_file /path/to/output/sample_list.csv
```

#### Extract Arguments

- `--vcf_path`: Path to the VCF file to extract metadata from
- `--output_file`: Path to save the extracted metadata as CSV

### Full Options

```bash
python main.py \
  --idat_path /path/to/idat/directory \
  --output_directory /path/to/output/directory \
  --bpm /path/to/beadpool_manifest.bpm \
  --bpm_csv /path/to/beadpool_manifest.csv \
  --egt /path/to/cluster_file.egt \
  --ref_fasta /path/to/reference_genome.fna \
  --iaap /path/to/iaap-cli/iaap-cli \
  --bcftools_plugins_path /path/to/bcftools/plugins \
  --cleanup \
  --debug \
  --bcftools_threads 8
```

### Required Arguments

- `--idat_path`: Directory containing the IDAT files (Red and Green)
- `--output_directory`: Where to save the processed data
- `--bpm`: Path to the Illumina BeadPool Manifest file (.bpm)
- `--bpm_csv`: Path to the BeadPool Manifest CSV file
- `--egt`: Path to the Illumina cluster file (.egt)
- `--ref_fasta`: Path to the reference genome FASTA file
- `--iaap`: Path to the Illumina Array Analysis Platform CLI executable
- `--bcftools_plugins_path`: Directory containing bcftools plugins, especially gtc2vcf

### Optional Arguments

- `--cleanup`: Delete intermediate files after processing
- `--debug`: Print detailed debugging information
- `--bcftools_threads`: Number of threads to use for bcftools operations (default: uses all available CPUs)

## Output Structure

For each IDAT directory processed, the pipeline creates:

```
output_directory/
└── barcode/
    ├── 207847320055_R01C01/  # Sample directory, partitioned by chromosome
    │   ├── chromosome=1/     # Chromosome partitions
    │   │   └── part.0.parquet
    │   ├── chromosome=2/
    │   │   └── part.0.parquet
    │   └── ... 
    ├── 207847320055_R02C01/
    └── ...
```

Each sample's data includes only the essential columns:
- `snpID`: The SNP identifier
- `BAF`: B Allele Frequency
- `LRR`: Log R Ratio
- `R`: Raw intensity value
- `THETA`: Theta value
- `GT`: Genotype (0=AA, 1=AB, 2=BB, -9=No Call)

## Example

### Process IDAT Files

```bash
python main.py process \
  --idat_path /data/illumina/runs/2023-05-15/207847320055 \
  --output_directory /data/processed/snp_metrics \
  --bpm /data/reference/illumina/NeuroBooster_20042459_A2.bpm \
  --bpm_csv /data/reference/illumina/NeuroBooster_20042459_A2.csv \
  --egt /data/reference/illumina/recluster_09092022.egt \
  --ref_fasta /data/reference/genome/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  --iaap /usr/local/bin/iaap-cli/iaap-cli \
  --bcftools_plugins_path /usr/local/lib/bcftools/plugins \
  --cleanup \
  --bcftools_threads 8
```

### Extract VCF Metadata

```bash
python main.py extract \
  --vcf_path /data/processed/vcfs/samples.vcf.gz \
  --output_file /data/metadata/sample_list.csv
```

## Processing Multiple Samples

To process multiple IDAT directories, you can create a simple bash script:

```bash
#!/bin/bash

# List of IDAT directories
IDAT_DIRS=(
  "/data/illumina/runs/2023-05-15/207847320055"
  "/data/illumina/runs/2023-05-15/207847320056"
)

# Common arguments
COMMON_ARGS="
  --output_directory /data/processed/snp_metrics \
  --bpm /data/reference/illumina/NeuroBooster_20042459_A2.bpm \
  --bpm_csv /data/reference/illumina/NeuroBooster_20042459_A2.csv \
  --egt /data/reference/illumina/recluster_09092022.egt \
  --ref_fasta /data/reference/genome/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna \
  --iaap /usr/local/bin/iaap-cli/iaap-cli \
  --bcftools_plugins_path /usr/local/lib/bcftools/plugins \
  --cleanup"

# Process each directory
for IDAT_DIR in "${IDAT_DIRS[@]}"; do
  echo "Processing: $IDAT_DIR"
  python main.py process --idat_path "$IDAT_DIR" $COMMON_ARGS
done
```