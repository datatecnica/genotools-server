# Carriers API

A microservice for processing genetic carrier information. This API allows you to extract carrier data from genotype files based on specified SNP lists.

## Overview

This service is part of the Genotools Server framework, designed to analyze genetic variant data across different ancestry populations. It provides an easy-to-use interface for extracting carrier information from PLINK2 formatted genotype files.

## Data Types Supported

The API supports three main types of genetic data:

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

## Installation

### Prerequisites

- Python 3.8+
- FastAPI
- Pandas
- PLINK2 (must be installed and available on PATH)

### Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd carriers_api
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. If using GCS storage:
   ```
   gcsfuse --implicit-dirs gp2tier2_vwb ~/gcs_mounts/gp2tier2_vwb
   gcsfuse --implicit-dirs genotools-server ~/gcs_mounts/genotools_server
   ```

## Usage

### Starting the API Server

Start the FastAPI server with:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Health Check
```
GET /health
```
Returns the health status of the API.

#### Process Carriers (NBA/WGS)
```
POST /process_carriers
```
Processes carrier information from single consolidated genotype files (NBA or WGS data).

Request Body:
```json
{
  "geno_path": "/path/to/plink/files/prefix",
  "snplist_path": "/path/to/snplist.csv",
  "out_path": "/path/to/output/prefix",
  "release_version": "10"
}
```

Response:
```json
{
  "status": "success",
  "outputs": {
    "var_info": "/path/to/output/prefix_var_info.parquet",
    "carriers_string": "/path/to/output/prefix_carriers_string.parquet",
    "carriers_int": "/path/to/output/prefix_carriers_int.parquet"
  }
}
```

#### Process Imputed Carriers
```
POST /process_imputed_carriers
```
Processes carrier information from chromosome-split imputed genotype files.

Request Body:
```json
{
  "ancestry": "AAC",
  "imputed_dir": "/path/to/imputed/genotypes/base/directory",
  "snplist_path": "/path/to/snplist.csv",
  "out_path": "/path/to/output/prefix",
  "release_version": "10"
}
```

Response:
```json
{
  "status": "success",
  "outputs": {
    "var_info": "/path/to/output/prefix_var_info.parquet",
    "carriers_string": "/path/to/output/prefix_carriers_string.parquet",
    "carriers_int": "/path/to/output/prefix_carriers_int.parquet"
  }
}
```

### Example Script Usage

The repository includes a comprehensive script `run_carriers.py` that demonstrates how to process all data types:

```bash
# Process all data types (NBA, WGS, Imputed)
python run_carriers.py --data-type all

# Process only NBA data
python run_carriers.py --data-type nba

# Process only WGS data  
python run_carriers.py --data-type wgs

# Process only imputed data
python run_carriers.py --data-type imputed

# Only run combination step (skip individual processing)
python run_carriers.py --data-type nba --combine-only
```

#### Command Line Options

```bash
python run_carriers.py [OPTIONS]

Options:
  --mnt-dir PATH           Mount directory path (default: /home/vitaled2/gcs_mounts)
  --release VERSION        Release version (default: 10)
  --api-url URL           API base URL (default: http://localhost:8000)
  --cleanup BOOL          Enable cleanup of existing files (default: True)
  --data-type TYPE        Type of data to process: nba, wgs, imputed, all (default: all)
  --combine-only          Only run combination step (skip individual processing)
  --carriers-dir PATH     Override carriers base directory
  --release-dir PATH      Override release data directory
  --wgs-dir PATH          Override WGS raw data directory
  --imputed-dir PATH      Override imputed data base directory
```

#### Directory Structure

The script automatically creates the following directory structure:

```
carriers_base_dir/
├── nba/release{version}/
│   ├── AAC/
│   ├── AFR/
│   └── ... (other ancestries)
├── wgs/release{version}/
└── imputed/release{version}/
    ├── AAC/
    ├── AFR/
    └── ... (other ancestries)
```

Each ancestry directory contains individual processing results, and a `combined/` subdirectory contains population-combined results.

## File Structure

- `main.py` - FastAPI server definition with endpoints for NBA/WGS and Imputed processing
- `run_carriers.py` - Comprehensive script to process all data types with command-line interface
- `src/core/` - Core implementation modules
  - `api_utils.py` - Utility functions for API interaction
  - `carrier_processor.py` - Unified processing logic for all data types (NBA, WGS, Imputed)
  - `harmonizer.py` - Allele harmonization utilities
  - `manager.py` - Orchestration of carrier analysis workflow
  - `carrier_validator.py` - Validation logic for carrier data
  - `data_repository.py` - Data access layer
  - `genotype_converter.py` - Utilities for genotype format conversion
  - `pipeline_config.py` - Configuration management for data paths and processing options
  - `file_manager.py` - File system operations and cleanup utilities

## Output Files

Processing carriers generates three output files per ancestry:

1. `*_var_info.parquet` - Comprehensive variant information including:
   - Original input metadata (snp_name, locus, rsid, hg38, hg19, etc.)
   - Harmonization results (genotype ID mapping)
   - PLINK statistics (frequencies, missingness rates)
   - PLINK metadata (chromosome format, positions, alleles)
2. `*_carriers_string.parquet` - Carriers in string format (e.g., A/G)
3. `*_carriers_int.parquet` - Carriers in integer format

### Combined Output Files

For NBA and Imputed data, the script also generates population-combined files:

1. `*_info.parquet` - Combined variant information with population-specific frequency columns
2. `*_string.parquet` - Combined carriers in string format across all ancestries
3. `*_int.parquet` - Combined carriers in integer format across all ancestries

### Population Labels

The API processes the following ancestry populations:
- **AAC** (African American Caribbean)
- **AFR** (African)
- **AJ** (Ashkenazi Jewish)
- **AMR** (Admixed American)
- **CAH** (Central Asian Hispanic)
- **CAS** (Central Asian)
- **EAS** (East Asian)
- **EUR** (European)
- **FIN** (Finnish)
- **MDE** (Middle Eastern)
- **SAS** (South Asian)

## Error Handling

The API provides detailed error responses including:
- Error message
- Stack trace (for debugging)

The `api_utils` module contains utilities for processing and displaying these errors in a readable format.

## Performance Considerations

### Processing Speed
- **NBA Data**: Fastest processing (single file per ancestry)
- **WGS Data**: Fast processing (single consolidated file)
- **Imputed Data**: Slower processing (chromosome-split files require individual processing and combination)

### Optimization Strategies
- **Multiprocessing**: The unified processor can be extended with parallel chromosome processing for imputed data
- **Cloud Deployment**: Consider using GCP with higher core counts for faster processing
- **Memory Management**: Large imputed datasets may require significant RAM for chromosome combination

### Data Volume Estimates
- **NBA**: ~200MB per ancestry file
- **WGS**: ~2-5GB consolidated file
- **Imputed**: ~25GB total across all chromosomes per ancestry