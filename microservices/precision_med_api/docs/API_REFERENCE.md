# Precision Medicine API - Complete Function & Class Reference

## 🎯 **Quick Navigation**

- [Processing Pipeline](#processing-pipeline) - Main extraction and harmonization classes
- [Configuration](#configuration) - Settings and path management  
- [Data Models](#data-models) - Pydantic models for data structures
- [Utilities](#utilities) - Helper functions and I/O operations
- [Module Relationships](#module-relationships) - How classes interact

---

## 🔄 **Processing Pipeline**

### **ExtractionCoordinator** (`app/processing/coordinator.py:96`)
**Purpose**: Main orchestrator for the entire extraction pipeline

#### **Public API:**
- **`run_full_extraction_pipeline(snp_list_path, data_types, ancestries, output_dir, ...)`**
  - **Main entry point** - Runs complete pipeline from SNP list to final output
  - Returns: Pipeline results with file paths and statistics

- **`execute_harmonized_extraction(plan, snp_list_df, parallel=True)`**
  - Orchestrates ProcessPool parallel extraction across all files
  - Returns: Combined DataFrame with harmonized genotypes

- **`plan_extraction(snp_list_ids, data_types, ancestries, chromosomes)`**
  - Creates extraction plan for multi-source processing
  - Returns: ExtractionPlan object with file targets and metadata

- **`export_results_cache_free(plan, output_dir, base_name, formats)`**
  - Exports results with real-time harmonization (no pre-built cache)
  - Returns: Dictionary of output file paths by format

- **`load_snp_list(file_path)`**
  - Loads and validates SNP list from CSV file
  - Returns: Validated pandas DataFrame

#### **Internal Methods**: 6 private methods for ProcessPool management, validation, and data type extraction

---

### **VariantExtractor** (`app/processing/extractor.py:29`)
**Purpose**: Extracts variants from PLINK files with real-time harmonization integration

#### **Public API:**
- **`extract_single_file_harmonized(pgen_path, snp_list_ids, parallel_safe=True)`**
  - **Main extraction method** - Extracts variants with real-time harmonization
  - Uses PLINK2 or simulation fallback
  - Returns: DataFrame with harmonized genotypes and metadata

#### **Internal Methods**: 10 private methods for PLINK interaction, simulation, harmonization application

---

### **HarmonizationEngine** (`app/processing/harmonizer.py:15`)
**Purpose**: Real-time allele harmonization using merge-based approach (no caching)

#### **Public API:**
- **`harmonize_variants(pvar_df, snp_list)`**
  - **Main harmonization method** - Direct allele comparison via merge
  - Flow: PVAR + SNP list → merge on chr:pos → allele comparison → action determination
  - Returns: List of HarmonizationRecord objects

- **`read_pvar_file(pgen_path)`**
  - Reads PVAR files with VCF-style header support (handles ##comments)
  - Returns: pandas DataFrame with variant information

#### **Internal Methods**: 4 private methods for SNP list preparation, data merging, allele comparison

---

### **GenotypeTransformer** (`app/processing/transformer.py:18`)
**Purpose**: Applies genotype transformations for allele harmonization

#### **Public API:**
- **`apply_transformation_by_formula(genotypes, formula)`**
  - **Main transform method** - Applies formula-based transformations
  - Formulas: `"2-x"` (swap), `"x"` or `None` (identity)
  - Returns: Transformed genotype array

- **`get_transformation_summary(harmonization_records)`**
  - Statistics on transformations needed across all records
  - Returns: Summary dictionary with transformation counts

- **`transform_for_swap(genotypes)`** → Allele swap transformation (0↔2, 1→1)
- **`transform_for_flip(genotypes)`** → Strand flip (no genotype change)
- **`validate_transformation(original, transformed, action)`** → Validates transformation correctness

#### **Batch Operations:**
- **`transform_matrix(genotype_matrix, actions)`** → Batch transformation for multiple variants
- **`batch_transform_by_formula(genotype_matrix, formulas)`** → Formula-based batch processing

---

### **TrawFormatter** (`app/processing/output.py:23`)
**Purpose**: Formats and exports harmonized results in multiple formats

#### **Public API:**
- **`export_multiple_formats(df, output_dir, base_name, formats, snp_list, harmonization_stats)`**
  - **Main export method** - Exports TRAW, Parquet, CSV, JSON formats
  - Creates QC reports and harmonization statistics
  - Returns: Dictionary mapping format to output file path

- **`write_traw(df, output_path, include_metadata=True)`**
  - PLINK TRAW format output with proper headers
  - Format: CHR SNP (C)M POS COUNTED ALT [sample genotypes...]

- **`create_qc_report(df, output_path)`**
  - Quality control statistics and metrics
  - Includes missing rates, allele frequencies, ancestry/chromosome breakdowns

#### **Additional Outputs:**
- **`write_harmonization_report()`** → Process statistics and metadata
- **`write_variant_summary()`** → Per-variant summary with HWE p-values
- **`format_harmonized_genotypes()`** → Prepares data for output with metadata

---

## ⚙️ **Configuration**

### **Settings** (`app/core/config.py:7`)
**Purpose**: Central configuration with auto-optimization based on machine specs

#### **Performance Optimization:**
- **`create_optimized(**overrides)`** → **Factory method** - Auto-detects machine specs and optimizes
- **`auto_detect_performance_settings()`** → Returns optimal settings dict based on CPU/RAM
- **`get_optimal_workers(total_files)`** → Calculates ProcessPool worker count

#### **Path Management:**
- **`get_nba_path(ancestry)`** → NBA file paths: `{ancestry}_release{version}_vwb`
- **`get_wgs_path()`** → WGS file paths: `R{version}_wgs_carrier_vars`
- **`get_imputed_path(ancestry, chrom)`** → Imputed paths: `chr{chrom}_{ancestry}_release{version}_vwb`
- **`get_output_path(job_id)`** → Output directory structure

#### **File Discovery:**
- **`validate_file_paths(data_type, ancestry, chrom)`** → Validates PLINK file existence
- **`list_available_ancestries(data_type)`** → Available ancestry groups for data type
- **`list_available_chromosomes(ancestry)`** → Available chromosomes with SNP filtering

#### **Performance Properties:**
```python
settings = Settings.create_optimized()
# Auto-detected based on machine:
# - max_workers: ProcessPool worker count (28 for 32 CPU system)  
# - chunk_size: Processing chunk size (50K for high-memory systems)
# - process_cap: Maximum concurrent processes (30)
# - cpu_reservation: CPUs reserved for OS (2-6)
```

---

## 📊 **Data Models**

### **Analysis Models** (`app/models/analysis.py`)
- **`DataType`** → Enum: NBA, WGS, IMPUTED
- **`AnalysisRequest`** → Request specification with validation
- **`AnalysisResult`** → Complete results container

### **Harmonization Models** (`app/models/harmonization.py`)
- **`HarmonizationAction`** → Enum: EXACT, SWAP, FLIP, FLIP_SWAP, INVALID, AMBIGUOUS
- **`HarmonizationRecord`** → Variant harmonization metadata
- **`ExtractionPlan`** → Multi-source extraction planning
- **`HarmonizationStats`** → Process statistics

### **Carrier Models** (`app/models/carrier.py`)
- **`GenotypeValue`** → Enum: 0/0, 0/1, 1/1, ./.
- **`Genotype`** → Individual genotype record
- **`CarrierReport`** → Analysis results

---

## 🛠️ **Utilities**

### **Path Management** (`app/utils/paths.py`)
- **`PgenFileSet(base_path)`** → PLINK file validation and metadata
- **`validate_pgen_files(base_path)`** → File existence check

### **Parquet I/O** (`app/utils/parquet_io.py`)
- **`save_parquet(df, path, partition_cols, compression)`** → Optimized genomic data storage
- **`read_parquet(path, filters, columns)`** → Efficient loading with filtering
- **`optimize_dtypes_for_genomics(df)`** → Memory optimization (categorical encoding, dtype optimization)

---

## 🔗 **Module Relationships**

### **Primary Dependencies:**

```python
# Main Pipeline Flow
run_carriers_pipeline.py 
    ↓
ExtractionCoordinator
    ↓ (orchestrates)
VariantExtractor + HarmonizationEngine + GenotypeTransformer + TrawFormatter
    ↓ (uses)
Settings + PgenFileSet + parquet_io
```

### **Class Interaction Patterns:**

#### **1. Coordinator → Extractor → Harmonizer**
```python
coordinator = ExtractionCoordinator(extractor, transformer, formatter, settings)
# coordinator calls:
extractor.extract_single_file_harmonized(pgen_path, snp_list_ids)
    # extractor calls:
    harmonizer.harmonize_variants(pvar_df, snp_list)
    transformer.apply_transformation_by_formula(genotypes, formula)
```

#### **2. ProcessPool Worker Pattern**
```python
# Module-level worker function for ProcessPoolExecutor serialization
def extract_single_file_process_worker(file_path, data_type, snp_list_ids, ...):
    extractor = VariantExtractor(settings)  # Fresh instance per process
    return extractor.extract_single_file_harmonized(file_path, snp_list_ids)
```

#### **3. Configuration Injection**
```python
settings = Settings.create_optimized()  # Auto-detect machine specs
extractor = VariantExtractor(settings)
harmonizer = HarmonizationEngine(settings)
coordinator = ExtractionCoordinator(extractor, transformer, formatter, settings)
```

### **Data Flow:**
1. **Input**: SNP list CSV + PLINK files (NBA/WGS/IMPUTED)
2. **Planning**: Extract target chromosomes, create file plan
3. **Parallel Extraction**: ProcessPool workers extract + harmonize each file
4. **Aggregation**: Combine results by data type
5. **Export**: Multiple formats (TRAW, Parquet, CSV) + QC reports

### **Key Design Benefits:**
- **ProcessPool Isolation**: Each worker has fresh instances (no shared state)
- **Real-Time Harmonization**: No caching overhead, always current
- **Auto-Optimization**: Machine-specific performance tuning
- **Multi-Source Integration**: Unified processing with separate outputs
- **Memory Efficiency**: Stream processing, optimal data types, <8GB usage

This API provides a comprehensive, high-performance system for genomic carrier screening with clear interfaces and excellent separation of concerns.