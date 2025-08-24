# Precision Medicine API - Comprehensive Code Documentation

## Overview

This is a **Genomic Carrier Screening FastAPI Application** designed for identifying carriers of pathogenic variants in large-scale genomic data from the GP2 (Global Parkinson's Genetics Program) cohort. The system processes ~400 pathogenic SNPs across 242+ PLINK 2.0 files from three data sources with different organizational structures.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:
- **Core Configuration**: Central settings and path management
- **Data Models**: Pydantic models for type safety and validation
- **Processing Pipeline**: Modular components for harmonization, extraction, and transformation
- **Utilities**: Helper functions for I/O operations and file management
- **Testing**: Comprehensive test suite

---

## Core Configuration (`app/core/`)

### `/app/core/config.py`

**Purpose**: Central configuration management for the genomic processing system.

#### Main Classes:

**`Settings(BaseModel)`**
- **Purpose**: Main configuration class that manages paths, parameters, and data organization
- **Key Properties**:
  - `release`: GP2 release version (default: "10")
  - `mnt_path`: Mount path for GCS buckets
  - `ANCESTRIES`: List of 11 ancestry groups ('AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS')
  - `CHROMOSOMES`: Valid chromosome identifiers (1-22, X, Y, MT)
  - `chunk_size`: Processing chunk size (default: 10000)
  - `max_workers`: Maximum parallel workers (default: 4)

#### Key Methods:
- `get_nba_path(ancestry)`: Returns NBA file path for ancestry
- `get_wgs_path()`: Returns WGS file path
- `get_imputed_path(ancestry, chrom)`: Returns imputed file path
- `get_cache_path()`: Returns cache directory path
- `get_harmonization_cache_path()`: Returns harmonization cache paths
- `validate_file_paths()`: Validates PLINK file existence
- `list_available_ancestries(data_type)`: Lists available ancestries for data type
- `list_available_chromosomes(ancestry)`: Lists available chromosomes for ancestry

#### Dependencies:
- `os`, `typing`, `functools.cached_property`
- `pydantic` (BaseModel, Field, field_validator, ConfigDict)

---

## Data Models (`app/models/`)

### `/app/models/analysis.py`

**Purpose**: Analysis workflow models for managing extraction requests and results.

#### Main Classes:

**`DataType(str, Enum)`**
- Values: NBA, WGS, IMPUTED
- Properties: `description` property for human-readable descriptions

**`AnalysisStatus(str, Enum)`**
- Values: PENDING, PROCESSING, COMPLETED, FAILED
- Properties: `is_terminal`, `is_active` for status checking

**`AnalysisRequest(BaseModel)`**
- **Purpose**: Request specification for variant analysis
- **Key Fields**:
  - `variant_list`: List of variants to analyze
  - `data_type`: Type of genomic data (NBA/WGS/IMPUTED)
  - `release`: GP2 release version
  - `ancestries`: Optional ancestry filter
  - `chromosomes`: Optional chromosome filter (for IMPUTED)
  - `include_clinical_data`: Whether to include clinical data
  - `output_format`: Output format (parquet, json, csv)

**`AnalysisMetadata(BaseModel)`**
- **Purpose**: Execution metadata and statistics
- **Key Fields**: Start/end times, duration, files processed, samples analyzed, errors, warnings

**`AnalysisResult(BaseModel)`**
- **Purpose**: Complete analysis results container
- **Key Fields**: job_id, status, request, carrier_reports, summary_statistics, metadata, output_files
- **Key Methods**: `is_complete`, `is_failed`, `total_carriers_found`, `add_error()`, `complete_analysis()`

#### Dependencies:
- `enum`, `typing`, `datetime`, `pydantic`
- Internal: `.variant.VariantList`, `.carrier.Carrier, CarrierReport`

### `/app/models/carrier.py`

**Purpose**: Models for genotypes and carrier analysis results.

#### Main Classes:

**`GenotypeValue(str, Enum)`**
- Values: HOMOZYGOUS_REF ("0/0"), HETEROZYGOUS ("0/1"), HOMOZYGOUS_ALT ("1/1"), MISSING ("./.")
- Properties: `is_carrier` (bool), `allele_count` (int)

**`Genotype(BaseModel)`**
- **Purpose**: Individual genotype record
- **Key Fields**: sample_id, variant_id, gt (GenotypeValue), ancestry
- **Properties**: `is_carrier`, `allele_count`
- **Validation**: variant_id format validation (chr:pos:ref:alt)

**`Carrier(Genotype)`**
- **Purpose**: Extended genotype with clinical metadata
- **Additional Fields**: sex, age, study_arm, clinical_status, family_history, phenotype, additional_metadata
- **Validation**: Sex validation (M/F/MALE/FEMALE)

**`CarrierStatistics(BaseModel)`**
- **Purpose**: Statistical summary for variant carriers
- **Key Fields**: total_samples, total_carriers, carrier_frequency, heterozygous_count, homozygous_count, missing_count
- **Properties**: `genotyped_samples`, `allele_frequency`

**`AncestryCarrierStats(CarrierStatistics)`**
- **Purpose**: Carrier statistics by ancestry group
- **Additional Field**: ancestry

**`CarrierReport(BaseModel)`**
- **Purpose**: Complete carrier analysis report for a variant
- **Key Fields**: variant_id, gene, inheritance_pattern, overall_statistics, ancestry_statistics, carriers, metadata
- **Key Methods**: `carriers_by_ancestry`, `carriers_by_genotype`, `get_ancestry_stats()`

#### Dependencies:
- `enum`, `typing`, `pydantic`

### `/app/models/genotype.py`

**Purpose**: Models for genotype data processing and carrier detection.

#### Main Classes:

**`GenotypeRecord(BaseModel)`**
- **Purpose**: Individual genotype data from PLINK TRAW files
- **Key Fields**: IID (sample ID), genotypes (Dict[str, Optional[float]])
- **Format**: Numeric genotypes (0=hom_ref, 1=het, 2=hom_alt, None=missing)

**`GenotypeCallFormat(str, Enum)`**
- Values: NUMERIC ("numeric"), STRING ("string")

**`GenotypeData(BaseModel)`**
- **Purpose**: Container for genotype dataset with metadata
- **Key Fields**: samples, variants, format, source, release, ancestry, chromosome
- **Properties**: `num_samples`, `num_variants`

**`CarrierStatus(BaseModel)`**
- **Purpose**: Carrier status results for analysis
- **Key Fields**: IID, variant_id, genotype, is_carrier, carrier_type, gene, snp_name

#### Dependencies:
- `typing`, `enum`, `pydantic`

### `/app/models/harmonization.py`

**Purpose**: Models for variant harmonization and allele transformation.

#### Main Classes:

**`HarmonizationAction(str, Enum)`**
- Values: EXACT, SWAP, FLIP, FLIP_SWAP, INVALID, AMBIGUOUS
- Properties: `requires_genotype_transform`, `is_valid`

**`HarmonizationRecord(BaseModel)`**
- **Purpose**: Record of variant harmonization between SNP list and PLINK file
- **Key Fields**: 
  - snp_list_id, pgen_variant_id
  - chromosome, position
  - snp_list_a1/a2, pgen_a1/a2 (alleles)
  - harmonization_action, genotype_transform
  - file_path, data_type, ancestry
- **Properties**: `variant_key`, `requires_transformation`, `is_strand_ambiguous`
- **Validation**: Chromosome and allele normalization

**`HarmonizationStats(BaseModel)`**
- **Purpose**: Statistics for harmonization results
- **Key Fields**: total_variants, exact_matches, swapped_alleles, flipped_strand, flip_and_swap, invalid_variants, ambiguous_variants
- **Properties**: `harmonized_variants`, `harmonization_rate`, `failure_rate`, `summary_dict`
- **Methods**: `update_from_records()`

**`ExtractionPlan(BaseModel)`**
- **Purpose**: Plan for multi-source variant extraction
- **Key Fields**: snp_list_ids, data_sources, expected_total_variants, expected_total_samples
- **Properties**: `num_files`, `data_types`
- **Methods**: `add_data_source()`, `get_files_for_data_type()`

#### Dependencies:
- `enum`, `typing`, `datetime`, `pydantic`

### `/app/models/variant.py`

**Purpose**: Models for genomic variants and variant lists.

#### Main Classes:

**`InheritancePattern(str, Enum)`**
- Values: AD (Autosomal Dominant), AR (Autosomal Recessive), XL (X-Linked), MT (Mitochondrial)

**`Variant(BaseModel)`**
- **Purpose**: Input genomic variants from SNP lists (pre-extraction)
- **Key Fields**: snp_name, snp_name_alt, locus, rsid, hg38, hg19, ancestry, precision_medicine, pipeline, submitter_email
- **Properties**: `chromosome`, `position`, `ref`, `alt`, `variant_id`, `gene_symbol` (extracted from coordinates)
- **Validation**: rsid format validation, precision_medicine normalization

**`ProcessedVariant(BaseModel)`**
- **Purpose**: Variants after PLINK processing with population statistics
- **Key Fields**: Similar to Variant plus ALT_FREQS, OBS_CT, F_MISS population statistics
- **Properties**: Similar coordinate extraction properties

**`VariantList(BaseModel)`**
- **Purpose**: Collection of variants with metadata
- **Key Fields**: variants, metadata, name, description, created_at, version
- **Properties**: `total_variants`, `variants_by_chromosome`, `variants_by_gene`, `inheritance_patterns`
- **Methods**: `get_variants_for_chromosome()`, `get_variants_for_gene()`

#### Dependencies:
- `enum`, `typing`, `pydantic`

### `/app/models/key_model.py`

**Purpose**: Clinical data key record model.

#### Main Classes:

**`KeyRecord(BaseModel)`**
- **Purpose**: GP2 clinical data master key record
- **Key Fields**: GP2ID, study, nba, wgs, clinical_exome, extended_clinical_data, GDPR, various QC fields, demographics
- **Configuration**: `populate_by_name = True` for flexible field mapping

#### Dependencies:
- `typing`, `pydantic`

### `/app/models/snp_model.py`

**Purpose**: SNP record model for variant metadata.

#### Main Classes:

**`SNPRecord(BaseModel)`**
- **Purpose**: SNP metadata record from variant lists
- **Key Fields**: snp_name, snp_name_alt, locus, rsid, hg38, hg19, ancestry, submitter_email, precision_medicine, pipeline
- **Configuration**: `populate_by_name = True`

#### Dependencies:
- `typing`, `pydantic`

---

## Processing Pipeline (`app/processing/`)

### `/app/processing/cache.py`

**Purpose**: Variant harmonization cache building and management.

#### Main Classes:

**`AlleleHarmonizer`**
- **Purpose**: Core allele harmonization logic
- **Key Methods**:
  - `complement_allele()`: Get complement for strand flipping
  - `check_strand_ambiguous()`: Check if allele pair is strand ambiguous
  - `get_all_representations()`: Get all possible orientations of allele pair
  - `determine_harmonization()`: Determine harmonization action needed
- **Constants**: 
  - `COMPLEMENT_MAP`: Base complements for strand flipping
  - `AMBIGUOUS_PAIRS`: Strand ambiguous allele pairs

**`CacheBuilder`**
- **Purpose**: Builds and manages variant harmonization caches
- **Key Methods**:
  - `build_harmonization_cache()`: Build cache for single PLINK file
  - `match_variants_with_harmonization()`: Match PVAR variants to SNP list
  - `save_cache()`, `load_cache()`: Cache file operations
  - `validate_harmonization()`: Validate cache and return statistics
  - `generate_harmonized_plink_files()`: Create harmonized PLINK files using PLINK2
  - `build_all_harmonization_caches()`: Build caches for all files of a data type
- **Private Methods**: PVAR file reading, quality scoring, PLINK2 integration

#### Dependencies:
- Standard: `os`, `pandas`, `numpy`, `typing`, `pathlib`, `logging`, `concurrent.futures`, `time`, `subprocess`, `tempfile`
- Internal: `..models.harmonization`, `..models.analysis`, `..core.config`, `..utils.parquet_io`

### `/app/processing/coordinator.py`

**Purpose**: High-level extraction coordination and orchestration.

#### Main Classes:

**`ExtractionCoordinator`**
- **Purpose**: Coordinates multi-source variant extraction with harmonization
- **Key Methods**:
  - `load_snp_list()`: Load and validate SNP list from file
  - `plan_extraction()`: Create extraction plan for variants and data types
  - `execute_harmonized_extraction()`: Execute harmonized extraction according to plan
  - `generate_harmonization_summary()`: Generate comprehensive harmonization summary
  - `export_results()`: Export extraction results in specified formats
  - `export_results_cache_free()`: Export using cache-free real-time harmonization
  - `run_full_extraction_pipeline()`: Complete pipeline from SNP list to output
- **Private Methods**: SNP list validation, data type extraction, harmonization summary generation

#### Dependencies:
- Standard: `os`, `pandas`, `numpy`, `typing`, `pathlib`, `logging`, `concurrent.futures`, `time`, `datetime`, `uuid`
- Internal: Multiple models and processing modules

### `/app/processing/extractor.py`

**Purpose**: Variant extraction engine with allele harmonization.

#### Main Classes:

**`VariantExtractor`**
- **Purpose**: Extracts variants from PLINK files and applies harmonization transformations
- **Key Methods**:
  - `extract_single_file_harmonized()`: Extract variants from single file with harmonization
  - `merge_harmonized_genotypes()`: Merge results from multiple sources
  - `extract_with_cache()`: Extract using pre-built harmonization cache
  - `extract_without_cache()`: Extract using real-time harmonization
- **Private Methods**: 
  - PLINK availability checking
  - PLINK2 extraction commands
  - TRAW file reading and processing
  - Genotype transformation application

#### Dependencies:
- Standard: `os`, `pandas`, `numpy`, `typing`, `pathlib`, `logging`, `concurrent.futures`, `subprocess`, `tempfile`, `time`
- Internal: Models, transformer, harmonizer

### `/app/processing/harmonizer.py`

**Purpose**: Improved variant harmonization engine.

#### Main Classes:

**`HarmonizationEngine`**
- **Purpose**: Enhanced harmonization engine based on proven logic
- **Key Features**:
  - Pre-computing all variant representations
  - Exact variant matching instead of position-based lookup
  - Deterministic, reproducible results
- **Key Methods**:
  - `read_pvar_file()`: Read PVAR file for PLINK file
  - `harmonize_variants()`: Harmonize SNP list against PLINK variants
  - `get_harmonization_records()`: Get harmonization records for file
- **Private Methods**: Variant lookup creation, exact matching logic

#### Dependencies:
- Standard: `os`, `pandas`, `typing`, `logging`
- Internal: `..models.harmonization`, `..core.config`, `.cache.AlleleHarmonizer`

### `/app/processing/output.py`

**Purpose**: Output formatting for harmonized genomic data.

#### Main Classes:

**`TrawFormatter`**
- **Purpose**: Formats harmonized genotypes into PLINK TRAW format and reports
- **Key Methods**:
  - `format_harmonized_genotypes()`: Format genotypes for output
  - `write_traw()`: Write PLINK TRAW format
  - `write_parquet()`: Write Parquet format
  - `write_csv()`: Write CSV format
  - `export_multiple_formats()`: Export in multiple formats
  - `create_qc_report()`: Generate quality control report
  - `write_harmonization_report()`: Write harmonization statistics
- **Private Methods**: Sample column identification, format conversion

#### Dependencies:
- Standard: `os`, `pandas`, `numpy`, `typing`, `pathlib`, `logging`, `json`, `datetime`
- Internal: `..models.harmonization`, `..utils.parquet_io`

### `/app/processing/transformer.py`

**Purpose**: Genotype transformation logic for allele harmonization.

#### Main Classes:

**`GenotypeTransformer`**
- **Purpose**: Handles genotype transformations for allele harmonization
- **Key Methods**:
  - `apply_transformation()`: Apply transformation based on harmonization action
  - `apply_transformation_by_formula()`: Apply transformation using formula string
  - `transform_for_swap()`: Transform for allele swap (A1<->A2)
  - `transform_for_flip()`: Transform for strand flip (identity)
  - `transform_for_flip_swap()`: Transform for both flip and swap
  - `get_transformation_summary()`: Generate transformation statistics
- **Static Methods**: `_transform_identity()`, `_transform_swap()`

#### Dependencies:
- Standard: `numpy`, `pandas`, `typing`, `logging`
- Internal: `..models.harmonization`

---

## Utilities (`app/utils/`)

### `/app/utils/parquet_io.py`

**Purpose**: Parquet I/O utilities for efficient genomic data storage.

#### Functions:

**`save_parquet()`**
- **Purpose**: Save DataFrame to parquet with optimal settings for genomic data
- **Parameters**: DataFrame, path, partition_cols, compression, index
- **Features**: Partitioning support, compression options, directory creation

**`read_parquet()`**
- **Purpose**: Read parquet with optional filtering and column selection
- **Parameters**: path, filters, columns, use_pandas_metadata
- **Features**: Handles both files and directories, PyArrow filters

**`append_parquet()`**
- **Purpose**: Append data to existing parquet dataset
- **Features**: Incremental data updates

**`query_parquet()`**
- **Purpose**: Query parquet dataset with filters
- **Features**: Efficient querying without full data load

**`optimize_dtypes_for_genomics()`**
- **Purpose**: Optimize DataFrame dtypes for genomic data
- **Features**: Memory optimization for large datasets

#### Dependencies:
- Standard: `os`, `pandas`, `pyarrow`, `typing`, `pathlib`, `logging`

### `/app/utils/paths.py`

**Purpose**: Path utilities and file validation for PLINK datasets.

#### Main Classes:

**`PgenFileSet`**
- **Purpose**: Represents a complete PLINK file set (.pgen, .pvar, .psam)
- **Key Properties**: base_path, pgen_file, pvar_file, psam_file, exists, file_sizes
- **Methods**: 
  - `validate()`: Check all files exist
  - `get_sample_count()`: Count samples in PSAM file
  - `get_variant_count()`: Count variants in PVAR file
  - `total_size_mb`: Total file size in MB

#### Functions:

**`validate_pgen_files()`**: Validate PLINK file set completeness
**`list_available_files()`**: List available files for data type and release
**`get_file_info()`**: Get detailed information about PLINK file set
**`find_matching_files()`**: Find files matching criteria
**`create_output_directory()`**: Create output directory for job
**`create_cache_directory()`**: Create cache directory structure
**`get_clinical_files()`**: Get clinical data file paths and existence status

#### Dependencies:
- Standard: `os`, `typing`, `pathlib`, `dataclasses`
- Internal: `app.models.analysis`, `app.core.config`

---

## Testing (`tests/`)

### `/tests/conftest.py`

**Purpose**: Pytest configuration and shared fixtures.

#### Fixtures:
- `test_settings()`: Create test settings with temporary paths
- `sample_snp_list()`: Create sample SNP list for testing
- Mock objects for testing isolation

#### Dependencies:
- `pytest`, `pandas`, `numpy`, `tempfile`, `os`, `unittest.mock`
- Internal: `app.core.config`, `app.models.harmonization`

### `/tests/test_cache.py`

**Purpose**: Tests for variant cache building functionality.

#### Test Classes:

**`TestCacheBuilder`**
- Tests cache building, harmonization matching, PVAR reading
- Mock PVAR files, SNP list validation
- Harmonization action detection

#### Dependencies:
- `pytest`, `pandas`, `numpy`, `tempfile`, `os`, `unittest.mock`
- Internal: `app.processing.cache`, `app.models.harmonization`, `app.core.config`

### `/tests/test_harmonization.py`

**Purpose**: Tests for variant harmonization functionality.

#### Test Classes:

**`TestAlleleHarmonizer`**
- Tests allele complement functions
- Strand ambiguity detection
- All possible allele representations
- Harmonization action determination

#### Dependencies:
- `pytest`, `pandas`, `numpy`, `typing`
- Internal: `app.models.harmonization`, `app.processing.cache`

### `/tests/test_transformer.py`

**Purpose**: Tests for genotype transformation functionality.

#### Test Classes:

**`TestGenotypeTransformer`**
- Tests transformation methods (identity, swap)
- Harmonization action-based transformations
- Formula-based transformations
- Missing data handling

#### Dependencies:
- `pytest`, `numpy`, `pandas`, `typing`
- Internal: `app.processing.transformer`, `app.models.harmonization`

### `/tests/test_multi_allelic.py`

**Purpose**: Tests for multi-allelic variant handling.

#### Test Classes:

**`TestMultiAllelicHandling`**
- Tests V499L/V499M scenario (multiple SNP list variants mapping to same PLINK variant)
- Multi-allelic site processing
- Harmonization record handling for complex cases

#### Dependencies:
- `pytest`, `pandas`, `numpy`, `unittest.mock`
- Internal: `app.processing.extractor`, `app.models.harmonization`

---

## Main Scripts

### `/test_nba_pipeline.py`

**Purpose**: Test script for NBA harmonization/extraction pipeline.

#### Functions:
- `parse_args()`: Command line argument parsing
- `main()`: Execute NBA pipeline test with AAC ancestry

#### Features:
- Configurable output paths
- Configurable ancestry selection
- Full pipeline testing from SNP list to output

#### Dependencies:
- Standard: `sys`, `os`, `logging`, `argparse`, `pathlib`
- Internal: Configuration, coordinator, extractor, transformer, analysis models

---

## Key System Features

### Harmonization Engine
- **Allele Orientation**: Handles strand flips and allele swaps between SNP lists and PLINK files
- **Multi-allelic Support**: Processes multiple variants at the same genomic position
- **Quality Scoring**: Prioritizes matches based on variant ID quality and allele characteristics
- **Cache System**: Pre-computes harmonization mappings for performance

### Data Processing Pipeline
- **Multi-source**: Supports NBA (11 files), WGS (1 file), and Imputed (242 files) data types
- **Memory Efficient**: Streaming and chunked processing for large datasets
- **Parallel Processing**: ThreadPoolExecutor for concurrent file processing
- **Format Support**: TRAW, Parquet, CSV, and JSON output formats

### Performance Optimizations
- **Variant Index Caching**: Parquet-based mapping for fast variant lookup
- **Compression**: Snappy compression for Parquet files
- **Partitioning**: Data partitioning by chromosome and ancestry
- **Memory Management**: Optimized dtypes and streaming for large datasets

### Quality Control
- **Validation**: Comprehensive input validation and error handling
- **Statistics**: Detailed harmonization and extraction statistics
- **Reporting**: QC reports with missing rates, allele frequencies, and processing metadata
- **Logging**: Structured logging throughout the pipeline

This system processes ~400 pathogenic SNPs across massive genomic datasets, reducing extraction time from days to <10 minutes through intelligent caching and harmonization strategies.