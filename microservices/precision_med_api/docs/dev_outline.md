# Genomic Carrier Screening FastAPI Application

## Project Overview
A production-ready FastAPI application for genomic carrier screening analysis that processes variant data from multiple sources (NBA, WGS, and imputed genotype data), identifies carriers of pathogenic variants, and provides ancestry-stratified reporting with statistical analysis.

## Core Objectives
1. Process ~400 pathogenic SNPs across multiple data types and ancestries
2. Harmonize variants to reference genome (build 38)
3. Identify carriers with statistical validation
4. Provide real-time analysis through REST API
5. Support background processing for large-scale batch analysis
6. Generate comprehensive carrier reports with ancestry stratification

## Technical Architecture

### Data Sources
- **NBA (NeuroBooster Array)**: Split by ancestry
- **WGS (Whole Genome Sequencing)**: No split
- **Imputed**: Split by both ancestry and chromosome (11 ancestries × 22 chromosomes)

### File Formats
- **Input**: 
  - PLINK 2.0 pgen/pvar/psam files for genotype data
  - CSV files for variant lists, clinical data, and metadata
- **Output**: 
  - Parquet files for results, reports, and cached indexes
  - JSON for API responses
- **Processing**: 
  - Parquet for intermediate data storage and caching

### Technology Stack
- **Core Framework**: FastAPI with Pydantic v2
- **Data Storage**: Parquet files (with potential PostgreSQL migration path)
- **Caching**: Redis for API response caching
- **Background Processing**: Celery with Redis broker
- **File Processing**: pgenlib for PLINK files
- **Scientific Computing**: NumPy, Pandas, SciPy
- **Deployment**: Docker with multi-stage builds
- **Monitoring**: Prometheus + Grafana

## Development Phases

### Phase 1: Foundation (Week 1)
1. **Data Models & Validation**
   - Variant, Carrier, and Report models
   - Genomic coordinate validation
   - Inheritance pattern enums
   
2. **File Discovery System**
   - Automatic detection of available files
   - Ancestry/chromosome mapping
   - File integrity validation

3. **Configuration Management**
   - Environment-specific configs
   - File path management
   - Output directory structure
   - Parquet file organization

### Phase 2: Core Processing (Week 2)
4. **Variant Index Cache**
   - Build persistent variant ID indexes as Parquet files
   - Store variant locations for quick lookup
   - Reduce file reading time from days to minutes
   - Support incremental updates to cache

5. **Variant Extraction Engine**
   - Extract variants from PLINK files
   - Memory-efficient processing
   - Parallel chromosome processing

6. **Harmonization Pipeline**
   - Allele harmonization to reference
   - Strand flip detection
   - Quality control metrics

### Phase 3: Analysis & Reporting (Week 3)
7. **Carrier Detection**
   - Identify carriers by genotype
   - Calculate carrier frequencies
   - Ancestry-stratified analysis

8. **Statistical Analysis**
   - Hardy-Weinberg equilibrium testing
   - Fisher's exact test for ancestry differences
   - Confidence intervals

9. **Report Generation**
   - Summary statistics exported to Parquet
   - Sample-level carrier status in Parquet format
   - Precision medicine insights
   - Structured output for downstream analysis

### Phase 4: API & Infrastructure (Week 4)
10. **FastAPI Endpoints**
    - `/analyze` - Submit analysis request
    - `/status/{job_id}` - Check job status
    - `/results/{job_id}` - Retrieve results
    - `/variants` - List available variants
    - `/files` - List available genotype files

11. **Background Processing**
    - Celery task queue
    - Progress tracking
    - Result storage

12. **Monitoring & Logging**
    - Structured logging
    - Performance metrics
    - Error tracking

## Key Features

### Performance Optimizations
- Variant index caching in Parquet format (reduce processing from days to minutes)
- Memory mapping for large PLINK files
- Concurrent processing by chromosome
- Redis caching for frequently accessed API responses
- Parquet columnar storage for efficient data retrieval
- Partitioned Parquet files by ancestry/chromosome for faster queries

### Data Quality & Validation
- Genomic coordinate validation
- Allele frequency checks
- Sample quality metrics
- Missing data handling
- Duplicate variant detection

### Security & Compliance
- HIPAA-compliant data handling
- File system permissions for genomic data
- Access control with JWT authentication
- Audit logging for all operations
- Secure file upload validation
- Output file encryption options

## File Structure
```
genomic-carrier-screening/
├── app/
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── analysis.py
│   │   │   ├── variants.py
│   │   │   └── reports.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── models/
│   │   ├── variant.py
│   │   ├── carrier.py
│   │   └── report.py
│   ├── processing/
│   │   ├── extractor.py
│   │   ├── harmonizer.py
│   │   ├── detector.py
│   │   └── cache.py
│   ├── tasks/
│   │   ├── analysis.py
│   │   └── reporting.py
│   └── utils/
│       ├── file_discovery.py
│       ├── parquet_io.py
│       └── statistics.py
├── data/
│   ├── cache/           # Variant index cache files
│   ├── results/         # Analysis results in Parquet
│   └── reports/         # Generated reports
├── tests/
├── docker/
├── config/
└── scripts/
```

## Data Storage Strategy

### Current Implementation (File-Based)
- **Input**: CSV files for variant lists and clinical data
- **Processing**: PLINK pgen files for genotype data
- **Cache**: Parquet files for variant indexes and intermediate results
- **Output**: Parquet files for carrier reports and analysis results
- **Organization**: Partitioned by release/ancestry/chromosome

### Future Migration Path
- PostgreSQL integration for metadata management
- Hybrid approach: Parquet for large datasets, PostgreSQL for queries
- Migration utilities to transfer Parquet data to database tables
- Backward compatibility with file-based outputs

## Development Workflow with Claude Code

### Prompt Strategy
Each feature will be developed with a specific, self-contained prompt that includes:
1. **Context**: Brief description of the component's role
2. **Requirements**: Specific technical requirements
3. **Input/Output**: Clear data flow specification
4. **Dependencies**: Required libraries and modules
5. **Testing**: Unit test requirements
6. **Error Handling**: Expected error scenarios

### Example Prompt Template
```
Create a [component name] for genomic carrier screening:

Context: [What this component does in the pipeline]

Requirements:
- [Specific requirement 1]
- [Specific requirement 2]

Input: [Data format and structure]
Output: [Expected results format]

Dependencies: [Required libraries]

Include:
- Comprehensive error handling
- Type hints with Pydantic models
- Docstrings with examples
- Unit tests with pytest
```

## Success Metrics
- Process 400 variants across 242 files in <1 hour (vs. days)
- API response time <500ms for cached queries
- Support concurrent analysis of 10+ jobs
- 99.9% uptime for production deployment
- Zero data integrity issues

## Next Steps
1. Set up development environment
2. Create initial project structure
3. Implement Phase 1 foundation components
4. Begin iterative development with Claude Code

## Notes for Claude Code
- Always use type hints and Pydantic models
- Include comprehensive error handling
- Write unit tests alongside implementation
- Use async/await for I/O operations where beneficial
- Follow genomics best practices for coordinate systems (0-based vs 1-based)
- Optimize for large file processing (streaming, chunking, memory mapping)
- Use Parquet for all persistent data storage with appropriate partitioning
- Design with future PostgreSQL migration in mind (clean data access layer)