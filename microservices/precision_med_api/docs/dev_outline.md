# Updated Development Plan - Genomic Carrier Screening API

## 📋 Development Progress Checklist

### Phase 1: Foundation
- [x] **Component 1**: Data Models & Validation (variant.py, carrier.py, analysis.py)
- [x] **Component 2**: Configuration Management (config.py with path methods)  
- [x] **Component 3**: File Discovery System (file detection and validation)

### Phase 2: Core Processing (Current Focus)
- [ ] **Component 4**: Variant Index Cache System (cache.py)
- [ ] **Component 5**: Variant Extraction Engine (extractor.py)
- [ ] **Component 6**: Harmonization Pipeline (harmonizer.py)

### Phase 3: Analysis & Reporting
- [ ] **Component 7**: Carrier Detection (detector.py)
- [ ] **Component 8**: Statistical Analysis (statistics.py)
- [ ] **Component 9**: Report Generation (reporting.py)

### Phase 4: API & Infrastructure  
- [ ] **Component 10**: FastAPI Endpoints (analysis.py, variants.py, reports.py)
- [ ] **Component 11**: Background Processing (Celery tasks)
- [ ] **Component 12**: Monitoring & Logging (Prometheus metrics)

---

## Phase 1: Foundation ✅ COMPLETED

### Component 1: Data Models & Validation ✅
**Implementation:** Pydantic v2 models with genomic coordinate validation

**Files Created:**
- `app/models/variant.py` - Variant, VariantList models
- `app/models/carrier.py` - Genotype, Carrier, CarrierReport models  
- `app/models/analysis.py` - DataType enum, AnalysisRequest, AnalysisResult models

**Key Features:**
- Chromosome validation (1-22, X, Y, MT)
- Inheritance pattern enums (AD/AR/XL/MT)
- Genomic coordinate validation
- Sample ancestry tracking

### Component 2: Configuration Management ✅
**Implementation:** Settings class with computed path properties

**Files Created:**
- `app/core/config.py` - Settings class with path methods

**Key Features:**
- Environment-specific configurations
- Dynamic path generation for NBA, WGS, Imputed data
- Ancestry and chromosome constants
- Clinical data path management

**Path Methods:**
```python
settings.get_nba_path("AAC")      # NBA files by ancestry
settings.get_wgs_path()           # WGS consolidated files  
settings.get_imputed_path("EUR", "1")  # Imputed by ancestry + chromosome
settings.get_clinical_paths()     # Clinical data files
```

### Component 3: File Discovery System ✅
**Implementation:** Automatic detection and validation of PLINK files

**Files Created:**
- `app/utils/file_discovery.py` - File scanning and validation
- `app/utils/paths.py` - PgenFileSet class for .pgen/.pvar/.psam triplets

**Key Features:**
- Validates PLINK file triplets (.pgen/.pvar/.psam)
- Scans available ancestries and chromosomes
- File integrity checking
- Missing file detection and reporting

---

## Phase 2: Core Processing (Current Focus)

### 🎯 Primary Objective
Build the variant index caching system and extraction engine to reduce processing time from **days to minutes** for 400+ pathogenic SNPs across 242+ PLINK files.

### Component 4: Variant Index Cache System
**Status:** 🔄 **READY TO IMPLEMENT**

**Key Performance Target:** Process 1M variants in <30 seconds, reduce file scanning from days to minutes

**Implementation Priority:**
1. `app/processing/cache.py` - CacheBuilder class with Parquet-based indexes
2. Cache validation and staleness detection
3. Parallel cache building for multiple ancestries/chromosomes
4. Memory-efficient cache loading

### Component 5: Variant Extraction Engine  
**Status:** 🔄 **NEXT UP**

**Key Performance Target:** Extract 400 variants from 242 files in <5 minutes, never exceed 8GB RAM

**Implementation Priority:**
1. `app/processing/extractor.py` - VariantExtractor class
2. Memory-mapped file access for large PLINK files
3. Batch extraction with concurrent processing
4. Streaming genotype access for memory efficiency

### Component 6: Harmonization Pipeline
**Status:** ⏳ **PHASE 2 FINAL**

**Implementation Priority:**
1. `app/processing/harmonizer.py` - Reference genome integration
2. Allele harmonization and strand flip detection
3. Variant normalization for indels
4. Quality control metrics

---

## Updated Technical Architecture

### Performance Optimizations (Phase 2 Focus)
- **Variant Index Caching**: Parquet-based indexes mapping variant IDs to file positions
- **Memory Mapping**: Large PLINK file access without loading entire files
- **Concurrent Processing**: Parallel extraction by chromosome/ancestry
- **Streaming Interface**: Process variants in chunks to control memory usage

### Data Flow (Phase 2)
```
PLINK Files → Cache Builder → Parquet Indexes → Variant Extractor → Harmonizer → Processed Variants
```

### File Organization (Updated)
```
~/gcs_mounts/genotools_server/carriers/
├── cache/                    # NEW: Variant index cache
│   └── release10/
│       ├── nba/{ancestry}_variant_index.parquet
│       ├── wgs/wgs_variant_index.parquet
│       └── imputed/{ancestry}/chr{chrom}_variant_index.parquet
├── results/                  # Analysis results in Parquet
└── reports/                  # Generated reports
```

---

## Claude Code Prompt Strategy (Updated)

### Phase 2 Prompt Template
Each Phase 2 component will use this enhanced template:

```
Create [component] for genomic carrier screening (Phase 2: Core Processing):

Context: [Component role in caching/extraction pipeline]
Performance Target: [Specific speed/memory requirements]
Cache Integration: [How component uses/builds Parquet indexes]

Requirements:
- [Specific technical requirements]
- [Performance constraints]
- [Cache dependencies]

Input: [Data format and structure]
Output: [Expected results format with Parquet schema]

Dependencies: [pgenlib, pandas, pyarrow for Parquet]

Include:
- Memory-efficient processing (streaming, chunking)
- Progress tracking with tqdm
- Cache validation and staleness checks
- Parallel processing capabilities
- Comprehensive error handling
- Unit tests with mock data
```

---

## Next Immediate Steps

### 1. Implement Variant Index Cache (This Week)
- Create `CacheBuilder` class in `app/processing/cache.py`
- Build Parquet indexes for all data types (NBA, WGS, Imputed)
- Implement cache validation and rebuild logic
- Add progress tracking for long-running cache builds

### 2. Variant Extraction Engine (Next Week)
- Create `VariantExtractor` class in `app/processing/extractor.py`
- Integrate with cache system for rapid variant lookup
- Implement memory-efficient genotype streaming
- Add concurrent processing for multiple files

### 3. Testing & Validation
- Unit tests with mock PLINK files
- Performance benchmarking against targets
- Memory usage profiling
- Integration testing with real data

---

## Success Metrics (Phase 2)

| Metric | Target | Current Status |
|--------|--------|----------------|
| Cache Build Time | <30 seconds for 1M variants | 🔄 Implementing |
| Variant Extraction | <5 minutes for 400 variants from 242 files | 🔄 Next |
| Memory Usage | <8GB peak RAM | 🔄 To validate |
| Cache Storage | <1GB total for all indexes | 🔄 To measure |

---

## Risk Mitigation

### Technical Risks
- **Large file memory usage**: Implement streaming and memory mapping
- **Cache corruption**: Add validation and automatic rebuild
- **Slow I/O**: Use Parquet columnar format and SSD storage

### Development Risks  
- **Complex PLINK format**: Use proven pgenlib library
- **Performance bottlenecks**: Profile and optimize iteratively
- **Integration complexity**: Build comprehensive test suite

---

## Future Phases (Post Phase 2)

### Phase 3: Analysis & Reporting (Week 3)
- Carrier detection algorithms
- Statistical analysis (Hardy-Weinberg, Fisher's exact test)
- Ancestry-stratified reporting

### Phase 4: API & Infrastructure (Week 4) 
- FastAPI endpoints implementation
- Celery background processing
- Redis caching and monitoring

---

## Development Environment Setup

### Required for Phase 2
```bash
# Core dependencies
pip install pgenlib pandas pyarrow fastapi pydantic

# Development tools  
pip install pytest tqdm black isort

# Performance monitoring
pip install memory-profiler py-spy
```

### Claude Code Integration
- Use the project document as context for each prompt
- Reference Phase 2 performance targets in prompts
- Include cache integration requirements
- Test each component independently before integration