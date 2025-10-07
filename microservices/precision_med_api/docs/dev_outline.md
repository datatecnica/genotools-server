# Genomic Carrier Screening API - Development Status

## 📋 Current Status Summary

### ✅ **Completed Phases**
- **Phase 1**: Foundation (Data models, config, file discovery)
- **Phase 2**: Core Processing (Merge-based harmonization, extraction, coordination)
- **Phase 3A**: ProcessPool Parallelization (True concurrent processing)
- **Phase 3B**: Data Quality & Organization (Correct allele counting, sample ID normalization, column organization)
- **Phase 3C**: Streamlit Viewer (Interactive web interface for result exploration)
- **Phase 3D**: Critical Bug Fixes (Multiple probe detection, sample counting accuracy)
- **Phase 4**: Postprocessing (Advanced analysis on extracted data)
- **Phase 4A**: Probe Selection Method (NBA probe quality analysis and selection against WGS ground truth)
- **Phase 4B**: Cross-Dataset Analysis & Clinical Integration (Multi-ancestry merge, genotype viewer, locus reports)

### 🎯 **Current Focus: Phase 4C - Imputed Dosage Support**
- 🔄 **Next Focus**: Imputed data dosage handling (continuous values 0.0-2.0)
- **Phase 5 (Future)**: API Development (FastAPI endpoints, authentication, background processing)

---

## 🏗️ Architecture Overview

### **Core Challenge**
Process ~400 pathogenic SNPs across 254 PLINK files from three data sources:
- **NBA**: 11 files (split by ancestry)
- **WGS**: 1 file (consolidated)  
- **IMPUTED**: 242 files (11 ancestries × 22 chromosomes)

### **Technical Solution**
- **Multiple Probe Detection**: Fixed NBA variants with multiple probes at same position being lost during deduplication
- **Correct Allele Counting**: Fixed to count pathogenic alleles instead of reference alleles
- **Advanced Genotype Transformation**: Proper handling of all harmonization scenarios with genotype flipping
- **Sample ID Normalization**: Consistent sample IDs across data types (removes '0_' prefixes, fixes WGS duplicates)
- **Accurate Sample Counting**: Fixed pipeline summary to report correct sample counts
- **Merge-Based Harmonization**: Real-time PVAR/SNP list merging with allele comparison
- **ProcessPool Parallelization**: True concurrent processing across all files
- **Memory-Efficient Processing**: Stream processing staying under 8GB RAM
- **Enhanced Streamlit Viewer**: Interactive web interface with multiple probes analysis and debug mode
- **Column Organization**: Metadata columns first, then sorted sample columns

---

## 🚀 Recent Major Achievements

### **Phase 4B: Enhanced Frontend & Cross-Dataset Analysis** - COMPLETE

**Locus Reports & Clinical Analysis** - COMPLETE
- **Feature**: Per-gene clinical phenotype statistics stratified by ancestry
- **Implementation**: LocusReportGenerator class with ancestry-stratified carrier analysis
- **Clinical Integration**: Clinical data from master key and extended clinical files (H&Y stage, MoCA scores, DAT imaging, disease duration)
- **Output**: Separate JSON and CSV reports for each data type (WGS, NBA, IMPUTED)
- **Frontend**: Interactive display with ancestry breakdowns and variant-level carrier counts (heterozygous/homozygous)
- **Bug Fix**: Fixed metadata column melting issue that prevented ancestry assignments in IMPUTED data
- **Components**: app/models/locus_report.py, app/processing/locus_report_generator.py, frontend/pages/locus_reports.py

**Multi-Ancestry Merge Fix** - COMPLETE
- **Problem**: Multiple ancestry files were concatenated instead of properly merged
- **Solution**: Implemented _merge_ancestry_results() with outer join on variant keys
- **Impact**: All samples now have genotypes for all variants across ancestries
- **Implementation**: Enhanced app/processing/coordinator.py with proper merge logic

**Genotype Viewer Frontend** - COMPLETE
- **New Page**: Interactive genotype matrix visualization
- **Features**: Real-time filtering by data type, genes/loci, carrier status; color-coded genotype display; carrier summary statistics
- **Architecture**: Modular component system with factory pattern
- **Components**: GenotypeDataLoader, GenotypeMatrixRenderer, CarrierSummaryRenderer, etc.

**Frontend Architecture Refactor** - COMPLETE
- **Migration**: From Streamlit to modular frontend with factory pattern
- **New Components**: 10+ new UI renderer components for genotype visualization
- **run_frontend.sh**: New launcher script replacing run_streamlit.sh
- **Pages**: Release Overview, Genotype Viewer, Locus Reports, Probe Validation

## 🚀 Previous Major Achievements

### **Phase 3B: Data Quality & Organization** ✅

**Critical Allele Counting Fix** ✅
- **Problem**: Pipeline was counting reference alleles instead of pathogenic alleles
- **Solution**: Implemented proper allele assignment and genotype transformation
- **Impact**: Genotype values now correctly represent pathogenic allele counts (0=none, 1=het carrier, 2=hom carrier)
- **Implementation**: Enhanced `_harmonize_extracted_genotypes()` with scenario-specific handling

**Sample ID Normalization** ✅  
- **WGS Duplicates**: `SAMPLE_001234_SAMPLE_001234` → `SAMPLE_001234`
- **NBA/IMPUTED Prefixes**: `0_SAMPLE_001234` → `SAMPLE_001234`
- **Result**: Consistent sample IDs across all data types for easy data integration
- **Implementation**: Added `_normalize_sample_id()` method with duplicate detection

**Column Organization** ✅
- **Metadata First**: 15 metadata columns in defined order, then sorted sample columns
- **Duplicate Handling**: Prevents column name conflicts after normalization
- **Implementation**: Enhanced `_reorder_dataframe_columns()` method in coordinator.py

**Code Quality Improvements** ✅
- **Test Suite Streamlining**: Reduced transformer.py by 83% (348→58 lines), removed 11 unused functions
- **Redundancy Analysis**: Identified 9 duplicate columns between parquet and variant_summary files
- **Performance**: 3 tests passing, zero warnings

### **Phase 3C: Streamlit Viewer** ✅ **COMPLETED**

**Interactive Web Interface** ✅
- **📊 Overview Tab**: Pipeline summary, sample counts, file information
- **🧬 Variant Browser**: Filter variants by harmonization action, chromosome, ancestry
- **📈 Statistics Tab**: Visualizations of harmonization distributions and variant counts  
- **💾 File Downloads**: Direct access to processed parquet and summary files

**Key Features** ✅
- **Real-time Data Loading**: Direct GCS mount access without file uploads
- **Redundancy Cleanup**: Streamlined to show only relevant columns (removed COUNTED/ALT duplicates)
- **User Education**: Clear explanation that genotypes represent pathogenic allele counts

### **Phase 3D: Critical Bug Fixes** ✅ **COMPLETED**

**Multiple Probe Detection Fix** ✅ **CRITICAL**
- **Problem**: NBA variants with multiple probes at the same genomic position were being lost during processing
- **Root Cause**: SNP ID mismatch between harmonization (coordinate-based) and plan creation (SNP names)
- **Solution**: 
  - Fixed harmonization to use `snp_name` instead of `variant_id` in records
  - Updated coordinator to pass `snp_name` as `snp_list_ids` 
  - Modified deduplication to include `variant_id` to preserve different probes
- **Impact**: 77 SNPs now correctly show multiple probes (316 variants vs 218 previously)
- **Implementation**: Updated `app/processing/harmonizer.py` and `app/processing/coordinator.py`

**Sample Counting Fix** ✅  
- **Problem**: Pipeline summary reported "Total samples: 0" instead of actual count
- **Root Cause**: `source_file` column missing from metadata list, sample aggregation not working
- **Solution**: 
  - Added `source_file` to `_get_sample_columns()` metadata list
  - Implemented proper sample count aggregation in `export_pipeline_results()`
- **Impact**: Accurate reporting of 1,215 samples in pipeline summaries
- **Implementation**: Updated `app/processing/output.py` and `app/processing/coordinator.py`

**Enhanced Streamlit Viewer** ✅
- **Multiple Probes Analysis**: New section showing SNPs with multiple probes detected
- **Debug Mode**: Optional job selection with `./run_streamlit.sh --debug` for development
- **Deprecation Fix**: Updated `use_container_width` to `width='stretch'` 
- **Production/Debug Split**: Clean interface for users, full access for developers

### **Phase 4: Postprocessing** ✅ **PHASE 4A COMPLETED**

### **Phase 4A: Probe Selection Method** ✅ **COMPLETED**

**NBA Probe Quality Analysis** ✅
- **Objective**: Validate NBA probe performance against WGS ground truth data for optimal probe selection
- **Problem**: Multiple NBA probes exist for the same genomic position with varying quality/accuracy
- **Solution**: Dual-metric analysis system comparing NBA probes to WGS data as ground truth

**Implementation Features** ✅
- **Automatic Detection**: Identifies mutations with multiple NBA probes via `snp_list_id` grouping
- **Dual Validation Approaches**:
  - **Diagnostic Classification**: Treats carrier detection as binary classification (sensitivity/specificity focus)
  - **Genotype Concordance**: Exact genotype-level agreement analysis (0/1/2 comparison with transition matrix)
- **Consensus Recommendations**: Combines both approaches with confidence scoring and disagreement analysis
- **Quality Thresholds**: Configurable sensitivity (80%), specificity (95%), and concordance (90%) thresholds

**Key Components** ✅
- **ProbeSelector** (`app/processing/probe_selector.py`): Core analysis engine with diagnostic/concordance metrics
- **ProbeRecommendationEngine** (`app/processing/probe_recommender.py`): Strategy-based probe selection with consensus analysis
- **Integration**: Seamless integration with pipeline (enabled by default, can use `--skip-probe-selection`)
- **Output**: Comprehensive JSON reports with per-mutation analysis and methodology comparison

**Clinical Impact** ✅
- **Evidence-Based Selection**: Quantitative metrics support optimal probe choices for carrier screening
- **Quality Assurance**: Identifies poor-performing probes that may need exclusion from clinical analysis
- **Methodology Validation**: Agreement analysis between diagnostic and concordance approaches
- **Precision Medicine**: Improves diagnostic accuracy by selecting highest-quality probes

**Usage** ✅
```bash
# Automatic (default behavior)
python run_carriers_pipeline.py

# On existing results (skip extraction, run probe selection)
python run_carriers_pipeline.py --skip-extraction

# Skip probe selection if results already exist
python run_carriers_pipeline.py --skip-probe-selection
```

**Output Files** ✅
- `{job_name}_probe_selection.json`: Comprehensive probe analysis report with recommendations

---

## 🎯 Next Phases: Advanced Analysis & Optimization

### **Phase 4B: Cross-Dataset Analysis & Clinical Integration** - COMPLETE
**Objective**: Complete cross-dataset analysis and clinical phenotype integration

**Completed Features**:
- **Multi-Ancestry Merge**: Proper merging of NBA/IMPUTED results across ancestries
- **Genotype Viewer**: Interactive matrix visualization with filtering
- **Frontend Architecture**: Modular component system with factory pattern
- **Locus Reports**: Per-gene clinical phenotype statistics with separate reports per data type
- **Clinical Integration**: H&Y stage, MoCA scores, DAT imaging, disease duration from phenotype files
- **Frontend Display**: Ancestry breakdowns, variant carrier counts (heterozygous/homozygous)
- **Bug Fix**: Fixed metadata column melting preventing ancestry assignments in IMPUTED data

### **Phase 4C: Imputed Dosage Support** - NEXT FOCUS
**Objective**: Support continuous dosage values in imputed data

**Planned Features**:
- **Imputed Dosage Handling**: Support continuous dosage values (0.0-2.0) in extraction
- **Dosage Visualization**: Update frontend to display dosage gradients
- **Cross-Dataset Concordance**: Compare dosages vs discrete genotypes
- **Carrier Thresholds**: Configurable thresholds for dosage-based carrier calling

### **Phase 4D: Variant Subset Preparation** - FUTURE
**Objective**: Prepare variant subsets with comprehensive metrics

**Planned Features**:
- **Per-Locus Metrics**: Variant-level statistics across all datasets
- **Population Stratification**: Ancestry-specific variant analysis
- **Quality Control Metrics**: Hardy-Weinberg equilibrium testing, call rate analysis
- **Variant Annotation**: Integration of clinical significance and pathogenicity scores
- **Subset Generation**: Create filtered variant sets based on quality thresholds

### **Phase 5: API Development** - FUTURE
**Objective**: Build production-ready API and web interface for genomic analysis

**Phase 5A: API Development** - PLANNED
- **FastAPI Endpoints**: RESTful API for variant querying, analysis submission, and result retrieval
- **Authentication**: User management and secure access to genomic data
- **Background Processing**: Celery + Redis for long-running analysis jobs
- **Result Caching**: Optimized storage and retrieval of analysis results

**Phase 5B: Web Interface** - PLANNED
- **Interactive Dashboard**: Advanced web UI replacing Streamlit prototype
- **Variant Explorer**: Production-grade variant browsing with filtering and visualization
- **Analysis Workflow**: Guided interface for carrier screening and variant subset creation
- **Result Visualization**: Advanced charts, plots, and statistical summaries

---

## 📊 Performance Targets

### **Current Performance** ✅
| Metric | Target | Status |
|--------|--------|---------|
| Single NBA File | <30s | ✅ 18.5s |
| Memory Usage | <8GB | ✅ Validated |
| ProcessPool Architecture | True parallelism | ✅ Complete |
| Data Quality | Correct allele counting | ✅ Fixed |
| Sample ID Consistency | Normalized across data types | ✅ Complete |
| Test Coverage | Streamlined essential tests | ✅ 3 tests passing |

### **Data Quality Achievements** ✅
| Metric | Target | Status |
|--------|--------|---------|
| Allele Counting | Pathogenic alleles counted | ✅ Fixed |
| Sample ID Format | Consistent across NBA/WGS/IMPUTED | ✅ Normalized |
| Column Organization | Metadata first, sorted samples | ✅ Complete |
| Code Redundancy | Eliminate unused functions | ✅ 83% reduction |
| Streamlit Interface | Interactive result exploration | ✅ Complete |

### **Next Targets** 🎯
| Metric | Target | Status |
|--------|--------|---------|
| File Redundancy | Eliminate variant_summary.csv | 🔄 Proposed |
| Storage Optimization | 25% reduction via consolidation | 🔄 Next |
| Enhanced Carrier Analysis | Population-level statistics | 🔄 Next |

---

## 🛠️ Technology Stack

### **Core Technologies**
- **Backend**: FastAPI, Pydantic v2
- **Processing**: pandas, numpy, ProcessPoolExecutor
- **Genomics**: pgenlib (PLINK file processing)
- **Storage**: Parquet files with Snappy compression
- **Testing**: pytest (31 tests, zero warnings)

### **File Paths**
```
Input:  ~/gcs_mounts/gp2tier2_vwb/release10/
Output: ~/gcs_mounts/genotools_server/precision_med/results/
```

### **Development Setup**
```bash
# Always activate virtual environment
source .venv/bin/activate

# Run streamlined tests
python -m pytest tests/ -v  # 3 tests, zero warnings

# Test pipelines with real data
python test_nba_pipeline.py        # NBA test
python test_imputed_pipeline.py    # IMPUTED test

# Run full pipeline
python run_carriers_pipeline.py --job-name my_analysis

# Skip extraction for rapid postprocessing development
python run_carriers_pipeline.py --job-name my_analysis --skip-extraction

# Launch Streamlit viewer
streamlit run streamlit_viewer.py
# Or use convenience script (production mode)
./run_streamlit.sh
# Debug mode with job selection
./run_streamlit.sh --debug
```

### **Key Files Updated**
```
# Recent Updates (Phase 4B)
app/processing/coordinator.py            # Added _merge_ancestry_results() for proper multi-ancestry merging
app/processing/locus_report_generator.py # Per-gene clinical phenotype statistics with per-datatype reports; fixed metadata melting bug
app/models/locus_report.py               # Locus report data models with variant details
frontend/main.py                         # Added Genotype Viewer and Locus Reports pages
frontend/pages/genotype_viewer.py        # Genotype viewer page implementation
frontend/pages/locus_reports.py          # Locus reports page with ancestry breakdowns and carrier counts
frontend/pages/overview.py               # Removed redundant probe selection notice
frontend/utils/data_loaders.py           # GenotypeDataLoader and locus report loaders
frontend/utils/ui_components.py          # UI renderer components for genotype visualization
frontend/utils/genotype_analysis.py      # Genotype analysis utilities
run_frontend.sh                          # Frontend launcher script (replaces run_streamlit.sh)

# Previous Updates
run_carriers_pipeline.py         # Added --skip-extraction, --skip-probe-selection flags
app/processing/harmonizer.py     # Fixed multiple probe detection with SNP name mapping
app/processing/output.py         # Added source_file to metadata columns
app/processing/extractor.py      # Fixed allele counting & genotype transformation
app/processing/probe_selector.py # NBA probe quality analysis against WGS ground truth
app/processing/probe_recommender.py # Probe selection recommendations
app/models/probe_validation.py   # Probe analysis data models
streamlit_viewer.py              # Enhanced with multiple probes analysis & debug mode
tests/test_transformer.py        # Streamlined to essential tests only
README.md                       # Updated with latest features and fixes
docs/dev_outline.md             # Updated development status
CLAUDE.md                       # Development instructions
```

---

## 🎯 Immediate Next Actions

### **Phase 4C: Imputed Dosage Support** (Immediate Priority)
1. **Dosage Extraction**: Implement pgenlib dosage reading for continuous values (0.0-2.0)
2. **Dosage Visualization**: Update frontend genotype viewer for gradient display
3. **Cross-Dataset Concordance**: Compare dosages vs discrete genotypes
4. **Carrier Thresholds**: Configurable thresholds for dosage-based carrier calling

### **Phase 4D: Variant Subset Preparation** (Future Priority)
1. **Per-Locus Statistics**: Calculate variant-level metrics (allele frequencies, carrier counts, call rates) across all datasets
2. **Cross-Dataset Quality Comparison**: Compare variant detection quality and concordance across NBA/WGS/IMPUTED sources
3. **Quality Control Implementation**: Hardy-Weinberg equilibrium testing, genotype concordance, and filtering thresholds
4. **Variant Annotation Integration**: Add clinical significance, gene context, and pathogenicity metadata
5. **Subset Generation Tools**: Create filtered variant sets based on quality and clinical relevance criteria

### **Phase 5: Frontend Development** (Future Priority)
7. **API Endpoints**: Develop FastAPI endpoints for variant querying and analysis submission
8. **Web Interface**: Build production-grade dashboard replacing Streamlit prototype
9. **Authentication & Security**: Implement user management and secure genomic data access
10. **Background Processing**: Set up Celery + Redis for long-running analysis workflows

---

## 📈 Success Metrics

### **Technical Metrics**
- ✅ ProcessPool implementation (Component 7A)
- ✅ Cross-DataType combination (Component 7)
- ✅ **Multiple probe detection fix** (Critical bug preventing variant loss)
- ✅ **Sample counting accuracy** (Fixed "Total samples: 0" reporting)
- ✅ Correct allele counting fix (Critical data quality issue)
- ✅ Sample ID normalization across data types
- ✅ Column organization (metadata first, sorted samples)
- ✅ Enhanced Streamlit viewer with debug mode and multiple probes analysis
- **Probe Selection Method** (NBA probe quality validation against WGS ground truth)
- **Locus Reports**: Per-gene clinical phenotype statistics with per-datatype reports and variant carrier counts
- **Clinical Data Integration**: Fixed ancestry join bug in IMPUTED data processing
- Zero deprecation warnings (Streamlit and Pydantic v2)
- Streamlined test suite (3 essential tests)
- Imputed dosage support for continuous values (0.0-2.0) - NEXT FOCUS
- Variant subset preparation with per-locus metrics and cross-dataset analysis - FUTURE

### **Quality Metrics**
- Process isolation and error handling
- Original allele transparency with harmonization tracking
- IMPUTED file format support
- Enhanced deduplication logic preserving multiple probes per SNP
- Critical Fix: Pathogenic allele counting instead of reference alleles
- Complete data capture: 77 SNPs with multiple probes (316 vs 218 variants)
- Accurate sample reporting: 1,215 samples correctly counted
- Consistent sample ID format across NBA/WGS/IMPUTED data types
- User-friendly Streamlit interface with multiple probes analysis
- Production/Debug interface split for optimal user experience
- Probe Quality Validation with dual-metric analysis (diagnostic + concordance)
- Evidence-Based Probe Selection with consensus recommendations and confidence scoring
- Locus Reports & Clinical Integration: Per-datatype ancestry-stratified statistics with variant carrier counts
- Clinical Data Bug Fix: Fixed metadata column melting preventing ancestry assignments
- Imputed dosage support for continuous genotype values - NEXT FOCUS
- Storage optimization (25% reduction via consolidation) - FUTURE

### **Impact Assessment**
The recent critical bug fixes, probe selection, and locus reports implementation represent a **major milestone** in pipeline development:
- **Data Completeness**: Fixed multiple probe detection ensuring 77 SNPs show all their probes (98 additional variants recovered)
- **Accuracy**: Pipeline summaries now correctly report 1,215 samples instead of misleading "0"
- **Correctness**: Fixed fundamental allele counting issue affecting all downstream analysis
- **Consistency**: Normalized sample IDs enable seamless cross-data-type integration
- **Usability**: Enhanced Streamlit interface with multiple probes analysis and production/debug modes
- **Maintainability**: Streamlined codebase with 83% reduction in unused code
- **Clinical Quality**: Evidence-based probe selection with WGS ground truth validation ensures optimal diagnostic accuracy
- **Methodology Validation**: Dual-metric analysis (diagnostic + concordance) provides comprehensive probe quality assessment
- **Clinical Integration**: Locus reports with ancestry-stratified clinical phenotype statistics enable population-level analysis

This comprehensive update ensures the pipeline now captures **all available genomic data**, produces **scientifically accurate** carrier frequency data, provides **evidence-based probe selection**, and integrates **clinical phenotype analysis** ready for population genetics research and clinical interpretation.