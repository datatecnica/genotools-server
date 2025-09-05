# CLAUDE.md
## Plan & Review

### Before Starting work
- always in plan mode to make a plan
- after getting the plan, write the plan to .claude/tasks/TASK_NAME.md
- the plan should be a detailed implementation plan and the reasoning behind them as well as tasks broken down
- if the task requires external knowledge or a certain package,  research to get the latest (use Task tool for research)
- don't over plan it, always think MVP
- once you write the plan, ask me to review it. do not continue until I approve the plan

### implementation
- update the plan as you work
- after you complete the tasks in the plan, you should update and append detailed descriptions of the changes you made so following tasks can be easily handed over to other engineers

## Project Overview

**Genomic Carrier Screening FastAPI Application** - A production-ready system for identifying carriers of pathogenic variants in large-scale genomic data from the GP2 (Global Parkinson's Genetics Program) cohort.

### Core Challenge
Process ~400 pathogenic SNPs across 242+ PLINK 2.0 files (>1M variants each) from three data sources with different organizational structures:
- **NBA**: 11 files split by ancestry
- **WGS**: 1 consolidated file  
- **Imputed**: 242 files (11 ancestries × 22 chromosomes)

### Technical Solution
- **Merge-Based Harmonization**: Direct merging of PVAR and SNP list data with real-time allele comparison
- **Allele Harmonization**: Handle strand flips and allele swaps to ensure correct genotype extraction
- **Memory-Efficient Processing**: Stream processing and memory mapping to stay under 8GB RAM
- **Output**: Harmonized genotypes in PLINK TRAW format for downstream analysis

### Performance Targets
- Reduce variant extraction from **days to <10 minutes** for 400 variants across all files
- Support concurrent analysis of 10+ jobs
- Real-time harmonization without pre-processing delays
- API response time <500ms for variant queries

### Technology Stack
- **Core**: FastAPI, Pydantic v2, pgenlib (PLINK file processing)
- **Storage**: Parquet files for caching and results
- **Processing**: NumPy, Pandas, ProcessPoolExecutor for true parallelization
- **Future**: Celery + Redis for background jobs, PostgreSQL migration path

### Current Status
- ✅ **Phase 1 Complete**: Data models, configuration, file discovery
- ✅ **Phase 2 Complete**: Merge-based harmonization, extraction engine, coordination system
- ✅ **Phase 3A Complete**: ProcessPool parallelization for concurrent file extraction
- 🎯 **Phase 3B Ready**: Within-datatype combination, carrier detection, statistical analysis (NEXT FOCUS)
- ⏳ **Phase 4 Planned**: REST API endpoints, background processing, monitoring

### Phase 2 Achievements (Completed)
- ✅ Merge-based harmonization engine with real-time processing
- ✅ Allele harmonization engine with strand flip detection
- ✅ Multi-source extraction engine (NBA/WGS/Imputed)
- ✅ Genotype transformation pipeline
- ✅ TRAW format output generation
- ✅ End-to-end pipeline integration and testing
- ✅ Comprehensive test coverage for core components

### Phase 3A Achievements (Recently Completed)
- ✅ ProcessPoolExecutor parallelization replacing ThreadPoolExecutor
- ✅ Process-isolated extraction worker for true concurrent processing
- ✅ Resource management with optimal process count calculation  
- ✅ Robust error handling with process failure isolation
- ✅ Progress tracking with tqdm integration
- ✅ Original allele transparency in variant summaries (pgen_a1/pgen_a2)
- ✅ Backwards compatibility with existing test scripts

### File Paths
```
Input:  ~/gcs_mounts/gp2tier2_vwb/release{10}/
Cache:  ~/gcs_mounts/genotools_server/precision_med/cache/
Output: ~/gcs_mounts/genotools_server/precision_med/output/
```

### Development Environment Setup
**IMPORTANT**: Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```
This ensures all required dependencies (pydantic, pandas, pgenlib, etc.) are available.