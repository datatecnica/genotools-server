# CLAUDE.md - Claude Code Development Instructions

## ⚠️ ALWAYS RUN FIRST ⚠️
```bash
source .venv/bin/activate
# Without this, imports will fail (pydantic, pandas, pgenlib, etc.)
```

## Project Status: Phase 4B - Dosage Support & Cross-Dataset Analysis
**Current Goal**: Complete imputed data dosage handling and cross-dataset analysis features.
✅ Multi-ancestry merge fix completed
✅ Genotype viewer frontend added
🔄 Next: Imputed dosage support (continuous values 0.0-2.0 instead of discrete 0/1/2)

## Implementation Rules
- **If changing >3 files or adding new module** → Create plan first in `.claude/tasks/TASK_NAME.md`
- **If fixing bug or enhancing existing function** → Direct implementation
- **If touching app/processing/extractor.py** → Run `test_nba_pipeline.py` after
- **If modifying frontend/** → Test with `./run_frontend.sh --debug`
- **If working on Phase 4B analysis** → Use `--skip-extraction` for rapid iteration

## Quick Development Patterns
- **Testing UI changes**: Edit `frontend/components/` → `./run_frontend.sh --debug`
- **Testing extraction logic**: Use `--skip-extraction` with existing job
- **Adding metrics**: Extend `app/models/probe_validation.py`, not new files
- **Rapid iteration**: `python run_carriers_pipeline.py --job-name existing_analysis --skip-extraction`

## Never Do This
- **DON'T** create new test files (use existing `tests/` directory)
- **DON'T** duplicate functionality (check `app/processing/` first)
- **DON'T** run full pipeline for testing (use `--skip-extraction`)
- **DON'T** create documentation unless explicitly asked
- **DON'T** create files unless absolutely necessary
- **DON'T** proactively create README or *.md files
- **ALWAYS** prefer editing existing files over creating new ones

## Recent Critical Fixes (Don't Regress)
- **Allele counting**: Now counts pathogenic, not reference alleles (fixed in extractor.py)
- **Sample IDs**: Normalized without '0_' prefix (WGS duplicates also fixed)
- **Multiple probes**: Fixed deduplication to preserve different variant_id values (77 SNPs with multiple probes)
- **Multi-ancestry merge**: Fixed NBA/IMPUTED to properly merge (not concat) across ancestries (coordinator.py)
- **Imputed data**: Currently extracts as discrete 0/1/2 via PLINK2, but data is actually dosages (0.0-2.0)

## Core Files (Edit with Caution)
```
app/processing/extractor.py     # Allele counting logic - test after changes
app/processing/coordinator.py   # ProcessPool orchestration - handles parallelization
app/models/harmonization.py     # Data models - many dependencies
app/processing/probe_selector.py # Probe quality validation against WGS
frontend/components/             # UI components - test with --debug mode
```

## Pipeline Execution Commands
```bash
# Full pipeline run (~45 minutes)
python run_carriers_pipeline.py --job-name my_analysis

# Rapid development (0.0s - reuses existing results)
python run_carriers_pipeline.py --job-name existing_analysis --skip-extraction

# Quick validation (5-10 minutes)
python run_carriers_pipeline.py --ancestries AAC AFR

# Probe selection (enabled by default, can be skipped)
python run_carriers_pipeline.py                        # probe selection enabled
python run_carriers_pipeline.py --skip-probe-selection # skip probe selection
```

## Frontend Interface
```bash
# Production mode
./run_frontend.sh

# Debug mode with job selection
./run_frontend.sh --debug

# Custom port
./run_frontend.sh 8502 --debug

# Legacy Streamlit viewer (deprecated)
streamlit run streamlit_viewer.py
```

### Frontend Development Notes
- **Streamlit API**: Use `width='stretch'` instead of deprecated `use_container_width=True`
  - `use_container_width` will be removed after 2025-12-31
  - `width='stretch'` for full width, `width='content'` for auto width
- **Component Architecture**: Use factory/facade patterns in `frontend/utils/ui_components.py`
- **Data Loading**: Use `DataLoaderFactory` for consistent data access patterns

## Test Execution
```bash
source .venv/bin/activate
python -m pytest tests/ -v  # 3 tests, zero warnings

# Pipeline integration tests
python test_nba_pipeline.py        # NBA ProcessPool test
python test_imputed_pipeline.py    # IMPUTED ProcessPool test
```

## File Paths
```
Input:  ~/gcs_mounts/gp2tier2_vwb/release{10}/
Cache:  ~/gcs_mounts/genotools_server/precision_med/cache/
Output: ~/gcs_mounts/genotools_server/precision_med/results/
```

## Architecture Context
- **Core Pipeline**: Stable with probe selection, allele counting fixes, sample ID normalization
- **Frontend**: Modular architecture with factory/facade patterns in `frontend/` directory
  - New genotype viewer page with interactive matrix visualization
  - Component-based UI system with specialized renderers
- **Phase 4B Focus**: Imputed dosage support, cross-dataset analysis, population stratification
  - Multi-ancestry merge fixed with proper outer join logic
  - Genotype viewer supports filtering by data type, genes, carrier status
- **Mature Codebase**: Prefer enhancing existing modules over creating new files
- **Performance**: ProcessPool parallelization, chunk_size auto-optimization, <10min for 400 variants

## Imputed Data Dosage Notes
- **Current State**: PLINK2 extraction produces discrete 0/1/2 genotypes via `.traw` format
- **Actual Data**: Imputed files contain continuous dosage values (0.0-2.0)
- **Required Change**: Need to use pgenlib's dosage reading capabilities or PLINK2's dosage export
- **Frontend Impact**: Genotype viewer will need gradient display for dosages vs discrete colors
- **Carrier Calling**: May need configurable thresholds (e.g., dosage > 0.5 = carrier)

## Important Development Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- Use `--skip-extraction` for rapid iteration during Phase 4B development
- Consider frontend architecture patterns when adding UI features
- Test changes incrementally - the pipeline is in production use