# CLAUDE.md - Claude Code Development Instructions

## ⚠️ ALWAYS RUN FIRST ⚠️
```bash
source .venv/bin/activate
# Without this, imports will fail (pydantic, pandas, pgenlib, etc.)
```

## Project Status: Phase 4B - Advanced Analysis (Mature Pipeline)
**Current Goal**: Variant subset preparation with per-locus metrics and cross-dataset analysis.
Core pipeline is stable with probe selection, frontend, and all critical fixes completed.

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

# Enable/disable probe selection (enabled by default)
python run_carriers_pipeline.py --enable-probe-selection
python run_carriers_pipeline.py --no-probe-selection
```

## Frontend Interface
```bash
# Production mode
./run_frontend.sh

# Debug mode with job selection
./run_frontend.sh --debug

# Custom port
./run_frontend.sh 8502 --debug

# Legacy Streamlit viewer
./run_streamlit.sh          # Production mode
./run_streamlit.sh --debug  # Debug mode
```

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
- **Phase 4B Focus**: Per-locus statistics, cross-dataset analysis, population stratification
- **Mature Codebase**: Prefer enhancing existing modules over creating new files
- **Performance**: ProcessPool parallelization, chunk_size auto-optimization, <10min for 400 variants

## Important Development Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files unless explicitly requested
- Use `--skip-extraction` for rapid iteration during Phase 4B development
- Consider frontend architecture patterns when adding UI features
- Test changes incrementally - the pipeline is in production use