# CLAUDE.md

## Claude Code Development Instructions

### Before Starting Work
- Always use plan mode to make a plan
- After getting the plan, write the plan to `.claude/tasks/TASK_NAME.md`
- The plan should be a detailed implementation plan with reasoning and broken-down tasks
- If the task requires external knowledge or packages, research to get the latest (use Task tool for research)
- Don't over-plan it, always think MVP
- Once you write the plan, ask me to review it. Do not continue until I approve the plan

### Implementation
- Update the plan as you work
- After completing tasks in the plan, update and append detailed descriptions of changes made so following tasks can be easily handed over to other engineers

### Development Environment Setup
**IMPORTANT**: Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```
This ensures all required dependencies (pydantic, pandas, pgenlib, etc.) are available.

### Test Execution
Run the test suite to verify functionality:
```bash
source .venv/bin/activate
python -m pytest tests/ -v  # 31 tests, zero warnings
```

Pipeline testing:
```bash
source .venv/bin/activate
python test_nba_pipeline.py        # NBA ProcessPool test
python test_imputed_pipeline.py    # IMPUTED ProcessPool test
```

### File Paths
```
Input:  ~/gcs_mounts/gp2tier2_vwb/release{10}/
Cache:  ~/gcs_mounts/genotools_server/precision_med/cache/
Output: ~/gcs_mounts/genotools_server/precision_med/results/
```

### Important Instructions
- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User