# Parallel Processing Overhaul: Sequential-Ancestry / Parallel-Chromosome

## Implementation Status

| Item | Status |
|------|--------|
| `--max-workers` correctly caps the pool | **Implemented** (2026-03-31) |
| RAM-based worker cap (`floor(RAM × 0.8 / 20 GB)`) | **Implemented** (2026-03-31) |
| `BrokenProcessPool` caught with actionable OOM message | **Implemented** (2026-03-31) |
| Option 3: Sequential-ancestry / parallel-chromosome refactor | **Planned** — not yet implemented |
| Option 4: Google Batch infrastructure | **Long-term** — not yet started |

The current stopgaps (RAM-based cap + `--max-workers`) are sufficient for normal
use. Option 3 is the recommended next step if OOM errors persist on very large
joint-calling WGS files even with a manual `--max-workers` override.

---

## Problem with the Current Approach

The pipeline uses a flat `ProcessPoolExecutor` that submits **all tasks at once** across all ancestries and chromosomes:

```
Submit all 220 tasks simultaneously:
  NBA/AAC, NBA/AFR, NBA/AJ, ...,
  WGS/AAC/chr1, WGS/AAC/chr2, ..., WGS/SAS/chrX
        ↓
OS schedules up to N workers concurrently
        ↓
Each worker memory-maps its PGEN file (~10–25 GB for WGS joint-calling)
        ↓
Peak RAM = N workers × ~20 GB per worker
```

### Why This Fails

1. **No memory awareness at submission time.** All tasks are queued before any have started, so the pool has no knowledge of how much RAM is currently in use. Workers start as fast as the OS allows.

2. **Thundering herd.** On a machine with many CPUs (e.g. 60), the auto-detected worker count was 54. 54 × ~20 GB = ~1 TB — exceeding physical RAM even on a 503 GB machine.

3. **Cascading failure.** When one worker is OOM-killed, Python's `BrokenProcessPool` exception propagates to all pending futures in the same pool. A single OOM event fails the entire remaining batch, not just one file.

4. **`max_workers` was silently ignored.** `_calculate_optimal_workers()` accepted a `max_workers` argument but discarded it, always deferring to `settings.get_optimal_workers()`. This was fixed (2026-03-31) but is a symptom of the underlying design problem.

5. **Worker count is CPU-bound, not memory-bound.** `get_optimal_workers()` used `cpu_count - reservation` as its primary heuristic. PLINK extraction is not CPU-bound — it memory-maps files and reads sequentially. CPU count is the wrong metric entirely.

### Current Mitigations (Stopgaps)

- `--max-workers` now correctly caps the pool (fixed 2026-03-31)
- `auto_detect_performance_settings()` now includes a RAM-based cap using `int(available_ram * 0.8 / 20GB)` (fixed 2026-03-31)
- `BrokenProcessPool` is caught and surfaced as a clear OOM error with a suggested `--max-workers` value

These mitigations are sufficient for now but rely on the user knowing a safe worker count. The underlying flat submission model remains.

---

## Option 3: Sequential-Ancestry / Parallel-Chromosome

### Core Idea

Instead of one flat pool across all files, restructure the work into two nested levels that match the natural structure of the data:

```
For each ancestry (sequential):          ← bounded RAM: only one ancestry at a time
    For each chromosome (parallel):      ← parallelism within a bounded scope
        Extract variants from PGEN
    Merge ancestry results
Merge all ancestry results
```

### Why This Matches the Data Structure

WGS data is already organised as `ancestry/chrN.pgen`. Each ancestry is an independent cohort — there is no cross-ancestry dependency during extraction. Running ancestries sequentially means peak RAM is bounded by:

```
peak RAM = n_chromosomes_in_parallel × RAM_per_chromosome
```

For WGS, there are ~20 chromosomes per ancestry (chr1–22, X). If we run 10 in parallel, peak RAM ≈ 10 × 20 GB = 200 GB — well within 503 GB with headroom. When that ancestry finishes, all its memory is freed before the next starts.

NBA data has one file per ancestry (not split by chromosome), so it fits naturally as a single task per ancestry-slot.

---

## Implementation Plan

### Step 1: Restructure task grouping in `coordinator.py`

Currently, `_run_processpool_extraction()` builds a flat list of `(file_path, data_type)` tuples and submits them all at once.

Refactor to group tasks by ancestry first:

```python
# Current (flat):
all_tasks = [(file1, dtype), (file2, dtype), ...]

# New (grouped):
tasks_by_ancestry = {
    'AAC': [(wgs/AAC/chr1.pgen, 'WGS'), (wgs/AAC/chr2.pgen, 'WGS'), ...],
    'AFR': [(wgs/AFR/chr1.pgen, 'WGS'), ...],
    'AAC_NBA': [(nba/AAC/AAC.pgen, 'NBA')],
    ...
}
```

This grouping is derivable from the file paths already in the plan — no new data needed.

### Step 2: Add `_run_ancestry_batch()` method

A new private method that takes one ancestry's file list and runs a bounded `ProcessPoolExecutor` over just those files:

```python
def _run_ancestry_batch(
    self,
    ancestry: str,
    tasks: List[Tuple[str, str]],
    snp_list_ids, snp_list_df,
    max_workers: int
) -> Tuple[List[pd.DataFrame], dict]:
    """Run extraction for a single ancestry using a bounded process pool."""
    ...
```

Key properties:
- Pool is created and destroyed per ancestry — memory is fully released between batches
- `max_workers` is set to `min(len(tasks), chromosome_parallelism_cap)` where cap is derived from available RAM / estimated RAM per chromosome
- Returns results and outcome stats in same format as current code so the merge logic is unchanged

### Step 3: Add `chromosome_parallelism_cap` to Settings

```python
chromosome_parallelism_cap: Optional[int] = Field(
    default=None,
    description="Max parallel chromosomes per ancestry batch. None = auto from RAM."
)
```

Auto-calculation in `get_optimal_workers()`:
```python
# For WGS: estimate ~20 GB per chromosome file
available_ram_gb = psutil.virtual_memory().available / (1024**3)
chromosome_parallelism_cap = max(1, int(available_ram_gb * 0.8 / 20))
```

Expose as `--chr-workers` CLI arg (separate from `--max-workers` which becomes the ancestry-level concurrency, default 1).

### Step 4: Update the outer loop in `_run_processpool_extraction()`

Replace the single flat pool with a sequential loop over ancestry batches:

```python
all_results = []
for ancestry, tasks in tasks_by_ancestry.items():
    logger.info(f"Processing ancestry: {ancestry} ({len(tasks)} files)")
    batch_results, batch_outcomes = self._run_ancestry_batch(
        ancestry, tasks, snp_list_ids, snp_list_df, chromosome_parallelism_cap
    )
    all_results.extend(batch_results)
    # merge batch_outcomes into extraction_outcomes
```

Progress bar updates per file as before, but now scoped per ancestry.

### Step 5: Update CLI args

| Arg | Purpose |
|-----|---------|
| `--chr-workers N` | Max parallel chromosomes within each ancestry batch (replaces `--max-workers` for WGS) |
| `--max-workers N` | Keep for backwards compatibility; maps to `chr-workers` if no explicit `--chr-workers` |

### Step 6: Update `_retry_failed_files()`

The retry path uses its own `ProcessPoolExecutor`. Apply the same ancestry-grouping logic so retries are also memory-bounded.

---

## Expected Outcomes

| Metric | Current | After Option 3 |
|--------|---------|----------------|
| Peak RAM | N_workers × 20 GB (unbounded) | N_chr_workers × 20 GB (bounded) |
| OOM risk | High with large joint-calling WGS | Low — each batch fits in available RAM |
| Speed | Faster if RAM holds, zero if OOM | Slightly slower but reliable |
| Failure blast radius | Entire pool on first OOM | Single ancestry batch only |
| Retry scope | All failed files | Per-ancestry, can resume mid-run |

## Files to Change

- `app/processing/coordinator.py` — main refactor (~150 lines changed)
- `app/core/config.py` — add `chromosome_parallelism_cap` field and auto-calc
- `run_carriers_pipeline.py` — add `--chr-workers` CLI arg

## What Does NOT Change

- Extraction logic (`extractor.py`) — untouched
- Output format / parquet structure — untouched
- Postprocessing (probe selection, locus reports, variant report) — untouched
- All existing CLI args remain valid

---

## Option 4: Google Batch (Longer-Term Architectural Direction)

Option 3 solves the memory problem in Python. Google Batch solves it at the
infrastructure level — and addresses several other limitations of the current
single-machine architecture at the same time.

### Why It's a Natural Fit

The pipeline's work units are already structured as independent tasks:
`data_type × ancestry × chromosome`. This is exactly the task-array model that
Batch is designed for. Each PLINK extraction task gets its own VM, sized
appropriately for one file — no shared memory, no `--max-workers` tuning, no OOM.

### Concrete Gains Over Option 3

| Problem | Option 3 | Google Batch |
|---------|----------|--------------|
| OOM on large WGS files | Bounded by sequential ancestry batches | Each task on its own VM — physically impossible to OOM |
| Manual `--max-workers` tuning | Still needed for chr-level parallelism | Gone — VM sizing handles it |
| 40-min sequential run | Slightly slower but reliable | 220 tasks truly parallel → finishes in time of one file |
| Failed task retry | `--retry-failed` + manual re-run | Automatic per-task retry with configurable attempts |
| Monolithic failure modes | Ancestry-scoped blast radius | Single task failure, rest continue |
| Missing file found after 40 min | Preflight check (see PREFLIGHT_CHECKS.md) | Job dependency DAG: validate → extract → postprocess |
| Cost | Fixed VM running full duration | Spot/preemptible VMs for extraction; pay per task-minute |

### Job Structure

The pipeline maps naturally to a three-stage Batch job DAG:

```
Stage 1: preflight          (1 task, standard VM, fast)
    ↓ only if all pass
Stage 2: extraction         (1 task per ancestry × chromosome × data_type)
    ↓ only if all pass                e.g. 220 tasks for R12 NBA+WGS
Stage 3: postprocessing     (1 task, standard VM)
    - probe selection
    - locus reports
    - variant report
```

Stage 2 tasks write parquet outputs to GCS. Stage 3 reads them. No single machine
needs to hold everything in memory.

### What Would Change

- **Containerisation required** — each worker runs in Docker. The pipeline and its
  dependencies (`plink2`, Python env) need to be packaged into an image and pushed
  to Artifact Registry.
- **GCS I/O explicit** — workers read PGEN files from GCS and write results to GCS.
  The current gcsfuse mount approach works inside containers but needs configuration.
- **Job submission replaces CLI** — `run_carriers_pipeline.py` becomes a job
  submission script that creates a Batch job definition rather than running
  extraction locally.
- **Debugging changes** — logs go to Cloud Logging instead of local log files.
  Individual task logs are accessible via `gcloud batch tasks describe` or the
  Cloud Console.

### What Would NOT Change

- Extraction logic (`extractor.py`) — each Batch task runs the same PLINK extraction
  code, just on one file
- Output format — same parquet files, same postprocessing, same variant report
- The SNP list, master key, and all other input files — same GCS paths

### Prerequisites

- Docker image with `plink2`, Python env, and pipeline code
- GCS bucket for intermediate parquet outputs between stages
- Batch API enabled on the GCP project
- Service account with Batch + GCS read/write permissions

### When to Consider This

Option 3 is the right near-term fix — it's ~150 lines of Python with no new
infrastructure. Google Batch becomes worth the investment when:

- Releases get larger and extraction time consistently exceeds ~1 hour
- Multiple releases need to run concurrently
- Cost optimisation on the VM becomes a priority
- The team wants fully automated, monitored pipeline runs without manual intervention
