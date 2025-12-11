# Precision Med API - Feedback Issues Tracker

## Issue 1: Missing Genes in Imputed Genotypes
**Status:** ✅ ROOT CAUSE FOUND - Ready to fix

**Description:** Imputed genotypes only include data on 5 genes. Many other genes are missing, including:
- GBA1 with the African intronic variant
- LRRK2 R1628P (note: feedback said LRRK1, likely meant LRRK2)
- Potentially many others

**Root Cause:**
The IMPUTED extraction for release11 **failed silently** - only 1 out of 192 variants was actually extracted with genotype data. The extraction plan showed 209 IMPUTED files were supposed to be processed, but only chr15_AAC was actually extracted.

**Evidence:**
- 192 unique variants in IMPUTED parquet
- Only 1 variant (`Arg93Alafs*25` on chr15) has `source_file` set and actual genotype data
- 191 variants have `source_file = None` and 0 samples with data
- Variants like G2019S, rs3115534 EXIST in the source PVAR files but weren't extracted
- The 5 genes showing (ATP7B, C19orf12, MAPT, PLA2G6, SPG11) are from the chr15 data only

**What SHOULD be present (based on SNP list mapping):**
- GBA1: 33 variants
- PRKN: 17 variants
- LRRK2: 6 variants (including Gly2019Ser, Arg1628Pro)
- PINK1: 10 variants
- And many more...

**Resolution:**
1. Implement extraction tracking (see "Agreed Implementation" below)
2. Re-run the pipeline for IMPUTED data

---

## Issue 2: Missing Carriers for Some Genes
**Status:** 🔴 Not Started

**Description:** Spot-checking revealed missing carriers:
- RAB32: Should have more than 1 carrier
- SNCA: Should have more than 0 carriers

**Root Cause:** TBD - Likely related to Issue 1 (IMPUTED extraction failure). Re-check after fixing Issue 1.

**Resolution:** TBD

---

## Issue 3: Total Variants Count Higher Than Carriers Count
**Status:** 🔴 Not Started

**Description:** Sometimes the number of total variants is higher than the number of carriers. This doesn't make sense if we only count variants detected in individuals - the carrier count shouldn't be lower than variant count.

**Root Cause:** TBD

**Resolution:** TBD

---

## Issue 4: Clinical Data Issues
**Status:** 🔴 Not Started

**Description:**
- Disease duration always shows 0 carriers (there is no column for this - needs to be calculated based on age and AAO)
- Some clinical items need editing (minor changes expected)

**Root Cause:** TBD

**Resolution:** TBD

---

## Investigation Log

### 2025-12-09 - Issue 1 Investigation

#### Summary
The IMPUTED extraction for release11 appears to have failed or been interrupted. While the extraction plan included 209 files across all ancestries and chromosomes, only one file was successfully processed (chr15_AAC_release11_vwb.pgen).

#### Key Findings:
1. **Extraction Plan vs Reality**
   - Plan: 209 IMPUTED files
   - Actual: Only 1 file extracted (chr15 AAC)
   - The pipeline reported "success" with no errors

2. **Variant Data Status**
   - 192 variants in IMPUTED parquet
   - 1 variant has actual data (source_file set)
   - 191 variants are placeholder rows with no genotypes

3. **Genes Affected**
   - GBA1: 33 variants planned, 0 extracted
   - LRRK2: 6 variants planned, 0 extracted
   - PRKN: 17 variants planned, 0 extracted
   - etc.

4. **Harmonization Works**
   - Tested harmonization for chr1 AAC GBA1 variants
   - Found 10 matching variants
   - The code CAN find the variants, but extraction didn't run

5. **Other Data Types Checked**
   - WGS: 444/444 (100%) variants have data ✅
   - NBA: 1328/1328 (100%) variants have data ✅
   - EXOMES: 229/232 (99%) variants have data ✅
   - Note: `source_file` tracking doesn't work properly for merged multi-ancestry data, but actual genotype data IS present

---

## Agreed Implementation: Extraction Tracking & Failed File Retry

### Problem Statement
The current pipeline:
- Does not distinguish between "exception" failures and "empty result" failures
- Reports "success" regardless of how many files actually extracted data
- Provides no way to retry only failed files

### Agreed Approach

**Key Principle:** The pipeline should NOT fail or interrupt when extractions fail. It should:
1. Let all extractions run to completion
2. Track and log ALL outcomes (successful, empty, failed)
3. Save failed file list for targeted retry
4. Always export whatever data succeeded
5. Keep `success = True` as long as pipeline didn't crash (unchanged)

### Implementation Details

#### 1. Track All Outcomes Per File
Modify `_execute_with_process_pool` in `coordinator.py` to track three categories:

```python
extraction_outcomes = {
    'successful': [],      # Files that returned data (non-empty DataFrame)
    'empty': [],           # Files that returned empty (no matching variants found)
    'failed': []           # Files that threw exceptions
}
```

Currently the code only tracks `failed_extractions` (exceptions). We need to also track when `result_df.empty` is True.

#### 2. Log Clear Summary at End
```
Extraction Summary:
  WGS: 76/209 successful, 133 empty, 0 failed
  NBA: 11/11 successful, 0 empty, 0 failed
  IMPUTED: 1/209 successful, 0 empty, 208 failed  ← Shows the problem clearly
  EXOMES: 15/19 successful, 4 empty, 0 failed
```

#### 3. Add Extraction Stats to Pipeline Results JSON
```json
{
  "extraction_stats": {
    "IMPUTED": {
      "planned_files": 209,
      "successful": 1,
      "empty": 0,
      "failed": 208,
      "failed_files": [
        "/path/to/chr1_AAC_release11_vwb.pgen",
        "/path/to/chr1_AFR_release11_vwb.pgen",
        "..."
      ]
    },
    "WGS": {
      "planned_files": 209,
      "successful": 76,
      "empty": 133,
      "failed": 0,
      "failed_files": []
    }
  }
}
```

#### 4. Save Failed Files for Retry (Separate JSON)
Write `{job_name}_failed_files.json`:
```json
{
  "job_name": "release11",
  "timestamp": "2025-12-09T...",
  "failed_files": {
    "IMPUTED": [
      "/path/to/chr1_AAC_release11_vwb.pgen",
      ...
    ]
  }
}
```

#### 5. `--retry-failed` Mode ✅ IMPLEMENTED
Re-run only failed files:
```bash
python run_carriers_pipeline.py --retry-failed release11_failed_files.json
```

This:
- Loads existing parquet results
- Runs extraction only on failed files
- Merges new results with existing data
- Updates failed_files.json if some still fail

### Code Location
The main changes are in `app/processing/coordinator.py`:
- `_execute_with_process_pool()` - Track outcomes per file ✅
- `export_pipeline_results()` - Include extraction_stats in results ✅
- `run_full_extraction_pipeline()` - Save failed files JSON ✅
- `retry_failed_extraction()` - New method for retry mode ✅

And in `run_carriers_pipeline.py`:
- Added `--retry-failed` CLI argument ✅

### Summary Table for Release 11 (Current State)
| Data Type | Planned Files | Processed* | Variants with Data | Status |
|-----------|---------------|------------|-------------------|--------|
| WGS | 209 | 76 | 444/444 (100%) | ✅ OK |
| NBA | 11 | 11 | 1328/1328 (100%) | ✅ OK |
| IMPUTED | 209 | 1 | 21/192 (11%) | ❌ FAILED |
| EXOMES | 19 | 15 | 229/232 (99%) | ✅ OK |

*"Processed" based on `source_file` tracking which has bugs for merged data. WGS/NBA actual data is complete despite showing low processed count.

---

## Implementation Status (2025-12-09)

### ✅ COMPLETED: Extraction Tracking & Retry

All tracking features have been implemented:

1. **Extraction outcome tracking per file**
   - Tracks: successful (non-empty), empty (no matching variants), failed (exception)
   - Logs clear summary at end of extraction:
     ```
     EXTRACTION SUMMARY
     ============================================================
       WGS: 76/209 successful, 133 empty, 0 failed ✅
       IMPUTED: 1/209 successful, 0 empty, 208 failed ❌
     ============================================================
     ```

2. **extraction_stats in pipeline results JSON**
   - Per-data-type breakdown in `{job_name}_pipeline_results.json`
   - Shows planned, successful, empty, failed counts

3. **Failed files JSON for retry**
   - Auto-saved to `{job_name}_failed_files.json` when failures occur
   - Contains list of failed file paths per data type

4. **`--retry-failed` CLI mode**
   - Usage: `python run_carriers_pipeline.py --retry-failed path/to/failed_files.json`
   - Loads existing parquet, extracts only failed files, merges results
   - Updates failed_files.json if some files still fail

---

## Next Steps

1. **Re-run IMPUTED extraction** for release11 using the new pipeline
   - The extraction tracking will now show exactly which files fail
   - Can use `--retry-failed` to retry any failures

2. **Re-check Issues 2-4** after IMPUTED is fixed - they may be resolved
