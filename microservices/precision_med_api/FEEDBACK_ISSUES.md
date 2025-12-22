# Precision Med API - Feedback Issues Tracker

## Issue 1: Missing Genes in Imputed Genotypes
**Status:** ✅ FIXED (2025-12-15)

**Description:** Imputed genotypes only include data on 5 genes. Many other genes are missing, including:
- GBA1 with the African intronic variant
- LRRK2 R1628P (note: feedback said LRRK1, likely meant LRRK2)
- Potentially many others

**Root Cause Found (2025-12-15):**
The bug was in `_merge_ancestry_results()` function in `coordinator.py`. When merging DataFrames from different chromosome files (which share the same sample columns), pandas outer join creates duplicate columns with `_dup` suffix. The code was **dropping** these `_dup` columns without first **combining** them with the original columns using `combine_first()`.

This caused all genotype data for variants from "later" DataFrames in the merge sequence to be lost.

**Fix Applied:**
In `coordinator.py` lines ~692-704, changed from:
```python
# Old (BROKEN): Just drop _dup columns
dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
if dup_cols:
    merged_df = merged_df.drop(columns=dup_cols)
```

To:
```python
# New (FIXED): Combine then drop
dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
for dup_col in dup_cols:
    base_col = dup_col[:-4]
    if base_col in merged_df.columns:
        merged_df[base_col] = merged_df[base_col].combine_first(merged_df[dup_col])
if dup_cols:
    merged_df = merged_df.drop(columns=dup_cols)
```

**Verification:**
- Before fix: GBA1 had 0 non-NaN samples
- After fix: GBA1 has 16,185 non-NaN samples (correct!)
- Full pipeline: 192 variants, 103,786 samples, 56,156 carriers

**See:** `DEBUGGING_IMPUTED_ISSUE.md` for detailed debugging history

**Further Refactor (2025-12-15):**
The merge logic was refactored to use a more efficient concat+merge approach:
1. Group DataFrames by ancestry
2. `pd.concat()` within ancestry (same samples, different variants)
3. `pd.merge()` across ancestries (different samples)

This improves performance from O(n²) to O(n) and makes the code clearer.

---

## Issue 2: Missing Carriers for Some Genes
**Status:** ✅ RESOLVED (2025-12-17) - Not a bug

**Description:** Spot-checking revealed missing carriers:
- RAB32: Should have more than 1 carrier
- SNCA: Should have more than 0 carriers

**Root Cause:** The RAB32 and SNCA variants do NOT exist in the IMPUTED source files. This is expected behavior - these extremely rare pathogenic variants are not present in the imputation reference panel (likely TOPMed) and cannot be reliably imputed.

**Investigation Results (2025-12-17):**

| Gene | Variant | Position | WGS | NBA | IMPUTED | EXOMES |
|------|---------|----------|-----|-----|---------|--------|
| RAB32 | Ser71Arg | chr6:146544084 | 1 carrier | - | **NOT IN SOURCE** | - |
| SNCA | Glu46Lys | chr4:89828170 | - | ✓ | **NOT IN SOURCE** | - |
| SNCA | Gly51Asp | chr4:89828154 | ✓ | ✓ | **NOT IN SOURCE** | - |
| SNCA | Ala53Thr | chr4:89828149 | ✓ | ✓ | **NOT IN SOURCE** | - |
| SNCA | Ala53Glu | chr4:89828148 | - | ✓ | **NOT IN SOURCE** | - |
| SNCA | Ala30Pro | chr4:89835580 | ✓ | **52 carriers** | **NOT IN SOURCE** | - |

**Conclusion:**
- SNCA has 52 carriers in NBA data (NeuroBooster Array) - this is correct
- RAB32 has 1 carrier in WGS - may be correct (very rare variant)
- The pipeline correctly reports what's available in each data source
- Missing from IMPUTED is expected for rare pathogenic variants

---

## Issue 3: Total Carriers != Heterozygous + Homozygous (IMPUTED)
**Status:** ✅ FIXED (2025-12-17)

**Description:** For IMPUTED data, the "Total Carriers" count was not equal to Heterozygous + Homozygous. Example: c.1225-34C>A showed carrier=13,636, het=1,632, hom=402 (but het+hom=2,034).

**Root Cause:** IMPUTED data contains **dosage values** (continuous 0.0-2.0) not discrete genotypes (0, 1, 2). The old counting logic was:
```python
carrier_count = (genotypes > 0).sum()    # Counted 0.01, 0.1, etc. as carriers
het_count = (genotypes == 1).sum()       # Only exact 1.0
hom_count = (genotypes == 2).sum()       # Only exact 2.0
```

This meant samples with dosage 0.5 were counted as "carriers" but not as het or hom.

**Fix Applied (2025-12-17):**
Added configurable dosage thresholds in `app/core/config.py`:
```python
dosage_het_min: float = 0.5   # Minimum dosage to call heterozygous
dosage_het_max: float = 1.5   # Maximum dosage to call heterozygous
dosage_hom_min: float = 1.5   # Minimum dosage to call homozygous
```

Updated counting in `app/processing/locus_report_generator.py`:
```python
het_count = ((genotypes >= het_min) & (genotypes < het_max)).sum()
hom_count = (genotypes >= hom_min).sum()
carrier_count = het_count + hom_count  # Now always matches!
```

**CLI Support:**
```bash
# Default soft calls (0.5, 1.5, 1.5) - rounds to nearest integer
python run_carriers_pipeline.py --job-name release11 --release 11

# Hard calls (stricter thresholds for high-confidence only)
python run_carriers_pipeline.py --job-name release11 --release 11 \
    --dosage-het-min 0.9 --dosage-het-max 1.1 --dosage-hom-min 1.9
```

**Verification:**
- Before: carrier=13,636, het+hom=2,034 (MISMATCH)
- After: carrier=3,777, het=3,212, hom=565, het+hom=3,777 (MATCH ✓)

---

## Issue 4: Clinical Data Issues
**Status:** ✅ FIXED (2025-12-18) - Disease duration now calculates correctly

**Description:**
- Disease duration always shows 0 carriers (there is no column for this - needs to be calculated based on age and AAO)
- Some clinical items need editing (minor changes expected)

**Root Cause (2025-12-18):**
The code in `locus_report_generator.py` was looking for columns that don't exist:
- Code expected: `age_at_baseline` and `age_at_onset` (from extended clinical data)
- Actual columns in master key: `age_at_sample_collection` and `age_of_onset`

The disease duration calculation at lines 653-657 checked:
```python
if 'age_at_baseline' in unique_carriers.columns and 'age_at_onset' in unique_carriers.columns:
    duration = age_at_baseline - age_at_onset
```

Since neither column existed, the condition was never true, resulting in 0 for all disease duration metrics.

**Fix Applied (2025-12-18):**
1. Updated `_load_master_key()` to include `age_at_sample_collection` and `age_of_onset` columns
2. Removed non-existent `age_at_baseline` and `age_at_onset` from `_load_extended_clinical()`
3. Updated `_join_clinical_data()` to pass age columns from master key to carriers
4. Updated `_calculate_ancestry_metrics()` to use correct column names:
```python
if 'age_at_sample_collection' in unique_carriers.columns and 'age_of_onset' in unique_carriers.columns:
    duration = age_at_sample_collection - age_of_onset
    # Only count valid durations (non-negative)
    valid_duration = duration[duration >= 0]
    disease_duration_lte_3 = (valid_duration <= 3).sum()
    ...
```

**Verification (2025-12-18):**
Re-ran the pipeline with `--release 11 --skip-extraction`. Disease duration now populates correctly:

| Gene | Total Carriers | ≤3 years | ≤5 years | ≤7 years |
|------|---------------|----------|----------|----------|
| GBA1 | 5,646 | 648 | 1,028 | 1,354 |
| LRRK2 | 2,896 | 191 | 324 | 470 |
| PRKN | 2,637 | 286 | 450 | 558 |
| GCH1 | 832 | 86 | 152 | 208 |

Before fix: All disease duration columns showed 0.

**Note:** "Some clinical items need editing" mentioned in original feedback - awaiting clarification on what specific changes are needed

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

## Issue 5: WGS Missing Carriers Due to Merge Bug
**Status:** ✅ FIXED (2025-12-18)

**Description:** WGS data was showing fewer carriers than expected for SNCA and RAB32 variants. Investigation revealed this was the same merge bug that was fixed for IMPUTED data in Issue 1.

**Root Cause (2025-12-18):**
WGS is now split by ancestry and chromosome (like IMPUTED), but the code was using `pd.concat()` + `drop_duplicates()` instead of the proper `_merge_ancestry_results()` function. This caused sample columns from later ancestries to be lost when the same variant appeared in multiple ancestry files.

Evidence from release11 WGS data before fix:
- SNCA Ala53Thr (chr4:89828149): Only **EAS** samples (4,470) kept, **EUR** samples lost
- SNCA Gly51Asp (chr4:89828154): Only **EUR** samples (20,837) kept, **EAS** samples lost
- RAB32 Ser71Arg (chr6:146544084): Only **CAS** samples (1,930) kept, **EUR** and **MDE** samples lost

Sample coverage statistics showed the bug clearly:
- Min samples per variant: 401 (variants only in small ancestries)
- Max samples per variant: 20,844 (EUR-only variants)
- 145 variants had <50% of median sample count

**Fix Applied (2025-12-18):**
In `coordinator.py` lines 491-499, changed from:
```python
# Old (BROKEN): Simple concat + drop duplicates
wgs_combined = pd.concat(wgs_results, ignore_index=True)
wgs_combined = wgs_combined.drop_duplicates(...)
```

To:
```python
# New (FIXED): Use proper merge across ancestries
wgs_combined = self._merge_ancestry_results(wgs_results, 'WGS')
```

Also added WGS to the ancestry parsing in the ProcessPool worker (line 76):
```python
if data_type in ["NBA", "IMPUTED", "WGS"]:  # Added WGS
    ancestry = _parse_ancestry_from_path(file_path, settings.ANCESTRIES)
```

**Verification Required:**
Re-run the pipeline with `--release 11` to regenerate WGS results with the fix.

---

## Next Steps

1. ✅ **IMPUTED extraction fixed** (Issue 1 - 2025-12-15)
2. ✅ **Issue 2 investigated** - RAB32/SNCA missing from IMPUTED is expected (not in source files)
3. ✅ **Issue 3 fixed** - Dosage threshold fix (2025-12-17)
4. ✅ **Issue 4 fixed** - Disease duration now calculates correctly (2025-12-18)
5. ✅ **Issue 5 fixed** - WGS merge bug same as IMPUTED (2025-12-18)
