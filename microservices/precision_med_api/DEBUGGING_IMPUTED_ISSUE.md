# Debugging: IMPUTED Extraction - Missing GBA1 Genotypes

## Date: 2025-12-11

## Problem Statement
The GBA1 variant `chr1:155235878:G:T` (snp_list_id: `c.1225-34C>A`) exists in IMPUTED source files but appears in the pipeline output with:
- `source_file = None`
- 0 non-NaN genotypes (all 103,786 samples are NaN)

## What Was Fixed

### Fix Applied: `source_file` tracking bug
**File:** `app/processing/coordinator.py` line 661-663

Added `'source_file'` to `metadata_cols` in `_merge_ancestry_results()` function.

**Before:**
```python
metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                 'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                 'pgen_a1', 'pgen_a2', 'data_type', 'maf_corrected', 'original_alt_af']
```

**After:**
```python
metadata_cols = ['chromosome', 'variant_id', '(C)M', 'position', 'COUNTED', 'ALT',
                 'counted_allele', 'alt_allele', 'harmonization_action', 'snp_list_id',
                 'pgen_a1', 'pgen_a2', 'data_type', 'source_file',
                 'maf_corrected', 'original_alt_af']
```

**Impact:** This fix prevents `source_file` from being incorrectly classified as a "sample column" during ancestry merge, which would cause the correct value to be lost when the `_dup` suffix columns are dropped.

**NOTE:** This fix does NOT address the genotype data loss issue - it only fixes metadata tracking.

---

## Debugging Flow

### Step 1: Confirmed Variant Exists in Source
```
PVAR file: /home/vitaled2/gcs_mounts/gp2tier2_vwb/release11/imputed_genotypes/AAC/chr1_AAC_release11_vwb.pvar
Position 155235878 has variant: chr1:155235878:G:T (REF=G, ALT=T) ✓
```

### Step 2: Confirmed SNP List Entry
```
snp_name: c.1225-34C>A
hg38: 1:155235878:G:T
chromosome: 1, position: 155235878, ref: G, alt: T ✓
```

### Step 3: Tested Direct Extraction (WORKS)
```python
# Testing extraction for chr1 variants only
aac_result = extractor.extract_single_file_harmonized(aac_pgen, snp_list_ids, chr1_snp)
# Result: 10 variants extracted, including GBA1 with actual genotype data!
```

**Key Finding:** Extraction WORKS when run directly with the extractor.

### Step 4: Tested Ancestry Merge (WORKS)
```python
# Simulating ancestry merge with AAC + EUR
merged = coordinator._merge_ancestry_results([aac_result, eur_result], 'IMPUTED')
# Result: GBA1 is in merged with sample values preserved!
```

**Key Finding:** Merge WORKS when run directly.

### Step 5: Ran Full Pipeline with Verbose Logging
```bash
python run_carriers_pipeline.py --job-name release11_debug --data-types IMPUTED --release 11
```

**Pipeline Log Shows Successful Extraction:**
```
Extracted and harmonized 10 variants from chr1_AAC_release11_vwb.pgen
Extracted and harmonized 18 variants from chr1_AFR_release11_vwb.pgen
Extracted and harmonized 16 variants from chr1_AMR_release11_vwb.pgen
... (successful extractions for all 11 ancestries)
```

### Step 6: Checked Pipeline Output (STILL BROKEN)
```python
# After pipeline completes
df = pd.read_parquet("release11_debug_IMPUTED.parquet")
gba1 = df[df['variant_id'] == 'chr1:155235878:G:T']
# Result: source_file=None, Non-NaN genotypes=0 !!!
```

---

## Current State

| What | Status | Notes |
|------|--------|-------|
| Variant exists in source | ✓ | Confirmed in PVAR file |
| SNP list entry correct | ✓ | Correct format and values |
| Direct extraction works | ✓ | Returns 10 variants including GBA1 |
| Direct merge works | ✓ | Preserves genotype data |
| Pipeline logs show success | ✓ | "Extracted and harmonized 10 variants from chr1_AAC" |
| Pipeline output has data | ✗ | 0 genotypes, source_file=None |

---

## Hypotheses for Remaining Issue

### Hypothesis 1: Data Loss in ProcessPool Worker
The extraction runs in a ProcessPoolExecutor. The worker function returns a DataFrame, but something might be happening during:
- Serialization/deserialization between processes
- The result collection logic in `_execute_with_process_pool()`

**Evidence:** Logs show extraction succeeds (worker-side), but output has no data (main process side).

### Hypothesis 2: Data Loss in Result Combining
After ProcessPool, results are combined in `_execute_with_process_pool()`:
```python
all_results.append(result_df)  # Line 430
```
Then grouped by data type and merged. Something might be losing data during:
- The list comprehension filtering: `imputed_results = [df for df in all_results if (df['data_type'] == 'IMPUTED').all()]`
- The merge call: `imputed_combined = self._merge_ancestry_results(imputed_results, 'IMPUTED')`

### Hypothesis 3: Empty DataFrame Placeholder Issue
There might be a code path that creates placeholder rows with just variant metadata but no genotype data, which then overwrites the real extracted data during merge.

---

## Next Steps to Investigate

1. **Add Debugging to ProcessPool Results Collection**
   - Log the number of columns and non-NaN values in each DataFrame that gets appended to `all_results`
   - Check if GBA1 is in `all_results` before merge

2. **Add Debugging to Merge**
   - Log the state of GBA1 before and after `_merge_ancestry_results()`
   - Check column names and values at each step

3. **Check Result Export**
   - Log what `imputed_combined` looks like before it's written to parquet

4. **Try Serial Execution**
   - Run without ProcessPool to see if the issue is parallelization-related

---

## Files Modified
- `app/processing/coordinator.py` - Added `source_file` to `metadata_cols`

## Related Files
- `app/processing/extractor.py` - Single file extraction
- `app/processing/harmonizer.py` - Variant harmonization
- `run_carriers_pipeline.py` - CLI entry point

## Related Issues
- See `FEEDBACK_ISSUES.md` for original issue tracking
