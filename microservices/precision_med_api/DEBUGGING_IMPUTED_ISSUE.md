# Debugging: IMPUTED Extraction - Missing GBA1 Genotypes

## Date: 2025-12-15 (RESOLVED)

## Problem Statement
The GBA1 variant `chr1:155235878:G:T` (snp_list_id: `c.1225-34C>A`) exists in IMPUTED source files but appears in the pipeline output with:
- `source_file = None`
- 0 non-NaN genotypes (all 103,786 samples are NaN)

## ✅ ROOT CAUSE FOUND AND FIXED (2025-12-15)

### The Bug: Duplicate Column Drop Without Combine

**Location:** `app/processing/coordinator.py`, function `_merge_ancestry_results()`, lines ~696-704

**Root Cause:** When merging DataFrames from different chromosome files (which share the same sample columns), pandas outer join creates duplicate columns with `_dup` suffix. The old code simply **dropped** these `_dup` columns, losing all genotype data for variants that only existed in the "right" DataFrame of each merge.

**Example of the bug:**
```
After pd.merge() with outer join:
variant_id          | sample_001 | sample_002 | sample_001_dup | sample_002_dup
--------------------|------------|------------|----------------|----------------
chr1:155235878:G:T  | 1.0        | 0.0        | NaN            | NaN      (from chr1 file)
chr22:50123456:A:G  | NaN        | NaN        | 0.0            | 1.0      (from chr22 file)

Old code did: merged_df.drop(columns=['sample_001_dup', 'sample_002_dup'])

Result: chr22 variant loses ALL genotype data!
```

### The Fix Applied

**File:** `app/processing/coordinator.py`, lines ~692-704

**Before (BROKEN):**
```python
# Drop any duplicate columns that might have been created
dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
if dup_cols:
    merged_df = merged_df.drop(columns=dup_cols)
```

**After (FIXED):**
```python
# Combine duplicate columns: take non-NaN value from either column
# This is necessary because different chromosome files have the same samples
# but different variants - an outer join creates _dup columns with NaN for
# variants that only exist in one file
dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
for dup_col in dup_cols:
    base_col = dup_col[:-4]  # Remove '_dup' suffix
    if base_col in merged_df.columns:
        # Combine: use base_col where not NaN, otherwise use dup_col
        merged_df[base_col] = merged_df[base_col].combine_first(merged_df[dup_col])
# Now drop the _dup columns
if dup_cols:
    merged_df = merged_df.drop(columns=dup_cols)
```

### Verification Results

**Before fix:**
- GBA1: 0 non-NaN samples after merge

**After fix:**
- GBA1: 16,185 non-NaN samples (AAC:1,382 + AFR:7,246 + AMR:3,816 + CAH:1,268 + MDE:2,473)
- Full pipeline: 192 variants, 103,786 samples, 56,156 carriers identified

---

## Previous Fix: `source_file` tracking bug (2025-12-11)

Added `'source_file'` to `metadata_cols` in `_merge_ancestry_results()` function. This was a secondary issue - the main data loss bug was the duplicate column drop.

---

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

## 🔧 PLANNED REFACTOR: Concat + Merge Approach

### Why Refactor?

The current `_merge_ancestry_results()` function uses merge for ALL DataFrames, but this is semantically incorrect and inefficient:

1. **Same ancestry, different chromosomes** → Should use `pd.concat` (stacking different variants)
2. **Different ancestries, same variants** → Should use `pd.merge` (adding sample columns)

The current approach works after the `combine_first` fix but does unnecessary work.

### Current Data Flow (Inefficient)

```
imputed_results = [
    chr1_AAC,   # 10 variants, 1382 samples
    chr22_AAC,  # 1 variant, 1382 samples (SAME samples as chr1_AAC!)
    chr1_AFR,   # 18 variants, 7246 samples
    chr22_AFR,  # 2 variants, 7246 samples (SAME samples as chr1_AFR!)
    ...
]

# Current: merge ALL 75 DataFrames one-by-one
# Each merge creates _dup columns for ALL sample columns, then combines them
# This is O(n²) work when it should be O(n)
```

### Proposed Data Flow (Efficient)

```
Step 1: Group by ancestry
ancestry_groups = {
    'AAC': [chr1_AAC, chr22_AAC, chr13_AAC, ...],  # Same samples
    'AFR': [chr1_AFR, chr22_AFR, chr13_AFR, ...],  # Same samples
    ...
}

Step 2: Concat within each ancestry (fast - just stacking rows)
ancestry_dfs = {
    'AAC': pd.concat([chr1_AAC, chr22_AAC, ...]),  # All AAC variants
    'AFR': pd.concat([chr1_AFR, chr22_AFR, ...]),  # All AFR variants
    ...
}

Step 3: Merge across ancestries (necessary - adding sample columns)
final = ancestry_dfs['AAC']
for ancestry in ['AFR', 'AMR', ...]:
    final = pd.merge(final, ancestry_dfs[ancestry], on=merge_keys, how='outer')
```

### Implementation Plan

**File:** `app/processing/coordinator.py`

**Function to modify:** `_merge_ancestry_results()` (lines ~625-714)

**Step-by-step changes:**

1. **Add ancestry grouping logic at the start:**
```python
def _merge_ancestry_results(self, result_dfs: List[pd.DataFrame], data_type: str) -> pd.DataFrame:
    if not result_dfs:
        return pd.DataFrame()

    if len(result_dfs) == 1:
        return result_dfs[0]

    # Step 1: Group DataFrames by ancestry (from source_file path)
    ancestry_groups = {}
    for df in result_dfs:
        # Extract ancestry from source_file or use 'unknown'
        if 'source_file' in df.columns and df['source_file'].notna().any():
            source = df['source_file'].iloc[0]
            ancestry = self._extract_ancestry_from_path(source)
        elif 'ancestry' in df.columns:
            ancestry = df['ancestry'].iloc[0]
        else:
            ancestry = 'unknown'

        if ancestry not in ancestry_groups:
            ancestry_groups[ancestry] = []
        ancestry_groups[ancestry].append(df)
```

2. **Concat within each ancestry group:**
```python
    # Step 2: Concat within each ancestry (same samples, different variants)
    ancestry_dfs = {}
    for ancestry, dfs in ancestry_groups.items():
        if len(dfs) == 1:
            ancestry_dfs[ancestry] = dfs[0]
        else:
            # Simple concat - same samples, just stacking variant rows
            combined = pd.concat(dfs, ignore_index=True)
            # Deduplicate variants within ancestry
            combined = combined.drop_duplicates(subset=merge_keys, keep='first')
            ancestry_dfs[ancestry] = combined
```

3. **Merge across ancestries:**
```python
    # Step 3: Merge across ancestries (different samples, may have overlapping variants)
    ancestry_list = list(ancestry_dfs.keys())
    merged_df = ancestry_dfs[ancestry_list[0]].copy()

    if 'ancestry' in merged_df.columns:
        merged_df = merged_df.drop(columns=['ancestry'])

    for ancestry in ancestry_list[1:]:
        df = ancestry_dfs[ancestry]
        df_sample_cols = [col for col in df.columns if col not in metadata_cols and col not in merge_keys and col != 'ancestry']
        cols_to_merge = merge_keys + df_sample_cols
        df_to_merge = df[cols_to_merge].copy()

        merged_df = pd.merge(merged_df, df_to_merge, on=merge_keys, how='outer', suffixes=('', '_dup'))

        # Combine duplicate columns (still needed for variants present in multiple ancestries)
        dup_cols = [col for col in merged_df.columns if col.endswith('_dup')]
        for dup_col in dup_cols:
            base_col = dup_col[:-4]
            if base_col in merged_df.columns:
                merged_df[base_col] = merged_df[base_col].combine_first(merged_df[dup_col])
        if dup_cols:
            merged_df = merged_df.drop(columns=dup_cols)

    return merged_df
```

4. **Add helper function:**
```python
def _extract_ancestry_from_path(self, path: str) -> str:
    """Extract ancestry code from file path like /path/AAC/chr1_AAC_release11.pgen"""
    if not path:
        return 'unknown'
    for ancestry in self.settings.ANCESTRIES:
        if f'/{ancestry}/' in path or f'_{ancestry}_' in path:
            return ancestry
    return 'unknown'
```

### Benefits of Refactor

1. **Performance:** O(n) concat + O(ancestries) merges instead of O(n²) merges
2. **Clarity:** Code matches the actual data semantics
3. **Fewer edge cases:** Less reliance on `combine_first` to fix incorrect merges

### Testing After Refactor

```bash
# Quick test with 2 ancestries
python run_carriers_pipeline.py --job-name refactor_test --data-types IMPUTED --ancestries AAC AFR --release 11

# Verify GBA1 has correct sample count
python -c "
import pandas as pd
df = pd.read_parquet('path/to/refactor_test_IMPUTED.parquet')
gba1 = df[df['variant_id'] == 'chr1:155235878:G:T']
print(f'GBA1 non-NaN samples: {gba1.iloc[:, 10:].notna().sum().sum()}')
# Should be AAC(1382) + AFR(7246) = 8628
"
```

---

## Historical Hypotheses (RESOLVED)

These hypotheses were investigated before finding the root cause:

### Hypothesis 1: Data Loss in ProcessPool Worker (NOT the issue)
The extraction runs correctly in ProcessPool - data is returned properly.

### Hypothesis 2: Data Loss in Result Combining (THIS WAS IT!)
The merge call `imputed_combined = self._merge_ancestry_results(imputed_results, 'IMPUTED')` was dropping `_dup` columns without combining them first.

### Hypothesis 3: Empty DataFrame Placeholder Issue (NOT the issue)
No placeholder rows were being created.

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
