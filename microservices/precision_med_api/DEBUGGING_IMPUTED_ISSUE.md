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

## ✅ COMPLETED REFACTOR: Concat + Merge Approach (2025-12-15)

### Why This Refactor Was Needed

The previous `_merge_ancestry_results()` function used merge for ALL DataFrames, which was semantically incorrect and inefficient:

1. **Same ancestry, different chromosomes** → Should use `pd.concat` (stacking different variants)
2. **Different ancestries, same variants** → Should use `pd.merge` (adding sample columns)

The old approach worked after the `combine_first` fix but did unnecessary work.

### New Data Flow (Implemented)

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

### Benefits of Refactor

1. **Performance:** O(n) concat + O(ancestries) merges instead of O(n²) merges
2. **Clarity:** Code matches the actual data semantics
3. **Fewer edge cases:** Less reliance on `combine_first` to fix incorrect merges

---

## ✅ DOSAGE THRESHOLD FIX (2025-12-17)

### Problem
For IMPUTED data, carrier counts showed `Total Carriers != Heterozygous + Homozygous`:
- Example: c.1225-34C>A showed carrier=13,636, het=1,632, hom=402, but het+hom=2,034

### Root Cause
IMPUTED data contains **dosage values** (0.0-2.0) not discrete genotypes (0, 1, 2).
The old counting logic:
```python
carrier_count = (genotypes > 0).sum()    # Counted 0.01, 0.1, etc. as carriers
het_count = (genotypes == 1).sum()       # Only exact 1.0
hom_count = (genotypes == 2).sum()       # Only exact 2.0
```

### Fix Applied
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

### CLI Support
```bash
# Default soft calls (0.5, 1.5, 1.5)
python run_carriers_pipeline.py --job-name release11 --release 11

# Hard calls (stricter thresholds)
python run_carriers_pipeline.py --job-name release11 --release 11 \
    --dosage-het-min 0.9 --dosage-het-max 1.1 --dosage-hom-min 1.9
```

### Verification
- Before: carrier=13,636, het+hom=2,034 (MISMATCH)
- After: carrier=3,777, het=3,212, hom=565, het+hom=3,777 (MATCH ✓)

---

## Files Modified

- `app/processing/coordinator.py` - Concat+merge refactor, combine_first fix
- `app/core/config.py` - Dosage threshold settings
- `app/processing/locus_report_generator.py` - Threshold-based carrier counting
- `frontend/app/utils/data_loaders.py` - Threshold-based counting (frontend)
- `run_carriers_pipeline.py` - CLI args for dosage thresholds

## Related Files
- `app/processing/extractor.py` - Single file extraction
- `app/processing/harmonizer.py` - Variant harmonization

## Related Issues
- See `FEEDBACK_ISSUES.md` for original issue tracking
