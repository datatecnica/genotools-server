# Pipeline Refactor Notes (2025-12-23)

## Summary

Refactored the imputation harmonizer to be a single-command pipeline that:
1. Runs the entire workflow (frequency generation → QC → PLINK corrections → cleanup)
2. Outputs per-chromosome files (VCF.GZ or PLINK1/PLINK2 format)
3. Generates a comprehensive JSON report with full variant-level details
4. Cleans up intermediate files automatically

## Changes Made

### 1. New Files

#### `imputation_harmonizer/writers/report.py`
- `VariantReport` dataclass: Stores per-variant decision details
- `ReportWriter` class: Collects variant results and writes JSON report
- JSON schema includes: metadata, statistics, full variant list, output files

### 2. Modified Files

#### `imputation_harmonizer/config.py`
Added new configuration fields:
```python
generate_report: bool = True
report_file: Path | None = None  # Default: {output_dir}/{stem}-report.json
```

#### `imputation_harmonizer/cli.py`
- Removed `--run-pipeline` flag (pipeline now runs directly by default)
- Added `--report-file` option for custom report path
- Added `--no-report` flag to skip JSON report generation (saves memory for large datasets)

#### `imputation_harmonizer/checks/comparator.py`
- Changed `check_variants()` return type from `Iterator[CheckResult]` to `Iterator[tuple[BimVariant, CheckResult]]`
- This allows the report writer to capture chr/pos information for each variant

#### `imputation_harmonizer/main.py`
- Integrated `ReportWriter` into the main processing loop
- Collects results during variant checking for JSON report
- Writes report after pipeline completes

#### `imputation_harmonizer/plink_runner.py`
Major refactoring:

1. **Added PLINK 1.9 support**:
   - `find_plink1()`: Locates PLINK 1.9 executable
   - `_run_plink1()`: Executes PLINK 1.9 commands
   - Required because PLINK2 doesn't support `--flip`

2. **Fixed `split_by_chromosome()`**:
   - Now reads BIM file to detect which chromosomes exist
   - Extracts each chromosome individually using `--chr`
   - Fixed path handling: uses `parent / f"{name}.pvar"` instead of `with_suffix()` to avoid replacing `.chr22` with `.pvar`

3. **Simplified `run_parallel_pipeline()`**:
   - Uses correction files already created by `check_variants()` in main.py
   - No longer re-runs variant checking per chromosome (was causing double-processing bug)
   - Pipeline now:
     1. Applies exclusions (PLINK2)
     2. Applies strand flips (PLINK 1.9 via bed conversion)
     3. Applies force-allele (PLINK2)
     4. Exports per-chromosome files

#### `imputation_harmonizer/writers/__init__.py`
- Added export for `ReportWriter`

### 3. Deleted Files

#### `imputation_harmonizer/writers/shell_script.py`
- No longer needed since pipeline runs directly

### 4. Test Updates

#### `tests/test_report_writer.py` (NEW)
- Unit tests for ReportWriter functionality
- Tests: add results, write JSON, atomic writes, metadata, statistics

#### `tests/test_integration.py`
- Updated to use new `check_variants()` return type
- Removed shell script tests
- Created `run_variant_check()` helper for testing without PLINK2

#### `tests/test_plink_runner.py`
- Updated split tests to create mock BIM files
- Added test for empty BIM returning empty chromosome list
- Added test for skipping non-autosome chromosomes

## Validation Results

Compared Python output against Perl script output on chr22 test data:

| Metric | Python | Perl | Match? |
|--------|--------|------|--------|
| Variant count | 20,416 | 20,416 | ✓ |
| Positions | All match | - | ✓ |
| Variant IDs | All match | - | ✓ |
| Alleles | Same alleles | Same alleles | ✓ (order differs) |

### Allele Order Difference

**Issue**: The A1/A2 columns are swapped between Python and Perl outputs.

```
# Python output (first 5 lines of BIM):
22  JHU_22.16650300  0  15327662  T  C
22  rs150338766      0  15479124  A  G

# Perl output (same variants):
22  JHU_22.16650300  0  15327662  C  T
22  rs150338766      0  15479124  G  A
```

**Cause**: Different PLINK commands handle allele ordering differently:
- Perl uses PLINK 1.9's `--a1-allele`
- Python uses PLINK 2's `--ref-allele force`

**Impact**: The genotypes are semantically equivalent. PLINK BIM files don't have a canonical REF/ALT definition - A1/A2 are just "allele 1" and "allele 2". The genotype encoding in the BED file is consistent with the BIM.

**TODO**: To get exact byte-for-byte match, we need to swap alleles. Options:
1. Use PLINK 1.9 for the entire pipeline instead of PLINK2
2. Add a post-processing step to swap A1↔A2 in the BIM file
3. Use PLINK2's `--a1-allele` instead of `--ref-allele force`

## Output Structure

After running the pipeline:
```
output_dir/
├── {stem}-chr1.vcf.gz (or .bed/.bim/.fam for plink1)
├── {stem}-chr2.vcf.gz
├── ...
├── {stem}-chr22.vcf.gz
├── {stem}-report.json
├── Exclude-{stem}-{panel}.txt
├── Strand-Flip-{stem}-{panel}.txt
├── Force-Allele1-{stem}-{panel}.txt
├── FreqPlot-{stem}-{panel}.txt
├── ID-{stem}-{panel}.txt
├── Position-{stem}-{panel}.txt
├── Chromosome-{stem}-{panel}.txt
└── LOG-{stem}-{panel}.txt
```

## JSON Report Schema

```json
{
  "metadata": {
    "version": "1.0.0",
    "tool": "imputation-harmonizer",
    "timestamp": "2025-12-23T...",
    "duration_seconds": 19.8,
    "input_files": {"bim_file": "...", "frq_file": "...", "ref_file": "..."},
    "reference_panel": "TOPMed",
    "population": "ALL",
    "thresholds": {"freq_diff": 0.2, "palindrome_maf": 0.4},
    "options": {"include_x": false, "output_format": "plink1"}
  },
  "statistics": {
    "total_variants": 28084,
    "excluded": {
      "total": 7668,
      "indels": 602,
      "alt_chromosome": 0,
      "not_in_reference": 3121,
      "palindromic_high_maf": 102,
      "freq_diff_too_high": 1306,
      "allele_mismatch": 1542,
      "duplicate": 995
    },
    "strand_flipped": 9885,
    "strand_ok": 10531,
    "ref_alt_swapped": 18401,
    "ref_alt_ok": 2015,
    "matching": {...}
  },
  "variants": [
    {
      "id": "rs123",
      "chr": "22",
      "pos": 12345,
      "action": "keep",
      "exclude_reason": null,
      "strand_flip": false,
      "force_ref": "A",
      "ref_freq": 0.30,
      "bim_freq": 0.28,
      "freq_diff": 0.02,
      "check_code": 1,
      "matched_by": "position"
    },
    ...
  ],
  "output_files": ["GP2_chr22-chr22.bed"]
}
```

## Commands Used for Testing

```bash
# Run pipeline with PLINK1 output
python -m imputation_harmonizer \
    --bim data/GP2_chr22.bim \
    --ref data/TOPMed_chr22.tab \
    --panel topmed \
    --output-format plink1 \
    --output-dir output_test \
    -v

# Compare outputs
wc -l output_test/GP2_chr22-chr22.bim output_perl/GP2_chr22-updated-chr22.bim
# Both: 20,416 lines

# Compare variant IDs (should be identical)
cut -f2 output_test/GP2_chr22-chr22.bim | sort > /tmp/new_ids.txt
cut -f2 output_perl/GP2_chr22-updated-chr22.bim | sort > /tmp/perl_ids.txt
diff /tmp/new_ids.txt /tmp/perl_ids.txt
# No difference

# Compare with sorted alleles (semantically equivalent)
awk '{if($5<$6) print $1,$2,$3,$4,$5,$6; else print $1,$2,$3,$4,$6,$5}' output_test/GP2_chr22-chr22.bim | sort > /tmp/new_sorted.txt
awk '{if($5<$6) print $1,$2,$3,$4,$5,$6; else print $1,$2,$3,$4,$6,$5}' output_perl/GP2_chr22-updated-chr22.bim | sort > /tmp/perl_sorted.txt
diff /tmp/new_sorted.txt /tmp/perl_sorted.txt
# No difference
```

## Next Steps

1. **Fix allele ordering** to get exact match with Perl output:
   - Investigate using `--a1-allele` instead of `--ref-allele force`
   - Or add post-processing to swap alleles in BIM

2. **Memory optimization** for large datasets:
   - Currently stores all 28K+ variant reports in memory
   - For 40M+ variants, consider streaming to JSON

3. **Parallel chromosome processing**:
   - Current implementation is sequential
   - Could parallelize the final export step

## Test Data Locations

- Input BIM: `data/GP2_chr22.bim` (28,084 variants)
- Reference: `data/TOPMed_chr22.tab` (5.8M variants)
- Perl output: `output_perl/GP2_chr22-updated-chr22.bim`
- Python output: `output_test/GP2_chr22-chr22.bim`
