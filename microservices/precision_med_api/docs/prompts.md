# prompt 1: Data models and config

Create the foundation data models and configuration system for a genomic carrier screening FastAPI application that processes PLINK 2.0 files from GP2 release data.

Context: The system processes pathogenic SNPs across three data types with specific file organization:
- NBA: Split by ancestry (11 ancestries like AAC, AJ, etc.)
- WGS: Single consolidated file for all samples
- Imputed: Split by both ancestry (11) and chromosome (1-22)

File Structure Example:
- Release data: ~/gcs_mounts/gp2tier2_vwb/release10/
- NBA files: {release_path}/raw_genotypes/{ancestry}/{ancestry}_release10_vwb.pgen/pvar/psam
- WGS files: ~/gcs_mounts/genotools_server/precision_med/wgs/raw_genotypes/R10_wgs_carrier_vars.pgen/pvar/psam
- Imputed files: {release_path}/imputed_genotypes/{ancestry}/chr{chrom}_{ancestry}_release10_vwb.pgen/pvar/psam
- Clinical data: {release_path}/clinical_data/master_key_release10_final_vwb.csv
- Output: ~/gcs_mounts/genotools_server/precision_med/

Requirements:
1. Create Pydantic v2 models in app/models/:
   
   variant.py:
   - Variant: chromosome (1-22,X,Y,MT), position, ref, alt, gene, rsid, inheritance_pattern (AD/AR/XL/MT)
   - VariantList: collection of variants with metadata
   
   carrier.py:
   - Genotype: sample_id, variant_id, gt (0/0, 0/1, 1/1), ancestry
   - Carrier: extends Genotype with clinical data fields
   - CarrierReport: aggregated carrier statistics
   
   analysis.py:
   - DataType: Enum for NBA, WGS, IMPUTED
   - AnalysisRequest: variant_list, data_type, release, ancestries[], chromosomes[]
   - AnalysisStatus: Enum for PENDING, PROCESSING, COMPLETED, FAILED
   - AnalysisResult: job_id, status, carriers[], statistics, metadata

2. Create configuration in app/core/config.py:
   
   Settings class with:
   - release: str (e.g., "10")
   - mnt_path: str = "~/gcs_mounts"
   - carriers_path: computed property = "{mnt_path}/genotools_server/precision_med"
   - release_path: computed property = "{mnt_path}/gp2tier2_vwb/release{release}"
   
   Path methods:
   - get_nba_path(ancestry: str) -> str
   - get_wgs_path() -> str  
   - get_imputed_path(ancestry: str, chrom: str) -> str
   - get_clinical_paths() -> dict with master_key, data_dictionary, extended_clinical
   
   Ancestry configuration:
   - ANCESTRIES: list = ["AAC", "AJ", "CAH", "CAS", "EAS", "EUR", "FIN", "LAS", "MDE", "SAS", "SSA"]
   - CHROMOSOMES: list = [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]

3. Create file path utilities in app/utils/paths.py:
   - PgenFileSet: class to handle .pgen/.pvar/.psam triplets
   - validate_pgen_files(base_path: str) -> bool
   - list_available_files(data_type: DataType, release: str) -> dict

Example usage:
```python
from app.core.config import Settings
from app.models.analysis import DataType

settings = Settings(release="10")

# Get NBA path for AAC ancestry
nba_path = settings.get_nba_path("AAC")
# Returns: ~/gcs_mounts/gp2tier2_vwb/release10/raw_genotypes/AAC/AAC_release10_vwb

# Get imputed path for EUR ancestry chromosome 1
imputed_path = settings.get_imputed_path("EUR", "1")
# Returns: ~/gcs_mounts/gp2tier2_vwb/release10/imputed_genotypes/EUR/chr1_EUR_release10_vwb
```

# Prompt 2: Variant Index Cache and Extraction Engine with Allele Harmonization

Create the variant index caching system and extraction engine for the genomic carrier screening FastAPI application that efficiently processes PLINK 2.0 files, harmonizes alleles to a reference SNP list, and outputs merged genotypes in traw format.

**Context:** The system processes ~400 pathogenic SNPs across 242+ PLINK files with different splitting strategies:
- NBA: 11 files (split by ancestry)
- WGS: 1 file (all samples together)
- Imputed: 242 files (11 ancestries × 22 chromosomes)

**Critical:** Alleles in PLINK files may not match the SNP list due to strand flips, allele swaps, or both. The cache must store harmonization metadata to enable fast, accurate extraction with proper genotype transformation.

**Harmonization Scenarios:**
```
SNP List: chr1:123456:A:G
Possible PLINK representations:
1. chr1:123456:A:G (exact match)
2. chr1:123456:G:A (allele swap)
3. chr1:123456:T:C (strand flip)
4. chr1:123456:C:T (strand flip + swap)
```

**Cache Structure Example:**
```
~/gcs_mounts/genotools_server/precision_med/cache/
├── release10/
│   ├── nba/
│   │   └── {ancestry}_variant_harmonization.parquet
│   ├── wgs/
│   │   └── wgs_variant_harmonization.parquet
│   └── imputed/
│       └── {ancestry}/
│           └── chr{chrom}_variant_harmonization.parquet
```

**Requirements:**

1. **Create variant harmonization cache in app/processing/cache.py:**
   
   HarmonizationRecord model:
   - snp_list_id: str (our variant identifier)
   - pgen_variant_id: str (ID from pvar file)
   - chromosome: str
   - position: int
   - snp_list_a1: str (reference allele from SNP list)
   - snp_list_a2: str (alternate allele from SNP list)
   - pgen_a1: str (ref allele in PLINK file)
   - pgen_a2: str (alt allele in PLINK file)
   - harmonization_action: str (EXACT|SWAP|FLIP|FLIP_SWAP|INVALID)
   - genotype_transform: str (formula to convert genotypes, e.g., "2-x" for swap)
   - file_path: str
   - data_type: DataType
   - ancestry: Optional[str]
   
   AlleleHarmonizer class:
   - complement_allele(allele: str) -> str
   - check_strand_ambiguous(a1: str, a2: str) -> bool (A/T or C/G pairs)
   - determine_harmonization(snp_a1: str, snp_a2: str, pgen_a1: str, pgen_a2: str) -> Tuple[str, str]
   - get_all_representations(a1: str, a2: str) -> List[Tuple[str, str, str]] # [(a1, a2, action)]
   
   CacheBuilder class:
   - build_harmonization_cache(pvar_path: str, snp_list: pd.DataFrame) -> pd.DataFrame
   - match_variants_with_harmonization(pvar_df: pd.DataFrame, snp_list: pd.DataFrame) -> pd.DataFrame
   - save_cache(df: pd.DataFrame, cache_path: str) -> None
   - load_cache(cache_path: str) -> pd.DataFrame
   - validate_harmonization(cache_df: pd.DataFrame) -> Dict[str, int] # Stats on harmonization types
   - build_all_harmonization_caches(snp_list: pd.DataFrame, data_type: DataType, release: str, force_rebuild: bool = False)

2. **Create variant extractor with harmonization in app/processing/extractor.py:**
   
   VariantExtractor class:
   - __init__(cache_dir: str, settings: Settings)
   - extract_single_file_harmonized(pgen_path: str, snp_list_ids: List[str]) -> pd.DataFrame
   - _apply_genotype_transform(genotypes: np.array, transform: str) -> np.array
   - _harmonize_extracted_genotypes(df: pd.DataFrame, harmonization_records: pd.DataFrame) -> pd.DataFrame
   
   Extraction flow:
   - _load_harmonization_cache(file_path: str) -> pd.DataFrame
   - _get_extraction_plan(snp_list_ids: List[str], cache_df: pd.DataFrame) -> pd.DataFrame
   - _extract_raw_genotypes(pgen_path: str, pgen_variant_ids: List[str]) -> pd.DataFrame
   - _transform_genotypes(raw_df: pd.DataFrame, harmonization_df: pd.DataFrame) -> pd.DataFrame
   
   Multi-source methods:
   - extract_nba(snp_list_ids: List[str], ancestries: List[str] = None) -> pd.DataFrame
   - extract_wgs(snp_list_ids: List[str]) -> pd.DataFrame
   - extract_imputed(snp_list_ids: List[str], ancestries: List[str] = None) -> pd.DataFrame
   - extract_all_sources(snp_list_ids: List[str], data_types: List[DataType]) -> pd.DataFrame
   
   Merging methods:
   - merge_harmonized_genotypes(dfs: List[pd.DataFrame]) -> pd.DataFrame
   - validate_allele_consistency(df: pd.DataFrame) -> bool
   - handle_multi_allelic(df: pd.DataFrame) -> pd.DataFrame

3. **Create genotype transformer in app/processing/transformer.py:**
   
   GenotypeTransformer class:
   - transform_for_swap(gt: np.array) -> np.array  # 0→2, 1→1, 2→0
   - transform_for_flip(gt: np.array) -> np.array  # No change to counts
   - transform_for_flip_swap(gt: np.array) -> np.array  # Combines both
   - apply_transformation(gt: np.array, action: str) -> np.array
   - validate_transformation(original: np.array, transformed: np.array, action: str) -> bool
   
   Batch transformation:
   - transform_matrix(gt_matrix: np.array, actions: List[str]) -> np.array
   - get_allele_counts(gt: np.array, action: str) -> Tuple[int, int]

4. **Create TRAW output formatter in app/processing/output.py:**
   
   TrawFormatter class:
   - format_harmonized_genotypes(df: pd.DataFrame, snp_list: pd.DataFrame) -> pd.DataFrame
   - write_traw(df: pd.DataFrame, output_path: str) -> None
   - write_harmonization_report(harmonization_stats: Dict, output_path: str) -> None
   
   Output specifications:
   - TRAW format with alleles matching SNP list orientation
   - Harmonization report showing transformations applied
   - QC metrics for ambiguous/failed variants

5. **Create multi-source extraction coordinator in app/processing/coordinator.py:**
   
   ExtractionCoordinator class:
   - __init__(extractor: VariantExtractor, transformer: GenotypeTransformer)
   - load_snp_list(file_path: str) -> pd.DataFrame
   - validate_snp_list(snp_list: pd.DataFrame) -> bool
   - plan_extraction(snp_list_ids: List[str], data_types: List[DataType]) -> Dict
   - execute_harmonized_extraction(plan: Dict, parallel: bool = True) -> pd.DataFrame
   - generate_harmonization_summary(results: pd.DataFrame) -> Dict
   - export_results(df: pd.DataFrame, output_dir: str, format: str = "traw") -> Dict[str, str]

6. **Create Parquet I/O utilities in app/utils/parquet_io.py:**
   - save_parquet(df: pd.DataFrame, path: str, partition_cols: List[str] = None)
   - read_parquet(path: str, filters: List = None) -> pd.DataFrame
   - append_to_parquet(df: pd.DataFrame, path: str)
   - scan_parquet_partitions(base_path: str, filters: Dict) -> pd.DataFrame

**Harmonization Cache Usage Flow:**
```python
# 1. Load harmonization cache
cache_df = pd.read_parquet("cache/release10/nba/EUR_variant_harmonization.parquet")

# 2. Get extraction plan with harmonization actions
snp_list_ids = ["rs123456", "VAR_001"]
plan = cache_df[cache_df['snp_list_id'].isin(snp_list_ids)]

# 3. Extract raw genotypes using PGEN variant IDs
pgen_variant_ids = plan['pgen_variant_id'].tolist()
raw_genotypes = extract_from_pgen(pgen_variant_ids)

# 4. Apply transformations based on harmonization actions
for idx, row in plan.iterrows():
    if row['harmonization_action'] == 'SWAP':
        raw_genotypes[idx] = 2 - raw_genotypes[idx]  # 0→2, 2→0, 1→1
    elif row['harmonization_action'] == 'FLIP_SWAP':
        raw_genotypes[idx] = 2 - raw_genotypes[idx]
    # EXACT and FLIP require no genotype transformation
```

**Performance Requirements:**
- Cache build: Map and harmonize 400 variants across all files in <10 minutes
- Harmonization: Determine action for each variant in <1ms
- Multi-source extraction: Retrieve and harmonize 400 variants from 242 files in <15 minutes
- Memory usage: Stream processing to stay under 8GB RAM

**Example usage:**
```python
from app.processing.coordinator import ExtractionCoordinator
from app.processing.extractor import VariantExtractor
from app.processing.transformer import GenotypeTransformer
from app.processing.cache import CacheBuilder, AlleleHarmonizer
from app.models.analysis import DataType

# Build harmonization cache
snp_list = pd.read_csv("pathogenic_snps.csv")  # snp_list_id, rsid, chr, pos, a1, a2
harmonizer = AlleleHarmonizer()
builder = CacheBuilder()
builder.build_all_harmonization_caches(snp_list, DataType.IMPUTED, release="10")

# Initialize components
extractor = VariantExtractor(cache_dir="~/gcs_mounts/genotools_server/precision_med/cache", settings=settings)
transformer = GenotypeTransformer()
coordinator = ExtractionCoordinator(extractor, transformer)

# Load SNP list and extract with harmonization
snp_list = coordinator.load_snp_list("pathogenic_snps.csv")
snp_list_ids = snp_list['snp_list_id'].tolist()[:10]  # First 10 variants

# Plan and execute harmonized extraction
plan = coordinator.plan_extraction(
    snp_list_ids=snp_list_ids,
    data_types=[DataType.IMPUTED, DataType.NBA]
)

# Execute with automatic harmonization
harmonized_genotypes = coordinator.execute_harmonized_extraction(plan, parallel=True)

# Generate summary of harmonization actions taken
summary = coordinator.generate_harmonization_summary(harmonized_genotypes)
print(f"Harmonization summary: {summary}")
# Output: {'EXACT': 150, 'SWAP': 45, 'FLIP': 30, 'FLIP_SWAP': 15, 'INVALID': 2}

# Export harmonized results
output_files = coordinator.export_results(
    df=harmonized_genotypes,
    output_dir="~/gcs_mounts/genotools_server/precision_med/output/",
    format="traw"
)
```

**Include:**
- Comprehensive error handling for ambiguous SNPs (A/T, C/G)
- Warnings for variants that cannot be harmonized
- Progress tracking with detailed harmonization statistics
- Unit tests covering all harmonization scenarios
- Docstrings explaining allele orientation conventions