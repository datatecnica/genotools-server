# prompt 1: Data models and config

Create the foundation data models and configuration system for a genomic carrier screening FastAPI application that processes PLINK 2.0 files from GP2 release data.

Context: The system processes pathogenic SNPs across three data types with specific file organization:
- NBA: Split by ancestry (11 ancestries like AAC, AJ, etc.)
- WGS: Single consolidated file for all samples
- Imputed: Split by both ancestry (11) and chromosome (1-22)

File Structure Example:
- Release data: ~/gcs_mounts/gp2tier2_vwb/release10/
- NBA files: {release_path}/raw_genotypes/{ancestry}/{ancestry}_release10_vwb.pgen/pvar/psam
- WGS files: ~/gcs_mounts/genotools_server/carriers/wgs/raw_genotypes/R10_wgs_carrier_vars.pgen/pvar/psam
- Imputed files: {release_path}/imputed_genotypes/{ancestry}/chr{chrom}_{ancestry}_release10_vwb.pgen/pvar/psam
- Clinical data: {release_path}/clinical_data/master_key_release10_final_vwb.csv
- Output: ~/gcs_mounts/genotools_server/carriers/

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
   - carriers_path: computed property = "{mnt_path}/genotools_server/carriers"
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

# Prompt 2: Variant Index Cache and Extraction Engine

Create the variant index caching system and extraction engine for the genomic carrier screening FastAPI application that efficiently processes PLINK 2.0 files across multiple data splits and outputs merged genotypes in traw format.

**Context:** The system processes ~400 pathogenic SNPs across 242+ PLINK files with different splitting strategies:
- NBA: 11 files (split by ancestry)
- WGS: 1 file (all samples together)
- Imputed: 242 files (11 ancestries × 22 chromosomes)
The engine must extract variants from all relevant files, merge results, and output in PLINK traw format for downstream analysis.

**Cache Structure Example:**
```
~/gcs_mounts/genotools_server/carriers/cache/
├── release10/
│   ├── nba/
│   │   └── {ancestry}_variant_index.parquet
│   ├── wgs/
│   │   └── wgs_variant_index.parquet
│   └── imputed/
│       └── {ancestry}/
│           └── chr{chrom}_variant_index.parquet
```

**Requirements:**

1. **Create variant cache builder in app/processing/cache.py:**
   
   VariantIndex model:
   - variant_id: str (chr:pos:ref:alt format)
   - rsid: Optional[str]
   - chromosome: str
   - position: int
   - ref: str
   - alt: str
   - row_index: int (position in pvar file)
   - file_path: str
   - data_type: DataType
   - ancestry: Optional[str]
   
   CacheBuilder class:
   - build_index(pvar_path: str) -> pd.DataFrame
   - save_cache(df: pd.DataFrame, cache_path: str) -> None
   - load_cache(cache_path: str) -> pd.DataFrame
   - validate_cache(cache_path: str, pvar_path: str) -> bool
   - build_all_indexes(data_type: DataType, release: str, force_rebuild: bool = False)
   - get_variant_file_map(variant_ids: List[str], data_type: DataType) -> Dict[str, List[str]]

2. **Create variant extractor in app/processing/extractor.py:**
   
   VariantExtractor class:
   - __init__(cache_dir: str, settings: Settings)
   - extract_single_file(pgen_path: str, variant_ids: List[str]) -> pd.DataFrame
   - extract_nba(variant_ids: List[str], ancestries: List[str] = None) -> pd.DataFrame
   - extract_wgs(variant_ids: List[str]) -> pd.DataFrame
   - extract_imputed(variant_ids: List[str], ancestries: List[str] = None) -> pd.DataFrame
   - extract_all_sources(variant_ids: List[str], data_types: List[DataType]) -> pd.DataFrame
   
   Merging methods:
   - merge_genotypes(dfs: List[pd.DataFrame], merge_strategy: str = "union") -> pd.DataFrame
   - handle_duplicate_samples(df: pd.DataFrame, priority: Dict[DataType, int]) -> pd.DataFrame
   - align_variants(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]

3. **Create TRAW output formatter in app/processing/output.py:**
   
   TrawFormatter class:
   - format_genotypes(df: pd.DataFrame) -> pd.DataFrame
   - write_traw(df: pd.DataFrame, output_path: str) -> None
   - write_tfam(samples: pd.DataFrame, output_path: str) -> None
   - validate_traw(file_path: str) -> bool
   
   Output specifications:
   - TRAW format: CHR SNP (CM) POS COUNTED ALT_ALLELE [sample genotypes as 0/1/2/NA]
   - Handle missing data as NA
   - Include sample metadata in accompanying tfam file

4. **Create multi-source extraction coordinator in app/processing/coordinator.py:**
   
   ExtractionCoordinator class:
   - __init__(extractor: VariantExtractor, harmonizer: VariantHarmonizer)
   - plan_extraction(variant_ids: List[str], data_types: List[DataType]) -> Dict
   - execute_extraction(plan: Dict, parallel: bool = True) -> pd.DataFrame
   - export_results(df: pd.DataFrame, output_dir: str, format: str = "traw") -> Dict[str, str]
   
   Extraction strategies:
   - by_ancestry(variant_ids: List[str], ancestries: List[str]) -> pd.DataFrame
   - by_chromosome(variant_ids: List[str], chromosomes: List[str]) -> pd.DataFrame
   - by_variant_batch(variant_ids: List[str], batch_size: int = 100) -> Iterator[pd.DataFrame]

5. **Create harmonization pipeline in app/processing/harmonizer.py:**
   
   VariantHarmonizer class:
   - harmonize_alleles(ref: str, alt: str, strand: str = "+") -> Tuple[str, str]
   - detect_strand_flip(ref: str, alt: str, ref_genome: str) -> bool
   - normalize_indels(ref: str, alt: str) -> Tuple[str, str]
   - reconcile_multi_source(variants: Dict[DataType, pd.DataFrame]) -> pd.DataFrame

6. **Create Parquet I/O utilities in app/utils/parquet_io.py:**
   - save_parquet(df: pd.DataFrame, path: str, partition_cols: List[str] = None)
   - read_parquet(path: str, filters: List = None) -> pd.DataFrame
   - append_to_parquet(df: pd.DataFrame, path: str)
   - scan_parquet_partitions(base_path: str, filters: Dict) -> pd.DataFrame

**Performance Requirements:**
- Cache build: Process 1M variants in <30 seconds
- Multi-source extraction: Retrieve 400 variants from all 242 imputed files in <10 minutes
- Memory usage: Stream processing to stay under 8GB RAM
- Parallel processing: Use ThreadPoolExecutor for I/O-bound operations

**Example usage:**
```python
from app.processing.coordinator import ExtractionCoordinator
from app.processing.extractor import VariantExtractor
from app.processing.harmonizer import VariantHarmonizer
from app.models.analysis import DataType

# Initialize components
extractor = VariantExtractor(cache_dir="~/gcs_mounts/genotools_server/carriers/cache", settings=settings)
harmonizer = VariantHarmonizer()
coordinator = ExtractionCoordinator(extractor, harmonizer)

# Extract variants from all imputed files for specific ancestries
variant_ids = ["1:123456:A:G", "2:234567:C:T", "10:456789:G:A"]
ancestries = ["EUR", "AAC", "EAS"]

# Plan and execute extraction
plan = coordinator.plan_extraction(
    variant_ids=variant_ids,
    data_types=[DataType.IMPUTED, DataType.NBA]
)
merged_genotypes = coordinator.execute_extraction(plan, parallel=True)

# Export to traw format
output_files = coordinator.export_results(
    df=merged_genotypes,
    output_dir="~/gcs_mounts/genotools_server/carriers/output/",
    format="traw"
)
# Returns: {"traw": "path/to/output.traw", "tfam": "path/to/output.tfam"}
```

**Include:**
- Comprehensive error handling for missing variants across files
- Progress tracking with tqdm for multi-file operations
- Structured logging for extraction planning and execution
- Unit tests with mock multi-source data
- Docstrings with complexity notes for different data configurations
