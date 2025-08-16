from dataclasses import dataclass
from typing import List, Optional
import logging


@dataclass
class PipelineConfig:
    """Configuration for carrier processing pipeline"""
    # Required paths
    mnt_dir: str
    release: str
    
    # Optional paths (can be overridden)
    carriers_base_dir: Optional[str] = None
    release_base_dir: Optional[str] = None
    wgs_raw_dir: Optional[str] = None
    nba_dir: Optional[str] = None
    imputed_dir: Optional[str] = None
    variant_cache_dir: Optional[str] = None
    
    # Processing options
    api_base_url: str = "http://localhost:8000"
    cleanup_enabled: bool = True
    labels: List[str] = None
    
    # Optimization settings (merged from OptimizationConfig)
    enable_variant_caching: bool = True
    enable_parallel_processing: bool = True
    max_chromosome_workers: int = 2  # Default to 2 vCPUs
    enable_polars: bool = True
    polars_fallback_on_error: bool = True
    enable_file_caching: bool = True
    cache_validation_method: str = "timestamp"
    enable_snp_splitting: bool = False
    snp_split_temp_dir: Optional[str] = None
    max_memory_usage_gb: float = 8.0
    enable_memory_monitoring: bool = True
    enable_performance_logging: bool = True
    performance_log_level: int = logging.INFO
    log_cache_stats: bool = True
    log_timing_details: bool = True
    
    def __post_init__(self):
        # Set defaults based on mnt_dir if not provided
        if self.carriers_base_dir is None:
            self.carriers_base_dir = f'{self.mnt_dir}/genotools_server/carriers'
        if self.release_base_dir is None:
            self.release_base_dir = f'{self.mnt_dir}/gp2tier2_vwb/release{self.release}'
        if self.wgs_raw_dir is None:
            self.wgs_raw_dir = f'{self.carriers_base_dir}/wgs/raw_genotypes'
        if self.nba_dir is None:
            self.nba_dir = f'{self.release_base_dir}/raw_genotypes'
        if self.imputed_dir is None:
            self.imputed_dir = f'{self.release_base_dir}/imputed_genotypes'
        if self.variant_cache_dir is None:
            self.variant_cache_dir = f'{self.carriers_base_dir}/variant_cache'
        if self.labels is None:
            self.labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
        
        # Validate optimization settings
        if self.max_chromosome_workers < 1:
            self.max_chromosome_workers = 1
        if self.cache_validation_method not in ["timestamp", "hash", "none"]:
            self.cache_validation_method = "timestamp"
    
    @property
    def summary_dir(self):
        return f'{self.carriers_base_dir}/summary_data'
    
    @property
    def snplist_path(self):
        # Always use the main carriers directory for SNP list, not the potentially overridden one
        main_carriers_dir = f'{self.mnt_dir}/genotools_server/carriers'
        return f'{main_carriers_dir}/summary_data/carriers_snp_list.csv'
    
 