from dataclasses import dataclass
from typing import List, Optional


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
    
    # Processing options
    api_base_url: str = "http://localhost:8000"
    cleanup_enabled: bool = True
    labels: List[str] = None
    
    def __post_init__(self):
        # Set defaults based on mnt_dir if not provided
        if self.carriers_base_dir is None:
            self.carriers_base_dir = f'{self.mnt_dir}/genotools_server/carriers'
        if self.release_base_dir is None:
            self.release_base_dir = f'{self.mnt_dir}/gp2_release{self.release}_staging/vwb/raw_genotypes'
        if self.wgs_raw_dir is None:
            self.wgs_raw_dir = f'{self.carriers_base_dir}/wgs/raw_genotypes'
        if self.labels is None:
            self.labels = ['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS']
    
    @property
    def summary_dir(self):
        return f'{self.carriers_base_dir}/summary_data'
    
    @property
    def snplist_path(self):
        return f'{self.summary_dir}/carriers_report_snps_complete_list.csv' 