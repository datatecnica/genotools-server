from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Configuration for browser preparation pipeline"""
    # Required paths
    mnt_dir: str
    release: int
    
    # Optional paths (can be overridden)
    browser_base_dir: Optional[str] = None # should be genotools-server bucket
    master_key_dir: Optional[str] = None # should be release bucket
    gt_base_dir: Optional[str] = None # should be release bucket
    
    # Processing options
    api_base_url: str = "http://localhost:8000"
    
    def __post_init__(self):
        # Set defaults based on mnt_dir if not provided
        if self.browser_base_dir is None:
            self.browser_base_dir = f'{self.mnt_dir}/genotools-server/cohort_browser'
        if self.master_key_dir is None:
            self.master_key_dir = f'{self.mnt_dir}/gp2tier2_vwb/release{self.release}/clinical_data'
        if self.gt_base_dir is None:
            self.gt_base_dir = f'{self.mnt_dir}/gp2tier2_vwb/release{self.release}/meta_data'