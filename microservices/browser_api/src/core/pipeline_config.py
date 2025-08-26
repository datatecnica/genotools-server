from dataclasses import dataclass
from typing import List, Optional


@dataclass
class PipelineConfig:
    """Configuration for browser preparation pipeline"""
    # Required paths
    mnt_dir: str
    release: int
    
    # Optional paths (can be overridden)
    browser_base_dir: Optional[str] = None # should be gt app utils output
    master_key_dir: Optional[str] = None # should be genotools data bucket for now
    gt_base_dir: Optional[str] = None # should be genotools data bucket
    
    # Processing options
    api_base_url: str = "http://localhost:8000"
    
    def __post_init__(self):
        # Set defaults based on mnt_dir if not provided
        if self.browser_base_dir is None:
            self.browser_base_dir = f'{self.mnt_dir}/gt_app_utils'
        if self.master_key_dir is None:
            self.master_key_dir = f'{self.mnt_dir}/gp2_release{self.release}/genotools_output'
        if self.gt_base_dir is None:
            self.gt_base_dir = f'{self.mnt_dir}/gp2_release{self.release}/genotools_output'