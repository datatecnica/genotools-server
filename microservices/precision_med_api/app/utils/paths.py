import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from app.models.analysis import DataType
from app.core.config import Settings


@dataclass
class PgenFileSet:
    base_path: str
    pgen_file: str
    pvar_file: str
    psam_file: str
    exists: bool = False
    file_sizes: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        self.pgen_file = f"{self.base_path}.pgen"
        self.pvar_file = f"{self.base_path}.pvar"
        self.psam_file = f"{self.base_path}.psam"
        self.exists = self.validate()
        
        if self.exists:
            self.file_sizes = {
                "pgen": os.path.getsize(self.pgen_file),
                "pvar": os.path.getsize(self.pvar_file),
                "psam": os.path.getsize(self.psam_file)
            }
    
    def validate(self) -> bool:
        return all([
            os.path.exists(self.pgen_file),
            os.path.exists(self.pvar_file),
            os.path.exists(self.psam_file)
        ])
    
    @property
    def is_valid(self) -> bool:
        return self.exists
    
    @property
    def total_size_mb(self) -> float:
        if not self.file_sizes:
            return 0.0
        total_bytes = sum(self.file_sizes.values())
        return total_bytes / (1024 * 1024)
    
    def get_sample_count(self) -> int:
        if not self.exists:
            return 0
        
        try:
            with open(self.psam_file, 'r') as f:
                # Skip header line
                lines = f.readlines()
                # First line is header, count remaining lines
                return len(lines) - 1
        except Exception:
            return 0
    
    def get_variant_count(self) -> int:
        if not self.exists:
            return 0
        
        try:
            with open(self.pvar_file, 'r') as f:
                # Skip header lines (lines starting with #)
                count = 0
                for line in f:
                    if not line.startswith('#'):
                        count += 1
                return count
        except Exception:
            return 0


def validate_pgen_files(base_path: str) -> bool:
    pgen_set = PgenFileSet(base_path)
    return pgen_set.is_valid











