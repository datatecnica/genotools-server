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


def list_available_files(data_type: DataType, release: str, 
                        settings: Optional[Settings] = None) -> Dict[str, Any]:
    if not settings:
        settings = Settings(release=release)
    
    result = {
        "data_type": data_type.value,
        "release": release,
        "available_files": [],
        "total_files": 0,
        "total_size_mb": 0.0
    }
    
    if data_type == DataType.WGS:
        # WGS has a single file
        base_path = settings.get_wgs_path()
        pgen_set = PgenFileSet(base_path)
        
        if pgen_set.is_valid:
            result["available_files"].append({
                "type": "WGS",
                "path": base_path,
                "exists": True,
                "samples": pgen_set.get_sample_count(),
                "variants": pgen_set.get_variant_count(),
                "size_mb": pgen_set.total_size_mb
            })
            result["total_files"] = 1
            result["total_size_mb"] = pgen_set.total_size_mb
    
    elif data_type == DataType.NBA:
        # NBA files split by ancestry
        for ancestry in settings.ANCESTRIES:
            try:
                base_path = settings.get_nba_path(ancestry)
                pgen_set = PgenFileSet(base_path)
                
                if pgen_set.is_valid:
                    result["available_files"].append({
                        "type": "NBA",
                        "ancestry": ancestry,
                        "path": base_path,
                        "exists": True,
                        "samples": pgen_set.get_sample_count(),
                        "variants": pgen_set.get_variant_count(),
                        "size_mb": pgen_set.total_size_mb
                    })
                    result["total_files"] += 1
                    result["total_size_mb"] += pgen_set.total_size_mb
            except ValueError:
                continue
    
    elif data_type == DataType.IMPUTED:
        # IMPUTED files split by ancestry and chromosome
        for ancestry in settings.ANCESTRIES:
            ancestry_files = []
            ancestry_size = 0.0
            
            for chrom in settings.CHROMOSOMES:
                try:
                    base_path = settings.get_imputed_path(ancestry, chrom)
                    pgen_set = PgenFileSet(base_path)
                    
                    if pgen_set.is_valid:
                        ancestry_files.append({
                            "chromosome": chrom,
                            "path": base_path,
                            "samples": pgen_set.get_sample_count(),
                            "variants": pgen_set.get_variant_count(),
                            "size_mb": pgen_set.total_size_mb
                        })
                        ancestry_size += pgen_set.total_size_mb
                except ValueError:
                    continue
            
            if ancestry_files:
                result["available_files"].append({
                    "type": "IMPUTED",
                    "ancestry": ancestry,
                    "chromosomes": len(ancestry_files),
                    "files": ancestry_files,
                    "total_size_mb": ancestry_size
                })
                result["total_files"] += len(ancestry_files)
                result["total_size_mb"] += ancestry_size
    
    return result


def get_file_info(file_path: str) -> Dict[str, Any]:
    pgen_set = PgenFileSet(file_path)
    
    if not pgen_set.is_valid:
        return {
            "path": file_path,
            "exists": False,
            "error": "PLINK file set incomplete or missing"
        }
    
    return {
        "path": file_path,
        "exists": True,
        "samples": pgen_set.get_sample_count(),
        "variants": pgen_set.get_variant_count(),
        "size_mb": pgen_set.total_size_mb,
        "files": {
            "pgen": pgen_set.pgen_file,
            "pvar": pgen_set.pvar_file,
            "psam": pgen_set.psam_file
        },
        "file_sizes_mb": {
            k: v / (1024 * 1024) for k, v in pgen_set.file_sizes.items()
        } if pgen_set.file_sizes else None
    }


def find_matching_files(data_type: DataType, ancestries: Optional[List[str]], 
                       chromosomes: Optional[List[str]], 
                       settings: Settings) -> List[PgenFileSet]:
    matching_files = []
    
    if data_type == DataType.WGS:
        # WGS has a single file, no filtering needed
        base_path = settings.get_wgs_path()
        pgen_set = PgenFileSet(base_path)
        if pgen_set.is_valid:
            matching_files.append(pgen_set)
    
    elif data_type == DataType.NBA:
        # NBA files split by ancestry only
        target_ancestries = ancestries if ancestries else settings.ANCESTRIES
        
        for ancestry in target_ancestries:
            if ancestry in settings.ANCESTRIES:
                try:
                    base_path = settings.get_nba_path(ancestry)
                    pgen_set = PgenFileSet(base_path)
                    if pgen_set.is_valid:
                        matching_files.append(pgen_set)
                except ValueError:
                    continue
    
    elif data_type == DataType.IMPUTED:
        # IMPUTED files split by both ancestry and chromosome
        target_ancestries = ancestries if ancestries else settings.ANCESTRIES
        target_chromosomes = chromosomes if chromosomes else settings.CHROMOSOMES
        
        for ancestry in target_ancestries:
            if ancestry not in settings.ANCESTRIES:
                continue
            
            for chrom in target_chromosomes:
                if chrom not in settings.CHROMOSOMES:
                    continue
                
                try:
                    base_path = settings.get_imputed_path(ancestry, chrom)
                    pgen_set = PgenFileSet(base_path)
                    if pgen_set.is_valid:
                        matching_files.append(pgen_set)
                except ValueError:
                    continue
    
    return matching_files


def create_output_directory(job_id: str, settings: Settings) -> str:
    output_path = settings.get_output_path(job_id)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path



def get_clinical_files(settings: Settings) -> Dict[str, Tuple[str, bool]]:
    clinical_paths = settings.get_clinical_paths()
    
    result = {}
    for key, path in clinical_paths.items():
        exists = os.path.exists(path)
        result[key] = (path, exists)
    
    return result