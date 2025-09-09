import os
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from functools import cached_property


class Settings(BaseModel):
    release: str = Field(default="10", description="GP2 release version")
    mnt_path: str = Field(
        default=os.path.expanduser("~/gcs_mounts"),
        description="Mount path for GCS buckets"
    )
    
    # Available ancestries in GP2 data
    ANCESTRIES: List[str] = Field(
        default=['AAC', 'AFR', 'AJ', 'AMR', 'CAH', 'CAS', 'EAS', 'EUR', 'FIN', 'MDE', 'SAS'],
        description="Available ancestry groups"
    )
    
    # Valid chromosomes
    CHROMOSOMES: List[str] = Field(
        default=[str(i) for i in range(1, 23)] + ["X", "Y", "MT"],
        description="Valid chromosome identifiers"
    )
    
    # Output formats
    output_format: str = Field(default="parquet", description="Default output format")
    
    # Performance parameters - Auto-detects based on machine specs
    chunk_size: int = Field(default=50000, description="Chunk size for processing large files")
    max_workers: int = Field(default=28, description="Maximum number of parallel workers")
    process_cap: int = Field(default=30, description="Maximum concurrent processes allowed")
    cpu_reservation: int = Field(default=2, description="CPU cores to reserve for OS")
    cache_enabled: bool = Field(default=True, description="Enable variant index caching")
    
    # Timeout parameters (seconds)
    plink_timeout_short: int = Field(default=10, description="Short PLINK operations timeout")
    plink_timeout_medium: int = Field(default=300, description="Medium PLINK operations timeout") 
    plink_timeout_long: int = Field(default=600, description="Long PLINK operations timeout")
    
    @field_validator('release')
    @classmethod
    def validate_release(cls, v: str) -> str:
        try:
            release_num = int(v)
            if release_num < 1 or release_num > 100:
                raise ValueError
        except ValueError:
            raise ValueError(f"Invalid release: {v}. Must be a valid release number")
        return v
    
    @field_validator('mnt_path')
    @classmethod
    def validate_mnt_path(cls, v: str) -> str:
        return os.path.expanduser(v)
    
    @classmethod
    def auto_detect_performance_settings(cls) -> Dict[str, int]:
        """Auto-detect optimal performance settings based on machine specs."""
        import psutil
        
        cpu_count = os.cpu_count() or 4
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Performance tier detection
        if cpu_count <= 4 and total_ram_gb <= 16:
            # Small: Development/laptop
            return {
                'chunk_size': 15000,
                'max_workers': max(2, cpu_count - 2),
                'process_cap': min(6, cpu_count),
                'cpu_reservation': 2
            }
        elif cpu_count <= 8 and total_ram_gb <= 32:
            # Medium: Small production
            return {
                'chunk_size': 25000,
                'max_workers': max(4, cpu_count - 2),
                'process_cap': min(12, cpu_count + 2),
                'cpu_reservation': 2
            }
        elif cpu_count <= 16 and total_ram_gb <= 64:
            # Large: Medium production
            return {
                'chunk_size': 40000,
                'max_workers': max(8, cpu_count - 2),
                'process_cap': min(20, cpu_count + 4),
                'cpu_reservation': 2
            }
        elif cpu_count <= 32 and total_ram_gb <= 128:
            # XLarge: High-end production (current system)
            return {
                'chunk_size': 50000,
                'max_workers': max(16, cpu_count - 4),
                'process_cap': min(30, cpu_count - 2),
                'cpu_reservation': 4
            }
        else:
            # XXLarge: Workstation/Server
            return {
                'chunk_size': 75000,
                'max_workers': max(24, cpu_count - 6),
                'process_cap': min(60, cpu_count - 4),
                'cpu_reservation': 6
            }
    
    def get_optimal_workers(self, total_files: int) -> int:
        """Get optimal worker count for given number of files."""
        cpu_count = os.cpu_count() or 4
        return min(
            total_files,
            max(1, cpu_count - self.cpu_reservation),
            self.max_workers,
            self.process_cap
        )
    
    @cached_property
    def carriers_path(self) -> str:
        return os.path.join(self.mnt_path, "genotools_server", "precision_med")
    
    @cached_property
    def snp_list_path(self) -> str:
        return os.path.join(self.carriers_path, "summary_data", "precision_med_snp_list.csv")
    
    def get_snp_chromosomes(self, snp_list_path: Optional[str] = None) -> List[str]:
        """
        Extract chromosome list from SNP list file.
        
        Args:
            snp_list_path: Path to SNP list file. Uses default if None.
            
        Returns:
            List of chromosomes that contain variants in the SNP list
        """
        import pandas as pd
        
        if snp_list_path is None:
            snp_list_path = self.snp_list_path
            
        try:
            # Load SNP list
            snp_list = pd.read_csv(snp_list_path)
            
            # Extract chromosomes from hg38 coordinates  
            if 'hg38' in snp_list.columns:
                coords = snp_list['hg38'].str.split(':', expand=True)
                if coords.shape[1] >= 1:
                    chromosomes = coords[0].str.replace('chr', '').str.upper().dropna().unique()
                    return sorted(chromosomes.tolist())
            
            # Fallback to all chromosomes if extraction fails
            return self.CHROMOSOMES
            
        except Exception:
            # Return all chromosomes if SNP list can't be loaded
            return self.CHROMOSOMES
    
    @cached_property
    def release_path(self) -> str:
        return os.path.join(self.mnt_path, "gp2tier2_vwb", f"release{self.release}")
    
    def get_nba_path(self, ancestry: str) -> str:
        if ancestry not in self.ANCESTRIES:
            raise ValueError(f"Invalid ancestry: {ancestry}. Must be one of {self.ANCESTRIES}")
        
        base_path = os.path.join(
            self.release_path,
            "raw_genotypes",
            ancestry,
            f"{ancestry}_release{self.release}_vwb"
        )
        return base_path
    
    def get_wgs_path(self) -> str:
        base_path = os.path.join(
            self.carriers_path,
            "wgs",
            "raw_genotypes",
            f"R{self.release}_wgs_carrier_vars"
        )
        return base_path
    
    def get_imputed_path(self, ancestry: str, chrom: str) -> str:
        if ancestry not in self.ANCESTRIES:
            raise ValueError(f"Invalid ancestry: {ancestry}. Must be one of {self.ANCESTRIES}")
        
        if chrom not in self.CHROMOSOMES:
            raise ValueError(f"Invalid chromosome: {chrom}. Must be one of {self.CHROMOSOMES}")
        
        base_path = os.path.join(
            self.release_path,
            "imputed_genotypes",
            ancestry,
            f"chr{chrom}_{ancestry}_release{self.release}_vwb"
        )
        return base_path
    
    def get_clinical_paths(self) -> Dict[str, str]:
        clinical_base = os.path.join(self.release_path, "clinical_data")
        
        return {
            "master_key": os.path.join(
                clinical_base,
                f"master_key_release{self.release}_final_vwb.csv"
            ),
            "data_dictionary": os.path.join(
                clinical_base,
                f"data_dictionary_release{self.release}.csv"
            ),
            "extended_clinical": os.path.join(
                clinical_base,
                f"extended_clinical_release{self.release}.csv"
            )
        }
    
    def get_output_path(self, job_id: str) -> str:
        return os.path.join(
            self.carriers_path,
            "results",
            f"release{self.release}",
            job_id
        )
    
    @cached_property
    def results_path(self) -> str:
        """Base results directory for current release."""
        return os.path.join(
            self.carriers_path,
            "results", 
            f"release{self.release}"
        )
    
    def get_cache_path(self) -> str:
        return os.path.join(
            self.carriers_path,
            "cache"
        )
    
    def get_pvar_file_path(self, data_type: str, ancestry: Optional[str] = None, 
                          chrom: Optional[str] = None) -> str:
        """Get path to PVAR file for given data type and parameters."""
        if data_type == "WGS":
            return self.get_wgs_path() + ".pvar"
        elif data_type == "NBA":
            if not ancestry:
                raise ValueError("Ancestry required for NBA PVAR file")
            return self.get_nba_path(ancestry) + ".pvar"
        elif data_type == "IMPUTED":
            if not ancestry or not chrom:
                raise ValueError("Ancestry and chromosome required for IMPUTED PVAR file")
            return self.get_imputed_path(ancestry, chrom) + ".pvar"
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    def get_pgen_file_path(self, data_type: str, ancestry: Optional[str] = None, 
                          chrom: Optional[str] = None) -> str:
        """Get path to PGEN file for given data type and parameters."""
        if data_type == "WGS":
            return self.get_wgs_path() + ".pgen"
        elif data_type == "NBA":
            if not ancestry:
                raise ValueError("Ancestry required for NBA PGEN file")
            return self.get_nba_path(ancestry) + ".pgen"
        elif data_type == "IMPUTED":
            if not ancestry or not chrom:
                raise ValueError("Ancestry and chromosome required for IMPUTED PGEN file")
            return self.get_imputed_path(ancestry, chrom) + ".pgen"
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    
    def validate_file_paths(self, data_type: str, ancestry: Optional[str] = None, 
                           chrom: Optional[str] = None) -> Dict[str, bool]:
        """Validate that required PLINK files exist."""
        results = {}
        
        try:
            pgen_path = self.get_pgen_file_path(data_type, ancestry, chrom)
            pvar_path = self.get_pvar_file_path(data_type, ancestry, chrom)
            psam_path = pgen_path.replace('.pgen', '.psam')
            
            results['pgen_exists'] = os.path.exists(pgen_path)
            results['pvar_exists'] = os.path.exists(pvar_path)
            results['psam_exists'] = os.path.exists(psam_path)
            results['all_files_exist'] = all(results.values())
            
            results['pgen_path'] = pgen_path
            results['pvar_path'] = pvar_path
            results['psam_path'] = psam_path
            
        except Exception as e:
            results['error'] = str(e)
            results['all_files_exist'] = False
        
        return results
    
    def list_available_ancestries(self, data_type: str) -> List[str]:
        if data_type == "WGS":
            return []  # WGS is not split by ancestry
        
        if data_type == "NBA":
            base_path = os.path.join(self.release_path, "raw_genotypes")
        elif data_type == "IMPUTED":
            base_path = os.path.join(self.release_path, "imputed_genotypes")
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
        if not os.path.exists(base_path):
            return []
        
        available = []
        for ancestry in self.ANCESTRIES:
            ancestry_path = os.path.join(base_path, ancestry)
            if os.path.exists(ancestry_path):
                available.append(ancestry)
        
        return available
    
    def list_available_chromosomes(self, ancestry: str, filter_by_snp_list: bool = True) -> List[str]:
        base_path = os.path.join(self.release_path, "imputed_genotypes", ancestry)
        
        if not os.path.exists(base_path):
            return []
        
        # Use SNP-based chromosome filtering by default
        chromosomes_to_check = self.get_snp_chromosomes() if filter_by_snp_list else self.CHROMOSOMES
        
        # Add validation logging for chromosome filtering
        if filter_by_snp_list and len(chromosomes_to_check) < len(self.CHROMOSOMES):
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Chromosome filtering active: checking {len(chromosomes_to_check)} SNP-list chromosomes ({chromosomes_to_check}) for ancestry {ancestry}")
        
        available = []
        for chrom in chromosomes_to_check:
            # Check if any of the PLINK files exist for this chromosome
            pgen_file = os.path.join(
                base_path,
                f"chr{chrom}_{ancestry}_release{self.release}_vwb.pgen"
            )
            if os.path.exists(pgen_file):
                available.append(chrom)
        
        # Add validation warning if no chromosomes match
        if not available and filter_by_snp_list:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"No available chromosomes found for ancestry {ancestry} after SNP-list filtering. Target chromosomes were: {chromosomes_to_check}")
        
        return available
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "release": "10",
                "mnt_path": "~/gcs_mounts",
                "output_format": "parquet",
                "chunk_size": 50000,
                "max_workers": 28,
                "process_cap": 30,
                "cpu_reservation": 2,
                "cache_enabled": True,
                "plink_timeout_short": 10,
                "plink_timeout_medium": 300,
                "plink_timeout_long": 600
            }
        }
    )
    
    @classmethod
    def create_optimized(cls, **overrides) -> 'Settings':
        """Create Settings instance with auto-detected performance optimization."""
        optimal_settings = cls.auto_detect_performance_settings()
        
        # Merge auto-detected settings with any manual overrides
        config_dict = {**optimal_settings, **overrides}
        
        return cls(**config_dict)


def get_settings() -> Settings:
    """Get Settings instance with auto-optimization based on environment."""
    # Check if AUTO_OPTIMIZE environment variable is set
    if os.getenv('AUTO_OPTIMIZE', 'true').lower() in ('true', '1', 'yes'):
        return Settings.create_optimized()
    else:
        return Settings()