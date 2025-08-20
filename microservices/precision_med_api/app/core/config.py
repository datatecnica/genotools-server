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
        default=["AAC", "AJ", "CAH", "CAS", "EAS", "EUR", "FIN", "LAS", "MDE", "SAS", "SSA"],
        description="Available ancestry groups"
    )
    
    # Valid chromosomes
    CHROMOSOMES: List[str] = Field(
        default=[str(i) for i in range(1, 23)] + ["X", "Y", "MT"],
        description="Valid chromosome identifiers"
    )
    
    # Output formats
    output_format: str = Field(default="parquet", description="Default output format")
    
    # Processing parameters
    chunk_size: int = Field(default=10000, description="Chunk size for processing large files")
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    cache_enabled: bool = Field(default=True, description="Enable variant index caching")
    
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
    
    @cached_property
    def carriers_path(self) -> str:
        return os.path.join(self.mnt_path, "genotools_server", "precision_med")
    
    @cached_property
    def snp_list_path(self) -> str:
        return os.path.join(self.carriers_path, "summary_data", "precision_med_snp_list.csv")
    
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
            f"R{self.release}_wgs_carriers_vars"
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
            job_id
        )
    
    def get_cache_path(self) -> str:
        return os.path.join(
            self.carriers_path,
            "cache"
        )
    
    def get_variant_index_path(self, data_type: str, ancestry: Optional[str] = None, 
                              chrom: Optional[str] = None) -> str:
        cache_base = self.get_cache_path()
        
        if data_type == "WGS":
            return os.path.join(cache_base, "wgs", "variant_index.parquet")
        elif data_type == "NBA":
            if not ancestry:
                raise ValueError("Ancestry required for NBA variant index")
            return os.path.join(cache_base, "nba", ancestry, "variant_index.parquet")
        elif data_type == "IMPUTED":
            if not ancestry or not chrom:
                raise ValueError("Ancestry and chromosome required for IMPUTED variant index")
            return os.path.join(
                cache_base, "imputed", ancestry, f"chr{chrom}_variant_index.parquet"
            )
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    def get_harmonization_cache_path(self, data_type: str, release: str, 
                                   ancestry: Optional[str] = None, 
                                   chrom: Optional[str] = None) -> str:
        """Get path to harmonization cache file."""
        cache_base = os.path.join(self.get_cache_path(), f"release{release}")
        
        if data_type == "WGS":
            return os.path.join(cache_base, "wgs", "wgs_variant_harmonization.parquet")
        elif data_type == "NBA":
            if not ancestry:
                raise ValueError("Ancestry required for NBA harmonization cache")
            return os.path.join(cache_base, "nba", f"{ancestry}_variant_harmonization.parquet")
        elif data_type == "IMPUTED":
            if not ancestry or not chrom:
                raise ValueError("Ancestry and chromosome required for IMPUTED harmonization cache")
            return os.path.join(
                cache_base, "imputed", ancestry, f"chr{chrom}_variant_harmonization.parquet"
            )
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    def get_harmonization_cache_dir(self, data_type: str, release: str) -> str:
        """Get directory containing harmonization cache files."""
        cache_base = os.path.join(self.get_cache_path(), f"release{release}")
        
        if data_type == "WGS":
            return os.path.join(cache_base, "wgs")
        elif data_type == "NBA":
            return os.path.join(cache_base, "nba")
        elif data_type == "IMPUTED":
            return os.path.join(cache_base, "imputed")
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
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
    
    def list_harmonization_cache_files(self, data_type: str, release: str) -> List[str]:
        """List all harmonization cache files for a data type."""
        cache_files = []
        
        if data_type == "WGS":
            cache_path = self.get_harmonization_cache_path("WGS", release)
            if os.path.exists(cache_path):
                cache_files.append(cache_path)
        
        elif data_type == "NBA":
            for ancestry in self.list_available_ancestries("NBA"):
                cache_path = self.get_harmonization_cache_path("NBA", release, ancestry=ancestry)
                if os.path.exists(cache_path):
                    cache_files.append(cache_path)
        
        elif data_type == "IMPUTED":
            for ancestry in self.list_available_ancestries("IMPUTED"):
                for chrom in self.list_available_chromosomes(ancestry):
                    cache_path = self.get_harmonization_cache_path(
                        "IMPUTED", release, ancestry=ancestry, chrom=chrom
                    )
                    if os.path.exists(cache_path):
                        cache_files.append(cache_path)
        
        return cache_files
    
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
    
    def list_available_chromosomes(self, ancestry: str) -> List[str]:
        base_path = os.path.join(self.release_path, "imputed_genotypes", ancestry)
        
        if not os.path.exists(base_path):
            return []
        
        available = []
        for chrom in self.CHROMOSOMES:
            # Check if any of the PLINK files exist for this chromosome
            pgen_file = os.path.join(
                base_path,
                f"chr{chrom}_{ancestry}_release{self.release}_vwb.pgen"
            )
            if os.path.exists(pgen_file):
                available.append(chrom)
        
        return available
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "release": "10",
                "mnt_path": "~/gcs_mounts",
                "output_format": "parquet",
                "chunk_size": 10000,
                "max_workers": 4,
                "cache_enabled": True
            }
        }
    )


def get_settings() -> Settings:
    return Settings()