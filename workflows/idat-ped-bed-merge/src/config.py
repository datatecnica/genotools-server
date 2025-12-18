"""Configuration management for IDAT Processor

Handles all paths, parameters, and settings required for the processing pipeline.
"""

from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ProcessorConfig:
    """Configuration container for SNP processing pipeline.
    
    All paths are converted to Path objects for consistent handling.
    Required paths are validated on initialization.
    """
    # Common Inputs
    calc_flag: int # 0 -> whole pipeline, 1 -> IDAT-PED, 2 -> PED-BED, 3 -> BEDS-MERGE
    log_file_path: Path #Please this should be the base address. For each step, we will create a subfolder with the step name (idat-ped, ped-bed and beds-merge).
    study_id: List[str]
    key_path: Path
    fam_path: Path
    raw_plink_path: Path
    batch_folder_path: Path
    exec_folder_path: str
    num_threads: int

    #idat-ped
    idat_path: Path
    barcodes_per_job: int
    #ped-bed
    codes_per_job: int #same for both ped-bed and beds-merge
    #beds merge
    clinical_key_dir: Path
    # codes_per_job: int
    # k8s server related
    service_account_name: str
    k8s_namespace: str
    pv_claim: str
    gke_nodepools: str
    #user_email: str


    def __post_init__(self):
        """Convert string paths to Path objects and validate required files."""
        self.calc_flag = self.calc_flag
        if not self.calc_flag in [0, 1, 2, 3]:  # Validate the 'calc_flag' value
            raise ValueError("Invalid value for 'calc_flag'. It should be 0, 1, 2, or 3.")
        # Convert all paths to Path objects
        self.study_id = self.study_id
        if not self.study_id:  # Check if the list is empty
            raise ValueError("The 'study_id' list cannot be empty. It must contain at least one element.")        
        self.key_path = Path(self.key_path)
        self.exec_folder_path = self.exec_folder_path
        self.idat_path = Path(self.idat_path)


        self.log_file_path = Path(self.log_file_path)
        #also create folder if it doesn't exist
        self.log_file_path.mkdir(parents=True, exist_ok=True)        

        self.fam_path = Path(self.fam_path)
        #also create folder if it doesn't exist
        self.fam_path.mkdir(parents=True, exist_ok=True)        
        
        self.raw_plink_path = Path(self.raw_plink_path)
        #also create folder if it doesn't exist
        self.raw_plink_path.mkdir(parents=True, exist_ok=True)        


        self.batch_folder_path = Path(self.batch_folder_path)
        #also create folder if it doesn't exist
        self.batch_folder_path.mkdir(parents=True, exist_ok=True)        
        
        self.num_threads = self.num_threads
        self.barcodes_per_job = self.barcodes_per_job
        #ped-bed
        self.codes_per_job = self.codes_per_job
        #beds-merge
        self.clinical_key_dir = Path(self.clinical_key_dir)

        #k8s server related
        self.service_account_name = self.service_account_name
        self.k8s_namespace = self.k8s_namespace
        self.pv_claim = self.pv_claim
        self.gke_nodepools = self.gke_nodepools
        # self.user_email = self.user_email


        
        # Validate required input files exist
        self._validate_required_files()
        
        # Create output directories
        # self._create_output_directories()
    
    def _validate_required_files(self):
        """Validate that required input files exist."""
        required_files = [
            self.key_path,
            self.log_file_path,
            self.fam_path,
            self.raw_plink_path,
            self.batch_folder_path,
            # self.exec_folder_path,
            self.idat_path,
            self.clinical_key_dir
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            raise ConfigurationError(
                f"Required files/inputs not found: {', '.join(str(f) for f in missing_files)}"
            )
                
class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass 