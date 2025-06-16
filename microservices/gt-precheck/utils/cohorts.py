import pandas as pd
from typing import Any, Optional, List, Dict, Tuple
import os

class CohortsExtrator:
    def extract_cohorts(self, 
                        geno_path: str) -> Tuple[str, List[str]]:
        """
        Extracts cohort names from the genotype file path and returns a list of cohort names.
        
        Args:
            geno_path: Path to PLINK file prefix            
        Returns:
            str: List of cohort names extracted from the genotype files.
        """
        all_files = os.listdir(geno_path)
        file_names = [f.split(".")[0] for f in all_files if os.path.isfile(os.path.join(geno_path, f))]
        prefix = '_'.join(file_names[0].split("_")[:-1])
        cohorts = [f.split("_")[-1] for f in file_names if os.path.isfile(os.path.join(geno_path, f))]
        #make sure to remove duplicates and empty strings
        cohorts = set(cohorts)
        cohorts = [c for c in cohorts if c!='']
        return tuple(prefix, cohorts)
