from typing import Dict, Any, Optional, List
from src.core.carrier_processor import CarrierProcessorFactory
from src.core.data_repository import DataRepository
from src.core.harmonizer import AlleleHarmonizer
from src.core.genotype_converter import StandardGenotypeConverter


class CarrierAnalysisManager:
    def __init__(self):
        # Initialize common dependencies
        self.data_repo = DataRepository()
        self.harmonizer = AlleleHarmonizer()
        self.genotype_converter = StandardGenotypeConverter()
        
        # Initialize processors using the factory
        factory = CarrierProcessorFactory()
        self.variant_processor = factory.create_variant_processor(self.harmonizer, self.data_repo)
        self.carrier_extractor = factory.create_carrier_extractor(
            self.variant_processor, self.genotype_converter, self.data_repo)
        self.carrier_combiner = factory.create_carrier_combiner(self.data_repo)
        self.validator = factory.create_validator(self.data_repo, self.genotype_converter)
    
    def extract_carriers(self, geno_path: str, snplist_path: str, out_path: str,
                        ancestry: str = None, release: str = None) -> Dict[str, str]:
        """Extract carrier information for given SNPs (unified method for all data types)"""
        return self.carrier_extractor.extract_carriers(geno_path, snplist_path, out_path, ancestry, release)
    
    def combine_carrier_files(self, results_by_label: Dict[str, Dict[str, str]], 
                             key_file: str, out_path: str) -> Dict[str, str]:
        """Combine carrier files from multiple ancestry labels"""
        return self.carrier_combiner.combine_carrier_files(results_by_label, key_file, out_path)
    
    def validate_carrier_data(self, traw_dir: str, combined_file: str, 
                             snp_info_file: str, samples_to_check: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Validate that combined carrier data matches original traw files"""
        return self.validator.validate_carrier_data(traw_dir, combined_file, snp_info_file, samples_to_check)