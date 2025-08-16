from typing import Dict, Any, Optional, List
from src.core.carrier_processor import CarrierProcessorFactory
from src.core.data_repository import DataRepository
from src.core.harmonizer import AlleleHarmonizer
from src.core.genotype_converter import StandardGenotypeConverter
from src.core.pipeline_config import PipelineConfig
from src.utils.performance_metrics import get_performance_tracker, log_performance_summary
import os
import logging

logger = logging.getLogger(__name__)


class CarrierAnalysisManager:
    def __init__(self, config: PipelineConfig = None):
        """
        Initialize CarrierAnalysisManager with pipeline configuration
        
        Args:
            config: Pipeline configuration containing optimization settings
        """
        if config is None:
            raise ValueError("PipelineConfig is required")
        
        self.config = config
        
        # Initialize common dependencies with configuration
        self.data_repo = DataRepository()
        self.harmonizer = AlleleHarmonizer(config=config)
        self.genotype_converter = StandardGenotypeConverter()
        
        # Initialize processors using the factory
        factory = CarrierProcessorFactory()
        self.variant_processor = factory.create_variant_processor(self.harmonizer, self.data_repo)
        self.carrier_extractor = factory.create_carrier_extractor(
            self.variant_processor, self.genotype_converter, self.data_repo)
        self.carrier_combiner = factory.create_carrier_combiner(self.data_repo)
        self.validator = factory.create_validator(self.data_repo, self.genotype_converter)
        
        # Setup performance tracking
        self.performance_tracker = get_performance_tracker()
        
        logger.info("CarrierAnalysisManager initialized with optimizations enabled")
    
    def extract_carriers(self, geno_path: str, snplist_path: str, out_path: str,
                        ancestry: str = None, release: str = None) -> Dict[str, str]:
        """Extract carrier information for given SNPs (unified method for all data types)"""
        # Configure parallel processing based on config
        if hasattr(self.carrier_extractor, '_process_chromosome_split') and self.config.enable_parallel_processing:
            # Monkey patch the max_workers parameter for parallel processing
            original_method = self.carrier_extractor._process_chromosome_split
            def enhanced_process_chromosome_split(base_dir, snplist_path, out_path, ancestry, release, max_workers=None):
                if max_workers is None:
                    max_workers = self.config.max_chromosome_workers
                return original_method(base_dir, snplist_path, out_path, ancestry, release, max_workers)
            self.carrier_extractor._process_chromosome_split = enhanced_process_chromosome_split
        
        return self.carrier_extractor.extract_carriers(geno_path, snplist_path, out_path, ancestry, release)
    
    def combine_carrier_files(self, results_by_label: Dict[str, Dict[str, str]], 
                             key_file: str, out_path: str) -> Dict[str, str]:
        """Combine carrier files from multiple ancestry labels"""
        return self.carrier_combiner.combine_carrier_files(results_by_label, key_file, out_path)
    
    def validate_carrier_data(self, traw_dir: str, combined_file: str, 
                             snp_info_file: str, samples_to_check: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Validate that combined carrier data matches original traw files"""
        return self.validator.validate_carrier_data(traw_dir, combined_file, snp_info_file, samples_to_check)
    
    def get_performance_metrics(self, operation: str = None) -> Dict[str, Any]:
        """Get performance metrics for analysis operations"""
        if not self.config.enable_performance_logging:
            return {"message": "Performance logging is disabled"}
        
        metrics = self.performance_tracker.get_metrics(operation)
        summary = self.performance_tracker.get_summary()
        
        return {
            "detailed_metrics": metrics,
            "summary": summary,
            "cache_stats": self._get_cache_stats() if self.config.log_cache_stats else None
        }
    
    def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}
        
        if self.config.enable_variant_caching and hasattr(self.harmonizer, 'cache') and self.harmonizer.cache:
            stats["variant_cache"] = self.harmonizer.cache.get_cache_stats()
        
        return stats
    
    def log_performance_summary(self):
        """Log performance summary for all operations"""
        if self.config.enable_performance_logging:
            log_performance_summary()
            
            # Also log cache stats if enabled
            if self.config.log_cache_stats:
                cache_stats = self._get_cache_stats()
                if cache_stats:
                    logger.info("=== Cache Statistics ===")
                    for cache_type, stats in cache_stats.items():
                        logger.info(f"{cache_type}: {stats}")
    
    def clear_performance_metrics(self):
        """Clear all stored performance metrics"""
        if self.config.enable_performance_logging:
            self.performance_tracker.clear_metrics()
            logger.info("Performance metrics cleared")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current optimization configuration"""
        return self.config.to_dict()