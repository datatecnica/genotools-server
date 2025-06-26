"""
Sample repository for GP2 Precision Medicine Data Browser.
Integrates clinical metadata with carrier data.
"""
from typing import List, Optional, Dict, Any
import logging

from .clinical_repo import ClinicalRepository
from .variant_repo import CarrierDataRepository
from ..models.sample import SampleCarrier, SampleFilterCriteria, SampleCarrierSummary
from ..models.clinical import ClinicalMetadata

logger = logging.getLogger(__name__)


class SampleRepository:
    """
    Repository for integrated sample data (clinical + carrier).
    
    This repository combines data from multiple sources:
    - Clinical metadata from master key
    - Carrier data from NBA and WGS sources
    """
    
    def __init__(self):
        """Initialize sample repository with underlying data repositories."""
        self.clinical_repo = ClinicalRepository()
        
        # Initialize carrier data repositories for both sources
        self.nba_carrier_repo = CarrierDataRepository(data_source="NBA")
        self.wgs_carrier_repo = CarrierDataRepository(data_source="WGS")
        
        # Cache for clinical lookup
        self._clinical_lookup: Optional[Dict[str, ClinicalMetadata]] = None
        
        logger.info("Initialized sample repository")
    
    def _get_clinical_lookup(self) -> Dict[str, ClinicalMetadata]:
        """Get or create clinical lookup dictionary."""
        if self._clinical_lookup is None:
            self._clinical_lookup = self.clinical_repo.create_clinical_lookup()
        return self._clinical_lookup
    
    def _create_sample_carrier(self, sample_id: str, data_source: str, 
                              carriers: Dict[str, Optional[float]],
                              clinical: Optional[ClinicalMetadata] = None) -> SampleCarrier:
        """Create a SampleCarrier instance."""
        return SampleCarrier(
            sample_id=sample_id,
            data_source=data_source,
            carriers=carriers,
            clinical=clinical
        )
    
    def get_sample_by_id(self, sample_id: str, data_source: Optional[str] = None) -> Optional[SampleCarrier]:
        """
        Get a sample by ID, optionally specifying data source.
        
        Args:
            sample_id: Sample ID (IID format for carrier data, GP2ID for clinical data)
            data_source: Preferred data source ("NBA", "WGS", or None for auto-detect)
            
        Returns:
            SampleCarrier instance or None if not found
        """
        clinical_lookup = self._get_clinical_lookup()
        
        # Try to get clinical data
        clinical = clinical_lookup.get(sample_id)
        
        # Determine data source priority
        if data_source:
            sources_to_try = [data_source.upper()]
        elif clinical:
            # Prioritize based on clinical data availability
            sources_to_try = []
            if clinical.has_wgs_data:
                sources_to_try.append("WGS")
            if clinical.has_nba_data:
                sources_to_try.append("NBA")
        else:
            # Try both sources
            sources_to_try = ["WGS", "NBA"]
        
        # Try to get carrier data from preferred sources
        for source in sources_to_try:
            if source == "WGS":
                carrier_repo = self.wgs_carrier_repo
            else:
                carrier_repo = self.nba_carrier_repo
            
            carriers = carrier_repo.get_carrier_data_for_sample(sample_id)
            if carriers:
                return self._create_sample_carrier(sample_id, source, carriers, clinical)
        
        # If no carrier data found but have clinical data, return with empty carriers
        if clinical:
            return self._create_sample_carrier(sample_id, "None", {}, clinical)
        
        return None
    
    def get_samples_by_data_source(self, data_source: str, 
                                  limit: Optional[int] = None,
                                  offset: int = 0) -> List[SampleCarrier]:
        """
        Get samples from a specific data source.
        
        Args:
            data_source: Data source ("NBA" or "WGS")
            limit: Maximum number of samples to return
            offset: Number of samples to skip
            
        Returns:
            List of SampleCarrier instances
        """
        clinical_lookup = self._get_clinical_lookup()
        
        if data_source.upper() == "WGS":
            carrier_repo = self.wgs_carrier_repo
        elif data_source.upper() == "NBA":
            carrier_repo = self.nba_carrier_repo
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        # Get sample IDs from carrier repository
        try:
            sample_ids = carrier_repo.get_sample_ids()
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"{data_source} data not available: {e}")
            return []
        
        # Apply pagination
        if offset > 0:
            sample_ids = sample_ids[offset:]
        if limit:
            sample_ids = sample_ids[:limit]
        
        # Create SampleCarrier instances
        samples = []
        for sample_id in sample_ids:
            carriers = carrier_repo.get_carrier_data_for_sample(sample_id)
            clinical = clinical_lookup.get(sample_id)
            
            sample = self._create_sample_carrier(sample_id, data_source.upper(), carriers, clinical)
            samples.append(sample)
        
        return samples
    
    def filter_samples(self, filters: SampleFilterCriteria) -> List[SampleCarrier]:
        """
        Filter samples based on criteria.
        
        Args:
            filters: Filter criteria
            
        Returns:
            List of filtered SampleCarrier instances
        """
        clinical_lookup = self._get_clinical_lookup()
        all_samples = []
        
        # Determine which data sources to query
        sources_to_query = filters.data_sources or ["WGS", "NBA"]
        
        for data_source in sources_to_query:
            if data_source.upper() == "WGS":
                carrier_repo = self.wgs_carrier_repo
            elif data_source.upper() == "NBA":
                carrier_repo = self.nba_carrier_repo
            else:
                continue
            
            sample_ids = carrier_repo.get_sample_ids()
            
            for sample_id in sample_ids:
                carriers = carrier_repo.get_carrier_data_for_sample(sample_id)
                clinical = clinical_lookup.get(sample_id)
                
                sample = self._create_sample_carrier(sample_id, data_source.upper(), carriers, clinical)
                
                # Apply filters
                if self._matches_filters(sample, filters):
                    all_samples.append(sample)
        
        # Apply pagination
        if filters.offset > 0:
            all_samples = all_samples[filters.offset:]
        if filters.limit:
            all_samples = all_samples[:filters.limit]
        
        return all_samples
    
    def _matches_filters(self, sample: SampleCarrier, filters: SampleFilterCriteria) -> bool:
        """Check if a sample matches the filter criteria."""
        
        # Clinical data availability filter
        if filters.has_clinical_data is not None:
            if filters.has_clinical_data != sample.has_clinical_data:
                return False
        
        # GDPR compliance filter
        if filters.gdpr_compliant_only and sample.clinical:
            if sample.clinical.gdpr != 1:
                return False
        
        # Study filter
        if filters.studies and sample.clinical:
            if sample.clinical.study not in filters.studies:
                return False
        
        # Ancestry filter
        if filters.ancestry_labels and sample.clinical:
            primary_ancestry = sample.clinical.primary_ancestry_label
            if not primary_ancestry or primary_ancestry not in filters.ancestry_labels:
                return False
        
        # Diagnosis filter
        if filters.diagnoses and sample.clinical:
            if not sample.clinical.diagnosis or sample.clinical.diagnosis not in filters.diagnoses:
                return False
        
        # Biological sex filter
        if filters.biological_sex and sample.clinical:
            if (not sample.clinical.biological_sex_for_qc or 
                sample.clinical.biological_sex_for_qc not in filters.biological_sex):
                return False
        
        # Carried variants filter
        if filters.carried_variants:
            carried_variants = sample.get_carried_variants()
            if not any(variant in carried_variants for variant in filters.carried_variants):
                return False
        
        # Carrier count filters
        carrier_count = sample.get_carrier_count()
        
        if filters.min_carrier_count is not None:
            if carrier_count < filters.min_carrier_count:
                return False
        
        if filters.max_carrier_count is not None:
            if carrier_count > filters.max_carrier_count:
                return False
        
        return True
    
    def get_carriers_for_variant(self, variant_id: str, data_source: Optional[str] = None) -> List[SampleCarrier]:
        """
        Get all samples that are carriers for a specific variant.
        
        Args:
            variant_id: Variant ID
            data_source: Data source to search ("NBA", "WGS", or None for both)
            
        Returns:
            List of carrier SampleCarrier instances
        """
        clinical_lookup = self._get_clinical_lookup()
        carriers = []
        
        sources_to_search = [data_source] if data_source else ["WGS", "NBA"]
        
        for source in sources_to_search:
            if source.upper() == "WGS":
                carrier_repo = self.wgs_carrier_repo
            elif source.upper() == "NBA":
                carrier_repo = self.nba_carrier_repo
            else:
                continue
            
            carrier_sample_ids = carrier_repo.get_carriers_for_variant(variant_id)
            
            for sample_id in carrier_sample_ids:
                carrier_data = carrier_repo.get_carrier_data_for_sample(sample_id)
                clinical = clinical_lookup.get(sample_id)
                
                sample = self._create_sample_carrier(sample_id, source.upper(), carrier_data, clinical)
                carriers.append(sample)
        
        return carriers
    
    def get_sample_summary(self) -> SampleCarrierSummary:
        """Get summary statistics for all samples."""
        clinical_lookup = self._get_clinical_lookup()
        
        # Get sample counts by source
        wgs_sample_ids = set(self.wgs_carrier_repo.get_sample_ids())
        
        # Try to get NBA sample IDs, but handle missing files gracefully
        try:
            nba_sample_ids = set(self.nba_carrier_repo.get_sample_ids())
        except (FileNotFoundError, Exception) as e:
            logger.warning(f"NBA data not available: {e}")
            nba_sample_ids = set()
        all_sample_ids = wgs_sample_ids.union(nba_sample_ids)
        
        # Count samples with clinical data
        samples_with_clinical = sum(1 for sid in all_sample_ids if sid in clinical_lookup)
        
        # Count by ancestry and study
        ancestry_counts = {}
        study_counts = {}
        
        for sample_id in all_sample_ids:
            clinical = clinical_lookup.get(sample_id)
            if clinical:
                # Ancestry counts
                ancestry = clinical.primary_ancestry_label
                if ancestry:
                    ancestry_counts[ancestry] = ancestry_counts.get(ancestry, 0) + 1
                
                # Study counts
                study = clinical.study
                if study:
                    study_counts[study] = study_counts.get(study, 0) + 1
        
        # Calculate average variants per sample (simplified for small datasets)
        total_carrier_observations = 0
        sample_count = 0
        
        # Process a reasonable subset for small datasets
        sample_subset = list(all_sample_ids)[:100] if len(all_sample_ids) > 100 else list(all_sample_ids)
        
        for sample_id in sample_subset:
            for source in ["WGS", "NBA"]:
                repo = self.wgs_carrier_repo if source == "WGS" else self.nba_carrier_repo
                try:
                    available_sample_ids = repo.get_sample_ids()
                    if sample_id in available_sample_ids:
                        carriers = repo.get_carrier_data_for_sample(sample_id)
                        total_carrier_observations += sum(1 for status in carriers.values() if status and status > 0)
                        sample_count += 1
                        break
                except (FileNotFoundError, Exception):
                    # Skip this source if data is not available
                    continue
        
        avg_variants_per_sample = total_carrier_observations / sample_count if sample_count > 0 else 0
        
        return SampleCarrierSummary(
            total_samples=len(all_sample_ids),
            samples_with_clinical=samples_with_clinical,
            samples_by_source={
                "WGS": len(wgs_sample_ids),
                "NBA": len(nba_sample_ids),
                "Both": len(wgs_sample_ids.intersection(nba_sample_ids))
            },
            samples_by_ancestry=ancestry_counts,
            samples_by_study=study_counts,
            avg_variants_per_sample=avg_variants_per_sample,
            total_carrier_observations=total_carrier_observations
        )
    
    def count_samples(self, filters: Optional[SampleFilterCriteria] = None) -> int:
        """
        Count samples matching filters.
        
        Args:
            filters: Optional filter criteria
            
        Returns:
            Number of matching samples
        """
        if filters:
            return len(self.filter_samples(filters))
        else:
            # Count all unique samples across both data sources
            wgs_sample_ids = set(self.wgs_carrier_repo.get_sample_ids())
            try:
                nba_sample_ids = set(self.nba_carrier_repo.get_sample_ids())
            except (FileNotFoundError, Exception):
                nba_sample_ids = set()
            return len(wgs_sample_ids.union(nba_sample_ids))
    
    def clear_cache(self):
        """Clear cached data."""
        self._clinical_lookup = None
        logger.info("Sample repository cache cleared") 