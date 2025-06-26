"""
Clinical repository for GP2 Precision Medicine Data Browser.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import logging

from .base import BaseRepository
from ..models.clinical import ClinicalMetadata, ClinicalFilterCriteria
from ..core.config import settings

logger = logging.getLogger(__name__)


class ClinicalRepository(BaseRepository[ClinicalMetadata, ClinicalFilterCriteria]):
    """Repository for clinical metadata from master key."""
    
    def __init__(self):
        """Initialize clinical repository."""
        super().__init__(settings.clinical_master_key_path)
        logger.info("Initialized clinical repository")
    
    def _load_data(self) -> pd.DataFrame:
        """Load clinical data from master key CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Clinical master key file not found: {self.data_path}")
        
        # Define data types for key columns
        dtype_mapping = {
            'GP2ID': 'string',
            'study': 'string',
            'nba': 'int8',
            'wgs': 'int8',
            'clinical_exome': 'int8',
            'extended_clinical_data': 'int8',
            'GDPR': 'int8',
            'nba_prune_reason': 'string',
            'nba_related': 'Int8',  # Nullable integer
            'nba_label': 'string',
            'wgs_prune_reason': 'string',
            'wgs_label': 'string',
            'study_arm': 'string',
            'study_type': 'string',
            'diagnosis': 'string',
            'baseline_GP2_phenotype_for_qc': 'string',
            'baseline_GP2_phenotype': 'string',
            'biological_sex_for_qc': 'string'
        }
        
        # Load CSV
        data = pd.read_csv(self.data_path, dtype=dtype_mapping, low_memory=False)
        
        # Convert numeric columns with missing values
        numeric_cols = ['age_at_sample_collection', 'age_of_onset']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Set GP2ID as index for efficient lookups
        if 'GP2ID' in data.columns:
            data.set_index('GP2ID', inplace=True)
        
        return data
    
    def _model_from_row(self, row: pd.Series) -> ClinicalMetadata:
        """Convert DataFrame row to ClinicalMetadata model."""
        # Add the index (GP2ID) back to the row data
        row_data = row.to_dict()
        row_data['GP2ID'] = row.name
        return ClinicalMetadata(**row_data)
    
    def _apply_filters(self, data: pd.DataFrame, filters: ClinicalFilterCriteria) -> pd.DataFrame:
        """Apply filtering criteria to clinical data."""
        filtered = data.copy()
        
        # Study filters
        if filters.studies:
            filtered = filtered[filtered['study'].isin(filters.studies)]
        
        # Data availability filters
        if filters.has_nba is not None:
            filtered = filtered[filtered['nba'] == (1 if filters.has_nba else 0)]
        
        if filters.has_wgs is not None:
            filtered = filtered[filtered['wgs'] == (1 if filters.has_wgs else 0)]
        
        if filters.has_clinical_exome is not None:
            filtered = filtered[filtered['clinical_exome'] == (1 if filters.has_clinical_exome else 0)]
        
        # Ancestry filters
        if filters.nba_labels:
            filtered = filtered[filtered['nba_label'].isin(filters.nba_labels)]
        
        if filters.wgs_labels:
            filtered = filtered[filtered['wgs_label'].isin(filters.wgs_labels)]
        
        if filters.ancestry_labels:
            # Filter by either NBA or WGS ancestry labels
            ancestry_mask = (
                filtered['nba_label'].isin(filters.ancestry_labels) |
                filtered['wgs_label'].isin(filters.ancestry_labels)
            )
            filtered = filtered[ancestry_mask]
        
        # Clinical filters
        if filters.diagnoses:
            filtered = filtered[filtered['diagnosis'].isin(filters.diagnoses)]
        
        if filters.study_arms:
            filtered = filtered[filtered['study_arm'].isin(filters.study_arms)]
        
        if filters.study_types:
            filtered = filtered[filtered['study_type'].isin(filters.study_types)]
        
        # Demographics filters
        if filters.biological_sex:
            filtered = filtered[filtered['biological_sex_for_qc'].isin(filters.biological_sex)]
        
        if filters.min_age_at_collection is not None:
            filtered = filtered[filtered['age_at_sample_collection'] >= filters.min_age_at_collection]
        
        if filters.max_age_at_collection is not None:
            filtered = filtered[filtered['age_at_sample_collection'] <= filters.max_age_at_collection]
        
        if filters.min_age_of_onset is not None:
            filtered = filtered[filtered['age_of_onset'] >= filters.min_age_of_onset]
        
        if filters.max_age_of_onset is not None:
            filtered = filtered[filtered['age_of_onset'] <= filters.max_age_of_onset]
        
        # GDPR compliance
        if filters.gdpr_compliant_only:
            filtered = filtered[filtered['GDPR'] == 1]
        
        return filtered
    
    def get_by_gp2_id(self, gp2_id: str) -> Optional[ClinicalMetadata]:
        """Get clinical metadata by GP2 ID."""
        return self.get_by_id(gp2_id, id_column='GP2ID')
    
    def get_samples_with_nba_data(self) -> List[ClinicalMetadata]:
        """Get all samples with NBA data available."""
        filters = ClinicalFilterCriteria(has_nba=True)
        return self.filter(filters)
    
    def get_samples_with_wgs_data(self) -> List[ClinicalMetadata]:
        """Get all samples with WGS data available."""
        filters = ClinicalFilterCriteria(has_wgs=True)
        return self.filter(filters)
    
    def get_by_study(self, study: str) -> List[ClinicalMetadata]:
        """Get all samples from a specific study."""
        filters = ClinicalFilterCriteria(studies=[study])
        return self.filter(filters)
    
    def get_by_ancestry(self, ancestry_label: str) -> List[ClinicalMetadata]:
        """Get all samples with a specific ancestry label."""
        filters = ClinicalFilterCriteria(ancestry_labels=[ancestry_label])
        return self.filter(filters)
    
    def get_studies(self) -> List[str]:
        """Get list of all available studies."""
        return self.get_unique_values('study')
    
    def get_nba_ancestry_labels(self) -> List[str]:
        """Get list of all NBA ancestry labels."""
        return self.get_unique_values('nba_label')
    
    def get_wgs_ancestry_labels(self) -> List[str]:
        """Get list of all WGS ancestry labels."""
        return self.get_unique_values('wgs_label')
    
    def get_all_ancestry_labels(self) -> List[str]:
        """Get combined list of all ancestry labels (NBA and WGS)."""
        nba_labels = set(self.get_nba_ancestry_labels())
        wgs_labels = set(self.get_wgs_ancestry_labels())
        return sorted(list(nba_labels.union(wgs_labels)))
    
    def get_diagnoses(self) -> List[str]:
        """Get list of all diagnoses."""
        return self.get_unique_values('diagnosis')
    
    def get_clinical_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of clinical data."""
        self.load()
        
        summary = {
            "total_samples": len(self._data),
            "samples_with_nba": (self._data['nba'] == 1).sum(),
            "samples_with_wgs": (self._data['wgs'] == 1).sum(),
            "samples_with_clinical_exome": (self._data['clinical_exome'] == 1).sum(),
            "gdpr_compliant": (self._data['GDPR'] == 1).sum(),
            
            # Study breakdown
            "samples_by_study": self._data['study'].value_counts().to_dict(),
            
            # Ancestry breakdown
            "samples_by_nba_ancestry": self._data['nba_label'].value_counts().to_dict(),
            "samples_by_wgs_ancestry": self._data['wgs_label'].value_counts().to_dict(),
            
            # Clinical characteristics
            "samples_by_diagnosis": self._data['diagnosis'].value_counts().to_dict(),
            "samples_by_sex": self._data['biological_sex_for_qc'].value_counts().to_dict(),
            
            # Age statistics
            "age_at_collection_stats": {
                "mean": self._data['age_at_sample_collection'].mean(),
                "std": self._data['age_at_sample_collection'].std(),
                "min": self._data['age_at_sample_collection'].min(),
                "max": self._data['age_at_sample_collection'].max(),
                "missing": self._data['age_at_sample_collection'].isnull().sum()
            },
            
            "age_of_onset_stats": {
                "mean": self._data['age_of_onset'].mean(),
                "std": self._data['age_of_onset'].std(),
                "min": self._data['age_of_onset'].min(),
                "max": self._data['age_of_onset'].max(),
                "missing": self._data['age_of_onset'].isnull().sum()
            }
        }
        
        return summary
    
    def get_sample_ids_for_data_source(self, data_source: str) -> List[str]:
        """
        Get sample IDs that have data available for a specific source.
        
        Args:
            data_source: "NBA" or "WGS"
            
        Returns:
            List of GP2 IDs
        """
        self.load()
        
        if data_source.upper() == "NBA":
            mask = self._data['nba'] == 1
        elif data_source.upper() == "WGS":
            mask = self._data['wgs'] == 1
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        return list(self._data[mask].index)
    
    def create_clinical_lookup(self) -> Dict[str, ClinicalMetadata]:
        """
        Create a lookup dictionary for efficient clinical data access.
        
        Returns:
            Dictionary mapping GP2ID to ClinicalMetadata
        """
        self.load()
        
        lookup = {}
        for gp2_id, row in self._data.iterrows():
            lookup[gp2_id] = self._model_from_row(row)
        
        return lookup 