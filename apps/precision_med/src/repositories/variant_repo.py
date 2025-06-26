"""
Variant repository for GP2 Precision Medicine Data Browser.
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import logging

from .base import BaseRepository
from ..models.variant import VariantInfo, FilterCriteria, VariantCarrierData
from ..core.config import settings

logger = logging.getLogger(__name__)


class VariantRepository(BaseRepository[VariantInfo, FilterCriteria]):
    """Repository for variant information data."""
    
    def __init__(self, data_source: str = "WGS"):
        """
        Initialize variant repository.
        
        Args:
            data_source: Data source ("WGS" or "NBA")
        """
        self.data_source = data_source.upper()
        
        if self.data_source == "WGS":
            data_path = settings.wgs_var_info_path
        elif self.data_source == "NBA":
            data_path = settings.nba_info_path
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        super().__init__(data_path)
        logger.info(f"Initialized {self.data_source} variant repository")
    
    def _load_data(self) -> pd.DataFrame:
        """Load variant info data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Variant info file not found: {self.data_path}")
        
        # Load CSV with appropriate data types
        dtype_mapping = {
            'id': 'string',
            'snp_name': 'string',
            'snp_name_alt': 'string',
            'locus': 'string',
            'rsid': 'string',
            'hg38': 'string',
            'hg19': 'string',
            'chrom': 'string',
            'pos': 'int64',
            'a1': 'string',
            'a2': 'string',
            'ancestry': 'string',
            'submitter_email': 'string',
            'precision_medicine': 'string',
            'pipeline': 'string'
        }
        
        data = pd.read_csv(self.data_path, dtype=dtype_mapping, low_memory=False)
        
        # Clean up column names and handle missing values
        data = data.rename(columns={
            'ALT_FREQS': 'alt_freqs',
            'OBS_CT': 'obs_ct',
            'F_MISS': 'f_miss'
        })
        
        # Convert numeric columns
        for col in ['alt_freqs', 'obs_ct', 'f_miss']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        return data
    
    def _model_from_row(self, row: pd.Series) -> VariantInfo:
        """Convert DataFrame row to VariantInfo model."""
        return VariantInfo(**row.to_dict())
    
    def _apply_filters(self, data: pd.DataFrame, filters: FilterCriteria) -> pd.DataFrame:
        """Apply filtering criteria to variant data."""
        filtered = data.copy()
        
        # Filter by loci/genes
        if filters.loci:
            filtered = filtered[filtered['locus'].isin(filters.loci)]
        
        # Quality filters
        if filters.min_obs_ct is not None:
            filtered = filtered[filtered['obs_ct'] >= filters.min_obs_ct]
        
        if filters.max_f_miss is not None:
            filtered = filtered[filtered['f_miss'] <= filters.max_f_miss]
        
        # Frequency filters
        if filters.min_alt_freq is not None:
            filtered = filtered[filtered['alt_freqs'] >= filters.min_alt_freq]
        
        if filters.max_alt_freq is not None:
            filtered = filtered[filtered['alt_freqs'] <= filters.max_alt_freq]
        
        # Ancestry filter
        if filters.ancestry:
            # Handle multiple ancestry annotations (e.g., "EUR (multi)")
            ancestry_mask = filtered['ancestry'].str.contains(
                '|'.join(filters.ancestry), 
                case=False, 
                na=False
            )
            filtered = filtered[ancestry_mask]
        
        # Precision medicine filter
        if filters.precision_medicine_only:
            filtered = filtered[filtered['precision_medicine'].notna()]
        
        return filtered
    
    def get_by_locus(self, locus: str) -> List[VariantInfo]:
        """Get all variants for a specific gene/locus."""
        filters = FilterCriteria(loci=[locus])
        return self.filter(filters)
    
    def get_precision_medicine_variants(self) -> List[VariantInfo]:
        """Get all precision medicine variants."""
        filters = FilterCriteria(precision_medicine_only=True)
        return self.filter(filters)
    
    def get_loci(self) -> List[str]:
        """Get list of all available loci/genes."""
        return self.get_unique_values('locus')
    
    def get_ancestry_labels(self) -> List[str]:
        """Get list of all ancestry labels."""
        return self.get_unique_values('ancestry')
    
    def get_variant_stats(self) -> Dict[str, Any]:
        """Get summary statistics for variants."""
        self.load()
        
        stats = self.get_summary_stats()
        
        # Add variant-specific statistics
        stats.update({
            "total_variants": len(self._data),
            "unique_loci": self._data['locus'].nunique(),
            "precision_medicine_variants": self._data['precision_medicine'].notna().sum(),
            "variants_by_chromosome": self._data['chrom'].value_counts().to_dict(),
            "variants_by_locus": self._data['locus'].value_counts().head(10).to_dict(),
        })
        
        return stats


class CarrierDataRepository(BaseRepository[VariantCarrierData, None]):
    """Repository for variant carrier data (sample-level genotype data)."""
    
    def __init__(self, data_source: str = "WGS", use_string_format: bool = False):
        """
        Initialize carrier data repository.
        
        Args:
            data_source: Data source ("WGS" or "NBA")
            use_string_format: Use string format file instead of integer format
        """
        self.data_source = data_source.upper()
        self.use_string_format = use_string_format
        
        if self.data_source == "WGS":
            if use_string_format:
                data_path = settings.wgs_carriers_string_path
            else:
                data_path = settings.wgs_carriers_int_path
        elif self.data_source == "NBA":
            if use_string_format:
                data_path = settings.nba_carriers_string_path
            else:
                data_path = settings.nba_carriers_int_path
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        super().__init__(data_path)
        self._variant_columns: Optional[List[str]] = None
        logger.info(f"Initialized {self.data_source} carrier data repository")
    
    def _load_data(self) -> pd.DataFrame:
        """Load carrier data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Carrier data file not found: {self.data_path}")
        
        # Load entire file for small datasets
        data = pd.read_csv(self.data_path, low_memory=False)
        
        # Set IID as index for efficient lookups
        data.set_index('IID', inplace=True)
        
        # Store variant column names (all columns except IID)
        self._variant_columns = list(data.columns)
        
        # Convert to numeric if using integer format
        if not self.use_string_format:
            data = data.apply(pd.to_numeric, errors='coerce')
        
        return data
    
    def _model_from_row(self, row: pd.Series) -> VariantCarrierData:
        """Convert DataFrame row to VariantCarrierData model."""
        # This repository returns raw data, not individual carrier records
        # Use get_carrier_data_for_sample method instead
        raise NotImplementedError("Use get_carrier_data_for_sample method")
    
    def _apply_filters(self, data: pd.DataFrame, filters: None) -> pd.DataFrame:
        """Carrier data doesn't use traditional filtering."""
        return data
    
    def get_carrier_data_for_sample(self, sample_id: str) -> Dict[str, Optional[float]]:
        """
        Get carrier status for all variants for a specific sample.
        
        Args:
            sample_id: Sample ID (IID)
            
        Returns:
            Dictionary mapping variant ID to carrier status
        """
        self.load()
        
        if sample_id not in self._data.index:
            return {}
        
        sample_data = self._data.loc[sample_id]
        return sample_data.to_dict()
    
    def get_carrier_data_for_variant(self, variant_id: str) -> Dict[str, Optional[float]]:
        """
        Get carrier status for all samples for a specific variant.
        
        Args:
            variant_id: Variant ID
            
        Returns:
            Dictionary mapping sample ID to carrier status
        """
        self.load()
        
        if variant_id not in self._data.columns:
            return {}
        
        variant_data = self._data[variant_id].dropna()
        return variant_data.to_dict()
    
    def get_carriers_for_variant(self, variant_id: str) -> List[str]:
        """
        Get list of sample IDs that are carriers for a specific variant.
        
        Args:
            variant_id: Variant ID
            
        Returns:
            List of sample IDs
        """
        carrier_data = self.get_carrier_data_for_variant(variant_id)
        return [sample_id for sample_id, status in carrier_data.items() if status and status > 0.0]
    
    def get_variant_ids(self) -> List[str]:
        """Get list of all variant IDs in the dataset."""
        self.load()
        return self._variant_columns or []
    
    def get_sample_ids(self) -> List[str]:
        """Get list of all sample IDs in the dataset."""
        self.load()
        return list(self._data.index)
    
    def get_carrier_stats(self) -> Dict[str, Any]:
        """Get summary statistics for carrier data."""
        self.load()
        
        total_samples = len(self._data)
        total_variants = len(self._data.columns)
        
        # Calculate carrier counts per variant
        carrier_counts = {}
        for variant_id in self._data.columns:
            variant_data = self._data[variant_id]
            carrier_counts[variant_id] = {
                "total_carriers": (variant_data > 0).sum(),
                "heterozygous": (variant_data == 1.0).sum(),
                "homozygous": (variant_data == 2.0).sum(),
                "missing": variant_data.isnull().sum()
            }
        
        return {
            "total_samples": total_samples,
            "total_variants": total_variants,
            "memory_usage_mb": self._data.memory_usage(deep=True).sum() / 1024 / 1024,
            "carrier_counts": carrier_counts
        } 