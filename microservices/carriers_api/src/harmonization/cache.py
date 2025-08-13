"""
Harmonization cache management.
"""

import os
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path
import hashlib
import json

from ..storage.repository import StorageRepository
from ..models.carrier import HarmonizationMapping


class HarmonizationCache:
    """
    Manages precomputed harmonization mappings.
    
    The cache stores mappings between reference variants and genotype file variants
    to speed up repeated harmonization operations.
    """
    
    def __init__(self, storage: StorageRepository, cache_dir: str):
        self.storage = storage
        self.cache_dir = Path(cache_dir)
        self._memory_cache: Dict[str, pd.DataFrame] = {}
    
    def _get_cache_key(self, dataset_type: str, release: str, 
                      ancestry: Optional[str] = None, 
                      chromosome: Optional[str] = None,
                      geno_prefix: Optional[str] = None) -> str:
        """Generate unique cache key for a dataset configuration."""
        parts = [dataset_type, release]
        if ancestry:
            parts.append(ancestry)
        if chromosome:
            parts.append(f"chr{chromosome}")
        if geno_prefix:
            # Use hash of prefix to keep key short
            prefix_hash = hashlib.md5(geno_prefix.encode()).hexdigest()[:8]
            parts.append(prefix_hash)
        return "_".join(parts)
    
    def _get_cache_path(self, dataset_type: str, release: str) -> str:
        """Get cache file path for a dataset type and release."""
        filename = f"{dataset_type}_release{release}_harmonization_map.parquet"
        return str(self.cache_dir / filename)
    
    async def get_mapping(self, dataset_key: str) -> Optional[pd.DataFrame]:
        """
        Get cached mapping if available.
        
        Args:
            dataset_key: Unique key for the dataset configuration
            
        Returns:
            Cached mapping DataFrame or None if not found
        """
        # Check memory cache first
        if dataset_key in self._memory_cache:
            return self._memory_cache[dataset_key].copy()
        
        # Try to load from disk
        # This would need to filter the full cache file for the specific key
        return None
    
    async def get_full_mapping(self, dataset_type: str, release: str) -> pd.DataFrame:
        """
        Get full mapping table for a dataset type and release.
        
        Args:
            dataset_type: Type of dataset ('nba', 'wgs', 'imputed')
            release: Release version
            
        Returns:
            Full mapping DataFrame or empty DataFrame if not found
        """
        cache_path = self._get_cache_path(dataset_type, release)
        
        if await self.storage.exists(cache_path):
            df = await self.storage.read_parquet(cache_path)
            # Store in memory cache for faster access
            self._memory_cache[f"{dataset_type}_{release}_full"] = df
            return df
        
        return pd.DataFrame()
    
    async def update_mapping(self, dataset_type: str, release: str, 
                           new_mappings: List[HarmonizationMapping]):
        """
        Update cache with new harmonization mappings.
        
        Args:
            dataset_type: Type of dataset
            release: Release version
            new_mappings: List of new mappings to add
        """
        if not new_mappings:
            return
        
        # Convert mappings to DataFrame
        new_df = pd.DataFrame([m.to_dict() for m in new_mappings])
        
        # Get existing mappings
        existing_df = await self.get_full_mapping(dataset_type, release)
        
        # Combine and deduplicate
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates based on key columns
            key_cols = ['variant_id_ref', 'variant_id_geno', 'ancestry', 'chromosome']
            key_cols = [col for col in key_cols if col in combined_df.columns]
            combined_df = combined_df.drop_duplicates(subset=key_cols, keep='last')
        else:
            combined_df = new_df
        
        # Save updated cache
        cache_path = self._get_cache_path(dataset_type, release)
        await self.storage.write_parquet(combined_df, cache_path)
        
        # Update memory cache
        self._memory_cache[f"{dataset_type}_{release}_full"] = combined_df
    
    async def get_filtered_mapping(self, dataset_type: str, release: str,
                                 ancestry: Optional[str] = None,
                                 chromosome: Optional[str] = None,
                                 geno_prefix: Optional[str] = None,
                                 target_variants: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get filtered mapping for specific dataset configuration.
        
        Args:
            dataset_type: Type of dataset
            release: Release version
            ancestry: Filter by ancestry
            chromosome: Filter by chromosome
            geno_prefix: Filter by genotype file prefix
            target_variants: Filter by specific variant IDs
            
        Returns:
            Filtered mapping DataFrame
        """
        # Get full mapping
        full_mapping = await self.get_full_mapping(dataset_type, release)
        
        if full_mapping.empty:
            return full_mapping
        
        # Apply filters
        mask = pd.Series([True] * len(full_mapping))
        
        if ancestry is not None:
            mask &= (full_mapping['ancestry'] == ancestry)
        
        if chromosome is not None:
            mask &= (full_mapping['chromosome'] == chromosome)
        
        if geno_prefix is not None:
            mask &= (full_mapping['geno_prefix'] == geno_prefix)
        
        if target_variants is not None:
            mask &= full_mapping['variant_id_ref'].isin(target_variants)
        
        return full_mapping[mask].copy()
    
    def clear_memory_cache(self):
        """Clear in-memory cache."""
        self._memory_cache.clear()
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "memory_cache_size": len(self._memory_cache),
            "cache_files": []
        }
        
        # List cache files
        if await self.storage.exists(str(self.cache_dir)):
            files = await self.storage.list_files(str(self.cache_dir), "*.parquet")
            for file in files:
                file_path = self.cache_dir / file
                if await self.storage.exists(str(file_path)):
                    # Get file info
                    df = await self.storage.read_parquet(str(file_path))
                    stats["cache_files"].append({
                        "file": file,
                        "rows": len(df),
                        "columns": list(df.columns)
                    })
        
        return stats
