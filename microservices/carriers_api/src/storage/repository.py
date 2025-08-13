"""
Abstract storage repository interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd
from pathlib import Path


class StorageRepository(ABC):
    """Abstract interface for storage operations."""
    
    @abstractmethod
    async def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file."""
        pass
    
    @abstractmethod
    async def write_csv(self, df: pd.DataFrame, path: str, **kwargs):
        """Write CSV file."""
        pass
    
    @abstractmethod
    async def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet file."""
        pass
    
    @abstractmethod
    async def write_parquet(self, df: pd.DataFrame, path: str, **kwargs):
        """Write Parquet file."""
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory matching pattern."""
        pass
    
    @abstractmethod
    async def makedirs(self, path: str):
        """Create directory and parents if needed."""
        pass
    
    @abstractmethod
    async def delete(self, path: str):
        """Delete file or directory."""
        pass
    
    @abstractmethod
    async def copy(self, src: str, dst: str):
        """Copy file from source to destination."""
        pass
    
    @abstractmethod
    async def move(self, src: str, dst: str):
        """Move file from source to destination."""
        pass


class CachedStorageRepository(StorageRepository):
    """Storage repository with caching capabilities."""
    
    def __init__(self, base_repository: StorageRepository):
        self.base = base_repository
        self._cache: Dict[str, pd.DataFrame] = {}
    
    async def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Read CSV with caching."""
        cache_key = f"csv:{path}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        df = await self.base.read_csv(path, **kwargs)
        self._cache[cache_key] = df.copy()
        return df
    
    async def write_csv(self, df: pd.DataFrame, path: str, **kwargs):
        """Write CSV and update cache."""
        await self.base.write_csv(df, path, **kwargs)
        cache_key = f"csv:{path}"
        self._cache[cache_key] = df.copy()
    
    async def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet with caching."""
        cache_key = f"parquet:{path}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        df = await self.base.read_parquet(path, **kwargs)
        self._cache[cache_key] = df.copy()
        return df
    
    async def write_parquet(self, df: pd.DataFrame, path: str, **kwargs):
        """Write Parquet and update cache."""
        await self.base.write_parquet(df, path, **kwargs)
        cache_key = f"parquet:{path}"
        self._cache[cache_key] = df.copy()
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        return await self.base.exists(path)
    
    async def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory."""
        return await self.base.list_files(directory, pattern)
    
    async def makedirs(self, path: str):
        """Create directory."""
        await self.base.makedirs(path)
    
    async def delete(self, path: str):
        """Delete file and clear from cache."""
        await self.base.delete(path)
        # Clear from cache
        for key in list(self._cache.keys()):
            if path in key:
                del self._cache[key]
    
    async def copy(self, src: str, dst: str):
        """Copy file."""
        await self.base.copy(src, dst)
    
    async def move(self, src: str, dst: str):
        """Move file and update cache."""
        await self.base.move(src, dst)
        # Update cache keys
        for key in list(self._cache.keys()):
            if src in key:
                new_key = key.replace(src, dst)
                self._cache[new_key] = self._cache.pop(key)
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)
