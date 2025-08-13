"""
Local file system storage implementation.
"""

import os
import shutil
import glob
import asyncio
from typing import List
from pathlib import Path
import pandas as pd

from .repository import StorageRepository


class FileStorageRepository(StorageRepository):
    """Local file system storage implementation."""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path.cwd()
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to base path."""
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj
        return self.base_path / path_obj
    
    async def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """Read CSV file."""
        resolved_path = self._resolve_path(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, pd.read_csv, resolved_path, kwargs)
    
    async def write_csv(self, df: pd.DataFrame, path: str, **kwargs):
        """Write CSV file."""
        resolved_path = self._resolve_path(path)
        # Ensure directory exists
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, df.to_csv, resolved_path, False, **kwargs)
    
    async def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """Read Parquet file."""
        resolved_path = self._resolve_path(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, pd.read_parquet, resolved_path, **kwargs)
    
    async def write_parquet(self, df: pd.DataFrame, path: str, **kwargs):
        """Write Parquet file."""
        resolved_path = self._resolve_path(path)
        # Ensure directory exists
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, df.to_parquet, resolved_path, False, **kwargs)
    
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        resolved_path = self._resolve_path(path)
        return resolved_path.exists()
    
    async def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in directory matching pattern."""
        resolved_dir = self._resolve_path(directory)
        if not resolved_dir.exists():
            return []
        
        # Use glob to find matching files
        pattern_path = resolved_dir / pattern
        files = glob.glob(str(pattern_path))
        
        # Return relative paths
        return [str(Path(f).relative_to(self.base_path)) for f in files]
    
    async def makedirs(self, path: str):
        """Create directory and parents if needed."""
        resolved_path = self._resolve_path(path)
        resolved_path.mkdir(parents=True, exist_ok=True)
    
    async def delete(self, path: str):
        """Delete file or directory."""
        resolved_path = self._resolve_path(path)
        if resolved_path.is_file():
            resolved_path.unlink()
        elif resolved_path.is_dir():
            shutil.rmtree(resolved_path)
    
    async def copy(self, src: str, dst: str):
        """Copy file from source to destination."""
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.copy2, src_path, dst_path)
    
    async def move(self, src: str, dst: str):
        """Move file from source to destination."""
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, shutil.move, str(src_path), str(dst_path))
