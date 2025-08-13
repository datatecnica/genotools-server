"""
Cloud storage implementation (Google Cloud Storage).
"""

from typing import List
import pandas as pd
from pathlib import Path

from .file_storage import FileStorageRepository


class CloudStorageRepository(FileStorageRepository):
    """
    Cloud storage implementation.
    
    Currently extends FileStorageRepository to work with GCS-mounted paths.
    In the future, this could be enhanced to use native GCS APIs for better performance.
    """
    
    def __init__(self, gcs_mount_path: str = "~/gcs_mounts"):
        """
        Initialize cloud storage repository.
        
        Args:
            gcs_mount_path: Path where GCS buckets are mounted
        """
        # Expand user path and use as base
        super().__init__(Path(gcs_mount_path).expanduser())
    
    def _get_bucket_and_path(self, path: str) -> tuple[str, str]:
        """
        Extract bucket name and path from a GCS path.
        
        Args:
            path: Path that may include bucket name
            
        Returns:
            Tuple of (bucket_name, path_within_bucket)
        """
        path_parts = Path(path).parts
        if len(path_parts) > 0:
            bucket = path_parts[0]
            remaining_path = str(Path(*path_parts[1:])) if len(path_parts) > 1 else ""
            return bucket, remaining_path
        return "", path
    
    # Override methods here if needed for native GCS operations
    # For now, we rely on the gcsfuse mount and use file operations
    
    async def validate_mount(self) -> bool:
        """Validate that GCS mount is accessible."""
        return await self.exists(".")
    
    def get_gcs_uri(self, path: str) -> str:
        """
        Convert local path to GCS URI format.
        
        Args:
            path: Local path within mount
            
        Returns:
            GCS URI (gs://bucket/path)
        """
        bucket, bucket_path = self._get_bucket_and_path(path)
        if bucket:
            return f"gs://{bucket}/{bucket_path}"
        return path
