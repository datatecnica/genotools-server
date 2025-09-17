"""
File system utilities for frontend operations.
"""

import os
from typing import Dict, List, Optional, Tuple


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: File size in bytes

    Returns:
        str: Formatted file size (e.g., '1.2 MB', '345 KB')
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def get_file_stats(file_path: str) -> Optional[Dict[str, any]]:
    """
    Get file statistics including size and modification time.

    Args:
        file_path: Path to file

    Returns:
        Dict with file stats or None if file doesn't exist
    """
    if not os.path.exists(file_path):
        return None

    try:
        stat = os.stat(file_path)
        return {
            'size_bytes': stat.st_size,
            'size_mb': round(stat.st_size / 1024 / 1024, 1),
            'size_formatted': format_file_size(stat.st_size),
            'modified_time': stat.st_mtime,
            'exists': True
        }
    except Exception:
        return None


def find_files_by_pattern(directory: str, patterns: List[str]) -> List[str]:
    """
    Find files matching given patterns in directory.

    Args:
        directory: Directory to search
        patterns: List of file patterns to match

    Returns:
        List of matching file paths
    """
    if not os.path.exists(directory):
        return []

    matching_files = []
    try:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                for pattern in patterns:
                    if pattern in file:
                        matching_files.append(file_path)
                        break
    except Exception:
        pass

    return matching_files


def validate_path_exists(path: str) -> bool:
    """
    Validate that a path exists and is accessible.

    Args:
        path: Path to validate

    Returns:
        bool: True if path exists and is accessible
    """
    try:
        return os.path.exists(path) and os.access(path, os.R_OK)
    except Exception:
        return False


def get_directory_size(directory: str) -> int:
    """
    Calculate total size of all files in directory.

    Args:
        directory: Directory path

    Returns:
        int: Total size in bytes
    """
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
    except Exception:
        pass

    return total_size


def parse_job_name_from_filename(filename: str, data_types: List[str]) -> Optional[Tuple[str, str]]:
    """
    Parse job name and data type from filename.

    Args:
        filename: Filename to parse
        data_types: List of valid data types

    Returns:
        Tuple of (job_name, data_type) or None if parsing fails
    """
    if filename.endswith('.parquet'):
        # Format: {job_name}_{data_type}.parquet
        parts = filename.replace('.parquet', '').split('_')
        if len(parts) >= 2 and parts[-1] in data_types:
            job_name = '_'.join(parts[:-1])
            data_type = parts[-1]
            return job_name, data_type

    elif filename.endswith('_pipeline_results.json'):
        # Format: {job_name}_pipeline_results.json
        job_name = filename.replace('_pipeline_results.json', '')
        return job_name, 'pipeline_results'

    return None