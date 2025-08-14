"""
Simplified file operations for gcsfuse mounted paths.
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, Any


def read_csv(path: str, **kwargs) -> pd.DataFrame:
    """Read CSV file with expanded path."""
    expanded_path = str(Path(path).expanduser())
    return pd.read_csv(expanded_path, **kwargs)


def write_csv(df: pd.DataFrame, path: str, **kwargs):
    """Write CSV file with expanded path."""
    expanded_path = Path(path).expanduser()
    expanded_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(expanded_path, index=False, **kwargs)


def read_parquet(path: str, **kwargs) -> pd.DataFrame:
    """Read Parquet file with expanded path."""
    expanded_path = str(Path(path).expanduser())
    return pd.read_parquet(expanded_path, **kwargs)


def write_parquet(df: pd.DataFrame, path: str, **kwargs):
    """Write Parquet file with expanded path."""
    expanded_path = Path(path).expanduser()
    expanded_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(expanded_path, index=False, **kwargs)


def exists(path: str) -> bool:
    """Check if file exists."""
    return Path(path).expanduser().exists()


def makedirs(path: str):
    """Create directory and parents if needed."""
    Path(path).expanduser().mkdir(parents=True, exist_ok=True)
