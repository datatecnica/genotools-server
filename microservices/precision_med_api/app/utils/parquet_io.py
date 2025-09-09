"""
Parquet I/O utilities for efficient genomic data storage and retrieval.

Provides optimized reading and writing of parquet files with:
- Compression and partitioning support
- Efficient filtering and querying
- Append operations for incremental data
- Partition scanning for large datasets
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def save_parquet(
    df: pd.DataFrame, 
    path: str, 
    partition_cols: Optional[List[str]] = None,
    compression: str = "snappy",
    index: bool = False
) -> None:
    """
    Save DataFrame to parquet with optimal settings for genomic data.
    
    Args:
        df: DataFrame to save
        path: Output path
        partition_cols: Columns to partition by (e.g., ['chromosome', 'ancestry'])
        compression: Compression algorithm (snappy, gzip, lz4)
        index: Whether to write DataFrame index
    """
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if partition_cols:
            # Partitioned dataset
            table = pa.Table.from_pandas(df, preserve_index=index)
            pq.write_to_dataset(
                table,
                root_path=path,
                partition_cols=partition_cols,
                compression=compression,
                use_dictionary=True,  # Efficient for categorical data
                row_group_size=100000,  # Optimize for genomic data
                existing_data_behavior="overwrite_or_ignore"
            )
        else:
            # Single file
            df.to_parquet(
                path,
                compression=compression,
                index=index,
                engine="pyarrow"
            )
        
        logger.info(f"Saved {len(df)} rows to {path}")
        
    except Exception as e:
        logger.error(f"Failed to save parquet to {path}: {e}")
        raise


def read_parquet(
    path: str, 
    filters: Optional[List] = None,
    columns: Optional[List[str]] = None,
    use_pandas_metadata: bool = True
) -> pd.DataFrame:
    """
    Read parquet file with optional filtering and column selection.
    
    Args:
        path: Path to parquet file or directory
        filters: PyArrow filters (e.g., [('chromosome', '=', '1')])
        columns: Columns to read (None for all)
        use_pandas_metadata: Use pandas metadata for proper dtypes
        
    Returns:
        DataFrame with requested data
    """
    try:
        if os.path.isdir(path):
            # Partitioned dataset
            dataset = pq.ParquetDataset(path)
            df = dataset.read_pandas(
                filters=filters,
                columns=columns,
                use_pandas_metadata=use_pandas_metadata
            ).to_pandas()
        else:
            # Single file
            df = pd.read_parquet(
                path,
                filters=filters,
                columns=columns,
                engine="pyarrow"
            )
        
        logger.info(f"Read {len(df)} rows from {path}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to read parquet from {path}: {e}")
        raise


def optimize_dtypes_for_genomics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes for genomic data to reduce memory usage.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    df_optimized = df.copy()
    
    # Optimize categorical columns commonly found in genomic data
    categorical_candidates = [
        'chromosome', 'ancestry', 'data_type', 'harmonization_action',
        'ref', 'alt', 'pgen_a1', 'pgen_a2', 'snp_list_a1', 'snp_list_a2'
    ]
    
    for col in categorical_candidates:
        if col in df_optimized.columns:
            if df_optimized[col].dtype == 'object':
                df_optimized[col] = df_optimized[col].astype('category')
    
    # Optimize integer columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        if col_min >= 0:
            # Unsigned integers
            if col_max < 255:
                df_optimized[col] = df_optimized[col].astype('uint8')
            elif col_max < 65535:
                df_optimized[col] = df_optimized[col].astype('uint16')
            elif col_max < 4294967295:
                df_optimized[col] = df_optimized[col].astype('uint32')
        else:
            # Signed integers
            if col_min >= -128 and col_max < 128:
                df_optimized[col] = df_optimized[col].astype('int8')
            elif col_min >= -32768 and col_max < 32768:
                df_optimized[col] = df_optimized[col].astype('int16')
            elif col_min >= -2147483648 and col_max < 2147483648:
                df_optimized[col] = df_optimized[col].astype('int32')
    
    # Optimize float columns
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        # Check if values can fit in float32 without precision loss
        try:
            float32_col = df_optimized[col].astype('float32')
            if (df_optimized[col] == float32_col).all():
                df_optimized[col] = float32_col
        except (ValueError, OverflowError):
            pass
    
    return df_optimized


