"""
Data utility functions for GP2 Precision Medicine Data Browser.
"""
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def validate_data_paths(config) -> Dict[str, bool]:
    """
    Validate that all configured data paths exist.
    
    Args:
        config: Settings configuration object
        
    Returns:
        Dictionary of path validation results
    """
    paths_to_check = {
        "clinical_master_key": config.clinical_master_key_path,
        "wgs_var_info": config.wgs_var_info_path,
        "wgs_carriers_int": config.wgs_carriers_int_path,
        "wgs_carriers_string": config.wgs_carriers_string_path,
        "nba_info": config.nba_info_path,
        "nba_carriers_int": config.nba_carriers_int_path,
        "nba_carriers_string": config.nba_carriers_string_path,
    }
    
    results = {}
    for name, path in paths_to_check.items():
        exists = path.exists() if isinstance(path, Path) else Path(path).exists()
        results[name] = exists
        if not exists:
            logger.warning(f"Data file not found: {name} at {path}")
        else:
            logger.info(f"Data file found: {name} at {path}")
    
    return results


def estimate_memory_usage(file_path: Path) -> float:
    """
    Estimate memory usage for loading a CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Estimated memory usage in MB
    """
    if not file_path.exists():
        return 0.0
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    # Estimate memory usage as ~2-3x file size for pandas DataFrame
    estimated_memory_mb = file_size_mb * 2.5
    
    return estimated_memory_mb


def sample_csv_data(file_path: Path, n_rows: int = 100) -> pd.DataFrame:
    """
    Sample data from a CSV file for inspection.
    
    Args:
        file_path: Path to CSV file
        n_rows: Number of rows to sample
        
    Returns:
        Sampled DataFrame
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, nrows=n_rows, low_memory=False)


def get_csv_info(file_path: Path) -> Dict[str, Any]:
    """
    Get basic information about a CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary with file information
    """
    if not file_path.exists():
        return {"exists": False}
    
    # Get file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    
    # Sample first few rows to get column info
    try:
        sample = pd.read_csv(file_path, nrows=10, low_memory=False)
        columns = list(sample.columns)
        dtypes = sample.dtypes.to_dict()
        
        # Count total rows (approximately)
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f) - 1  # Subtract header
        
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return {
            "exists": True,
            "file_size_mb": file_size_mb,
            "error": str(e)
        }
    
    return {
        "exists": True,
        "file_size_mb": file_size_mb,
        "estimated_memory_mb": estimate_memory_usage(file_path),
        "total_rows": total_lines,
        "total_columns": len(columns),
        "columns": columns,
        "dtypes": {k: str(v) for k, v in dtypes.items()}
    }


def clean_ancestry_label(label: Optional[str]) -> Optional[str]:
    """
    Clean and standardize ancestry labels.
    
    Args:
        label: Raw ancestry label
        
    Returns:
        Cleaned ancestry label
    """
    if not label or pd.isna(label):
        return None
    
    # Convert to string and clean whitespace
    label = str(label).strip()
    
    # Handle common patterns
    if "(" in label:
        # Extract main ancestry from patterns like "EUR (multi)"
        main_ancestry = label.split("(")[0].strip()
        return main_ancestry
    
    return label


def standardize_variant_id(variant_id: str) -> str:
    """
    Standardize variant ID format.
    
    Args:
        variant_id: Raw variant ID
        
    Returns:
        Standardized variant ID
    """
    if not variant_id:
        return variant_id
    
    # Remove any extra whitespace
    variant_id = str(variant_id).strip()
    
    # Ensure chromosome format (e.g., "chr1" instead of "1")
    parts = variant_id.split(":")
    if len(parts) >= 1 and not parts[0].startswith("chr"):
        parts[0] = f"chr{parts[0]}"
        variant_id = ":".join(parts)
    
    return variant_id


def merge_carrier_data(sample_id: str, 
                      wgs_carriers: Dict[str, Optional[float]], 
                      nba_carriers: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """
    Merge carrier data from multiple sources, prioritizing WGS over NBA.
    
    Args:
        sample_id: Sample identifier
        wgs_carriers: WGS carrier data
        nba_carriers: NBA carrier data
        
    Returns:
        Merged carrier data
    """
    merged = {}
    
    # Start with NBA data
    merged.update(nba_carriers)
    
    # Overlay WGS data (WGS takes priority)
    for variant_id, status in wgs_carriers.items():
        if status is not None:  # Only override if WGS has actual data
            merged[variant_id] = status
    
    return merged


def calculate_carrier_frequency(carriers: Dict[str, Optional[float]]) -> Dict[str, float]:
    """
    Calculate carrier frequencies for variants.
    
    Args:
        carriers: Dictionary mapping variant ID to carrier status values
        
    Returns:
        Dictionary mapping variant ID to carrier frequency
    """
    frequencies = {}
    
    for variant_id, statuses in carriers.items():
        if not statuses:
            frequencies[variant_id] = 0.0
            continue
        
        # Convert to list if it's a single value
        if not isinstance(statuses, list):
            statuses = [statuses]
        
        # Count carriers (status > 0) and non-missing values
        carrier_count = sum(1 for s in statuses if s is not None and s > 0)
        total_count = sum(1 for s in statuses if s is not None)
        
        frequency = carrier_count / total_count if total_count > 0 else 0.0
        frequencies[variant_id] = frequency
    
    return frequencies


def validate_sample_id_format(sample_id: str, expected_format: str = "GP2") -> bool:
    """
    Validate sample ID format.
    
    Args:
        sample_id: Sample ID to validate
        expected_format: Expected format ("GP2", "IID", etc.)
        
    Returns:
        True if format is valid
    """
    if not sample_id:
        return False
    
    sample_id = str(sample_id).strip()
    
    if expected_format.upper() == "GP2":
        # GP2 IDs typically follow pattern like "STUDY_XXXXXX"
        return "_" in sample_id and len(sample_id) > 5
    elif expected_format.upper() == "IID":
        # IID format is more flexible, usually alphanumeric
        return len(sample_id) > 0 and sample_id.replace("_", "").replace("-", "").isalnum()
    
    return True  # Default to valid for unknown formats


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 10000) -> List[pd.DataFrame]:
    """
    Split a DataFrame into chunks for memory-efficient processing.
    
    Args:
        df: DataFrame to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of DataFrame chunks
    """
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)
    
    return chunks


def get_memory_usage_mb(obj: Any) -> float:
    """
    Get memory usage of an object in MB.
    
    Args:
        obj: Object to measure
        
    Returns:
        Memory usage in MB
    """
    if isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum() / (1024 * 1024)
    elif isinstance(obj, dict):
        # Rough estimate for dictionary
        return len(str(obj)) / (1024 * 1024)
    else:
        return 0.0


def log_data_summary(data_type: str, total_records: int, memory_mb: float, 
                    unique_values: Optional[Dict[str, int]] = None):
    """
    Log a summary of loaded data.
    
    Args:
        data_type: Type of data (e.g., "variants", "samples")
        total_records: Total number of records
        memory_mb: Memory usage in MB
        unique_values: Dictionary of unique value counts per column
    """
    logger.info(f"Loaded {data_type} data:")
    logger.info(f"  - Total records: {total_records:,}")
    logger.info(f"  - Memory usage: {memory_mb:.2f} MB")
    
    if unique_values:
        for column, count in unique_values.items():
            logger.info(f"  - Unique {column}: {count:,}")


class DataValidator:
    """Data validation utility class."""
    
    @staticmethod
    def validate_variant_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate variant data DataFrame."""
        issues = []
        
        required_columns = ['id', 'locus', 'chrom', 'pos']
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        
        if 'pos' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['pos']):
                issues.append("Position column should be numeric")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_records": len(df),
            "null_counts": df.isnull().sum().to_dict()
        }
    
    @staticmethod
    def validate_clinical_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate clinical data DataFrame."""
        issues = []
        
        required_columns = ['GP2ID', 'study']
        for col in required_columns:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
        
        if 'GP2ID' in df.columns:
            if df['GP2ID'].duplicated().any():
                issues.append("Duplicate GP2ID values found")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_records": len(df),
            "null_counts": df.isnull().sum().to_dict()
        }
    
    @staticmethod
    def validate_carrier_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate carrier data DataFrame."""
        issues = []
        
        if 'IID' not in df.columns and df.index.name != 'IID':
            issues.append("Missing IID column or index")
        
        # Check for valid carrier values (should be 0, 1, 2, or NaN)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            unique_vals = df[col].dropna().unique()
            invalid_vals = [v for v in unique_vals if v not in [0.0, 1.0, 2.0]]
            if invalid_vals:
                issues.append(f"Invalid carrier values in {col}: {invalid_vals}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_records": len(df),
            "total_variants": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
        } 