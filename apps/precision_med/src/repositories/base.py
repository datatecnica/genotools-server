"""
Base repository class for GP2 Precision Medicine Data Browser.
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import logging


# Type variables for generic repository
T = TypeVar('T')
FilterT = TypeVar('FilterT')

logger = logging.getLogger(__name__)


class BaseRepository(ABC, Generic[T, FilterT]):
    """
    Abstract base repository implementing common data access patterns.
    
    Provides:
    - Lazy loading of data
    - Basic CRUD operations
    - Filtering capabilities
    - Type safety
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize repository with optional data path."""
        self.data_path = data_path
        self._data: Optional[pd.DataFrame] = None
        self._loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Check if data is loaded in memory."""
        return self._loaded and self._data is not None
    
    @abstractmethod
    def _load_data(self) -> pd.DataFrame:
        """Load data from source. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _model_from_row(self, row: pd.Series) -> T:
        """Convert DataFrame row to model instance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _apply_filters(self, data: pd.DataFrame, filters: FilterT) -> pd.DataFrame:
        """Apply filters to DataFrame. Must be implemented by subclasses."""
        pass
    
    def load(self, force_reload: bool = False) -> None:
        """
        Load data into memory.
        
        Args:
            force_reload: Force reload even if data is already loaded
        """
        if self.is_loaded and not force_reload:
            return
        
        try:
            logger.info(f"Loading data from {self.data_path}")
            self._data = self._load_data()
            self._loaded = True
            logger.info(f"Loaded {len(self._data)} records")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    

    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Get all records as model instances.
        
        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of model instances
        """
        self.load()
        
        # Simple pagination for small datasets
        if offset > 0 or limit is not None:
            end_idx = offset + limit if limit else None
            subset = self._data.iloc[offset:end_idx]
        else:
            subset = self._data
        
        return [self._model_from_row(row) for _, row in subset.iterrows()]
    
    def get_by_id(self, record_id: str, id_column: str = "id") -> Optional[T]:
        """
        Get a single record by ID.
        
        Args:
            record_id: ID of the record to retrieve
            id_column: Name of the ID column
            
        Returns:
            Model instance or None if not found
        """
        self.load()
        
        matches = self._data[self._data[id_column] == record_id]
        if matches.empty:
            return None
        
        return self._model_from_row(matches.iloc[0])
    
    def filter(self, filters: FilterT) -> List[T]:
        """
        Filter records based on criteria.
        
        Args:
            filters: Filter criteria
            
        Returns:
            List of filtered model instances
        """
        self.load()
        
        filtered_data = self._apply_filters(self._data, filters)
        
        # Simple pagination from filters
        limit = getattr(filters, 'limit', None)
        offset = getattr(filters, 'offset', 0)
        
        if offset > 0:
            filtered_data = filtered_data.iloc[offset:]
        if limit:
            filtered_data = filtered_data.head(limit)
        
        return [self._model_from_row(row) for _, row in filtered_data.iterrows()]
    
    def count(self, filters: Optional[FilterT] = None) -> int:
        """
        Count records matching filters.
        
        Args:
            filters: Optional filter criteria
            
        Returns:
            Number of matching records
        """
        self.load()
        
        if filters:
            filtered_data = self._apply_filters(self._data, filters)
            return len(filtered_data)
        
        return len(self._data)
    
    def get_unique_values(self, column: str) -> List[str]:
        """
        Get unique values from a column.
        
        Args:
            column: Column name
            
        Returns:
            List of unique values
        """
        self.load()
        
        if column not in self._data.columns:
            return []
        
        return self._data[column].dropna().unique().tolist()
    
    def get_summary_stats(self, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summary statistics for the dataset.
        
        Args:
            columns: Specific columns to summarize
            
        Returns:
            Dictionary of summary statistics
        """
        self.load()
        
        if columns:
            subset = self._data[columns]
        else:
            subset = self._data
        
        stats = {
            "total_records": len(self._data),
            "columns": list(self._data.columns),
            "memory_usage_mb": self._data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Simple column stats for small datasets
        for col in subset.columns:
            if subset[col].dtype in ['int64', 'float64']:
                stats[f"{col}_stats"] = {
                    "mean": subset[col].mean(),
                    "std": subset[col].std(),
                    "min": subset[col].min(),
                    "max": subset[col].max(),
                    "null_count": subset[col].isnull().sum()
                }
            else:
                stats[f"{col}_stats"] = {
                    "unique_count": subset[col].nunique(),
                    "null_count": subset[col].isnull().sum(),
                    "most_common": subset[col].value_counts().head(5).to_dict()
                }
        
        return stats
    

    
    def __len__(self) -> int:
        """Get the number of records in the repository."""
        self.load()
        return len(self._data)
    
    def __contains__(self, record_id: str) -> bool:
        """Check if a record exists by ID."""
        return self.get_by_id(record_id) is not None 