"""
Custom exceptions for precision medicine data access.
Kept minimal - only what's needed for clear error handling.
"""


class DataAccessError(Exception):
    """Base exception for data access related errors."""
    pass


class ConfigurationError(DataAccessError):
    """Raised when there are configuration issues (paths, releases, etc.)."""
    pass


class DataValidationError(DataAccessError):
    """Raised when data doesn't meet expected format or validation criteria."""
    pass


class FileNotFoundError(DataAccessError):
    """Raised when expected data files are not found."""
    pass 