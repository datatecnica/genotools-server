"""
Centralized logging configuration for precision medicine pipeline.

Provides:
- Console handler: Shows only high-level progress (WARNING level for most loggers)
- File handler: Captures all details with rotation (DEBUG level)
- Progress logger: Always prints to console for step announcements
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


# Module-level state
_logging_initialized = False
_log_file_path: Optional[str] = None


def setup_logging(
    log_dir: Optional[str] = None,
    job_name: str = "pipeline",
    console_level: int = logging.WARNING,
    file_level: int = logging.DEBUG,
    max_bytes: int = 5 * 1024 * 1024,  # 5MB
    backup_count: int = 3
) -> str:
    """
    Initialize logging with console and rotating file handlers.

    Args:
        log_dir: Directory for log files. If None, logs to current directory.
        job_name: Name prefix for log file.
        console_level: Log level for console output (default: WARNING).
        file_level: Log level for file output (default: DEBUG).
        max_bytes: Maximum size per log file before rotation.
        backup_count: Number of backup files to keep.

    Returns:
        Path to the log file.
    """
    global _logging_initialized, _log_file_path

    # Avoid re-initialization
    if _logging_initialized:
        return _log_file_path or ""

    # Create log directory
    if log_dir:
        log_path = Path(log_dir) / "logs"
    else:
        log_path = Path.cwd() / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{job_name}_{timestamp}.log"
    _log_file_path = str(log_file)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything, handlers filter

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler - minimal output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler - detailed output with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(file_level)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy_logger in ["urllib3", "matplotlib", "PIL"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _logging_initialized = True

    return _log_file_path


def get_progress_logger() -> logging.Logger:
    """
    Get a logger that always prints to console.

    Use this for high-level progress messages that should always be visible:
    - Step announcements (Step 1/4, Step 2/4, etc.)
    - Pipeline start/completion
    - Final summaries
    - Errors

    Returns:
        Logger configured to always output to console.
    """
    logger = logging.getLogger("progress")

    # Ensure progress logger always shows on console
    # by setting its level to INFO (below WARNING threshold)
    # and adding a dedicated console handler
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        # Prevent propagation to avoid duplicate messages
        logger.propagate = False

    return logger


def get_log_file_path() -> Optional[str]:
    """Get the path to the current log file."""
    return _log_file_path


def reset_logging():
    """Reset logging state. Useful for testing."""
    global _logging_initialized, _log_file_path
    _logging_initialized = False
    _log_file_path = None

    # Clear root logger handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Clear progress logger handlers
    progress_logger = logging.getLogger("progress")
    progress_logger.handlers.clear()
