"""
Centralized logging configuration for the AutoMap project.

This module sets up logging to both console and rotating log files.
"""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    log_dir: str = 'logs',
    log_level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure logging to write to both console and rotating log files.

    Args:
        log_dir: Directory for log files (default: 'logs')
        log_level: Logging level (default: INFO)
        max_bytes: Maximum size of each log file before rotation (default: 10 MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Configured root logger
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create log filename (consistent name)
    log_file = log_path / 'autolog.log'

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger and set to WARNING (suppress verbose logs)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)  # Handler accepts all levels
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Configure module-specific log levels
    # Only prepare.py and feature_eng.py log at INFO level
    logging.getLogger('prepare').setLevel(logging.INFO)
    logging.getLogger('utilities.feature_eng').setLevel(logging.INFO)

    # Everything else (models, other utilities, train.py) stays at WARNING
    # This suppresses verbose logs from:
    # - models.ardl_model, models.var_model, etc.
    # - utilities.data_prep, utilities.data_io
    # - __main__ (train.py)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
