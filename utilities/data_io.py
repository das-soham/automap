"""
Data I/O utilities for loading transformed data.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(data_file: str = 'data.parquet') -> pd.DataFrame:
    """
    Load transformed data from parquet file.

    Args:
        data_file: Path to parquet file

    Returns:
        DataFrame with all transformed series
    """
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    logger.info(f"Loading data from {data_file}...")
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded data shape: {data.shape}")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    return data
