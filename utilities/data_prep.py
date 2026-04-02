"""
Data preparation utilities for time series modeling.
"""

import logging
from typing import List, Union, Tuple
import pandas as pd

from config import TRAIN_START_DT, TRAIN_END_DT

logger = logging.getLogger(__name__)


def parse_exogenous_vars(exog_spec: Union[str, List[str]]) -> List[str]:
    """
    Parse exogenous variable specification from MEV_REL tuple.

    Args:
        exog_spec: Either a string ('GPR_EU') or list (['CISS_BOND', 'CISS_EQ'])

    Returns:
        List of exogenous variable names
    """
    if isinstance(exog_spec, str):
        return [exog_spec] if exog_spec else []
    elif isinstance(exog_spec, list):
        return [var for var in exog_spec if var]
    else:
        return []


def prepare_training_data(
    data: pd.DataFrame,
    endogenous: str,
    exogenous_vars: List[str],
    train_start: str = None,
    train_end: str = None
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Prepare data for time series model estimation (model-agnostic).

    Returns current values only - individual models are responsible for creating lags
    as needed (e.g., ARDL creates lags automatically, VAR handles it differently).

    Args:
        data: Full dataset
        endogenous: Endogenous variable name
        exogenous_vars: All exogenous variables (primary + additional)
        train_start: Training period start (defaults to TRAIN_START_DT from config)
        train_end: Training period end (defaults to TRAIN_END_DT from config)

    Returns:
        Tuple of (y_train, X_train, y_test, X_test)
    """
    logger.info(f"\nPreparing training data for: {endogenous}")
    logger.info(f"  Exogenous variables: {exogenous_vars}")

    # Get endogenous variable
    y = data[endogenous].copy()

    # Get exogenous variables (current values only)
    exog_cols = [var for var in exogenous_vars if var in data.columns and var != endogenous]
    X = data[exog_cols].copy()

    # Drop rows with NaN in either y or X
    valid_idx = y.notna() & X.notna().all(axis=1)
    y = y[valid_idx]
    X = X[valid_idx]

    logger.info(f"  Exogenous variables: {len(X.columns)}")
    logger.info(f"  Total observations (after dropping NaN): {len(y)}")

    # Use config defaults if not provided
    if train_start is None:
        train_start = TRAIN_START_DT
    if train_end is None:
        train_end = TRAIN_END_DT

    # Split train/test
    train_mask = (y.index >= train_start) & (y.index <= train_end)
    test_mask = y.index > train_end

    y_train = y[train_mask]
    X_train = X[train_mask]
    y_test = y[test_mask]
    X_test = X[test_mask]

    logger.info(f"  Train period: {y_train.index[0]} to {y_train.index[-1]} ({len(y_train)} obs)")
    logger.info(f"  Test period: {y_test.index[0]} to {y_test.index[-1]} ({len(y_test)} obs)")

    return y_train, X_train, y_test, X_test
