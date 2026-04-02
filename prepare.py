"""
Model Evaluation Module
Calculates RMSE for fitted ARDL models on test dataset.
"""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

# Note: Logging is typically configured by train.py
# If running prepare.py standalone, call setup_logging() from utilities
logger = logging.getLogger(__name__)

# Test period constants
TEST_START_DT = '2021-01-31'
TEST_END_DT = '2025-12-31'


def prepare_model_data(
    data: pd.DataFrame,
    endogenous: str,
    exogenous: list
) -> Tuple[pd.Series, pd.DataFrame]:
    """Prepare data for model prediction."""
    y = data[endogenous].copy()
    exog_cols = [var for var in exogenous if var in data.columns and var != endogenous]
    X = data[exog_cols].copy()

    # Drop NaN
    valid_idx = y.notna() & X.notna().all(axis=1)
    return y[valid_idx], X[valid_idx]


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate Root Mean Squared Error."""
    common_idx = actual.index.intersection(predicted.index)
    actual_aligned = actual.loc[common_idx]
    predicted_aligned = predicted.loc[common_idx]
    return np.sqrt(np.mean((actual_aligned - predicted_aligned) ** 2))


def format_exogenous_with_lags(exogenous: list, lags_exogenous: list) -> str:
    """Format exogenous variables with their lag counts."""
    if not exogenous or not lags_exogenous:
        return ""

    formatted = []
    for var, lag in zip(exogenous, lags_exogenous):
        formatted.append(f"{var}({lag})")

    return ", ".join(formatted)


def evaluate_single_model(
    model_info: dict,
    full_data: pd.DataFrame,
    test_start: str,
    test_end: str,
    model_name: str
) -> Tuple[float, int]:
    """Evaluate a single time series model on test data (ARDL, VAR, VECM, etc.)."""

    try:
        fitted_model = model_info['model']
        endogenous = model_info['endogenous']
        exogenous = model_info['exogenous']

        # Prepare full dataset (train + test for ARDL lags)
        y_full, X_full = prepare_model_data(full_data, endogenous, exogenous)

        if len(y_full) == 0:
            logger.warning(f"  No data for {model_name}")
            return np.nan, 0

        # Filter test period for actual values and exog data
        test_mask = (y_full.index >= test_start) & (y_full.index <= test_end)
        y_test = y_full[test_mask]
        X_test = X_full[test_mask]

        if len(y_test) == 0:
            logger.warning(f"  No test data for {model_name}")
            return np.nan, 0

        # Generate out-of-sample predictions (model-type specific)
        model_type = model_info.get('model_type', 'ARDL')  # Default to ARDL for backward compatibility

        if model_type == 'ARDL':
            # ARDL: Use date-based indexing with exog_oos
            predictions = fitted_model.predict(
                start=y_test.index[0],
                end=y_test.index[-1],
                exog_oos=X_test
            )

        elif model_type == 'VAR':
            # VAR: Use forecast() with steps and last observations
            n_steps = len(y_test)
            k_ar = fitted_model.k_ar  # VAR lag order

            # Get last k_ar observations from y_full (needed for VAR initialization)
            train_mask = y_full.index < y_test.index[0]
            y_train_full = y_full[train_mask]

            if len(y_train_full) < k_ar:
                logger.warning(f"  Insufficient history for VAR (need {k_ar}, have {len(y_train_full)})")
                return np.nan, 0

            y_last = y_train_full.iloc[-k_ar:].values

            # VAR forecast returns ndarray, need to map back to dates
            forecast_array = fitted_model.forecast(y_last, steps=n_steps, exog_future=X_test.values)

            # If VAR returns multivariate, extract the endogenous variable column
            # (For now, assume univariate or take first column)
            if forecast_array.ndim > 1:
                forecast_array = forecast_array[:, 0]

            predictions = pd.Series(forecast_array, index=y_test.index)

        elif model_type == 'VECM':
            # VECM: Use predict() with steps
            n_steps = len(y_test)

            # VECM may need exog_fc and exog_coint_fc separately
            # For simplicity, assuming no cointegrating exogenous terms
            forecast_array = fitted_model.predict(
                steps=n_steps,
                exog_fc=X_test.values if X_test is not None and len(X_test) > 0 else None
            )

            # If VECM returns multivariate, extract the endogenous variable column
            if forecast_array.ndim > 1:
                forecast_array = forecast_array[:, 0]

            predictions = pd.Series(forecast_array, index=y_test.index)

        elif model_type == 'BVAR':
            # BVAR not implemented in statsmodels
            logger.error(f"  BVAR model type not supported (not in statsmodels)")
            return np.nan, 0

        else:
            logger.error(f"  Unknown model type: {model_type}")
            return np.nan, 0

        rmse = calculate_rmse(y_test, predictions)

        return rmse, len(predictions)

    except Exception as e:
        logger.error(f"  Error evaluating {model_name}: {e}")
        return np.nan, 0


def evaluate_model_dict(
    model_dict: Dict,
    data_file: str = 'data_gpr.parquet',
    test_start: str = TEST_START_DT,
    test_end: str = TEST_END_DT
) -> Tuple[float, pd.DataFrame]:
    """
    Evaluate all models in MODEL_DICT on test data.

    Args:
        model_dict: Dictionary of fitted models from train.py
        data_file: Path to data parquet file
        test_start: Test period start date
        test_end: Test period end date

    Returns:
        Tuple of (total_rmse, results_df)
        - total_rmse: Sum of all RMSEs
        - results_df: DataFrame with [model_name, endogenous, rmse, n_predictions]
    """

    # Load full dataset (ARDL needs full history for lags)
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    full_data = pd.read_parquet(data_file)
    logger.info(f"Loaded full dataset: {full_data.shape}")
    logger.info(f"Test period: {test_start} to {test_end}")

    results = []
    for model_name, model_info in model_dict.items():
        rmse, n_pred = evaluate_single_model(model_info, full_data, test_start, test_end, model_name)

        # Format exogenous variables with lags
        exog_formatted = format_exogenous_with_lags(
            model_info['exogenous'],
            model_info['lags_exogenous']
        )

        results.append({
            'model_name': model_name,
            'model_type': model_info.get('model_type', 'Unknown'),
            'endogenous': model_info['endogenous'],
            'exogenous_vars': exog_formatted,
            'n_exog': model_info['n_exog'],
            'rmse': rmse if rmse is not np.nan else 100.0,
            'n_predictions': n_pred
        })

    results_df = pd.DataFrame(results)
    total_rmse = results_df['rmse'].fillna(100.0).sum()

    # Log detailed evaluation summary for grep-ability
    log_evaluation_summary(results_df, total_rmse)

    return total_rmse, results_df


def evaluate_from_pickle(
    model_dict_file: str = 'MODEL_DICT.pkl',
    data_file: str = 'data.parquet'
) -> Tuple[float, pd.DataFrame]:
    """
    Load MODEL_DICT from pickle and evaluate.

    Args:
        model_dict_file: Path to pickled MODEL_DICT
        data_file: Path to data parquet file

    Returns:
        Tuple of (total_rmse, results_df)
    """
    import pickle

    logger.info(f"Loading MODEL_DICT from {model_dict_file}...")
    with open(model_dict_file, 'rb') as f:
        model_dict = pickle.load(f)
    return evaluate_model_dict(model_dict, data_file)


def log_evaluation_summary(results_df: pd.DataFrame, total_rmse: float) -> None:
    """
    Log evaluation results in a grep-friendly format.

    Each result is logged with prefixes for easy searching:
    - EVAL_RESULT: Individual model results (one per line)
    - EVAL_SUMMARY: Overall statistics
    - EVAL_BEST: Best performing model
    - EVAL_WORST: Worst performing model

    Example grep usage:
        grep "EVAL_RESULT:" logs/autolog.log          # All model results
        grep "EVAL_SUMMARY:" logs/autolog.log         # Summary stats
        grep "model_type=ARDL" logs/autolog.log       # Filter by model type
        grep "rmse=" logs/autolog.log | sort -t= -k6n # Sort by RMSE
    """
    logger.info("="*70)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("="*70)

    # Log each model result on a single line for easy grep
    sorted_df = results_df.sort_values('rmse')
    for _, row in sorted_df.iterrows():
        logger.info(
            f"EVAL_RESULT: model_name={row['model_name']}, "
            f"model_type={row['model_type']}, "
            f"endogenous={row['endogenous']}, "
            f"n_exog={row['n_exog']}, "
            f"rmse={row['rmse']:.4f}, "
            f"n_predictions={row['n_predictions']}"
        )

    # Log summary statistics
    logger.info("="*70)
    mean_rmse = results_df['rmse'].mean()
    logger.info(
        f"EVAL_SUMMARY: total_rmse={total_rmse:.4f}, "
        f"mean_rmse={mean_rmse:.4f}, "
        f"n_models={len(results_df)}"
    )

    # Log best and worst models
    best = sorted_df.iloc[0]
    worst = sorted_df.iloc[-1]
    logger.info(
        f"EVAL_BEST: model_name={best['model_name']}, "
        f"endogenous={best['endogenous']}, "
        f"rmse={best['rmse']:.4f}"
    )
    logger.info(
        f"EVAL_WORST: model_name={worst['model_name']}, "
        f"endogenous={worst['endogenous']}, "
        f"rmse={worst['rmse']:.4f}"
    )
    logger.info("="*70)

if __name__ == '__main__':
    import logging
    from utilities.logging_config import setup_logging
    setup_logging()
    logging.getLogger(__name__).setLevel(logging.INFO)  # Enable logging for __main__
    total_rmse, results_df = evaluate_from_pickle()