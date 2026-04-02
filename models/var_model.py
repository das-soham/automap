"""
VAR (Vector Autoregression) model implementation.

This is a STUB implementation to demonstrate the experimenter workflow.
Replace with actual VAR implementation using statsmodels.tsa.vector_ar.var_model.VAR
"""

import logging
from typing import Any, Dict, List
import pandas as pd

from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


class VARModel(BaseTimeSeriesModel):
    """
    VAR model implementation (STUB for demonstration).

    In a real implementation, this would use statsmodels VAR:
    from statsmodels.tsa.vector_ar.var_model import VAR
    """

    def __init__(self, max_lag: int = 12):
        """
        Initialize VAR model.

        Args:
            max_lag: Maximum lags for all variables
        """
        self.max_lag = max_lag

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame, **kwargs) -> Any:
        """
        Fit VAR model (STUB - would use statsmodels VAR in real implementation).

        Args:
            y_train: Training target (endogenous variable)
            X_train: Training features (exogenous variables)
            **kwargs: Additional parameters

        Returns:
            Fitted VAR model object (currently returns None as stub)
        """
        endogenous = y_train.name if y_train.name else 'y'
        logger.info(f"\nFitting VAR model for {endogenous}...")

        # STUB: Real implementation would combine y_train and X_train into multivariate series
        # and fit using statsmodels VAR
        logger.info(f"  [STUB] VAR model would be fitted here")
        logger.info(f"  [STUB] Would use max_lag={self.max_lag}")

        # Return a stub object with minimal attributes for demonstration
        class VARStub:
            def __init__(self):
                self.aic = 999.99
                self.bic = 999.99
                self.nobs = len(y_train) - max_lag

        return VARStub()

    def to_model_dict(
        self,
        fitted_model: Any,
        endogenous: str,
        exogenous: List[str],
        y_train: pd.Series,
        **metadata
    ) -> Dict:
        """
        Convert fitted VAR model to standard MODEL_DICT format.

        Args:
            fitted_model: Fitted VAR object (stub)
            endogenous: Endogenous variable name
            exogenous: List of exogenous variable names
            y_train: Training target (for extracting date range)
            **metadata: Additional metadata

        Returns:
            Dictionary with 10 required keys for prepare.py compatibility
        """
        # VAR treats all variables symmetrically, but we maintain the endogenous/exogenous
        # distinction for compatibility with the MODEL_DICT structure
        model_dict = {
            'model': fitted_model,
            'endogenous': endogenous,
            'exogenous': exogenous,
            'lags_exogenous': [self.max_lag] * len(exogenous),
            'lags_endogenous': self.max_lag,
            'n_exog': len(exogenous),
            'n_obs_train': fitted_model.nobs,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'train_period': (str(y_train.index[0]), str(y_train.index[-1])),
            'model_type': 'VAR'
        }

        return model_dict
