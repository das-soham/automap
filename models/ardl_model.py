"""
ARDL (Autoregressive Distributed Lag) model implementation.
"""

import logging
from typing import Any, Dict, List
import pandas as pd
from statsmodels.tsa.ardl import ARDL

from .base_model import BaseTimeSeriesModel

logger = logging.getLogger(__name__)


class ARDLModel(BaseTimeSeriesModel):
    """
    ARDL model implementation using statsmodels.

    ARDL automatically creates lags for:
    - Endogenous variable: lags 1 to max_lag
    - Exogenous variables: lags 0 to max_lag (includes current value)
    """

    def __init__(self, max_lag: int = 12):
        """
        Initialize ARDL model.

        Args:
            max_lag: Maximum lags for all variables (endogenous and exogenous)
        """
        self.max_lag = max_lag

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame, **kwargs) -> Any:
        """
        Fit ARDL model using statsmodels.

        Args:
            y_train: Training target (endogenous variable)
            X_train: Training features (exogenous variables, current values)
            **kwargs: Additional parameters (ignored for ARDL)

        Returns:
            Fitted ARDLResults object from statsmodels
        """
        endogenous = y_train.name if y_train.name else 'y'
        logger.info(f"\nFitting ARDL model for {endogenous}...")

        try:
            # Fit ARDL model
            # ARDL will automatically create lags for:
            # - Endogenous variable: lags 1 to max_lag
            # - Exogenous variables: lags 0 to max_lag (includes current value)
            model = ARDL(
                endog=y_train,
                lags=self.max_lag,  # Lags for endogenous
                exog=X_train,
                order=self.max_lag,  # Lags for exogenous (same for all)
                trend='c'  # Include constant
            )
            fitted_model = model.fit()

            logger.info(f"  Model fitted successfully")
            logger.info(f"  AIC: {fitted_model.aic:.2f}")
            logger.info(f"  BIC: {fitted_model.bic:.2f}")
            logger.info(f"  N observations: {fitted_model.nobs}")

            return fitted_model

        except Exception as e:
            logger.error(f"  Error fitting ARDL model for {endogenous}: {e}")
            logger.error(f"  Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def to_model_dict(
        self,
        fitted_model: Any,
        endogenous: str,
        exogenous: List[str],
        y_train: pd.Series,
        **metadata
    ) -> Dict:
        """
        Convert fitted ARDL model to standard MODEL_DICT format.

        Args:
            fitted_model: Fitted ARDLResults object
            endogenous: Endogenous variable name
            exogenous: List of exogenous variable names
            y_train: Training target (for extracting date range)
            **metadata: Additional metadata (ignored)

        Returns:
            Dictionary with 10 required keys for prepare.py compatibility
        """
        model_dict = {
            'model': fitted_model,
            'endogenous': endogenous,
            'exogenous': exogenous,
            'lags_exogenous': [self.max_lag] * len(exogenous),  # All use same lags
            'lags_endogenous': self.max_lag,
            'n_exog': len(exogenous),
            'n_obs_train': fitted_model.nobs,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'train_period': (str(y_train.index[0]), str(y_train.index[-1])),
            'model_type': 'ARDL'
        }

        return model_dict
