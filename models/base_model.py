"""
Abstract base class for time series models.

This interface enforces the contract that prepare.py depends on - all models
must return a dictionary with exactly these 11 keys:
- model: Fitted model object
- endogenous: Endogenous variable name
- exogenous: List of exogenous variable names
- lags_exogenous: Lags used for exogenous variables
- lags_endogenous: Lags used for endogenous variable
- n_exog: Number of exogenous variables
- n_obs_train: Number of training observations
- aic: Akaike Information Criterion
- bic: Bayesian Information Criterion
- train_period: Tuple of (start_date, end_date) as strings
- model_type: Model type identifier ('ARDL', 'VAR', 'BVAR', 'VECM')
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series models.

    Any model implementation (ARDL, VAR, VECM, BVAR, etc.) must implement
    this interface to ensure compatibility with prepare.py.
    """

    @abstractmethod
    def fit(self, y_train: pd.Series, X_train: pd.DataFrame, **kwargs) -> Any:
        """
        Fit the time series model.

        Args:
            y_train: Training target (endogenous variable)
            X_train: Training features (exogenous variables)
            **kwargs: Model-specific parameters

        Returns:
            Fitted model object (model-specific type)
        """
        pass

    @abstractmethod
    def to_model_dict(
        self,
        fitted_model: Any,
        endogenous: str,
        exogenous: List[str],
        y_train: pd.Series,
        **metadata
    ) -> Dict:
        """
        Convert fitted model to standard MODEL_DICT format.

        This method MUST return a dictionary with these 11 required keys:
        - model: Fitted model object
        - endogenous: Endogenous variable name
        - exogenous: List of exogenous variable names
        - lags_exogenous: Lags used for exogenous variables
        - lags_endogenous: Lags used for endogenous variable
        - n_exog: Number of exogenous variables
        - n_obs_train: Number of training observations
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
        - train_period: Tuple of (start_date, end_date) as strings
        - model_type: Model type identifier ('ARDL', 'VAR', 'BVAR', 'VECM')

        Args:
            fitted_model: The fitted model object
            endogenous: Endogenous variable name
            exogenous: List of exogenous variable names
            y_train: Training target (for extracting metadata)
            **metadata: Additional model-specific metadata

        Returns:
            Dictionary with standardized model information
        """
        pass
