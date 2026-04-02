"""
ECB Data Transformation Pipeline
Handles frequency alignment, stationarity testing, and data transformation
for statistical analysis of ECB time series.

Target start date: 2004-09-01 (latest common start across all 23 series)
All data is actual/observed - no backcasting needed.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import pandas as pd
import numpy as np
from pandas import DataFrame
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from statsmodels.tsa.stattools import adfuller, kpss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Date Parsing
# ============================================================================

class DateParser:
    """Handles parsing of multiple date formats from ECB data."""

    @staticmethod
    def parse_obs_date(date_str: str, freq_code: str) -> pd.Timestamp:
        """
        Parse observation date based on frequency code.

        Args:
            date_str: Date string from database
            freq_code: Frequency code ('D', 'B', 'M', 'Q', 'A')

        Returns:
            pd.Timestamp normalized to midnight (00:00:00)

        Examples:
            >>> DateParser.parse_obs_date('2004-09-06', 'B')
            Timestamp('2004-09-06 00:00:00')
            >>> DateParser.parse_obs_date('2004-09', 'M')
            Timestamp('2004-09-30 00:00:00')
            >>> DateParser.parse_obs_date('2004-Q3', 'Q')
            Timestamp('2004-07-01 00:00:00')
        """
        if freq_code in ['D', 'B']:
            # Daily/Business Daily: YYYY-MM-DD
            return pd.to_datetime(date_str).normalize()
        elif freq_code == 'M':
            # Monthly: YYYY-MM → convert to month-END to match resampled series
            return (pd.to_datetime(date_str + '-01') + pd.offsets.MonthEnd(0)).normalize()
        elif freq_code == 'Q':
            # Quarterly: YYYYQ# → convert via Period
            return pd.PeriodIndex([date_str], freq='Q').to_timestamp()[0].normalize()
        elif freq_code == 'A':
            # Annual: YYYY
            return pd.to_datetime(date_str + '-01-01').normalize()
        else:
            raise ValueError(f"Unsupported frequency code: {freq_code}")

    @staticmethod
    def create_datetime_index(dates: List[str], freq_code: str) -> pd.DatetimeIndex:
        """
        Create DatetimeIndex from list of date strings.

        Args:
            dates: List of date strings
            freq_code: Frequency code

        Returns:
            pd.DatetimeIndex
        """
        return pd.DatetimeIndex([
            DateParser.parse_obs_date(d, freq_code) for d in dates
        ])


# ============================================================================
# Frequency Alignment
# ============================================================================

class FrequencyAligner:
    """Handles frequency conversion with appropriate aggregation methods."""

    # Default resampling methods for each frequency conversion
    RESAMPLE_METHODS = {
        ('D', 'ME'): 'mean',      # Daily to Monthly: average
        ('B', 'ME'): 'mean',      # Business Daily to Monthly: average
        ('Q', 'ME'): 'interpolate', # Quarterly to Monthly: interpolation
        ('ME', 'ME'): 'identity',  # Monthly to Monthly: no-op
        ('ME', 'Q'): 'mean',      # Monthly to Quarterly: average
    }

    def __init__(self, source_freq: str, target_freq: str):
        """
        Initialize frequency aligner.

        Args:
            source_freq: Source frequency code ('D', 'B', 'ME', 'Q')
            target_freq: Target frequency code ('ME', 'Q')
        """
        self.source_freq = source_freq
        self.target_freq = target_freq
        self.default_method = self.RESAMPLE_METHODS.get(
            (source_freq, target_freq), 'mean'
        )

    def align(self, series: pd.Series, method: Optional[str] = None) -> pd.Series:
        """
        Align series to target frequency.

        Args:
            series: Time series with DatetimeIndex
            method: Resampling method ('mean', 'last', 'median', 'interpolate')
                   If None, uses default for frequency pair

        Returns:
            Resampled series
        """
        if method is None:
            method = self.default_method

        # Identity transformation
        if self.source_freq == self.target_freq:
            return series

        # Downsample (higher to lower frequency)
        if self.source_freq in ['D', 'B'] and self.target_freq == 'ME':
            if method == 'mean':
                result = series.resample(self.target_freq).mean()
            elif method == 'last':
                result = series.resample(self.target_freq).last()
            elif method == 'median':
                result = series.resample(self.target_freq).median()
            else:
                result = series.resample(self.target_freq).agg(method)
            # Normalize index to remove time components
            result.index = result.index.normalize()
            return result

        # Upsample (lower to higher frequency)
        elif self.source_freq == 'Q' and self.target_freq == 'ME':
            # Remove any duplicate indices before resampling
            if series.index.duplicated().any():
                series = series[~series.index.duplicated(keep='last')]

            # Reindex to monthly frequency first to establish the grid
            # Create a monthly date range covering the series span
            monthly_index = pd.date_range(
                start=series.index.min(),
                end=series.index.max(),
                freq='ME'
            ).normalize()  # Normalize to midnight

            # Reindex and interpolate
            series_monthly = series.reindex(series.index.union(monthly_index)).sort_index()

            if method == 'interpolate':
                result = series_monthly.interpolate(method='cubic')[monthly_index]
            elif method == 'linear':
                result = series_monthly.interpolate(method='linear')[monthly_index]
            else:
                # Forward fill
                result = series_monthly.ffill()[monthly_index]

            # Normalize index to remove time components
            result.index = result.index.normalize()
            return result

        # Monthly to Quarterly
        elif self.source_freq == 'ME' and self.target_freq == 'Q':
            if method == 'mean':
                result = series.resample('Q').mean()
            elif method == 'last':
                result = series.resample('Q').last()
            else:
                result = series.resample('Q').agg(method)
            # Normalize index to remove time components
            result.index = result.index.normalize()
            return result

        else:
            raise ValueError(
                f"Unsupported frequency conversion: {self.source_freq} -> {self.target_freq}"
            )


# ============================================================================
# Stationarity Testing
# ============================================================================

@dataclass
class ADFTestResult:
    """Container for ADF test results from statsmodels."""
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    used_lag: int
    n_obs: int
    is_stationary: bool  # p_value < 0.05

    def summary(self) -> str:
        """Return formatted summary of test results."""
        return (
            f"ADF Test:\n"
            f"  Statistic: {self.test_statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Is Stationary: {self.is_stationary}\n"
            f"  Critical values: {self.critical_values}\n"
            f"  Lags used: {self.used_lag}, N obs: {self.n_obs}"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'critical_values': self.critical_values,
            'used_lag': self.used_lag,
            'n_obs': self.n_obs,
            'is_stationary': self.is_stationary
        }


@dataclass
class KPSSTestResult:
    """Container for KPSS test results from statsmodels."""
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    used_lag: int
    is_stationary: bool  # p_value > 0.05 (null: stationary)

    def summary(self) -> str:
        """Return formatted summary of test results."""
        return (
            f"KPSS Test:\n"
            f"  Statistic: {self.test_statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Is Stationary: {self.is_stationary}\n"
            f"  Critical values: {self.critical_values}\n"
            f"  Lags used: {self.used_lag}"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_statistic': self.test_statistic,
            'p_value': self.p_value,
            'critical_values': self.critical_values,
            'used_lag': self.used_lag,
            'is_stationary': self.is_stationary
        }


@dataclass
class StationarityResult:
    """Combined results from multiple stationarity tests."""
    series_name: Optional[str]
    n_obs: int
    adf_result: Optional[ADFTestResult]
    kpss_result: Optional[KPSSTestResult]
    overall_stationary: bool
    recommended_transform: str  # 'none', 'diff', 'log', 'log_diff'

    def summary(self) -> str:
        """Return formatted summary of all test results."""
        lines = [
            "="*60,
            f"Stationarity Test Results",
            "="*60,
            f"Series Name: {self.series_name}",
            f"Observations: {self.n_obs}",
            ""
        ]

        if self.adf_result:
            lines.append(self.adf_result.summary())
            lines.append("")

        if self.kpss_result:
            lines.append(self.kpss_result.summary())
            lines.append("")

        lines.extend([
            f"Overall Stationary: {self.overall_stationary}",
            f"Recommended Transformation: {self.recommended_transform}",
            "="*60
        ])

        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        data = {
            'series_name': [self.series_name],
            'n_obs': [self.n_obs],
            'overall_stationary': [self.overall_stationary],
            'recommended_transform': [self.recommended_transform]
        }

        if self.adf_result:
            data['adf_statistic'] = [self.adf_result.test_statistic]
            data['adf_pvalue'] = [self.adf_result.p_value]
            data['adf_stationary'] = [self.adf_result.is_stationary]

        if self.kpss_result:
            data['kpss_statistic'] = [self.kpss_result.test_statistic]
            data['kpss_pvalue'] = [self.kpss_result.p_value]
            data['kpss_stationary'] = [self.kpss_result.is_stationary]

        return pd.DataFrame(data)


class StationarityTester:
    """Performs battery of stationarity tests using statsmodels."""

    def __init__(self, series: pd.Series, series_name: Optional[str] = None):
        """
        Initialize stationarity tester.

        Args:
            series: Time series to test
            series_name: Optional series name for display
        """
        self.series = series.dropna()  # Remove NaN for testing
        self.series_name = series_name

    def test_adf(self, regression: str = 'c', maxlag: Optional[int] = None) -> ADFTestResult:
        """
        Perform Augmented Dickey-Fuller test.

        Args:
            regression: Type of regression ('c', 'ct', 'ctt', 'n')
            maxlag: Maximum lag to use (None = auto)

        Returns:
            ADFTestResult
        """
        if len(self.series) < 50:
            logger.warning(
                f"Series has only {len(self.series)} observations. "
                f"ADF test may be unreliable (recommended: ≥50 obs)"
            )

        result = adfuller(self.series, regression=regression, maxlag=maxlag)

        return ADFTestResult(
            test_statistic=result[0],
            p_value=result[1],
            critical_values=result[4],
            used_lag=result[2],
            n_obs=result[3],
            is_stationary=(result[1] < 0.05)  # Reject null (unit root) if p < 0.05
        )

    def test_kpss(self, regression: str = 'c', nlags: Optional[int] = None) -> KPSSTestResult:
        """
        Perform KPSS test.

        Args:
            regression: Type of regression ('c', 'ct')
            nlags: Number of lags (None = auto)

        Returns:
            KPSSTestResult
        """
        if len(self.series) < 50:
            logger.warning(
                f"Series has only {len(self.series)} observations. "
                f"KPSS test may be unreliable (recommended: ≥50 obs)"
            )

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            result = kpss(self.series, regression=regression, nlags=nlags)

        return KPSSTestResult(
            test_statistic=result[0],
            p_value=result[1],
            critical_values=result[3],
            used_lag=result[2],
            is_stationary=(result[1] > 0.05)  # Do NOT reject null (stationary) if p > 0.05
        )

    def test_all(self, tests: List[str] = ['adf', 'kpss']) -> StationarityResult:
        """
        Run all specified stationarity tests.

        Args:
            tests: List of tests to run ('adf', 'kpss')

        Returns:
            StationarityResult with combined analysis
        """
        adf_result = None
        kpss_result = None

        if 'adf' in tests:
            adf_result = self.test_adf()

        if 'kpss' in tests:
            kpss_result = self.test_kpss()

        # Determine overall stationarity and recommendation
        overall_stationary, recommended_transform = self._recommend_transformation(
            adf_result, kpss_result
        )

        return StationarityResult(
            series_name=self.series_name,
            n_obs=len(self.series),
            adf_result=adf_result,
            kpss_result=kpss_result,
            overall_stationary=overall_stationary,
            recommended_transform=recommended_transform
        )

    def _recommend_transformation(
        self,
        adf_result: Optional[ADFTestResult],
        kpss_result: Optional[KPSSTestResult]
    ) -> Tuple[bool, str]:
        """
        Recommend transformation based on test results.

        Returns:
            (overall_stationary: bool, recommended_transform: str)
        """
        # If only one test ran, use its result
        if adf_result and not kpss_result:
            return adf_result.is_stationary, ('none' if adf_result.is_stationary else 'diff')

        if kpss_result and not adf_result:
            return kpss_result.is_stationary, ('none' if kpss_result.is_stationary else 'diff')

        # Both tests ran: check agreement
        if adf_result and kpss_result:
            adf_stationary = adf_result.is_stationary
            kpss_stationary = kpss_result.is_stationary

            if adf_stationary and kpss_stationary:
                # Both say stationary
                return True, 'none'
            elif not adf_stationary and not kpss_stationary:
                # Both say non-stationary
                return False, 'diff'
            else:
                # Conflicting results: be conservative, suggest differencing
                logger.warning(
                    f"Conflicting stationarity test results (ADF: {adf_stationary}, "
                    f"KPSS: {kpss_stationary}). Recommending differencing to be safe."
                )
                return False, 'diff'

        # No tests ran (shouldn't happen)
        return False, 'diff'


# ============================================================================
# Series Transformation
# ============================================================================

@dataclass
class TransformedSeriesResult:
    """Container for transformed series with metadata."""
    original_series: pd.Series
    transformed_series: pd.Series
    transformations: List[str]
    stationarity_before: Optional[StationarityResult]
    stationarity_after: Optional[StationarityResult]

    def summary(self) -> str:
        """Return formatted summary."""
        lines = [
            "="*60,
            "Transformation Results",
            "="*60,
            f"Transformations applied: {' -> '.join(self.transformations)}",
            f"Original obs: {len(self.original_series)}",
            f"Transformed obs: {len(self.transformed_series)}",
            ""
        ]

        if self.stationarity_before:
            lines.append("BEFORE TRANSFORMATION:")
            lines.append(self.stationarity_before.summary())
            lines.append("")

        if self.stationarity_after:
            lines.append("AFTER TRANSFORMATION:")
            lines.append(self.stationarity_after.summary())

        return "\n".join(lines)


class SeriesTransformer:
    """Applies transformations to achieve stationarity."""

    def __init__(self, series: pd.Series):
        """
        Initialize transformer.

        Args:
            series: Time series to transform
        """
        self.series = series
        self.transformations_applied = []

    def difference(self, order: int = 1, seasonal: bool = False,
                  seasonal_periods: int = 12) -> pd.Series:
        """
        Apply differencing.

        Args:
            order: Difference order
            seasonal: Apply seasonal differencing
            seasonal_periods: Seasonal period (e.g., 12 for monthly)

        Returns:
            Differenced series
        """
        if seasonal:
            result = self.series.diff(periods=seasonal_periods)
            self.transformations_applied.append(f'seasonal_diff({seasonal_periods})')
        else:
            result = self.series.diff(periods=order)
            self.transformations_applied.append(f'diff({order})')

        logger.info(f"Differencing lost {result.isna().sum()} observations (NaN at start)")
        return result

    def log_transform(self) -> pd.Series:
        """
        Apply log transformation.

        Returns:
            Log-transformed series

        Raises:
            ValueError: If series has non-positive values
        """
        if (self.series <= 0).any():
            raise ValueError(
                "Cannot apply log transform to series with non-positive values. "
                "Consider adding a constant or using a different transformation."
            )

        result = np.log(self.series)
        self.transformations_applied.append('log')
        return result

    def log_difference(self, order: int = 1) -> pd.Series:
        """
        Apply log-difference (approximates percentage change).

        Args:
            order: Difference order

        Returns:
            Log-differenced series
        """
        if (self.series <= 0).any():
            raise ValueError(
                "Cannot apply log transform to series with non-positive values."
            )

        result = np.log(self.series).diff(periods=order)
        self.transformations_applied.append(f'log_diff({order})')
        return result

    def percentage_change(self) -> pd.Series:
        """
        Apply percentage change.

        Returns:
            Percentage change series
        """
        result = self.series.pct_change()
        self.transformations_applied.append('pct_change')
        return result

    def apply_transform_chain(self, transforms: List[str],
                              **kwargs) -> TransformedSeriesResult:
        """
        Apply chain of transformations.

        Args:
            transforms: List of transformation names ('diff', 'log', 'log_diff', 'pct_change')
            **kwargs: Additional arguments for specific transformations

        Returns:
            TransformedSeriesResult
        """
        result_series = self.series.copy()
        transformations = []

        for transform in transforms:
            if transform == 'diff':
                result_series = result_series.diff()
                transformations.append('diff')
            elif transform == 'log':
                if (result_series <= 0).any():
                    logger.warning("Cannot apply log to series with non-positive values, skipping")
                    continue
                result_series = np.log(result_series)
                transformations.append('log')
            elif transform == 'log_diff':
                if (result_series <= 0).any():
                    logger.warning("Cannot apply log_diff to series with non-positive values, skipping")
                    continue
                result_series = np.log(result_series).diff()
                transformations.append('log_diff')
            elif transform == 'pct_change':
                result_series = result_series.pct_change()
                transformations.append('pct_change')
            elif transform == 'seasonal_diff':
                periods = kwargs.get('seasonal_periods', 12)
                result_series = result_series.diff(periods=periods)
                transformations.append(f'seasonal_diff({periods})')
            else:
                logger.warning(f"Unknown transformation: {transform}, skipping")

        return TransformedSeriesResult(
            original_series=self.series,
            transformed_series=result_series,
            transformations=transformations,
            stationarity_before=None,
            stationarity_after=None
        )


# ============================================================================
# Main Transformer Class
# ============================================================================

class ECBDataTransformer:
    """
    Handles frequency alignment, stationarity testing, and transformation
    of ECB time series data for statistical analysis.

    Target start date: 2004-09-01 (latest common start across all 23 series)
    All data is actual/observed - no backcasting needed.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the transformation handler.

        Args:
            connection_string: PostgreSQL connection string.
                             If None, loads from DATABASE_URL environment variable.
        """
        load_dotenv()

        if connection_string is None:
            connection_string = os.getenv('DATABASE_URL')
            if connection_string is None:
                raise ValueError(
                    "No database connection string provided. "
                    "Set DATABASE_URL environment variable or pass connection_string."
                )

        self.engine = create_engine(connection_string)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        logger.info("Database connection established for transformation pipeline")

    # ========================================================================
    # Loading Methods
    # ========================================================================

    def load_series(self, series_id: int, parse_dates: bool = True) -> pd.Series:
        """
        Load a single time series from database.

        Args:
            series_id: Series ID from series_metadata table
            parse_dates: Convert obs_date to DatetimeIndex

        Returns:
            pd.Series with DatetimeIndex (if parse_dates=True) or string index
        """
        # Get series metadata
        metadata_query = text("""
            SELECT s.series_id, s.fact_table, s.frequency_code, s.series_label
            FROM series_metadata s
            WHERE s.series_id = :sid
        """)

        with self.engine.connect() as conn:
            metadata = conn.execute(metadata_query, {'sid': series_id}).fetchone()

            if metadata is None:
                raise ValueError(f"Series ID {series_id} not found in database")

            fact_table, freq_code, series_label = metadata[1], metadata[2], metadata[3]

            # Load observations
            obs_query = text(f"""
                SELECT obs_date, obs_value
                FROM {fact_table}
                WHERE series_id = :sid
                ORDER BY obs_date
            """)

            result = conn.execute(obs_query, {'sid': series_id})
            dates, values = [], []

            for row in result:
                dates.append(row[0])
                values.append(row[1])

        # Create Series
        if parse_dates:
            index = DateParser.create_datetime_index(dates, freq_code)
        else:
            index = dates

        # Use series_label if available, otherwise fallback to series_id
        series_name = series_label if series_label else f"series_{series_id}"
        series = pd.Series(values, index=index, name=series_name)

        logger.info(
            f"Loaded series {series_id} ({series_name}) from {fact_table}: "
            f"{len(series)} obs from {series.index[0]} to {series.index[-1]}"
        )

        return series

    def load_fact_table(self, table_name: str, parse_dates: bool = True) -> pd.DataFrame:
        """
        Load all series from a fact table as wide DataFrame.

        Args:
            table_name: One of 'fsi', 'yc', 'mmr', 'inf', 'ir', 'lfi', 'mpd'
            parse_dates: Convert obs_date to DatetimeIndex

        Returns:
            pd.DataFrame with columns for each series_id in that table
        """
        # Get all series IDs for this table
        query = text("""
            SELECT series_id, frequency_code
            FROM series_metadata
            WHERE fact_table = :table
            ORDER BY series_id
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {'table': table_name.lower()})
            series_info = [(row[0], row[1]) for row in result]

        if not series_info:
            raise ValueError(f"No series found for fact table: {table_name}")

        # Load each series
        series_dict = {}
        for sid, freq_code in series_info:
            loaded_series = self.load_series(sid, parse_dates=parse_dates)
            series_dict[loaded_series.name] = loaded_series

        # Combine into DataFrame
        df = pd.DataFrame(series_dict)

        logger.info(f"Loaded {len(series_info)} series from {table_name}: shape {df.shape}")

        return df

    def load_all_series(self, freq: Optional[str] = None) -> Dict[int, pd.Series]:
        """
        Load all series from database.

        Args:
            freq: Optional frequency filter ('D', 'B', 'M', 'Q')

        Returns:
            Dictionary of {series_id: pd.Series}
        """
        query = text("""
            SELECT series_id
            FROM series_metadata
            ORDER BY series_id
        """)

        if freq:
            query = text("""
                SELECT series_id
                FROM series_metadata
                WHERE frequency_code = :freq
                ORDER BY series_id
            """)

        with self.engine.connect() as conn:
            if freq:
                result = conn.execute(query, {'freq': freq})
            else:
                result = conn.execute(query)
            series_ids = [row[0] for row in result]

        logger.info(f"Loading {len(series_ids)} series from database...")

        series_dict = {}
        for sid in series_ids:
            series_dict[sid] = self.load_series(sid, parse_dates=True)

        return series_dict

    # ========================================================================
    # Frequency Alignment
    # ========================================================================

    def align_to_frequency(
        self,
        series_dict: Dict[int, pd.Series],
        target_freq: str = 'ME',
        method: str = 'auto',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        fill_method: str = 'ffill',
        fill_limit: int = 3
    ) -> pd.DataFrame:
        """
        Align multiple time series to common frequency.

        Args:
            series_dict: {series_id: pd.Series} with DatetimeIndex
            target_freq: 'D', 'ME', 'Q' (default: 'ME')
            method: 'auto', 'mean', 'last', 'median', 'interpolate'
            start_date: Trim to start date (e.g., '2004-09-01')
            end_date: Trim to end date (e.g., '2026-01-01')
            fill_method: How to handle missing after alignment
            fill_limit: Max consecutive NaN to fill

        Returns:
            pd.DataFrame with aligned series as columns, DatetimeIndex
        """
        # Detect source frequencies
        freq_map = {}
        for sid, series in series_dict.items():
            # Infer frequency from index
            inferred_freq = pd.infer_freq(series.index[:min(10, len(series))])
            if inferred_freq is None or inferred_freq not in ['D', 'B', 'M', 'Q', 'MS', 'QS']:
                # Query database for frequency
                query = text("SELECT frequency_code FROM series_metadata WHERE series_id = :sid")
                with self.engine.connect() as conn:
                    freq_code = conn.execute(query, {'sid': sid}).scalar()
                # Map database freq codes to pandas codes
                db_freq_mapping = {'D': 'D', 'B': 'B', 'M': 'ME', 'Q': 'Q'}
                freq_map[sid] = db_freq_mapping.get(freq_code, freq_code)
            else:
                # Map pandas freq to ECB freq codes
                freq_mapping = {'D': 'D', 'B': 'B', 'M': 'ME', 'MS': 'ME', 'Q': 'Q', 'QS': 'Q'}
                freq_map[sid] = freq_mapping.get(inferred_freq, 'ME')

        # Align each series
        aligned_series = {}
        for sid, series in series_dict.items():
            source_freq = freq_map[sid]

            # Use series.name if available, otherwise fallback to series_id
            col_name = series.name if hasattr(series, 'name') and series.name else f"series_{sid}"

            if source_freq == target_freq:
                aligned_series[col_name] = series
            else:
                aligner = FrequencyAligner(source_freq, target_freq)
                aligned_series[col_name] = aligner.align(series, method if method != 'auto' else None)

        # Combine into DataFrame
        df = pd.DataFrame(aligned_series)

        # Normalize index to ensure all timestamps are at midnight
        df.index = df.index.normalize()

        # Handle any duplicate index values (keep last)
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate dates after alignment, consolidating...")
            # Group by index and take last non-NaN value for each group
            df = df.groupby(level=0).last()

        # Determine date range
        if start_date is None:
            # Use latest common start date
            start_date = df.index.max()
            for col in df.columns:
                first_valid = df[col].first_valid_index()
                if first_valid and first_valid > start_date:
                    start_date = first_valid
        else:
            start_date = pd.to_datetime(start_date)

        if end_date is not None:
            end_date = pd.to_datetime(end_date)

        # Trim to date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # Handle missing values
        if fill_method == 'ffill':
            df = df.ffill(limit=fill_limit)
        elif fill_method == 'bfill':
            df = df.bfill(limit=fill_limit)
        elif fill_method == 'interpolate':
            df = df.interpolate(method='linear', limit=fill_limit)

        logger.info(
            f"Aligned {len(series_dict)} series to {target_freq} frequency: "
            f"shape {df.shape}, range {df.index.min()} to {df.index.max()}"
        )

        return df

    # ========================================================================
    # Stationarity Testing
    # ========================================================================

    def test_stationarity(
        self,
        series: pd.Series,
        tests: List[str] = ['adf', 'kpss'],
        adf_regression: str = 'c',
        kpss_regression: str = 'c',
        verbose: bool = True
    ) -> StationarityResult:
        """
        Run battery of stationarity tests using statsmodels.

        Args:
            series: Time series to test (must not have NaN)
            tests: List of tests to run ['adf', 'kpss']
            adf_regression: ADF regression type ('c', 'ct', 'ctt', 'n')
            kpss_regression: KPSS regression type ('c', 'ct')
            verbose: Print test results

        Returns:
            StationarityResult with all test outcomes and recommendation
        """
        series_name = series.name

        tester = StationarityTester(series, series_name)
        result = tester.test_all(tests=tests)

        if verbose:
            print(result.summary())

        return result

    # ========================================================================
    # Transformation
    # ========================================================================

    def transform_series(
        self,
        series: pd.Series,
        transformations: List[str] = ['diff'],
        test_after: bool = True,
        **kwargs
    ) -> TransformedSeriesResult:
        """
        Apply transformation chain to series.

        Args:
            series: Time series to transform
            transformations: Ordered list from ['diff', 'log', 'log_diff',
                            'pct_change', 'seasonal_diff']
            test_after: Run stationarity tests on transformed series
            **kwargs: Additional args for specific transformations
                - seasonal_periods: For seasonal_diff (default: 12)

        Returns:
            TransformedSeriesResult containing original, transformed,
            and test results
        """
        # Test before transformation
        stationarity_before = None
        if test_after:
            stationarity_before = self.test_stationarity(series, verbose=False)

        # Apply transformations
        transformer = SeriesTransformer(series)
        result = transformer.apply_transform_chain(transformations, **kwargs)

        # Test after transformation
        if test_after:
            stationarity_after = self.test_stationarity(
                result.transformed_series.dropna(),
                verbose=False
            )
            result.stationarity_before = stationarity_before
            result.stationarity_after = stationarity_after

        return result

    def transform_until_stationary(
        self,
        series: pd.Series,
        max_iterations: int = 3,
        min_observations: int = 50,
        significance_level: float = 0.05,
        verbose: bool = True
    ) -> Tuple[pd.Series, List[str], StationarityResult]:
        """
        Iteratively transform a series until it becomes stationary.

        Applies transformations in a loop, testing after each transformation,
        until the series becomes stationary or max_iterations is reached.

        Args:
            series: Time series to transform
            max_iterations: Maximum number of transformation iterations (default: 3)
            min_observations: Minimum observations required after transformation
            significance_level: P-value threshold for stationarity tests
            verbose: Log transformation steps

        Returns:
            Tuple of:
                - Transformed series (or original if already stationary)
                - List of transformations applied (e.g., ['diff', 'diff'] for I(2))
                - Final stationarity test result

        Raises:
            ValueError: If series has insufficient observations
        """
        current_series = series.dropna().copy()
        transform_history = []

        # Check minimum observations
        if len(current_series) < min_observations:
            raise ValueError(
                f"Series has {len(current_series)} observations, "
                f"minimum {min_observations} required"
            )

        # Initial stationarity test
        current_result = self.test_stationarity(
            current_series,
            verbose=False
        )

        if verbose:
            logger.info(
                f"Initial test for {series.name}: "
                f"Stationary={current_result.overall_stationary}"
            )

        # Iterative transformation loop
        iteration = 0
        while not current_result.overall_stationary and iteration < max_iterations:
            iteration += 1

            # Determine transformation to apply based on current test results
            recommended = current_result.recommended_transform

            if recommended == 'none':
                # Edge case: recommended none but overall_stationary is False
                # This means conflicting results - apply diff anyway
                recommended = 'diff'

            # Apply transformation
            if recommended == 'diff':
                transformed = current_series.diff().dropna()
            elif recommended == 'log_diff':
                if (current_series > 0).all():
                    transformed = np.log(current_series).diff().dropna()
                else:
                    # Fallback to simple diff if log not applicable
                    transformed = current_series.diff().dropna()
                    recommended = 'diff'
            elif recommended == 'log':
                if (current_series > 0).all():
                    transformed = np.log(current_series).dropna()
                else:
                    # Fallback to diff
                    transformed = current_series.diff().dropna()
                    recommended = 'diff'
            else:
                # Default to diff
                transformed = current_series.diff().dropna()
                recommended = 'diff'

            transform_history.append(recommended)

            # Check if we still have enough observations
            if len(transformed) < min_observations:
                logger.warning(
                    f"Iteration {iteration}: Insufficient observations after {recommended} "
                    f"({len(transformed)} < {min_observations}). Stopping transformation."
                )
                break

            if verbose:
                logger.info(
                    f"Iteration {iteration}: Applied '{recommended}' "
                    f"({len(current_series)} → {len(transformed)} obs)"
                )

            # Update current series and re-test
            current_series = transformed
            current_result = self.test_stationarity(
                current_series,
                verbose=False
            )

            if verbose:
                adf_p = current_result.adf_result.p_value if current_result.adf_result else None
                kpss_p = current_result.kpss_result.p_value if current_result.kpss_result else None
                adf_str = f"{adf_p:.6f}" if adf_p is not None else "N/A"
                kpss_str = f"{kpss_p:.6f}" if kpss_p is not None else "N/A"
                logger.info(
                    f"Iteration {iteration}: Stationary={current_result.overall_stationary} "
                    f"(ADF p={adf_str}, KPSS p={kpss_str})"
                )

        # Final status
        if current_result.overall_stationary:
            if verbose:
                logger.info(
                    f"✓ Achieved stationarity after {len(transform_history)} "
                    f"transformation(s): {' → '.join(transform_history) if transform_history else 'none'}"
                )
        else:
            if iteration >= max_iterations:
                logger.warning(
                    f"✗ Failed to achieve stationarity after {max_iterations} iterations. "
                    f"Series may be I({max_iterations + 1}) or have structural breaks."
                )
            else:
                logger.warning(
                    f"✗ Transformation stopped due to insufficient observations."
                )

        return current_series, transform_history, current_result

    # ========================================================================
    # Master Pipeline
    # ========================================================================

    def create_analysis_dataset(
        self,
        target_freq: str = 'ME',
        start_date: str = '2004-09-01',
        end_date: Optional[str] = None,
        include_projections: bool = False,
        auto_transform: bool = False,
        max_transform_iterations: int = 3,
        min_observations_after_transform: int = 50,
        series_ids: Optional[List[int]] = None,
        output_format: str = 'dataframe',
        return_stationarity_report: bool = True
    ) -> tuple[DataFrame, DataFrame, DataFrame] | DataFrame:
        """
        End-to-end pipeline: load → align → test → transform → export.

        Args:
            target_freq: Target frequency ('ME' recommended)
            start_date: Trim start (default: '2004-09-01')
            end_date: Trim end (None = latest actual data)
            include_projections: Include MPD projections beyond 2026
            auto_transform: Automatically transform non-stationary series iteratively
            max_transform_iterations: Maximum transformation iterations (default: 3)
                Allows handling I(2), I(3) processes by repeated differencing
            min_observations_after_transform: Minimum obs to retain after transform (default: 50)
            series_ids: Specific series to include (None = all 23)
            output_format: 'dataframe' (only supported format for now)
            return_stationarity_report: Also return test results DataFrame

        Returns:
            If return_stationarity_report=False:
                Aligned (and optionally transformed) DataFrame
            If return_stationarity_report=True:
                Tuple of (data_df, stationarity_report_df)
                Note: stationarity_report includes 'transform_applied' column showing
                transformation history (e.g., 'diff→diff' for second differencing)
        """
        # Set default end date if not including projections
        if end_date is None and not include_projections:
            end_date = '2026-01-01'

        # Load series
        if series_ids:
            logger.info(f"Loading {len(series_ids)} specified series...")
            series_dict = {sid: self.load_series(sid) for sid in series_ids}
        else:
            logger.info("Loading all 23 series...")
            series_dict = self.load_all_series()

        # Align to target frequency
        logger.info(f"Aligning to {target_freq} frequency from {start_date}...")
        aligned_data = self.align_to_frequency(
            series_dict=series_dict,
            target_freq=target_freq,
            start_date=start_date,
            end_date=end_date
        )

        # Test stationarity
        logger.info("Testing stationarity for all series...")
        stationarity_results = []

        for col in aligned_data.columns:
            series = aligned_data[col].dropna()
            if len(series) < 10:
                logger.warning(f"Skipping {col}: insufficient data ({len(series)} obs)")
                continue

            result = self.test_stationarity(series, verbose=False)
            stationarity_results.append(result.to_dataframe())

        stationarity_report = pd.concat(stationarity_results, ignore_index=True)

        # Auto-transform if requested (iterative approach)
        if auto_transform:
            logger.info("Auto-transforming non-stationary series (iterative)...")
            transformed_data = aligned_data.copy()
            final_stationarity_results = []

            for col in aligned_data.columns:
                series = aligned_data[col].dropna()

                if len(series) < min_observations_after_transform:
                    logger.warning(
                        f"Skipping {col}: insufficient data ({len(series)} obs)"
                    )
                    final_stationarity_results.append({
                        'series_name': col,
                        'overall_stationary': False,
                        'transform_applied': 'skipped',
                        'reason': 'insufficient_observations'
                    })
                    continue

                try:
                    # Iteratively transform until stationary
                    transformed_series, transform_history, final_result = \
                        self.transform_until_stationary(
                            series,
                            max_iterations=max_transform_iterations,
                            min_observations=min_observations_after_transform,
                            verbose=True
                        )

                    # Update the DataFrame with transformed series
                    transformed_data[col] = transformed_series

                    # Record transformation history
                    result_dict = final_result.to_dataframe().iloc[0].to_dict()
                    result_dict['transform_applied'] = ' → '.join(transform_history) if transform_history else 'none'
                    result_dict['iterations'] = len(transform_history)
                    final_stationarity_results.append(result_dict)

                except Exception as e:
                    logger.error(f"Failed to transform {col}: {e}")
                    # Keep original series on error
                    result_dict = stationarity_report[
                        stationarity_report['series_name'] == col
                    ].iloc[0].to_dict()
                    result_dict['transform_applied'] = 'error'
                    result_dict['iterations'] = 0
                    final_stationarity_results.append(result_dict)

            aligned_data = transformed_data

            # Update stationarity report with final results
            final_stationarity_results = pd.DataFrame(final_stationarity_results)

        else:
            final_stationarity_results = None

        logger.info(f"Analysis dataset created: shape {aligned_data.shape}")

        if return_stationarity_report:
            return aligned_data, stationarity_report, final_stationarity_results
        else:
            return aligned_data

    # ========================================================================
    # Export
    # ========================================================================

    def export_to_table(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = 'replace'
    ) -> Dict[str, Any]:
        """
        Export DataFrame to PostgreSQL table.

        Args:
            data: DataFrame to export
            table_name: Target table name
            if_exists: 'replace', 'append', or 'fail'

        Returns:
            Dictionary with export statistics
        """
        logger.info(f"Exporting {data.shape[0]} rows to table '{table_name}'...")

        data.to_sql(
            name=table_name,
            con=self.engine,
            if_exists=if_exists,
            index=True,
            index_label='obs_date'
        )

        stats = {
            'table': table_name,
            'rows': len(data),
            'columns': len(data.columns),
            'date_range': (str(data.index.min()), str(data.index.max()))
        }

        logger.info(f"Export completed: {stats}")

        return stats


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """Command-line interface for transform module."""
    import argparse

    parser = argparse.ArgumentParser(description='ECB Data Transformation Pipeline')
    parser.add_argument('--create-dataset', action='store_true',
                       help='Create complete analysis dataset')
    parser.add_argument('--start-date', default='2004-09-01',
                       help='Start date (default: 2004-09-01)')
    parser.add_argument('--end-date', default='2026-01-01',
                       help='End date (default: 2026-01-01)')
    parser.add_argument('--auto-transform', action='store_true',
                       help='Automatically transform non-stationary series')
    parser.add_argument('--output', default='aligned_data.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    transformer = ECBDataTransformer()

    if args.create_dataset:
        logger.info("Creating complete analysis dataset...")
        data, report = transformer.create_analysis_dataset(
            start_date=args.start_date,
            end_date=args.end_date,
            auto_transform=args.auto_transform,
            return_stationarity_report=True
        )

        # Save data
        data.to_csv(args.output)
        logger.info(f"Data saved to {args.output}")

        # Save stationarity report
        report_path = args.output.replace('.csv', '_stationarity_report.csv')
        report.to_csv(report_path, index=False)
        logger.info(f"Stationarity report saved to {report_path}")

        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        print(f"Shape: {data.shape}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"\nStationarity Report:")
        print(report.to_string(index=False))
        print("="*80)


if __name__ == "__main__":
    main()
