"""
Feature engineering utilities for economic indicators.
"""

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


# GDP weights (billion EUR, 2020) - economic importance
# All 18 countries in dataset (EU members + major neighbors)
GDP_WEIGHTS = {
    'GPR_BEL': 432.2,   # Belgium
    'GPR_CHE': 631.9,   # Switzerland
    'GPR_DEU': 3332.0,  # Germany
    'GPR_DNK': 302.5,   # Denmark
    'GPR_ESP': 1121.0,  # Spain
    'GPR_FIN': 236.8,   # Finland
    'GPR_FRA': 2303.0,  # France
    'GPR_GBR': 2407.0,  # United Kingdom
    'GPR_HUN': 139.8,   # Hungary
    'GPR_ITA': 1658.0,  # Italy
    'GPR_NLD': 771.2,   # Netherlands
    'GPR_NOR': 326.9,   # Norway
    'GPR_POL': 524.0,   # Poland
    'GPR_PRT': 200.3,   # Portugal
    'GPR_RUS': 1483.0,  # Russia
    'GPR_SWE': 475.3,   # Sweden
    'GPR_TUR': 648.0,   # Turkey
    'GPR_UKR': 155.6,   # Ukraine
}


def create_gpr_composite(
    data_file: str = 'data.parquet',
    start_date: str = '2004-09-01',
    end_date: str = '2026-01-01',
    output_file: str = None
) -> pd.Series:
    """
    Create GDP-weighted geopolitical risk composite from all 18 countries.

    Includes: EU members (12) + neighbors (UK, Switzerland, Norway, Russia, Turkey, Ukraine).

    Args:
        data_file: Path to parquet file with transformed data
        start_date: Start date for data (for filtering)
        end_date: End date for data (for filtering)
        output_file: Output Parquet file path (optional). If provided, saves entire DataFrame
                     with non-GPR columns plus the GPR_EU composite column.

    Returns:
        Composite GPR series
    """
    # Check if data file exists
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Please run transform.py first to create the transformed dataset."
        )

    logger.info(f"Loading data from {data_file}...")

    # Load transformed data from parquet
    data = pd.read_parquet(data_file)
    logger.info(f"Loaded data shape: {data.shape}")

    # Filter by date range
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    logger.info(f"Filtered to {start_date} - {end_date}: {data.shape}")

    # Extract GPR columns
    gpr_cols = [col for col in data.columns if col in GDP_WEIGHTS]
    non_gpr_cols = [col for col in data.columns if col not in GDP_WEIGHTS]
    gpr_data = data[gpr_cols]

    logger.info(f"Found {len(gpr_cols)} GPR series: {gpr_cols}")

    # Normalize GDP weights to sum to 1
    total_gdp = sum(GDP_WEIGHTS[col] for col in gpr_cols)
    weights = {col: GDP_WEIGHTS[col] / total_gdp for col in gpr_cols}

    logger.info(f"Top 3 weights: {sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]}")

    # Create weighted composite
    composite = pd.Series(0.0, index=gpr_data.index, name='GPR_EU')
    for col in gpr_cols:
        composite += gpr_data[col].fillna(0) * weights[col]

    # Save to Parquet
    if output_file:
        data = data[non_gpr_cols]
        data['GPR_EU'] = composite
        data.to_parquet(output_file)
        logger.info(f"Saved composite to {output_file}")

    # Summary
    logger.info(f"\nComposite Summary:")
    logger.info(f"  Observations: {len(composite.dropna())}")
    logger.info(f"  Date range: {composite.index[0]} to {composite.index[-1]}")
    logger.info(f"  Mean: {composite.mean():.4f}")
    logger.info(f"  Std: {composite.std():.4f}")

    return composite
