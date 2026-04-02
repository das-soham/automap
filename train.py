"""
Economic Modeling Framework
1. GPR Composite Index Construction - Blends 18 country GPR indices
2. ARDL Model Building - Fits macroeconomic variable relationships
"""

import logging
from typing import Dict
import pandas as pd


from config import MEV_REL, TRAIN_START_DT, TRAIN_END_DT, MAX_LAGS, MIN_LAGS

# Import utilities
from utilities import (
    load_data,
    parse_exogenous_vars,
    prepare_training_data,
    create_gpr_composite,
    print_model_summary,
    setup_logging
)

# Import models
from models import ARDLModel

# Configure logging (will be set up in main)
logger = logging.getLogger(__name__)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Hard-coded random selection of additional exogenous variables (3 variables)
RANDOM_EXOG_SUBSET = ['CISS_SOV', 'EUR_3M', 'AAA_5Y']

# Model selection - Experimenter changes this to swap models!
ModelClass = ARDLModel


# ============================================================================
# MODEL BUILDING ORCHESTRATION
# ============================================================================

def build_model_dict(data_file: str = 'data.parquet') -> Dict:
    """
    Build MODEL_DICT for all relationships in MEV_REL.

    Args:
        data_file: Path to parquet data file

    Returns:
        MODEL_DICT with all fitted models
    """
    logger.info("="*70)
    logger.info("BUILDING MODEL DICTIONARY")
    logger.info("="*70)

    # Load data
    data = load_data(data_file)

    # Get available variables
    available_vars = [col for col in data.columns if col in data.columns]

    logger.info(f"\nAvailable variables: {len(available_vars)}")
    logger.info(f"Random exogenous subset: {RANDOM_EXOG_SUBSET}")
    logger.info(f"Max lags: {MAX_LAGS}")
    logger.info(f"Training period: {TRAIN_START_DT} to {TRAIN_END_DT}")

    MODEL_DICT = {}

    # Process each relationship
    for idx, (exog_spec, endog) in enumerate(MEV_REL, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Relationship {idx}/{len(MEV_REL)}: {exog_spec} -> {endog}")
        logger.info(f"{'='*70}")

        # Skip if endogenous is empty
        if not endog:
            logger.warning(f"Skipping empty endogenous variable")
            continue

        # Parse exogenous variables
        exog_primary = parse_exogenous_vars(exog_spec)

        # Skip if no valid exogenous
        if not exog_primary:
            logger.warning(f"Skipping - no valid exogenous variables")
            continue

        # Check if variables exist
        if endog not in data.columns:
            logger.error(f"Endogenous variable {endog} not found in data")
            continue

        missing_exog = [var for var in exog_primary if var not in data.columns]
        if missing_exog:
            logger.error(f"Exogenous variables not found: {missing_exog}")
            continue

        # Add random subset of other variables
        # exog_other = [var for var in RANDOM_EXOG_SUBSET
        #               if var in data.columns and var != endog and var not in exog_primary]

        # Combine all exogenous variables - use only primary for simplicity
        all_exog = exog_primary  # + exog_other

        # Prepare data
        try:
            y_train, X_train, y_test, X_test = prepare_training_data(
                data=data,
                endogenous=endog,
                exogenous_vars=all_exog
            )

            # Initialize model builder
            model_builder = ModelClass(max_lag=3)  # Further reduced

            # Fit model
            fitted_model = model_builder.fit(y_train=y_train, X_train=X_train)

            # Convert to standard MODEL_DICT format
            model_info = model_builder.to_model_dict(
                fitted_model=fitted_model,
                endogenous=endog,
                exogenous=all_exog,
                y_train=y_train
            )

            # Store in MODEL_DICT with unique key
            model_key = f"{endog}_model"

            # Handle duplicate endogenous variables
            if model_key in MODEL_DICT:
                counter = 2
                while f"{endog}_model_{counter}" in MODEL_DICT:
                    counter += 1
                model_key = f"{endog}_model_{counter}"
                logger.info(f"  ⚠ Duplicate endogenous '{endog}' detected, using unique key")

            MODEL_DICT[model_key] = model_info
            logger.info(f"  ✓ Model stored as '{model_key}'")

        except Exception as e:
            logger.error(f"Failed to build model for {endog}: {e}")
            continue

    logger.info(f"\n{'='*70}")
    logger.info(f"MODEL DICTIONARY COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total models fitted: {len(MODEL_DICT)}/{len(MEV_REL)}")
    logger.info(f"Models: {list(MODEL_DICT.keys())}")

    return MODEL_DICT


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import argparse

    # Setup logging to both console and file
    setup_logging(log_dir='logs', log_level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Economic Modeling Framework: GPR Composite + ARDL Models'
    )
    parser.add_argument(
        '--mode',
        choices=['gpr', 'model', 'both'],
        default='both',
        help='Mode: gpr (composite only), model (models only), both (default)'
    )
    parser.add_argument('--data', default='data_gpr.parquet', help='Input parquet file')
    parser.add_argument('--start-date', default='2004-09-01', help='Start date')
    parser.add_argument('--end-date', default='2026-01-01', help='End date')
    parser.add_argument('--gpr-output', default='data_gpr.parquet', help='GPR output file')
    parser.add_argument('--model-output', default='MODEL_DICT.pkl', help='Model dict output file')

    args = parser.parse_args()

    if args.mode in ['gpr', 'both']:
        print("\n" + "="*70)
        print("CREATING GPR COMPOSITE")
        print("="*70)
        composite = create_gpr_composite(
            data_file=args.data,
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=args.gpr_output
        )
        print(f"\nStatistics:\n{composite.describe()}")

    if args.mode in ['model', 'both']:
        print("\n" + "="*70)
        print("BUILDING MODELS")
        print("="*70)
        MODEL_DICT = build_model_dict(data_file=args.data)
        print_model_summary(MODEL_DICT)

        # Save MODEL_DICT
        import pickle
        with open(args.model_output, 'wb') as f:
            pickle.dump(MODEL_DICT, f)
        logger.info(f"\nSaved MODEL_DICT to {args.model_output}")

    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
