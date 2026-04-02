"""
Display utilities for model summaries.
"""

from typing import Dict


def print_model_summary(model_dict: Dict) -> None:
    """Print summary of all fitted models."""
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)

    for model_key, model_info in model_dict.items():
        print(f"\n{model_key}:")
        print(f"  Model Type: {model_info.get('model_type', 'Unknown')}")
        print(f"  Endogenous: {model_info['endogenous']}")
        print(f"  Exogenous: {', '.join(model_info['exogenous'])}")
        print(f"  Lags (exog): {model_info['lags_exogenous']}")
        print(f"  Lags (endog): {model_info['lags_endogenous']}")
        print(f"  N exog vars: {model_info['n_exog']}")
        print(f"  Train obs: {model_info['n_obs_train']}")
        print(f"  AIC: {model_info['aic']:.2f}")
        print(f"  BIC: {model_info['bic']:.2f}")
