# Experimenter's Guide: How to Swap Models

## Overview

The refactored `train.py` makes it trivial to experiment with different time series models. You only need to change **ONE LINE** to switch between models!

## Architecture

```
automap/
├── train.py              # Orchestration + Experimentation (210 lines)
├── utilities/            # Reusable infrastructure
│   ├── data_io.py       # Data loading
│   ├── data_prep.py     # Data preparation
│   ├── feature_eng.py   # Feature engineering (GPR composite)
│   └── display.py       # Display utilities
└── models/              # Swappable model implementations
    ├── base_model.py    # Abstract interface (contract)
    ├── ardl_model.py    # ARDL implementation
    └── var_model.py     # VAR implementation (stub)
```

## Current Model: ARDL

By default, `train.py` uses ARDL (line 38):

```python
# Model selection - Experimenter changes this to swap models!
ModelClass = ARDLModel
```

## How to Swap to VAR

### Step 1: Import the new model

Add VAR to the imports (line 21):

```python
# Import models
from models import ARDLModel, VARModel
```

### Step 2: Change ONE line

Update the model selection (line 38):

```python
# OLD:
ModelClass = ARDLModel

# NEW:
ModelClass = VARModel
```

### Step 3: Run training

```bash
python train.py --mode model
```

That's it! The MODEL_DICT.pkl will be generated with VAR models instead of ARDL, and `prepare.py` will work unchanged.

## How to Add a New Model (e.g., VECM)

### Step 1: Create the model class

Create `models/vecm_model.py`:

```python
from .base_model import BaseTimeSeriesModel
import pandas as pd
from typing import Any, Dict, List

class VECMModel(BaseTimeSeriesModel):
    def __init__(self, max_lag: int = 12):
        self.max_lag = max_lag

    def fit(self, y_train: pd.Series, X_train: pd.DataFrame, **kwargs) -> Any:
        # Implement VECM fitting using statsmodels
        # from statsmodels.tsa.vector_ar.vecm import VECM
        pass

    def to_model_dict(self, fitted_model: Any, endogenous: str,
                     exogenous: List[str], y_train: pd.Series, **metadata) -> Dict:
        # Convert to standard MODEL_DICT format with 10 required keys
        return {
            'model': fitted_model,
            'endogenous': endogenous,
            'exogenous': exogenous,
            'lags_exogenous': [self.max_lag] * len(exogenous),
            'lags_endogenous': self.max_lag,
            'n_exog': len(exogenous),
            'n_obs_train': fitted_model.nobs,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'train_period': (str(y_train.index[0]), str(y_train.index[-1]))
        }
```

### Step 2: Export the model

Update `models/__init__.py`:

```python
from .base_model import BaseTimeSeriesModel
from .ardl_model import ARDLModel
from .var_model import VARModel
from .vecm_model import VECMModel

__all__ = ['BaseTimeSeriesModel', 'ARDLModel', 'VARModel', 'VECMModel']
```

### Step 3: Use it in train.py

```python
from models import ARDLModel, VARModel, VECMModel

# Change this line:
ModelClass = VECMModel
```

## The MODEL_DICT Contract

All models MUST return a dictionary with these 10 keys (enforced by `BaseTimeSeriesModel`):

1. **model**: Fitted model object
2. **endogenous**: Endogenous variable name (str)
3. **exogenous**: List of exogenous variable names (List[str])
4. **lags_exogenous**: Lags used for exogenous variables (List[int])
5. **lags_endogenous**: Lags used for endogenous variable (int)
6. **n_exog**: Number of exogenous variables (int)
7. **n_obs_train**: Number of training observations (int or float)
8. **aic**: Akaike Information Criterion (float)
9. **bic**: Bayesian Information Criterion (float)
10. **train_period**: Tuple of (start_date, end_date) as strings

This guarantees that `prepare.py` will work unchanged with any model!

## Experimenter Workflow Summary

```
┌─────────────────────────────────────┐
│ 1. Implement NewModel class         │
│    (in models/new_model.py)         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 2. Update models/__init__.py        │
│    (export NewModel)                │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 3. Change ONE line in train.py      │
│    ModelClass = NewModel            │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 4. Run: python train.py --mode model│
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ 5. prepare.py works unchanged!      │
└─────────────────────────────────────┘
```

## Analyzing Evaluation Results with Grep

### Overview

Model evaluation results are automatically logged to `logs/autolog.log` in a grep-friendly format. Each result line includes searchable prefixes for easy data extraction and analysis.

### Log Format

The evaluation logger writes four types of entries:

1. **EVAL_RESULT:** - Individual model performance (one per model)
2. **EVAL_SUMMARY:** - Overall statistics (total RMSE, mean RMSE, model count)
3. **EVAL_BEST:** - Best performing model
4. **EVAL_WORST:** - Worst performing model

### Sample Log Output

```
2026-04-02 19:29:31 - prepare - INFO - EVAL_RESULT: model_name=LOAN_CORP_model, model_type=ARDL, endogenous=LOAN_CORP, n_exog=4, rmse=0.0136, n_predictions=60
2026-04-02 19:29:31 - prepare - INFO - EVAL_RESULT: model_name=CISS_BOND_model, model_type=ARDL, endogenous=CISS_BOND, n_exog=4, rmse=0.0182, n_predictions=60
...
2026-04-02 19:29:31 - prepare - INFO - EVAL_SUMMARY: total_rmse=10.6042, mean_rmse=0.9640, n_models=11
2026-04-02 19:29:31 - prepare - INFO - EVAL_BEST: model_name=LOAN_CORP_model, endogenous=LOAN_CORP, rmse=0.0136
2026-04-02 19:29:31 - prepare - INFO - EVAL_WORST: model_name=HICP_NRG_model, endogenous=HICP_NRG, rmse=9.4063
```

### Common Grep Patterns

#### Get All Model Results
```bash
grep "EVAL_RESULT:" logs/autolog.log
```

#### Get Summary Statistics
```bash
grep "EVAL_SUMMARY:" logs/autolog.log
```

#### Get Best and Worst Models
```bash
grep "EVAL_BEST:\|EVAL_WORST:" logs/autolog.log
```

#### Filter by Model Type
```bash
# ARDL models only
grep "EVAL_RESULT:" logs/autolog.log | grep "model_type=ARDL"

# VAR models only
grep "EVAL_RESULT:" logs/autolog.log | grep "model_type=VAR"

# VECM models only
grep "EVAL_RESULT:" logs/autolog.log | grep "model_type=VECM"
```

#### Filter by Endogenous Variable
```bash
# All models predicting CISS
grep "EVAL_RESULT:" logs/autolog.log | grep "endogenous=CISS"

# All models predicting HICP_TOT
grep "EVAL_RESULT:" logs/autolog.log | grep "endogenous=HICP_TOT"
```

#### Extract RMSE Values
```bash
# Get just RMSE values
grep "EVAL_RESULT:" logs/autolog.log | grep -o "rmse=[0-9.]*" | cut -d= -f2

# Get model name and RMSE
grep "EVAL_RESULT:" logs/autolog.log | sed 's/.*model_name=\([^,]*\).*rmse=\([0-9.]*\).*/\1: \2/'
```

#### Sort Models by Performance
```bash
# Sort by RMSE (ascending - best to worst)
grep "EVAL_RESULT:" logs/autolog.log | sort -t= -k6 -n

# Sort by RMSE (descending - worst to best)
grep "EVAL_RESULT:" logs/autolog.log | sort -t= -k6 -rn
```

#### Find Models with High RMSE
```bash
# Models with RMSE > 1.0
grep "EVAL_RESULT:" logs/autolog.log | awk -F'rmse=' '$2 > 1.0'

# Models with RMSE > 0.5
grep "EVAL_RESULT:" logs/autolog.log | awk -F'rmse=' '$2 > 0.5'
```

#### Count Models by Type
```bash
grep "EVAL_RESULT:" logs/autolog.log | grep -o "model_type=[A-Z]*" | sort | uniq -c
```

**Example output:**
```
    11 model_type=ARDL
     3 model_type=VAR
     2 model_type=VECM
```

#### Compare Multiple Runs
```bash
# Get all summary lines with timestamps
grep "EVAL_SUMMARY:" logs/autolog.log

# Compare mean RMSE across runs
grep "EVAL_SUMMARY:" logs/autolog.log | grep -o "mean_rmse=[0-9.]*"
```

### Advanced Analysis Examples

#### Compare ARDL vs VAR Performance
```bash
# Average RMSE for ARDL models
grep "EVAL_RESULT:" logs/autolog.log | grep "model_type=ARDL" | \
  grep -o "rmse=[0-9.]*" | cut -d= -f2 | \
  awk '{sum+=$1; count++} END {print "ARDL avg:", sum/count}'

# Average RMSE for VAR models
grep "EVAL_RESULT:" logs/autolog.log | grep "model_type=VAR" | \
  grep -o "rmse=[0-9.]*" | cut -d= -f2 | \
  awk '{sum+=$1; count++} END {print "VAR avg:", sum/count}'
```

#### Find Best Model for Each Endogenous Variable
```bash
for var in HICP_NRG HICP_TOT EUR_1M CISS; do
  echo "Best model for $var:"
  grep "EVAL_RESULT:" logs/autolog.log | \
    grep "endogenous=$var" | \
    sort -t= -k6 -n | head -1
done
```

#### Track Performance Over Time
```bash
# Get mean RMSE from each evaluation run
grep "EVAL_SUMMARY:" logs/autolog.log | \
  sed 's/\(^[^ ]* [^ ]*\).*mean_rmse=\([0-9.]*\).*/\1: \2/'
```

**Example output:**
```
2026-04-02 10:30:15: 0.9640
2026-04-02 14:22:31: 0.8723
2026-04-02 19:29:31: 0.8156
```

### Tips for Effective Log Analysis

1. **Use tail for latest results**: `grep "EVAL_SUMMARY:" logs/autolog.log | tail -1`

2. **Combine with other commands**: Pipe grep output to `awk`, `sed`, `sort`, `uniq` for custom analysis

3. **Export to CSV**: 
   ```bash
   echo "model_name,model_type,endogenous,n_exog,rmse,n_predictions" > results.csv
   grep "EVAL_RESULT:" logs/autolog.log | \
     sed 's/.*model_name=\([^,]*\), model_type=\([^,]*\), endogenous=\([^,]*\), n_exog=\([^,]*\), rmse=\([^,]*\), n_predictions=\([0-9]*\)/\1,\2,\3,\4,\5,\6/' \
     >> results.csv
   ```

4. **Monitor in real-time**: `tail -f logs/autolog.log | grep "EVAL_"`

5. **Search across log rotations**: `grep "EVAL_SUMMARY:" logs/autolog.log*`

### Integration with Scripts

You can automate analysis with shell scripts:

```bash
#!/bin/bash
# analyze_models.sh - Automated model performance analysis

LOG_FILE="logs/autolog.log"

echo "=== Latest Evaluation Summary ==="
grep "EVAL_SUMMARY:" $LOG_FILE | tail -1

echo -e "\n=== Best Performers (RMSE < 0.05) ==="
grep "EVAL_RESULT:" $LOG_FILE | awk -F'rmse=' '$2 < 0.05'

echo -e "\n=== Models Needing Improvement (RMSE > 1.0) ==="
grep "EVAL_RESULT:" $LOG_FILE | awk -F'rmse=' '$2 > 1.0'

echo -e "\n=== Performance by Model Type ==="
for type in ARDL VAR VECM; do
  count=$(grep "EVAL_RESULT:" $LOG_FILE | grep "model_type=$type" | wc -l)
  avg=$(grep "EVAL_RESULT:" $LOG_FILE | grep "model_type=$type" | \
        grep -o "rmse=[0-9.]*" | cut -d= -f2 | \
        awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
  echo "$type: $count models, avg RMSE: $avg"
done
```

Save the script, make it executable (`chmod +x analyze_models.sh`), and run it after each evaluation:
```bash
./analyze_models.sh
```

This grep-friendly logging makes it easy to track model performance, compare different approaches, and identify areas for improvement! 📊🔍
