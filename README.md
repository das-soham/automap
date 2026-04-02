# automap

The idea - developed on the lines of Andrej Karpathy's autoresearch - we develop **automap** for an AI agent to
iteratively figure out the best model hyperparameters and sparingly select among a narrow set of conceptually defensible models to
build the system.

The core idea is that a human should not touch any of the Python files like one normally would as a researcher.
Instead, one is programming the `program.md` Markdown files that provide context to the AI agents and set up the autonomous research org.
The default `program.md` in this repo is intentionally kept as a bare-bones baseline, though it's obvious how one would
iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc.
A bit more context on this project is here in this:

## How it works

On the lines of Karpathy's work, the repo spiritually mirrors the small footprint and has three files that matter:

- **`transform.py`** — fixed constants, one-time data prep (downloads training data,transformations), and other runtime utilities (dataloader, evaluation).
- **`train.py`** — the single file the agent edits. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

---

# Project Validation & Readiness Assessment

## Current State: 100% Ready for Autoresearch

**Last validated**: 2026-04-02

The project is fully ready to run autoresearch with **zero human setup required**. All core infrastructure is in place, and the agent will autonomously implement advanced models (VAR, VECM, GIRF) during the 30-minute experimentation loop.

### Core Components ✓

1. **Three-File Architecture** - Complete
   - `program.md` - Agent instructions (340 lines)
   - `train.py` - Agent-editable orchestration (213 lines)
   - `prepare.py` - Read-only evaluation harness (298 lines)

2. **Data Pipeline** - Functional
   - 23 ECB time series across 7 datasets
   - `data_gpr.parquet` - 256 rows × 25 columns with GPR_EU composite
   - Monthly frequency, 2004-2023 training period

3. **Model Infrastructure** - Pluggable
   - `models/base_model.py` - Abstract interface contract
   - `models/ardl_model.py` - **Production ARDL** (baseline RMSE: 10.6042)
   - `models/var_model.py` - Stub template for agent implementation

4. **Supporting Utilities** - Complete
   - GDP-weighted GPR composite (18 countries)
   - Train/test splitting, logging, data I/O
   - Grep-friendly evaluation format

5. **Tracking** - Ready
   - `results.tsv` created for experiment history
   - Git-based experimentation workflow

### Economic Transmission Models

**11 causal relationships (MEV_REL)**:
1. GPR_EU → HICP_NRG (geopolitical risk → energy inflation)
2. HICP_NRG → HICP_TOT (energy → total inflation)
3. HICP_TOT → EUR_1M (inflation → money market rates)
4. EUR_1M → LOAN_CORP (rates → corporate lending)
5. EUR_1M → LOAN_HH (rates → household lending)
6. HICP_TOT → UNEMP (inflation → unemployment)
7. CISS_FIN → CISS (financial stress → overall stress)
8. CISS → AAA_1Y (stress → bond yields)
9. GPR_EU → CISS_EQ (geopolitical → equity stress)
10. GPR_EU → CISS_BOND (geopolitical → bond stress)
11. [CISS_BOND, CISS_EQ] → CISS (multi-input relationship)

**Baseline Performance** (ARDL, 11 models):
- Total RMSE: 10.6042
- Best: LOAN_CORP (0.0136)
- Worst: HICP_NRG (9.4063)
- Test period: 2021-01-31 to 2025-12-31

### Agent Capabilities During Autoresearch

The agent will autonomously:
- ✓ Create new model files (`models/vecm_model.py`, custom models)
- ✓ Implement VAR/VECM from scratch using statsmodels
- ✓ Add GIRF impulse response analysis (required for GPR_EU shocks)
- ✓ Modify feature engineering (`utilities/feature_eng.py`)
- ✓ Redesign architectures in `train.py`
- ✓ Experiment with lag selection algorithms
- ✓ Test different exogenous variable combinations
- ✓ Make git commits for improvements, reset for regressions

**The stub files are intentional templates** - VAR/VECM stubs show the agent how to structure new models following the base contract.

### How to Start Autoresearch

Simply provide the agent with a run tag:

```
"Start autoresearch with tag apr2"
```

The agent will automatically:
1. Agree on run tag (e.g., `apr2`)
2. Create branch: `git checkout -b automap/<tag>`
3. Read all in-scope files for context
4. Verify `data_gpr.parquet` exists
5. Begin 30-minute experimentation loop
6. Track results in `results.tsv`

No human setup commands required.

### Verification

```bash
# Check data exists
python -c "import pandas as pd; df=pd.read_parquet('data_gpr.parquet'); print(df.shape, 'GPR_EU' in df.columns)"
# Expected: (256, 25) True

# Run baseline
uv run train.py > run.log 2>&1
grep "EVAL_SUMMARY:" logs/autolog.log | tail -1
# Expected: total_rmse=10.6042, mean_rmse=0.9640, n_models=11

# Check tracking
head -1 results.tsv
# Expected: commit	Total_RMSE	status	description
```

### What Agent Cannot Modify

Per `program.md`:
- ❌ `prepare.py` - Evaluation harness (read-only)
- ❌ `config.py` - Training constants (read-only)
- ❌ `models/base_model.py` - Interface contract (read-only)

