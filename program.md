# automap

This is an experiment to have the LLM do its own research on macro-economic transmission map.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr2`). The branch `automap/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b automap/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `EXPERIMENTER_GUIDE.md` — experimentation manual.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Explicit instruction: NEVER TOUCH THIS. Do not modify.
   - `train.py` — the file you modify. Time Series Models, Lags, Transformations. Everything is a fair game.
   - `feature_eng.py` - the file you modify to create a geopolitical risk indicator for EU. The list of variables to use are: 
   ['GPR_BEL', 'GPR_CHE', 'GPR_DEU', 'GPR_DNK', 'GPR_ESP', 'GPR_FIN', 'GPR_FRA', 'GPR_GBR', 'GPR_HUN', 'GPR_ITA', 'GPR_NLD', 'GPR_NOR', 'GPR_POL', 'GPR_PRT', 'GPR_RUS', 'GPR_SWE', 'GPR_TUR', 'GPR_UKR'].
   You are allowed to transform them in any way (provided all of them undergo the same transformation), you are allowed to combine them linearly only with/without transformation.
4. **Verify data exists**: Check that `data_gpr.parquet` exists. If not, tell the human to run `uv run train.py --mode both` 
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation Workflow

You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, composite calculation, lag size, model size, etc. You can create bespoke models for each of the relationships or you may create 1 model for everything - as per your desire.
- Modify `utilities/feature_eng.py` - this file you can edit in an economically rational way - all transformations need to be consistent across the various GPRC time series; you may choose to drop or keep a subset of the variables - but whatever you keep needs to be transformed in a consistent manner; When building composite - the operation has to be linear
- Create `models/*_model.py` - you can create new time series models as per config.py MODELS list (currently it lists - ARDL, VAR and VECM) - so you can create a vecm_model.py in models/

**What you MUST do:**
- In `train.py`: model GPR_EU shocks as generalized impulse response function. You have no freedom to change this relationship. Everything else you are allowed to do. The hyperparameters to model the GIRF is left to you. 

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, sequence length, etc).
- Modify `config.py`. It is read only. It also contains the training constants.
- Modify `base_model.py`: It is the "contract" that lets a time series model "talk" with utilities
- Blend GPR variables in a economically-non-rational way - for example having inconsistent transformations, or having non-linear operations on them.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_*` function in `prepare.py` are the ground truth metric.

**The goal is simple: get the lowest total RMSE.** Everything is fair game: change the model type, use separate models for separate relationships, the number of lags, the number of exogenous variables for each relationship (limited to 6), the model size. The only constraint is that the code runs without crashing.

**Number of exogenous variables used to model relationships** is a soft constraint. Adding an extra variable or an extra lag is acceptable for meaningful RMSE improvements, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. An addition to a model specification that adds no tangible RMSE improvements is not worth it. Conversely, removing a variable and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the model specification against the RMSE improvement. A 0.1% RMSE improvement that adds 5 extra variables or stretches the lagged variables to maximum? Probably not worth it. A 0.1% RMSE improvement by removing a single lag ? Definitely keep. An improvement of ~0 but much simpler model specification? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
=====================================================================
2026-04-02 20:14:36 - __main__ - INFO - MODEL EVALUATION RESULTS
2026-04-02 20:14:36 - __main__ - INFO - ======================================================================
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=LOAN_CORP_model, model_type=ARDL, endogenous=LOAN_CORP, n_exog=4, rmse=0.0136, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=CISS_BOND_model, model_type=ARDL, endogenous=CISS_BOND, n_exog=4, rmse=0.0182, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=LOAN_HH_model, model_type=ARDL, endogenous=LOAN_HH, n_exog=4, rmse=0.0322, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=CISS_EQ_model, model_type=ARDL, endogenous=CISS_EQ, n_exog=4, rmse=0.0361, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=UNEMP_model, model_type=ARDL, endogenous=UNEMP, n_exog=4, rmse=0.0688, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=EUR_1M_model, model_type=ARDL, endogenous=EUR_1M, n_exog=4, rmse=0.1468, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=CISS_model, model_type=ARDL, endogenous=CISS, n_exog=4, rmse=0.1591, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=AAA_1Y_model, model_type=ARDL, endogenous=AAA_1Y, n_exog=4, rmse=0.1639, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=CISS_model_2, model_type=ARDL, endogenous=CISS, n_exog=5, rmse=0.1661, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=HICP_TOT_model, model_type=ARDL, endogenous=HICP_TOT, n_exog=4, rmse=0.3932, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - EVAL_RESULT: model_name=HICP_NRG_model, model_type=ARDL, endogenous=HICP_NRG, n_exog=4, rmse=9.4063, n_predictions=60
2026-04-02 20:14:36 - __main__ - INFO - ======================================================================
2026-04-02 20:14:36 - __main__ - INFO - EVAL_SUMMARY: total_rmse=10.6042, mean_rmse=0.9640, n_models=11
2026-04-02 20:14:36 - __main__ - INFO - EVAL_BEST: model_name=LOAN_CORP_model, endogenous=LOAN_CORP, rmse=0.0136
2026-04-02 20:14:36 - __main__ - INFO - EVAL_WORST: model_name=HICP_NRG_model, endogenous=HICP_NRG, rmse=9.4063
2026-04-02 20:14:36 - __main__ - INFO - ======================================================================
```

```
Read EXPERIMENTER_GUIDE.md to extract the results from log through grep
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	Total_RMSE	status	description
```

1. git commit hash (short, 7 chars)
2. Total RMSE achieved (e.g. 1.234567) — use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	  status   	description
a1b2c3d	0.997900  keep  	baseline
b2c3d4e	0.993200  keep  	increase the number of lags to 3
c3d4e5f	1.005000  discard	switch to ARIMA 
d4e5f6g	0.000000  crash 	switched to OLS
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `automap/apr2` ).

LOOP FOR 30MINUTES:

1. Look at the git state: the current branch/commit we're on
2. Modify `train.py` with an experimental idea by directly hacking the code.
3. Run the experiment: `uv run train.py` (redirect everything — do NOT use tee or let output flood your context)
4. Evaluate the RMSE on the hold-out data `uv run prepare.py`. if any of the model is not working then it will return a 100.0 as the rmse value.
5. Read out the results: `grep "EVAL_SUMMARY:" logs/autolog.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 logs/autolog.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If total_rmse improved (lower), commit the branch and "advance" the branch, keeping the git commit
9. If total_rmse is equal or worse, you undo the changes and rebuild 

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**TIME BUDGET**: The experiment loop has a fixed wall-clock budget of 30 minutes from the moment the first experiment begins. Before starting each new experiment, 
check elapsed time. If less than 30 minutes remain AND that time is insufficient to complete a full experiment, do not start a new one. Instead, write a brief 
summary of completed experiments to results.tsv and stop cleanly.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!