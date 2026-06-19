# hpo.py
import sys
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pipeline import pipeline

FIXED = dict(
    DATASET_KEY="BNCI2014_001",
    EPOCHS=100,
    BATCH_SIZE=32,
    N_STEPS_TRAIN=4,
    N_STEPS_EVAL=20,
    RUN_ZSCORE=False,   # always disabled
    RUN_BANDPASS=True,  # always enabled
)

NORM_AXIS_MAP = {
    "full":       (1, 2, 3),
    "no_channel": (1, 3),
}


def objective(trial):
    params = dict(
        FLOW             = trial.suggest_float("FLOW",             1.0,  20.0),
        FHIGH            = trial.suggest_float("FHIGH",            24.0, 120.0),
        LR_EXP           = trial.suggest_float("LR_EXP",          -4.5,  -2.0),
        DROPOUT          = trial.suggest_float("DROPOUT",          0.1,   0.75),
        BETA             = trial.suggest_float("BETA",             0.5,   0.99),
        SPIKE_GRAD_SLOPE = trial.suggest_float("SPIKE_GRAD_SLOPE", 0.1, 60.0),

        TEMPORAL_FILTERS      = trial.suggest_int("TEMPORAL_FILTERS",      4,  32),
        DEPTH_MULTIPLIER      = trial.suggest_int("DEPTH_MULTIPLIER",      1,   4),
        POINTWISE_FILTERS     = trial.suggest_int("POINTWISE_FILTERS",     8,  64),
        TEMPORAL_KERNEL_DIV   = trial.suggest_int("TEMPORAL_KERNEL_DIV",   2,   8),
        SEPARABLE_KERNEL_SIZE = trial.suggest_int("SEPARABLE_KERNEL_SIZE", 4,  32),
        POOL1_SIZE            = trial.suggest_int("POOL1_SIZE",            2,   8),
        POOL2_SIZE            = trial.suggest_int("POOL2_SIZE",            2,   8),

        NORM_AXIS        = NORM_AXIS_MAP[trial.suggest_categorical("NORM_AXIS", list(NORM_AXIS_MAP))],
    )

    return pipeline(**params, **FIXED, trial=trial, save_plots=False)


def status_callback(study, frozen_trial):
    """Write current progress to a status file after every trial."""
    n_completed = len([
        t for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    ])
    n_total = len(study.trials)

    try:
        best_value = study.best_value
        best_number = study.best_trial.number
    except ValueError:
        best_value = None
        best_number = None

    with open("optuna_status.txt", "w") as f:
        f.write(f"last_trial_number : {frozen_trial.number}\n")
        f.write(f"last_trial_state  : {frozen_trial.state.name}\n")
        f.write(f"last_trial_value  : {frozen_trial.value}\n")
        f.write(f"trials_completed  : {n_completed}\n")
        f.write(f"trials_total      : {n_total}\n")
        f.write(f"best_trial_number : {best_number}\n")
        f.write(f"best_value        : {best_value}\n")

    print(f"[status] trial {frozen_trial.number} done "
          f"(state={frozen_trial.state.name}, value={frozen_trial.value}) "
          f"-- {n_completed} completed total")


if __name__ == "__main__":
    n_trials = 200
    tpe_trails = 20

    study = optuna.create_study(
        direction="maximize",
        study_name=f"snn_eegnet_v2_{n_trials}_{tpe_trails}",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        # n_warmup_steps is now in units of SUBJECTS (since pruning is
        # evaluated once per subject for true LOSO), not epochs:
        # a trial won't be pruned until at least 3 subjects have finished.
        pruner=MedianPruner(n_startup_trials=tpe_trails, n_warmup_steps=3),
        sampler=TPESampler(n_startup_trials=tpe_trails, multivariate=True, group=True)
    )

    # timeout in seconds: stop starting new trials after this long so the
    # current trial can finish and write results before SLURM kills the job.
    # 23h leaves ~1h margin under a 24h SLURM time limit.
    study.optimize(objective, n_trials=n_trials, n_jobs=1,
                    timeout=23 * 3600, callbacks=[status_callback])

    print(f"\nBest trial #{study.best_trial.number}")
    print(f"  Value : {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Re-running the best trial with plots happens in a separate
    # final job (e.g. `python3 main.py --plot-best`), not on every
    # chained HPO job, since chained jobs each time out before reaching
    # n_trials and would otherwise re-run this every time.
    if "--plot-best" in sys.argv:
        print("\nRe-running best trial to generate plots...")
        best = dict(study.best_params)
        best["NORM_AXIS"] = NORM_AXIS_MAP[best["NORM_AXIS"]]
        pipeline(**best, **FIXED, save_plots=True)