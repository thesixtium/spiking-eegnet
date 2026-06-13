# hpo.py
import optuna
from optuna.pruners import MedianPruner
from pipeline import pipeline

FIXED = dict(
    DATASET_KEY="BNCI2014_001",
    TEST_SUBJECT_IDX=0,
    EPOCHS=10,
    BATCH_SIZE=32,
    N_STEPS_TRAIN=4,
    N_STEPS_EVAL=20,
)

NORM_AXIS_MAP = {
    "full":       (1, 2, 3),
    "no_channel": (1, 3),
}


def objective(trial):
    params = dict(
        FLOW             = trial.suggest_float("FLOW",             1.0,  40.0),
        FHIGH            = trial.suggest_float("FHIGH",            8.0, 120.0),
        LR_EXP           = trial.suggest_float("LR_EXP",          -4.5,  -2.0),
        DROPOUT          = trial.suggest_float("DROPOUT",          0.1,   0.75),
        BETA             = trial.suggest_float("BETA",             0.5,   0.99),
        SPIKE_GRAD_SLOPE = trial.suggest_float("SPIKE_GRAD_SLOPE", 5.0, 100.0),

        TEMPORAL_FILTERS      = trial.suggest_int("TEMPORAL_FILTERS",      4,  32),
        DEPTH_MULTIPLIER      = trial.suggest_int("DEPTH_MULTIPLIER",      1,   4),
        POINTWISE_FILTERS     = trial.suggest_int("POINTWISE_FILTERS",     8,  64),
        TEMPORAL_KERNEL_DIV   = trial.suggest_int("TEMPORAL_KERNEL_DIV",   2,   8),
        SEPARABLE_KERNEL_SIZE = trial.suggest_int("SEPARABLE_KERNEL_SIZE", 4,  32),
        POOL1_SIZE            = trial.suggest_int("POOL1_SIZE",            2,   8),
        POOL2_SIZE            = trial.suggest_int("POOL2_SIZE",            2,   8),

        NORM_AXIS        = NORM_AXIS_MAP[trial.suggest_categorical("NORM_AXIS", list(NORM_AXIS_MAP))],
        RUN_QUANTIZATION = trial.suggest_categorical("RUN_QUANTIZATION", [True, False]),
        RUN_ZSCORE       = trial.suggest_categorical("RUN_ZSCORE",       [True, False]),
        RUN_BANDPASS     = trial.suggest_categorical("RUN_BANDPASS",     [True, False]),
        QUANT_BITS       = trial.suggest_categorical("QUANT_BITS",       [2, 4, 8, 16]),
    )

    return pipeline(**params, **FIXED, trial=trial, save_plots=False)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="snn_eegnet",
        storage="sqlite:///optuna_study.db",
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=100, n_jobs=1)

    print(f"\nBest trial #{study.best_trial.number}")
    print(f"  Value : {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")

    # Re-run best trial with plots enabled
    print("\nRe-running best trial to generate plots...")
    best = dict(study.best_params)
    best["NORM_AXIS"] = NORM_AXIS_MAP[best["NORM_AXIS"]]
    pipeline(**best, **FIXED, save_plots=True)