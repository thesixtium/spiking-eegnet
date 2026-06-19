# pipeline.py
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import optuna

from zscore_normalize import zscore_normalize
from load_moabb_dataset import load_moabb_dataset
from bandpass_filter import bandpass_filter
from experiment_loso import experiment_loso_all
from make_loader import make_loader
from evaluate import evaluate


def _log_trial_to_csv(csv_path, row: dict):
    """Append one trial's results to the CSV, writing the header if the file is new."""
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_results(OUTPUT_DIR, DATASET_KEY, FLOW, FHIGH, histories, accs, meta):
    chance = 1 / meta["n_classes"]
    subjects = sorted(histories.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for subj in subjects:
        hist = histories[subj]
        epochs_run = range(1, len(hist["loss"]) + 1)
        ax1.plot(epochs_run, hist["loss"], linewidth=1.2, alpha=0.8, label=f"subj {subj}")
        ax2.plot(epochs_run, hist["bal_acc"], linewidth=1.2, alpha=0.8, label=f"subj {subj}")

    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{DATASET_KEY} — True LOSO (all {len(subjects)} subjects) | "
                  f"{FLOW}–{FHIGH} Hz | mean acc={sum(accs)/len(accs):.3f}")
    ax1.grid(alpha=0.3)
    ax2.axhline(chance, color="grey", linestyle="--", linewidth=1, label=f"Chance ({chance:.2f})")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_curves.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / 'loso_curves.png'}")


def pipeline(
    # Continuous Float
    FLOW=4.0,               # uniform float [1.0, 40.0]
    FHIGH=40.0,             # uniform float [8.0, 120.0]
    LR_EXP=-3.52,           # uniform float [-4.5, -2.0]
    DROPOUT=0.5,            # uniform float [0.1, 0.75]
    BETA=0.95,              # uniform float [0.5, 0.99]
    SPIKE_GRAD_SLOPE=25.0,  # uniform float [5.0, 100.0]

    # Continuous Int
    TEMPORAL_FILTERS=8,        # uniform int [4, 32]
    DEPTH_MULTIPLIER=2,        # uniform int [1, 4]
    POINTWISE_FILTERS=16,      # uniform int [8, 64]
    TEMPORAL_KERNEL_DIV=2,     # uniform int [2, 8]
    SEPARABLE_KERNEL_SIZE=16,  # uniform int [4, 32]
    POOL1_SIZE=4,              # uniform int [2, 8]
    POOL2_SIZE=4,              # uniform int [2, 8]

    # Discrete
    NORM_AXIS=(1, 2, 3),    # [(1,2,3), (1,3)]
    RUN_ZSCORE=False,       # always disabled for the Optuna study
    RUN_BANDPASS=True,      # always enabled for the Optuna study

    # Experiment Parameters
    DATASET_KEY="BNCI2014_001",
    EPOCHS=10,
    BATCH_SIZE=32,
    N_STEPS_TRAIN=4,
    N_STEPS_EVAL=20,
    EARLY_STOPPING_PATIENCE=None,

    # Optuna
    trial=None,
    save_plots=False,
):
    LR = 10 ** LR_EXP

    TRAIN_CFG = {
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
        "n_steps_train": N_STEPS_TRAIN, "n_steps_eval": N_STEPS_EVAL,
        "patience": EARLY_STOPPING_PATIENCE,
        # NOTE: trial is intentionally left out here. Per-epoch pruning
        # inside run_training() is disabled for true LOSO; pruning is
        # instead evaluated once per subject in experiment_loso_all().
        "trial": None,
    }
    MODEL_CFG = {
        "temporal_filters": TEMPORAL_FILTERS, "depth_multiplier": DEPTH_MULTIPLIER,
        "pointwise_filters": POINTWISE_FILTERS, "temporal_kernel_div": TEMPORAL_KERNEL_DIV,
        "separable_kernel_size": SEPARABLE_KERNEL_SIZE, "pool1_size": POOL1_SIZE,
        "pool2_size": POOL2_SIZE, "dropout": DROPOUT, "beta": BETA,
        "spike_grad_slope": SPIKE_GRAD_SLOPE,
    }

    OUTPUT_DIR = Path("results") / DATASET_KEY
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, subject_ids, meta = load_moabb_dataset(DATASET_KEY)
    print(f"Dataset meta: {meta}")

    if RUN_ZSCORE:
        X = zscore_normalize(X, axis=NORM_AXIS)

    if RUN_BANDPASS:
        print(f"\nApplying bandpass filter: {FLOW}–{FHIGH} Hz")
        X = bandpass_filter(X, sfreq=meta["sfreq"], flow=FLOW, fhigh=FHIGH)

    histories, accs, mean_acc = experiment_loso_all(
        X, y, subject_ids, meta, device, TRAIN_CFG,
        model_kwargs=MODEL_CFG,
        trial=trial,
    )

    results = {
        "dataset":   DATASET_KEY,
        "filter":    {"flow": FLOW, "fhigh": FHIGH},
        "model_cfg": MODEL_CFG,
        "meta":      {k: v for k, v in meta.items() if k != "subject_list"},
        "train_cfg": {k: v for k, v in TRAIN_CFG.items() if k != "trial"},
        "loso_all_subjects": {
            "per_subject_acc": dict(zip(sorted(histories.keys()), accs)),
            "mean_bal_acc": mean_acc,
            "histories": {str(k): v for k, v in histories.items()},
        },
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {OUTPUT_DIR / 'results.json'}")

    # ── CSV trial log ─────────────────────────────────────────────────────────
    import datetime
    csv_row = {
        "timestamp":            datetime.datetime.now().isoformat(timespec="seconds"),
        "trial_number":         trial.number if trial is not None else "",
        # Outputs
        "mean_bal_acc":         round(mean_acc, 6),
        "n_subjects":           len(accs),
        # Preprocessing
        "run_zscore":           RUN_ZSCORE,
        "run_bandpass":         RUN_BANDPASS,
        "norm_axis":            str(NORM_AXIS),
        "flow":                 FLOW,
        "fhigh":                FHIGH,
        # Training
        "lr_exp":               LR_EXP,
        "lr":                   round(LR, 8),
        "epochs":               EPOCHS,
        "batch_size":           BATCH_SIZE,
        "n_steps_train":        N_STEPS_TRAIN,
        "n_steps_eval":         N_STEPS_EVAL,
        # Model
        "temporal_filters":     TEMPORAL_FILTERS,
        "depth_multiplier":     DEPTH_MULTIPLIER,
        "pointwise_filters":    POINTWISE_FILTERS,
        "temporal_kernel_div":  TEMPORAL_KERNEL_DIV,
        "separable_kernel_size":SEPARABLE_KERNEL_SIZE,
        "pool1_size":           POOL1_SIZE,
        "pool2_size":           POOL2_SIZE,
        "dropout":              DROPOUT,
        "beta":                 BETA,
        "spike_grad_slope":     SPIKE_GRAD_SLOPE,
     }
    _log_trial_to_csv(OUTPUT_DIR / "trials.csv", csv_row)
    print(f"Trial logged to {OUTPUT_DIR / 'trials.csv'}")

    if save_plots:
        plot_results(OUTPUT_DIR, DATASET_KEY, FLOW, FHIGH, histories, accs, meta)

    return mean_acc