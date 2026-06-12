import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from zscore_normalize import zscore_normalize
from load_moabb_dataset import load_moabb_dataset
from bandpass_filter import bandpass_filter
from experiment_loso import experiment_loso

if __name__ == "__main__":

    # ═══════════════════════════════════════════════════════════════════════════
    # ALL TUNABLE PARAMETERS — edit here, everything else is wired up below
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Dataset ───────────────────────────────────────────────────────────────
    DATASET_KEY      = "BNCI2014_001"
    TEST_SUBJECT_IDX = 0               # which subject to hold out in LOSO

    # ── Preprocessing ─────────────────────────────────────────────────────────
    FLOW      = 4.0        # float  uniform [1.0,  40.0]  — bandpass low  cutoff (Hz)
    FHIGH     = 40.0       # float  uniform [8.0, 124.0]  — bandpass high cutoff (Hz); must be > FLOW+2
    NORM_AXIS = (1, 2, 3)  # (1,2,3) = whole-epoch z-score | (1,3) = per-channel z-score

    # ── Training ──────────────────────────────────────────────────────────────
    EPOCHS        = 10
    BATCH_SIZE    = 32
    LR            = 3e-4
    N_STEPS_TRAIN = 4    # SNN time-steps during training  (fewer = faster)
    N_STEPS_EVAL  = 20   # SNN time-steps during eval      (more = stabler spike rate)

    # ── SpikingEEGNet architecture ────────────────────────────────────────────
    # Filter counts
    TEMPORAL_FILTERS   = 8    # int   uniform [4,  32]  — temporal conv output channels
    DEPTH_MULTIPLIER   = 2    # int   uniform [1,   4]  — spatial filters per temporal filter
    POINTWISE_FILTERS  = 16   # int   uniform [8,  64]  — separable pointwise output channels

    # Filter sizes
    TEMPORAL_KERNEL_DIV    = 2   # int   uniform [2,  8]  — temporal kernel = num_samples // div
    SEPARABLE_KERNEL_SIZE  = 16  # int   uniform [4, 32]  — separable depthwise kernel width

    # Pooling
    POOL1_SIZE = 4   # int   uniform [2, 8]  — avg-pool after Block 1
    POOL2_SIZE = 4   # int   uniform [2, 8]  — avg-pool after Block 2

    # Regularisation
    DROPOUT = 0.5    # float uniform [0.10, 0.75]  — applied after each pooling stage

    # SNN dynamics
    BETA             = 0.95   # float uniform [0.50,  0.99]   — LIF membrane decay rate
    SPIKE_GRAD_SLOPE = 25.0   # float uniform [5.0,  100.0]  — fast-sigmoid surrogate slope

    # ═══════════════════════════════════════════════════════════════════════════
    # Derived configs (no need to edit below this line)
    # ═══════════════════════════════════════════════════════════════════════════

    TRAIN_CFG = {
        "epochs":        EPOCHS,
        "batch_size":    BATCH_SIZE,
        "lr":            LR,
        "n_steps_train": N_STEPS_TRAIN,
        "n_steps_eval":  N_STEPS_EVAL,
    }

    MODEL_CFG = {
        "temporal_filters":      TEMPORAL_FILTERS,
        "depth_multiplier":      DEPTH_MULTIPLIER,
        "pointwise_filters":     POINTWISE_FILTERS,
        "temporal_kernel_div":   TEMPORAL_KERNEL_DIV,
        "separable_kernel_size": SEPARABLE_KERNEL_SIZE,
        "pool1_size":            POOL1_SIZE,
        "pool2_size":            POOL2_SIZE,
        "dropout":               DROPOUT,
        "beta":                  BETA,
        "spike_grad_slope":      SPIKE_GRAD_SLOPE,
    }

    OUTPUT_DIR = Path("results") / DATASET_KEY
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load (cache-friendly, no filtering) ───────────────────────────────────
    X, y, subject_ids, meta = load_moabb_dataset(DATASET_KEY)
    print(f"Dataset meta: {meta}")

    # --- Zscore Normalize
    X = zscore_normalize(X, axis=NORM_AXIS)

    # ── Bandpass filter (searchable) ──────────────────────────────────────────
    print(f"\nApplying bandpass filter: {FLOW}–{FHIGH} Hz")
    X = bandpass_filter(X, sfreq=meta["sfreq"], flow=FLOW, fhigh=FHIGH)

    # ── Experiment ────────────────────────────────────────────────────────────
    hist_loso, acc_loso = experiment_loso(
        X, y, subject_ids, meta, device, TRAIN_CFG,
        test_subject_idx=TEST_SUBJECT_IDX,
        model_kwargs=MODEL_CFG,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "dataset":    DATASET_KEY,
        "filter":     {"flow": FLOW, "fhigh": FHIGH},
        "model_cfg":  MODEL_CFG,
        "meta":       {k: v for k, v in meta.items() if k != "subject_list"},
        "train_cfg":  TRAIN_CFG,
        "loso_subject_0": {
            "history":       hist_loso,
            "final_bal_acc": acc_loso,
        },
    }
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    epochs_range = range(1, TRAIN_CFG["epochs"] + 1)
    chance = 1 / meta["n_classes"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(epochs_range, hist_loso["loss"], color="steelblue", linewidth=1.8)
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{DATASET_KEY} — LOSO (Subject 0 held out) | {FLOW}–{FHIGH} Hz")
    ax1.grid(alpha=0.3)

    ax2.plot(epochs_range, hist_loso["bal_acc"], color="darkorange", linewidth=1.8)
    ax2.axhline(chance, color="grey", linestyle="--",
                linewidth=1, label=f"Chance ({chance:.2f})")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_curves.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / 'loso_curves.png'}")