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

    # ── Config ────────────────────────────────────────────────────────────────
    DATASET_KEY = "BNCI2014_001"

    # Filter parameters — searchable in genetic algorithm
    # Search spaces documented in bandpass_filter.py
    FLOW  = 4.0    # uniform float [1.0, 40.0]
    FHIGH = 40.0   # uniform float [8.0, 124.0]

    TRAIN_CFG = {
        "epochs":        10,
        "batch_size":    32,
        "lr":            3e-4,
        "n_steps_train": 4,
        "n_steps_eval":  20,
    }

    OUTPUT_DIR = Path("results") / DATASET_KEY
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load (cache-friendly, no filtering) ───────────────────────────────────
    X, y, subject_ids, meta = load_moabb_dataset(DATASET_KEY)
    print(f"Dataset meta: {meta}")

    # --- Zscore Normalize
    X = zscore_normalize(X, axis=(1,2,3))

    # ── Bandpass filter (searchable) ──────────────────────────────────────────
    print(f"\nApplying bandpass filter: {FLOW}–{FHIGH} Hz")
    X = bandpass_filter(X, sfreq=meta["sfreq"], flow=FLOW, fhigh=FHIGH)

    # ── Experiment ────────────────────────────────────────────────────────────
    hist_loso, acc_loso = experiment_loso(
        X, y, subject_ids, meta, device, TRAIN_CFG,
        test_subject_idx=0,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "dataset":    DATASET_KEY,
        "filter":     {"flow": FLOW, "fhigh": FHIGH},
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