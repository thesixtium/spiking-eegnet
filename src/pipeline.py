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
from experiment_loso import experiment_loso
from make_loader import make_loader
from evaluate import evaluate
from quantize_model import quantize_model


def _log_trial_to_csv(csv_path, row: dict):
    """Append one trial's results to the CSV, writing the header if the file is new."""
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def plot_results(OUTPUT_DIR, DATASET_KEY, TEST_SUBJECT_IDX, FLOW, FHIGH,
                 hist_loso, meta, acc_fp32=None, acc_q=None, QUANT_BITS=8):
    chance     = 1 / meta["n_classes"]
    epochs_run = range(1, len(hist_loso["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(epochs_run, hist_loso["loss"], color="steelblue", linewidth=1.8)
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{DATASET_KEY} — LOSO (Subject {TEST_SUBJECT_IDX} held out) | {FLOW}–{FHIGH} Hz")
    ax1.grid(alpha=0.3)
    ax2.plot(epochs_run, hist_loso["bal_acc"], color="darkorange", linewidth=1.8)
    ax2.axhline(chance, color="grey", linestyle="--", linewidth=1, label=f"Chance ({chance:.2f})")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "loso_curves.png", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {OUTPUT_DIR / 'loso_curves.png'}")

    if acc_fp32 is not None and acc_q is not None:
        label_q = f"INT{QUANT_BITS}"
        fig2, ax = plt.subplots(figsize=(5, 5))
        bars = ax.bar(["fp32", label_q], [acc_fp32, acc_q],
                      color=["steelblue", "darkorange"], width=0.4, zorder=3)
        for bar, acc in zip(bars, [acc_fp32, acc_q]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.axhline(chance, color="grey", linestyle="--", linewidth=1,
                   label=f"Chance ({chance:.2f})", zorder=2)
        drop = acc_fp32 - acc_q
        ax.set_title(
            f"{DATASET_KEY} — fp32 vs {label_q}\n"
            f"Subject {TEST_SUBJECT_IDX} held out | Δ = {drop:+.4f}"
        )
        ax.set_ylabel("Balanced Accuracy")
        ax.set_ylim(0, min(1.0, max(acc_fp32, acc_q) + 0.15))
        ax.legend()
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        fig2.savefig(OUTPUT_DIR / "quantization_accuracy.png", dpi=150)
        plt.close(fig2)
        print(f"Quantization plot saved to {OUTPUT_DIR / 'quantization_accuracy.png'}")


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
    RUN_QUANTIZATION=True,  # [True, False]
    RUN_ZSCORE=True,        # [True, False]
    RUN_BANDPASS=True,      # [True, False]
    QUANT_BITS=8,           # [2, 4, 8, 16]

    # Experiment Parameters
    DATASET_KEY="BNCI2014_001",
    TEST_SUBJECT_IDX=0,
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
        "trial": trial,
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

    hist_loso, acc_loso, trained_model = experiment_loso(
        X, y, subject_ids, meta, device, TRAIN_CFG,
        test_subject_idx=TEST_SUBJECT_IDX,
        model_kwargs=MODEL_CFG,
    )

    acc_fp32 = acc_q = None
    if RUN_QUANTIZATION:
        print(f"\n{'='*60}\nPost-Training Quantization (INT{QUANT_BITS})\n{'='*60}")
        test_mask  = subject_ids == TEST_SUBJECT_IDX
        val_loader = make_loader(X[test_mask], y[test_mask], BATCH_SIZE, shuffle=False)
        trained_model.eval()
        acc_fp32 = evaluate(trained_model, val_loader, device, N_STEPS_EVAL)
        print(f"fp32        balanced accuracy : {acc_fp32:.4f}")
        q_model = quantize_model(trained_model, bits=QUANT_BITS)
        acc_q   = evaluate(q_model, val_loader, torch.device("cpu"), N_STEPS_EVAL)
        print(f"INT{QUANT_BITS:<8}balanced accuracy : {acc_q:.4f}")
        print(f"Accuracy drop               : {acc_fp32 - acc_q:+.4f}")

    results = {
        "dataset":   DATASET_KEY,
        "filter":    {"flow": FLOW, "fhigh": FHIGH},
        "model_cfg": MODEL_CFG,
        "meta":      {k: v for k, v in meta.items() if k != "subject_list"},
        "train_cfg": {k: v for k, v in TRAIN_CFG.items() if k != "trial"},
        "loso_subject_0": {"history": hist_loso, "final_bal_acc": acc_loso},
    }
    if RUN_QUANTIZATION:
        results["quantization"] = {
            "bits": QUANT_BITS, "acc_fp32": acc_fp32,
            f"acc_int{QUANT_BITS}": acc_q, "drop": acc_fp32 - acc_q,
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
        "acc_loso":             round(acc_loso, 6),
        "acc_fp32":             round(acc_fp32, 6) if acc_fp32 is not None else "",
        "acc_q":                round(acc_q,    6) if acc_q    is not None else "",
        "quant_drop":           round(acc_fp32 - acc_q, 6) if (acc_fp32 is not None and acc_q is not None) else "",
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
        # Quantization
        "run_quantization":     RUN_QUANTIZATION,
        "quant_bits":           QUANT_BITS,
    }
    _log_trial_to_csv(OUTPUT_DIR / "trials.csv", csv_row)
    print(f"Trial logged to {OUTPUT_DIR / 'trials.csv'}")

    if save_plots:
        plot_results(OUTPUT_DIR, DATASET_KEY, TEST_SUBJECT_IDX, FLOW, FHIGH,
                     hist_loso, meta, acc_fp32, acc_q, QUANT_BITS)

    return acc_loso