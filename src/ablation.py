# ablation.py
"""
Two analyses in one script, run back-to-back so the per-subject LOSO models
trained in Phase 1 can be reused for the channel-importance part of Phase 2
instead of re-training from scratch.

Lives in src/ alongside the rest of the pipeline (bandpass_filter.py,
build_model.py, pipeline.py, etc.) and imports them directly, the same way
experiment_loso.py / pipeline.py do.

By default this reads the best hyperparameters straight out of
optuna_study.db (no separate best_params.json needed) — it auto-detects the
study name if there's only one study in the database, matching the
sqlite:///optuna_study.db file main.py already writes to in the repo root.

PHASE 1 — Per-subject LOSO breakdown
    True Leave-One-Subject-Out (same protocol as experiment_loso_all): for
    each subject, train on every other subject and test on the held-out one.
    Reports which subjects the model does best/worst on, with confusion
    matrices, per-class accuracy, training curves, and summary statistics —
    in the same publication-style format as statistic.py.

PHASE 2 — Ablation study
    a) Component ablations: re-run LOSO with one design choice flipped at a
       time (z-score on/off + axis, bandpass on/off, readout mode) and
       compare mean balanced accuracy against the Phase-1 baseline.
    b) Channel importance: for each subject's Phase-1 model, zero out one
       EEG channel at a time in the held-out test set and measure the drop
       in balanced accuracy (occlusion importance). No re-training needed —
       this reuses the already-trained Phase-1 models, so it's cheap.

Outputs (all under --out-dir):
  Phase 1:
    01_per_subject_accuracy.png
    02_confusion_matrices.png
    03_per_class_accuracy_heatmap.png
    04_training_curves.png
    05_accuracy_distribution.png
    per_subject_summary.csv
  Phase 2:
    06_channel_importance.png
    07_channel_importance_heatmap.png
    08_ablation_comparison.png
    ablation_summary.csv
    channel_importance.csv
  recommendations.txt   (combined write-up, same style as statistic.py)

Usage (run from the repo root, e.g. inside ablation.slurm):
    python3 src/ablation.py
        # reads sqlite:///optuna_study.db automatically, writes to
        # results/BNCI2014_001/ablation_study

    python3 src/ablation.py --out-dir results/BNCI2014_001/ablation_study \
        --optuna-db sqlite:///optuna_study.db --study-name snn_eegnet_v3_200_20

    python3 src/ablation.py --params-json best_params.json
        # use a JSON file instead of reading the live study
"""


import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

from load_moabb_dataset import load_moabb_dataset
from zscore_normalize import zscore_normalize
from bandpass_filter import bandpass_filter
from make_loader import make_loader
from build_model import build_model
from run_training import run_training
from train_one_epoch import aggregate_logits


# ----------------------------------------------------------------------
# Publication-style global plot settings (matches statistic.py)
# ----------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
SAVE_KW = dict(dpi=300, bbox_inches="tight")

# Known electrode layouts (used only for nicer channel labels; falls back
# to generic "Ch0".."ChN-1" for unknown datasets or mismatched channel counts)
CHANNEL_NAMES = {
    "BNCI2014_001": [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz",
        "C2", "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz",
        "P2", "POz",
    ],
}

# FIXED params held constant during the Optuna study (main.py FIXED dict).
# These don't live in best_params.json (search space only), so we supply
# sensible defaults here, overridable via CLI.
DEFAULT_FIXED = dict(
    DATASET_KEY="BNCI2014_001",
    EPOCHS=50,
    BATCH_SIZE=32,
    NORM_AXIS=(1, 3),
    RUN_ZSCORE=False,
    RUN_BANDPASS=True,
)

READOUT_MODES = ["spk_mean", "spk_last", "spk_sum", "mem_last"]


def pretty(col: str) -> str:
    names = {
        "bal_acc": "Balanced Accuracy", "f1_macro": "Macro F1 Score",
        "subject": "Subject", "run_zscore": "Z-Score Normalization",
        "run_bandpass": "Bandpass Filtering", "readout_mode": "Readout Mode",
        "norm_axis": "Normalization Axis", "mean_bal_acc": "Mean Balanced Accuracy",
        "std_bal_acc": "Std. Dev. Balanced Accuracy",
    }
    return names.get(col, col.replace("_", " ").title())


def get_channel_names(dataset_key, n_channels):
    names = CHANNEL_NAMES.get(dataset_key)
    if names and len(names) == n_channels:
        return names
    return [f"Ch{i}" for i in range(n_channels)]


# ----------------------------------------------------------------------
# Config assembly — mirrors pipeline.py's param -> TRAIN_CFG/MODEL_CFG mapping
# ----------------------------------------------------------------------
def make_cfgs(params: dict):
    LR = 10 ** params["LR_EXP"]
    train_cfg = {
        "epochs": params["EPOCHS"], "batch_size": params["BATCH_SIZE"], "lr": LR,
        "n_steps_train": params["N_STEPS_TRAIN"], "n_steps_eval": params["N_STEPS_EVAL"],
        "readout_mode": params["READOUT_MODE"],
        "patience": params.get("EARLY_STOPPING_PATIENCE"),
    }
    model_cfg = {
        "temporal_filters": params["TEMPORAL_FILTERS"], "depth_multiplier": params["DEPTH_MULTIPLIER"],
        "pointwise_filters": params["POINTWISE_FILTERS"], "temporal_kernel_div": params["TEMPORAL_KERNEL_DIV"],
        "separable_kernel_size": params["SEPARABLE_KERNEL_SIZE"], "pool1_size": params["POOL1_SIZE"],
        "pool2_size": params["POOL2_SIZE"], "dropout": params["DROPOUT"], "beta": params["BETA"],
        "spike_grad_slope": params["SPIKE_GRAD_SLOPE"],
    }
    return train_cfg, model_cfg


def resolve_study_name(storage, study_name):
    """If --study-name wasn't given, auto-detect it: works as long as the
    database has exactly one study (the normal case for this repo's
    optuna_study.db, written by main.py)."""
    if study_name:
        return study_name
    import optuna
    summaries = optuna.study.get_all_study_summaries(storage)
    if not summaries:
        raise ValueError(f"No Optuna studies found in {storage}")
    if len(summaries) == 1:
        name = summaries[0].study_name
        print(f"Auto-detected Optuna study: '{name}'")
        return name
    names = [s.study_name for s in summaries]
    raise ValueError(
        f"Multiple Optuna studies found in {storage}: {names}. "
        f"Pass --study-name to pick one."
    )


def load_best_params(args):
    if args.params_json:
        with open(args.params_json) as f:
            best = json.load(f)
        print(f"Loaded best params from {args.params_json}")
    else:
        import optuna
        study_name = resolve_study_name(args.optuna_db, args.study_name)
        study = optuna.load_study(study_name=study_name, storage=args.optuna_db)
        best = dict(study.best_params)
        print(f"Loaded best params from study '{study_name}' in {args.optuna_db} "
              f"(trial #{study.best_trial.number}, value={study.best_value:.4f})")

    fixed = dict(DEFAULT_FIXED)
    fixed["DATASET_KEY"] = args.dataset_key or fixed["DATASET_KEY"]
    if args.epochs is not None:
        fixed["EPOCHS"] = args.epochs
    if args.batch_size is not None:
        fixed["BATCH_SIZE"] = args.batch_size

    params = dict(fixed)
    params.update(best)
    return params


def prepare_data(params, override_zscore=None, override_bandpass=None, override_norm_axis=None):
    """Loads raw data and applies (optionally overridden) preprocessing."""
    X, y, subject_ids, meta = load_moabb_dataset(params["DATASET_KEY"])

    run_zscore = params["RUN_ZSCORE"] if override_zscore is None else override_zscore
    run_bandpass = params["RUN_BANDPASS"] if override_bandpass is None else override_bandpass
    norm_axis = params["NORM_AXIS"] if override_norm_axis is None else override_norm_axis

    if run_zscore:
        X = zscore_normalize(X, axis=norm_axis)
    if run_bandpass:
        X = bandpass_filter(X, sfreq=meta["sfreq"], flow=params["FLOW"], fhigh=params["FHIGH"])

    return X, y, subject_ids, meta


@torch.no_grad()
def predict_all(model, loader, device, n_steps, readout_mode):
    """Like evaluate(), but returns raw predictions/labels for confusion
    matrices and per-class breakdowns instead of just a scalar accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        spk, mem = model(xb, num_steps=n_steps)
        if readout_mode == "mem_last":
            logits = model.classifier(mem[-1].flatten(1))
        else:
            logits = aggregate_logits(spk, mem, readout_mode)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def channel_importance_for_model(model, X_test, y_test, device, n_steps, readout_mode,
                                  batch_size, channel_names, baseline_acc):
    """Occlusion-based channel importance: zero one channel at a time in the
    held-out test set and measure the balanced-accuracy drop vs. baseline.
    Larger drop = more important channel. No re-training required."""
    n_channels = X_test.shape[2]
    importances = {}
    for ch in range(n_channels):
        X_ablated = X_test.copy()
        X_ablated[:, :, ch, :] = 0.0
        loader = make_loader(X_ablated, y_test, batch_size, shuffle=False)
        preds, labels = predict_all(model, loader, device, n_steps, readout_mode)
        acc = balanced_accuracy_score(labels, preds)
        importances[channel_names[ch]] = baseline_acc - acc
    return importances


# ----------------------------------------------------------------------
# PHASE 1 — per-subject true LOSO
# ----------------------------------------------------------------------
def run_phase1(X, y, subject_ids, meta, device, train_cfg, model_cfg,
               channel_names, do_channels, batch_size, n_steps_eval, readout_mode):
    subjects_internal = sorted(set(int(s) for s in subject_ids))
    subject_labels = meta.get("subject_list", subjects_internal)

    per_subject = {}
    channel_importance_rows = []

    for i, subj in enumerate(subjects_internal):
        real_id = subject_labels[i] if i < len(subject_labels) else subj
        print(f"\n=== Phase 1: LOSO subject {i + 1}/{len(subjects_internal)} "
              f"(internal idx {subj}, subject id {real_id}) ===")

        test_mask = subject_ids == subj
        train_mask = ~test_mask
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        print(f"  Train: {X_tr.shape[0]} trials | Test: {X_te.shape[0]} trials")

        train_loader = make_loader(X_tr, y_tr, train_cfg["batch_size"])
        val_loader = make_loader(X_te, y_te, train_cfg["batch_size"], shuffle=False)

        try:
            model = build_model(meta, device, **model_cfg)
            history = run_training(
                model, train_loader, val_loader,
                epochs=train_cfg["epochs"], lr=train_cfg["lr"], device=device,
                n_steps_train=train_cfg["n_steps_train"], n_steps_eval=train_cfg["n_steps_eval"],
                readout_mode=train_cfg["readout_mode"], eval_every_epoch=True,
                patience=train_cfg.get("patience"),
            )
        except Exception as e:
            print(f"  [SKIPPED subject {real_id}] training failed: {e}")
            continue

        preds, labels = predict_all(model, val_loader, device, train_cfg["n_steps_eval"], train_cfg["readout_mode"])
        bal_acc = balanced_accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        cm = confusion_matrix(labels, preds, labels=list(range(meta["n_classes"])))
        per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

        per_subject[real_id] = dict(
            internal_idx=subj, history=history, bal_acc=bal_acc, f1_macro=f1_macro,
            confusion_matrix=cm, per_class_acc=per_class_acc,
            n_train=int(train_mask.sum()), n_test=int(test_mask.sum()),
        )
        print(f"  -> bal_acc={bal_acc:.4f}  f1_macro={f1_macro:.4f}")

        if do_channels:
            try:
                imp = channel_importance_for_model(
                    model, X_te, y_te, device, n_steps_eval, readout_mode,
                    batch_size, channel_names, bal_acc,
                )
                for ch_name, drop in imp.items():
                    channel_importance_rows.append({"subject": real_id, "channel": ch_name, "importance": drop})
            except Exception as e:
                print(f"  [channel importance skipped for subject {real_id}] {e}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return per_subject, channel_importance_rows


# ----------------------------------------------------------------------
# PHASE 1 plots
# ----------------------------------------------------------------------
def plot_per_subject_accuracy(per_subject, meta, out_dir):
    subjects = list(per_subject.keys())
    accs = [per_subject[s]["bal_acc"] for s in subjects]
    order = np.argsort(accs)[::-1]
    subjects_sorted = [subjects[i] for i in order]
    accs_sorted = [accs[i] for i in order]
    chance = 1 / meta["n_classes"]
    mean_acc = float(np.mean(accs))

    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(subjects)), 5))
    colors = ["#2a7f3f" if a >= mean_acc else "#b3492e" for a in accs_sorted]
    ax.bar([str(s) for s in subjects_sorted], accs_sorted, color=colors, edgecolor="white")
    ax.axhline(mean_acc, color="black", linestyle="--", linewidth=1.2, label=f"Mean = {mean_acc:.3f}")
    ax.axhline(chance, color="grey", linestyle=":", linewidth=1.2, label=f"Chance = {chance:.3f}")
    for x, a in enumerate(accs_sorted):
        ax.text(x, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Per-Subject Leave-One-Subject-Out Accuracy (Best → Worst)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_per_subject_accuracy.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '01_per_subject_accuracy.png'}")
    return subjects_sorted, accs_sorted


def plot_confusion_matrices(per_subject, meta, out_dir):
    subjects = list(per_subject.keys())
    n = len(subjects)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])
    class_names = meta.get("class_names", list(range(meta["n_classes"])))

    for i, subj in enumerate(subjects):
        cm = per_subject[subj]["confusion_matrix"]
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i],
                    annot_kws={"size": 8}, vmin=0, vmax=1)
        axes[i].set_title(f"Subject {subj} (acc={per_subject[subj]['bal_acc']:.3f})", fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Per-Subject Confusion Matrices (Row-Normalized)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "02_confusion_matrices.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '02_confusion_matrices.png'}")


def plot_per_class_heatmap(per_subject, meta, out_dir):
    subjects = list(per_subject.keys())
    class_names = meta.get("class_names", list(range(meta["n_classes"])))
    mat = np.array([per_subject[s]["per_class_acc"] for s in subjects])

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(class_names)), max(5, 0.45 * len(subjects))))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=[str(s) for s in subjects],
                cbar_kws={"label": "Per-Class Accuracy"}, ax=ax)
    ax.set_xlabel("Class")
    ax.set_ylabel("Subject")
    ax.set_title("Per-Class Accuracy by Subject")
    fig.tight_layout()
    fig.savefig(out_dir / "03_per_class_accuracy_heatmap.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '03_per_class_accuracy_heatmap.png'}")


def plot_training_curves(per_subject, meta, out_dir):
    chance = 1 / meta["n_classes"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for subj, d in per_subject.items():
        hist = d["history"]
        epochs_run = range(1, len(hist["loss"]) + 1)
        ax1.plot(epochs_run, hist["loss"], linewidth=1.2, alpha=0.8, label=f"subj {subj}")
        ax2.plot(epochs_run, hist["bal_acc"], linewidth=1.2, alpha=0.8, label=f"subj {subj}")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Per-Subject Training Curves (True LOSO)")
    ax2.axhline(chance, color="grey", linestyle="--", linewidth=1, label=f"Chance ({chance:.2f})")
    ax2.set_ylabel("Balanced Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "04_training_curves.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '04_training_curves.png'}")


def plot_accuracy_distribution(per_subject, meta, out_dir):
    accs = [d["bal_acc"] for d in per_subject.values()]
    chance = 1 / meta["n_classes"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(y=accs, ax=ax, color="#4C72B0", width=0.3)
    sns.stripplot(y=accs, ax=ax, color="black", alpha=0.6, size=7)
    ax.axhline(chance, color="grey", linestyle=":", label=f"Chance ({chance:.2f})")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Distribution of Per-Subject Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "05_accuracy_distribution.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '05_accuracy_distribution.png'}")


def write_per_subject_csv(per_subject, out_dir):
    rows = []
    for subj, d in per_subject.items():
        rows.append({
            "subject": subj, "bal_acc": d["bal_acc"], "f1_macro": d["f1_macro"],
            "n_train": d["n_train"], "n_test": d["n_test"],
        })
    df = pd.DataFrame(rows).sort_values("bal_acc", ascending=False)
    path = out_dir / "per_subject_summary.csv"
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return df


# ----------------------------------------------------------------------
# PHASE 2a — component ablations
# ----------------------------------------------------------------------
def build_ablation_configs(baseline_params):
    """Each ablation flips ONE design choice relative to the baseline best
    config; everything else stays identical so the comparison isolates that
    one factor's effect."""
    configs = []
    base_zscore, base_bandpass = baseline_params["RUN_ZSCORE"], baseline_params["RUN_BANDPASS"]
    base_axis, base_readout = baseline_params["NORM_AXIS"], baseline_params["READOUT_MODE"]

    configs.append({"name": "no_bandpass", "run_bandpass": False, "run_zscore": base_zscore, "norm_axis": base_axis, "readout_mode": base_readout})
    configs.append({"name": "zscore_per_channel_on", "run_bandpass": base_bandpass, "run_zscore": True, "norm_axis": (1, 3), "readout_mode": base_readout})
    configs.append({"name": "zscore_per_epoch_on", "run_bandpass": base_bandpass, "run_zscore": True, "norm_axis": (1, 2, 3), "readout_mode": base_readout})
    for mode in READOUT_MODES:
        if mode == base_readout:
            continue
        configs.append({"name": f"readout_{mode}", "run_bandpass": base_bandpass, "run_zscore": base_zscore, "norm_axis": base_axis, "readout_mode": mode})
    return configs


def run_ablations(params, ablation_subjects_internal, meta_template, device, batch_size_override=None):
    results = []
    configs = build_ablation_configs(params)
    train_cfg_base, model_cfg = make_cfgs(params)

    for cfg in configs:
        print(f"\n=== Phase 2a: ablation '{cfg['name']}' "
              f"(zscore={cfg['run_zscore']}, bandpass={cfg['run_bandpass']}, "
              f"axis={cfg['norm_axis']}, readout={cfg['readout_mode']}) ===")
        X, y, subject_ids, meta = prepare_data(
            params, override_zscore=cfg["run_zscore"], override_bandpass=cfg["run_bandpass"],
            override_norm_axis=cfg["norm_axis"],
        )
        train_cfg = dict(train_cfg_base)
        train_cfg["readout_mode"] = cfg["readout_mode"]

        accs = []
        for subj in ablation_subjects_internal:
            test_mask = subject_ids == subj
            train_mask = ~test_mask
            train_loader = make_loader(X[train_mask], y[train_mask], train_cfg["batch_size"])
            val_loader = make_loader(X[test_mask], y[test_mask], train_cfg["batch_size"], shuffle=False)
            try:
                model = build_model(meta, device, **model_cfg)
                history = run_training(
                    model, train_loader, val_loader,
                    epochs=train_cfg["epochs"], lr=train_cfg["lr"], device=device,
                    n_steps_train=train_cfg["n_steps_train"], n_steps_eval=train_cfg["n_steps_eval"],
                    readout_mode=train_cfg["readout_mode"], eval_every_epoch=True,
                    patience=train_cfg.get("patience"),
                )
                accs.append(history["bal_acc"][-1])
            except Exception as e:
                print(f"  [SKIPPED subject {subj} for ablation '{cfg['name']}'] {e}")
                continue
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if not accs:
            print(f"  -> ablation '{cfg['name']}' produced no successful runs, skipping from results")
            continue

        results.append({
            "name": cfg["name"], "mean_bal_acc": float(np.mean(accs)),
            "std_bal_acc": float(np.std(accs)), "n_subjects": len(accs),
        })
        print(f"  -> mean_bal_acc={np.mean(accs):.4f} (n={len(accs)} subjects)")

    return results


def plot_ablation_comparison(baseline_mean, baseline_std, ablation_results, out_dir, baseline_label="baseline (best config)"):
    names = [baseline_label] + [r["name"] for r in ablation_results]
    means = [baseline_mean] + [r["mean_bal_acc"] for r in ablation_results]
    stds = [baseline_std] + [r["std_bal_acc"] for r in ablation_results]

    order = np.argsort(means)[::-1]
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]
    colors = ["#2a7f3f" if n == baseline_label else "#4C72B0" for n in names]

    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(names)), 5))
    ax.bar(names, means, yerr=stds, capsize=4, color=colors, edgecolor="white")
    for x, m in enumerate(means):
        ax.text(x, m + 0.015, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Mean Balanced Accuracy (± std across evaluated subjects)")
    ax.set_title("Ablation Study: Effect of Each Design Choice")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "08_ablation_comparison.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '08_ablation_comparison.png'}")


# ----------------------------------------------------------------------
# PHASE 2b — channel importance plots
# ----------------------------------------------------------------------
def plot_channel_importance(channel_rows, out_dir):
    df = pd.DataFrame(channel_rows)
    if df.empty:
        print("No channel-importance data to plot (use --do-channels).")
        return df
    agg = df.groupby("channel")["importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.32 * len(agg))))
    colors = ["#b3492e" if v > 0 else "#4C72B0" for v in agg["mean"]]
    ax.barh(agg.index[::-1], agg["mean"][::-1], xerr=agg["std"].fillna(0)[::-1],
            color=colors[::-1], capsize=3, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Accuracy Drop When Channel Is Zeroed Out (across subjects)")
    ax.set_title("Channel Importance (Occlusion-Based)")
    fig.tight_layout()
    fig.savefig(out_dir / "06_channel_importance.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '06_channel_importance.png'}")

    pivot = df.pivot(index="subject", columns="channel", values="importance")
    pivot = pivot[agg.index]  # order columns by overall importance
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * pivot.shape[1]), max(4, 0.45 * pivot.shape[0])))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, ax=ax,
                cbar_kws={"label": "Accuracy Drop When Zeroed Out"})
    ax.set_xlabel("Channel")
    ax.set_ylabel("Subject")
    ax.set_title("Channel Importance by Subject")
    fig.tight_layout()
    fig.savefig(out_dir / "07_channel_importance_heatmap.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '07_channel_importance_heatmap.png'}")

    path = out_dir / "channel_importance.csv"
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return agg


# ----------------------------------------------------------------------
# Recommendations write-up
# ----------------------------------------------------------------------
def write_recommendations(subject_df, ablation_results, baseline_mean, channel_agg, out_dir, meta):
    lines = []
    lines.append("=" * 70)
    lines.append("PER-SUBJECT + ABLATION STUDY — SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    chance = 1 / meta["n_classes"]
    lines.append("-" * 70)
    lines.append("PER-SUBJECT PERFORMANCE")
    lines.append("-" * 70)
    lines.append(f"  Mean balanced accuracy : {subject_df['bal_acc'].mean():.4f}")
    lines.append(f"  Std balanced accuracy  : {subject_df['bal_acc'].std():.4f}")
    lines.append(f"  Chance level            : {chance:.4f}")
    best_row = subject_df.iloc[0]
    worst_row = subject_df.iloc[-1]
    lines.append(f"  Best subject  : {best_row['subject']}  (acc={best_row['bal_acc']:.4f})")
    lines.append(f"  Worst subject : {worst_row['subject']} (acc={worst_row['bal_acc']:.4f})")
    spread = best_row['bal_acc'] - worst_row['bal_acc']
    lines.append(f"  Best-worst spread : {spread:.4f}")
    if spread > 0.25:
        lines.append("  [FLAG: large inter-subject variance -- consider subject-specific")
        lines.append("   fine-tuning or zero-shot calibration tailored to hard subjects.]")
    lines.append("")
    lines.append("  Full ranking (best -> worst):")
    for _, row in subject_df.iterrows():
        lines.append(f"      subject {row['subject']:>4}: bal_acc={row['bal_acc']:.4f}  f1_macro={row['f1_macro']:.4f}")
    lines.append("")

    if ablation_results:
        lines.append("-" * 70)
        lines.append("ABLATION STUDY (design-choice comparison)")
        lines.append("-" * 70)
        lines.append(f"  Baseline (best config) mean_bal_acc = {baseline_mean:.4f}")
        for r in sorted(ablation_results, key=lambda x: -x["mean_bal_acc"]):
            delta = r["mean_bal_acc"] - baseline_mean
            sign = "+" if delta >= 0 else ""
            lines.append(f"      {r['name']:25s} mean_bal_acc={r['mean_bal_acc']:.4f} "
                         f"({sign}{delta:.4f} vs baseline, n={r['n_subjects']} subjects)")
        worst_ablation = min(ablation_results, key=lambda x: x["mean_bal_acc"])
        lines.append(f"  -> Most harmful change when removed/swapped: {worst_ablation['name']} "
                     f"({worst_ablation['mean_bal_acc'] - baseline_mean:+.4f})")
        lines.append("")

    if channel_agg is not None and not channel_agg.empty:
        lines.append("-" * 70)
        lines.append("CHANNEL IMPORTANCE (top 10 most important)")
        lines.append("-" * 70)
        for ch, row in channel_agg.head(10).iterrows():
            lines.append(f"      {ch:8s} mean_drop={row['mean']:.4f}  std={row['std']:.4f}")
        lines.append("")
        lines.append("  Bottom 5 (least important / possibly removable):")
        for ch, row in channel_agg.tail(5).iterrows():
            lines.append(f"      {ch:8s} mean_drop={row['mean']:.4f}  std={row['std']:.4f}")
        lines.append("")
        negative = channel_agg[channel_agg["mean"] < 0]
        if not negative.empty:
            lines.append(f"  [NOTE: {len(negative)} channel(s) show negative mean drop -- removing them")
            lines.append("   did not hurt (and slightly helped) accuracy on average. These are")
            lines.append("   candidates for a reduced-channel-count edge deployment.]")
        lines.append("")

    path = out_dir / "recommendations.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved recommendations -> {path}")
    print("\n" + "\n".join(lines[:25]) + "\n  ... (see recommendations.txt for full report)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params-json", type=str, default=None,
                  help="JSON file of best hyperparameters. If omitted (the default), "
                       "best params are read live from --optuna-db instead.")
    p.add_argument("--optuna-db", type=str, default="sqlite:///optuna_study.db",
                  help="Optuna storage URL (default: sqlite:///optuna_study.db, "
                       "matching the file main.py writes to in the repo root)")
    p.add_argument("--study-name", type=str, default=None,
                  help="Optuna study name. If omitted, auto-detected as long as the "
                       "database contains exactly one study.")
    p.add_argument("--dataset-key", type=str, default=None, help="Override DATASET_KEY (default: BNCI2014_001)")
    p.add_argument("--epochs", type=int, default=None, help="Override EPOCHS for all runs")
    p.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE")
    p.add_argument("--out-dir", type=str, default=None,
                  help="Output directory (default: results/<DATASET_KEY>/ablation_study)")
    p.add_argument("--skip-ablation", action="store_true", help="Skip Phase 2a component ablations")
    p.add_argument("--skip-channels", action="store_true", help="Skip Phase 2b channel-importance analysis")
    p.add_argument("--ablation-subjects", type=str, default=None,
                  help="Comma-separated internal subject indices to use for Phase 2a "
                       "(default: all subjects -- expensive, ~7x extra LOSO runs since "
                       "there are 7 non-baseline ablation configs)")
    return p.parse_args()


def main():
    args = parse_args()

    params = load_best_params(args)
    print("Resolved config:")
    for k, v in params.items():
        print(f"  {k:25s} = {v}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / params["DATASET_KEY"] / "ablation_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    X, y, subject_ids, meta = prepare_data(params)
    print(f"Dataset meta: {meta}")
    channel_names = get_channel_names(params["DATASET_KEY"], meta["n_channels"])

    train_cfg, model_cfg = make_cfgs(params)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    per_subject, channel_rows = run_phase1(
        X, y, subject_ids, meta, device, train_cfg, model_cfg,
        channel_names, do_channels=not args.skip_channels,
        batch_size=train_cfg["batch_size"], n_steps_eval=train_cfg["n_steps_eval"],
        readout_mode=train_cfg["readout_mode"],
    )

    plot_per_subject_accuracy(per_subject, meta, out_dir)
    plot_confusion_matrices(per_subject, meta, out_dir)
    plot_per_class_heatmap(per_subject, meta, out_dir)
    plot_training_curves(per_subject, meta, out_dir)
    plot_accuracy_distribution(per_subject, meta, out_dir)
    subject_df = write_per_subject_csv(per_subject, out_dir)

    baseline_mean = float(subject_df["bal_acc"].mean())
    baseline_std = float(subject_df["bal_acc"].std())

    # ── Phase 2a: component ablations ───────────────────────────────────
    ablation_results = []
    if not args.skip_ablation:
        internal_subjects = sorted(set(int(s) for s in subject_ids))
        if args.ablation_subjects:
            ablation_subjects = [int(s) for s in args.ablation_subjects.split(",")]
        else:
            ablation_subjects = internal_subjects
        ablation_results = run_ablations(params, ablation_subjects, meta, device)
        plot_ablation_comparison(baseline_mean, baseline_std, ablation_results, out_dir)
        pd.DataFrame(
            [{"name": "baseline", "mean_bal_acc": baseline_mean, "std_bal_acc": baseline_std,
              "n_subjects": len(per_subject)}] + ablation_results
        ).to_csv(out_dir / "ablation_summary.csv", index=False)
        print(f"Saved -> {out_dir / 'ablation_summary.csv'}")

    # ── Phase 2b: channel importance ────────────────────────────────────
    channel_agg = None
    if not args.skip_channels:
        channel_agg = plot_channel_importance(channel_rows, out_dir)

    write_recommendations(subject_df, ablation_results, baseline_mean, channel_agg, out_dir, meta)

    print("\nDone. All outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()