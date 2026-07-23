# ablation.py
"""
Two analyses in one script, run back-to-back so the per-subject LOSO models
trained in Phase 1 can be reused for the channel-importance part of Phase 2
instead of re-training from scratch.

Lives in src/ alongside the rest of the pipeline (build_model.py, pipeline.py,
etc.) and imports them directly, the same way
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
    This is a genuine ablation study: each analysis below removes or degrades
    one specific part of the model or the input and measures the resulting
    drop in balanced accuracy, isolating that one part's contribution.

    a) Model-component ablations (re-trained): each config strips out ONE
       architectural/dynamical piece of the tuned model, holding everything
       else fixed, then re-runs LOSO:
         - no_depthwise      : depth_multiplier -> 1 (removes the depthwise
                                spatial-filter expansion)
         - no_separable      : pointwise_filters -> 1 (collapses the
                                separable/pointwise conv's capacity)
         - no_dropout        : dropout -> 0.0 (removes regularization)
         - single_timestep   : n_steps_train = n_steps_eval -> 1 (removes the
                                spiking model's multi-step temporal dynamics)
         - no_leak           : beta -> ~1.0 (removes the leaky-integrate
                                decay of the LIF neuron)
         - readout_<mode>    : swap READOUT_MODE (kept from before; now one
                                genuine ablation among several)
       Configs that would be a no-op against the baseline (e.g. the baseline
       already has depth_multiplier=1) are skipped automatically.
    b) Single-channel occlusion importance: for each subject's Phase-1 model,
       zero out one EEG channel at a time in the held-out test set and
       measure the drop in balanced accuracy. No re-training needed.
    c) Progressive channel removal: using the per-subject importance ranking
       from (b), cumulatively zero channels out most-important-first and
       least-important-first, tracing two accuracy-vs-channels-removed
       curves. Shows how quickly the model collapses when it loses the
       channels it relies on most, versus how many "useless" channels could
       be dropped for free.
    d) Channel-group (scalp-region) ablation: zero out whole anatomical
       regions (frontal / central / centro-parietal / parietal, from the
       known 10-20 layout; falls back to index-quartiles for unknown
       datasets) at once, to see which broad regions matter.
    e) Temporal-window ablation: zero out one quarter of the trial's time
       axis at a time, to see which part of the trial timeline carries the
       discriminative signal.
    (c)-(e) all reuse the already-trained Phase-1 models, so they're cheap;
    only (a) requires re-training.

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
    09_progressive_channel_removal.png
    10_channel_group_importance.png
    11_temporal_window_importance.png
    ablation_summary.csv
    channel_importance.csv
    progressive_channel_removal.csv
    channel_group_importance.csv
    temporal_window_importance.csv
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

# Anatomical scalp-region groupings used for the channel-group ablation
# (Phase 2d). Falls back to index-quartiles (get_channel_groups) for any
# dataset not listed here.
CHANNEL_GROUPS = {
    "BNCI2014_001": {
        "Frontal": ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4"],
        "Central": ["C5", "C3", "C1", "Cz", "C2", "C4", "C6"],
        "Centro-parietal": ["CP3", "CP1", "CPz", "CP2", "CP4"],
        "Parietal/Occipital": ["P1", "Pz", "P2", "POz"],
    },
}

# FIXED params held constant during the Optuna study (main.py FIXED dict).
# These don't live in best_params.json (search space only), so we supply
# sensible defaults here, overridable via CLI. Z-score normalization
# (RUN_ZSCORE / NORM_AXIS) has been removed from the pipeline entirely, so
# it's no longer part of this config.
DEFAULT_FIXED = dict(
    DATASET_KEY="BNCI2014_001",
    EPOCHS=50,
    BATCH_SIZE=32,
)

# main.py's HPO search space now only searches READOUT_MODE over
# ["spk_mean", "spk_sum"] (spk_last and mem_last were dropped -- mem_last
# had a known shape-mismatch bug, spk_last added little on top of spk_mean).
# Ablation configs mirror that same restricted set so we don't burn compute
# re-testing readout modes the actual HPO study can no longer select.
READOUT_MODES = ["spk_mean", "spk_sum"]


def pretty(col: str) -> str:
    names = {
        "bal_acc": "Balanced Accuracy", "f1_macro": "Macro F1 Score",
        "subject": "Subject", "readout_mode": "Readout Mode",
        "mean_bal_acc": "Mean Balanced Accuracy",
        "std_bal_acc": "Std. Dev. Balanced Accuracy",
    }
    return names.get(col, col.replace("_", " ").title())


def get_channel_names(dataset_key, n_channels):
    names = CHANNEL_NAMES.get(dataset_key)
    if names and len(names) == n_channels:
        return names
    return [f"Ch{i}" for i in range(n_channels)]


def get_channel_groups(dataset_key, channel_names):
    """Anatomical scalp regions for the channel-group ablation. Falls back to
    four index-quartiles (labelled generically) when the dataset isn't in
    CHANNEL_GROUPS or the known layout doesn't match the actual channel
    names (e.g. generic "Ch0".."ChN-1" names from get_channel_names)."""
    groups = CHANNEL_GROUPS.get(dataset_key)
    if groups and all(c in channel_names for cs in groups.values() for c in cs):
        return groups
    n = len(channel_names)
    n_groups = min(4, n) if n > 0 else 0
    fallback = {}
    edges = np.linspace(0, n, n_groups + 1).astype(int)
    for i in range(n_groups):
        fallback[f"Group {i + 1}"] = channel_names[edges[i]:edges[i + 1]]
    return fallback


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


def prepare_data(params):
    """Loads the dataset. Preprocessing is now just what load_moabb_dataset
    itself does -- z-score normalization has been removed from the pipeline,
    so there's no longer any override/branch to handle here."""
    return load_moabb_dataset(params["DATASET_KEY"])


@torch.no_grad()
def predict_all(model, loader, device, n_steps, readout_mode):
    """Like evaluate(), but returns raw predictions/labels for confusion
    matrices and per-class breakdowns instead of just a scalar accuracy.

    Only spk_mean / spk_sum readout modes are supported now (matching
    main.py's search space) -- aggregate_logits() handles both directly, so
    no special-casing is needed here anymore.
    """
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        spk, mem = model(xb, num_steps=n_steps)
        logits = aggregate_logits(spk, mem, readout_mode)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(yb.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def channel_importance_for_model(model, X_test, y_test, device, n_steps, readout_mode,
                                  batch_size, channel_names, baseline_acc):
    """Occlusion-based channel importance: zero one channel at a time in the
    held-out test set and measure the balanced-accuracy drop vs. baseline.
    Larger drop = more important channel. No re-training required.

    Returns (importances_by_name, order_desc) where order_desc is an array
    of channel *indices* sorted most-important-first, used downstream by
    progressive_channel_removal_for_model."""
    n_channels = X_test.shape[2]
    importances = {}
    drops = np.zeros(n_channels)
    for ch in range(n_channels):
        X_ablated = X_test.copy()
        X_ablated[:, :, ch, :] = 0.0
        loader = make_loader(X_ablated, y_test, batch_size, shuffle=False)
        preds, labels = predict_all(model, loader, device, n_steps, readout_mode)
        acc = balanced_accuracy_score(labels, preds)
        drop = baseline_acc - acc
        importances[channel_names[ch]] = drop
        drops[ch] = drop
    order_desc = np.argsort(drops)[::-1]
    return importances, order_desc


def progressive_channel_removal_for_model(model, X_test, y_test, device, n_steps, readout_mode,
                                           batch_size, order_desc):
    """Cumulative ablation curve: zero channels out one at a time, in order
    of importance, tracing how balanced accuracy degrades as more channels
    are removed. Run in both directions:
      - most_important_first : removes the channels the model relies on most
                                first -- shows how fast it collapses.
      - least_important_first : removes the "least useful" channels first --
                                 shows how many channels could be dropped
                                 for free before accuracy suffers.
    Index 0 in each curve corresponds to zero channels removed (i.e. the
    unmodified baseline test accuracy)."""
    n_channels = X_test.shape[2]
    results = {}
    for order_name, order in [
        ("most_important_first", order_desc),
        ("least_important_first", order_desc[::-1]),
    ]:
        accs = []
        X_ablated = X_test.copy()
        for k in range(n_channels + 1):
            if k > 0:
                X_ablated[:, :, order[k - 1], :] = 0.0
            loader = make_loader(X_ablated, y_test, batch_size, shuffle=False)
            preds, labels = predict_all(model, loader, device, n_steps, readout_mode)
            accs.append(balanced_accuracy_score(labels, preds))
        results[order_name] = accs
    return results


def channel_group_importance_for_model(model, X_test, y_test, device, n_steps, readout_mode,
                                        batch_size, channel_names, baseline_acc, groups):
    """Zero out an entire anatomical group of channels at once (e.g. all
    frontal electrodes together) to measure how much the model relies on
    that broad scalp region, rather than any single electrode."""
    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    importances = {}
    for group_name, group_channels in groups.items():
        idxs = [name_to_idx[c] for c in group_channels if c in name_to_idx]
        if not idxs:
            continue
        X_ablated = X_test.copy()
        X_ablated[:, :, idxs, :] = 0.0
        loader = make_loader(X_ablated, y_test, batch_size, shuffle=False)
        preds, labels = predict_all(model, loader, device, n_steps, readout_mode)
        acc = balanced_accuracy_score(labels, preds)
        importances[group_name] = baseline_acc - acc
    return importances


def temporal_window_importance_for_model(model, X_test, y_test, device, n_steps, readout_mode,
                                          batch_size, baseline_acc, n_windows=4):
    """Zero out one time-window (e.g. one quarter of the trial) at a time to
    see which part of the trial's timeline the model actually needs."""
    T = X_test.shape[3]
    edges = np.linspace(0, T, n_windows + 1).astype(int)
    importances = {}
    for w in range(n_windows):
        lo, hi = edges[w], edges[w + 1]
        if hi <= lo:
            continue
        X_ablated = X_test.copy()
        X_ablated[:, :, :, lo:hi] = 0.0
        loader = make_loader(X_ablated, y_test, batch_size, shuffle=False)
        preds, labels = predict_all(model, loader, device, n_steps, readout_mode)
        acc = balanced_accuracy_score(labels, preds)
        importances[f"window_{w + 1}_of_{n_windows}"] = baseline_acc - acc
    return importances


# ----------------------------------------------------------------------
# PHASE 1 — per-subject true LOSO
# ----------------------------------------------------------------------
def run_phase1(X, y, subject_ids, meta, device, train_cfg, model_cfg,
               channel_names, do_channels, do_progressive, do_groups, do_temporal,
               batch_size, n_steps_eval, readout_mode, dataset_key, n_temporal_windows=4):
    subjects_internal = sorted(set(int(s) for s in subject_ids))
    subject_labels = meta.get("subject_list", subjects_internal)
    channel_groups = get_channel_groups(dataset_key, channel_names)

    per_subject = {}
    channel_importance_rows = []
    progressive_rows = []
    channel_group_rows = []
    temporal_window_rows = []

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

        order_desc = None
        if do_channels or do_progressive:
            try:
                imp, order_desc = channel_importance_for_model(
                    model, X_te, y_te, device, n_steps_eval, readout_mode,
                    batch_size, channel_names, bal_acc,
                )
                for ch_name, drop in imp.items():
                    channel_importance_rows.append({"subject": real_id, "channel": ch_name, "importance": drop})
            except Exception as e:
                print(f"  [channel importance skipped for subject {real_id}] {e}")

        if do_progressive and order_desc is not None:
            try:
                prog = progressive_channel_removal_for_model(
                    model, X_te, y_te, device, n_steps_eval, readout_mode,
                    batch_size, order_desc,
                )
                for order_name, accs in prog.items():
                    for k, acc in enumerate(accs):
                        progressive_rows.append({
                            "subject": real_id, "order": order_name,
                            "k_removed": k, "accuracy": acc,
                        })
            except Exception as e:
                print(f"  [progressive channel removal skipped for subject {real_id}] {e}")

        if do_groups:
            try:
                grp = channel_group_importance_for_model(
                    model, X_te, y_te, device, n_steps_eval, readout_mode,
                    batch_size, channel_names, bal_acc, channel_groups,
                )
                for group_name, drop in grp.items():
                    channel_group_rows.append({"subject": real_id, "group": group_name, "importance": drop})
            except Exception as e:
                print(f"  [channel group importance skipped for subject {real_id}] {e}")

        if do_temporal:
            try:
                twin = temporal_window_importance_for_model(
                    model, X_te, y_te, device, n_steps_eval, readout_mode,
                    batch_size, bal_acc, n_windows=n_temporal_windows,
                )
                for window_name, drop in twin.items():
                    temporal_window_rows.append({"subject": real_id, "window": window_name, "importance": drop})
            except Exception as e:
                print(f"  [temporal window importance skipped for subject {real_id}] {e}")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return per_subject, channel_importance_rows, progressive_rows, channel_group_rows, temporal_window_rows


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
def build_ablation_configs(train_cfg_base, model_cfg_base):
    """Each config removes or degrades ONE architectural/dynamical part of
    the tuned model relative to the baseline, holding everything else fixed,
    so the resulting accuracy delta isolates that one component's
    contribution. This is a genuine component-ablation set (not just a
    hyperparameter swap):

      no_depthwise     : depth_multiplier -> 1 (removes the depthwise
                         spatial-filter expansion)
      no_separable     : pointwise_filters -> 1 (collapses the separable/
                         pointwise conv down to minimal capacity)
      no_dropout       : dropout -> 0.0 (removes regularization)
      single_timestep  : n_steps_train = n_steps_eval -> 1 (removes the
                         spiking model's multi-step temporal dynamics)
      no_leak          : beta -> 0.999 (removes the leaky-integrate decay
                         of the LIF neuron, i.e. "no leak")
      readout_<mode>   : swap READOUT_MODE (kept from before as one
                         ablation among several)

    A config is skipped if the baseline is already at that value (e.g. the
    tuned model already has depth_multiplier=1), since that wouldn't be a
    meaningful ablation."""
    configs = []

    def maybe_add(name, key_section, overrides):
        base_dict = model_cfg_base if key_section == "model_overrides" else train_cfg_base
        if all(base_dict.get(k) == v for k, v in overrides.items()):
            print(f"  [skipping ablation '{name}': baseline already matches {overrides}]")
            return
        configs.append({"name": name, key_section: overrides})

    maybe_add("no_depthwise", "model_overrides", {"depth_multiplier": 1})
    maybe_add("no_separable", "model_overrides", {"pointwise_filters": 1})
    maybe_add("no_dropout", "model_overrides", {"dropout": 0.0})
    maybe_add("single_timestep", "train_overrides", {"n_steps_train": 1, "n_steps_eval": 1})
    maybe_add("no_leak", "model_overrides", {"beta": 0.999})

    base_readout = train_cfg_base["readout_mode"]
    for mode in READOUT_MODES:
        if mode == base_readout:
            continue
        maybe_add(f"readout_{mode}", "train_overrides", {"readout_mode": mode})

    return configs


def run_ablations(params, ablation_subjects_internal, meta, device, X, y, subject_ids):
    """Re-runs LOSO training once per config in build_ablation_configs(),
    each config stripping out one model component or dynamical piece (or,
    for readout_<mode>, swapping the readout hyperparameter). Reuses the
    dataset already loaded in main()."""
    results = []
    train_cfg_base, model_cfg_base = make_cfgs(params)
    configs = build_ablation_configs(train_cfg_base, model_cfg_base)

    for cfg in configs:
        train_cfg = dict(train_cfg_base)
        train_cfg.update(cfg.get("train_overrides", {}))
        model_cfg = dict(model_cfg_base)
        model_cfg.update(cfg.get("model_overrides", {}))
        overrides_desc = {**cfg.get("model_overrides", {}), **cfg.get("train_overrides", {})}
        print(f"\n=== Phase 2a: ablation '{cfg['name']}' (overrides={overrides_desc}) ===")

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
# PHASE 2c — progressive (cumulative) channel-removal curve
# ----------------------------------------------------------------------
def plot_progressive_channel_removal(progressive_rows, meta, out_dir):
    df = pd.DataFrame(progressive_rows)
    if df.empty:
        print("No progressive channel-removal data to plot (use --skip-progressive to disable).")
        return df

    chance = 1 / meta["n_classes"]
    colors = {"most_important_first": "#b3492e", "least_important_first": "#4C72B0"}
    labels = {
        "most_important_first": "Remove most-important channels first",
        "least_important_first": "Remove least-important channels first",
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for order_name, sub in df.groupby("order"):
        agg = sub.groupby("k_removed")["accuracy"].agg(["mean", "std"]).sort_index()
        std = agg["std"].fillna(0.0)
        ax.plot(agg.index, agg["mean"], color=colors.get(order_name, "black"),
                label=labels.get(order_name, order_name), linewidth=2)
        ax.fill_between(agg.index, agg["mean"] - std, agg["mean"] + std,
                         color=colors.get(order_name, "black"), alpha=0.15)
    ax.axhline(chance, color="grey", linestyle=":", linewidth=1.2, label=f"Chance = {chance:.3f}")
    ax.set_xlabel("Number of Channels Zeroed Out")
    ax.set_ylabel("Balanced Accuracy (mean ± std across subjects)")
    ax.set_title("Progressive Channel Removal (Cumulative Ablation Curve)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "09_progressive_channel_removal.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '09_progressive_channel_removal.png'}")

    path = out_dir / "progressive_channel_removal.csv"
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return df


# ----------------------------------------------------------------------
# PHASE 2d — channel-group (scalp-region) ablation
# ----------------------------------------------------------------------
def plot_channel_group_importance(channel_group_rows, out_dir):
    df = pd.DataFrame(channel_group_rows)
    if df.empty:
        print("No channel-group importance data to plot (use --skip-groups to disable).")
        return df
    agg = df.groupby("group")["importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.6 * len(agg))))
    colors = ["#b3492e" if v > 0 else "#4C72B0" for v in agg["mean"]]
    ax.barh(agg.index[::-1], agg["mean"][::-1], xerr=agg["std"].fillna(0)[::-1],
            color=colors[::-1], capsize=4, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Accuracy Drop When Entire Region Is Zeroed Out (across subjects)")
    ax.set_title("Channel-Group (Scalp-Region) Ablation")
    fig.tight_layout()
    fig.savefig(out_dir / "10_channel_group_importance.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '10_channel_group_importance.png'}")

    path = out_dir / "channel_group_importance.csv"
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return agg


# ----------------------------------------------------------------------
# PHASE 2e — temporal-window ablation
# ----------------------------------------------------------------------
def plot_temporal_window_importance(temporal_window_rows, out_dir):
    df = pd.DataFrame(temporal_window_rows)
    if df.empty:
        print("No temporal-window importance data to plot (use --skip-temporal to disable).")
        return df
    # keep windows in chronological order rather than sorted by importance
    order = sorted(df["window"].unique(), key=lambda w: int(w.split("_")[1]))
    agg = df.groupby("window")["importance"].agg(["mean", "std"]).reindex(order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#b3492e" if v > 0 else "#4C72B0" for v in agg["mean"]]
    ax.bar(agg.index, agg["mean"], yerr=agg["std"].fillna(0), color=colors, capsize=4, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Trial Time Window (chronological order)")
    ax.set_ylabel("Mean Accuracy Drop When Window Is Zeroed Out (across subjects)")
    ax.set_title("Temporal-Window Ablation")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "11_temporal_window_importance.png", **SAVE_KW)
    plt.close(fig)
    print(f"Saved -> {out_dir / '11_temporal_window_importance.png'}")

    path = out_dir / "temporal_window_importance.csv"
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")
    return agg


# ----------------------------------------------------------------------
# Recommendations write-up
# ----------------------------------------------------------------------
def write_recommendations(subject_df, ablation_results, baseline_mean, channel_agg,
                           progressive_df, group_agg, window_agg, out_dir, meta):
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

    if progressive_df is not None and not progressive_df.empty:
        lines.append("-" * 70)
        lines.append("PROGRESSIVE CHANNEL REMOVAL (cumulative ablation curve)")
        lines.append("-" * 70)
        mif = progressive_df[progressive_df["order"] == "most_important_first"]
        lif = progressive_df[progressive_df["order"] == "least_important_first"]
        n_channels = int(progressive_df["k_removed"].max())
        if not mif.empty:
            mif_curve = mif.groupby("k_removed")["accuracy"].mean()
            below_chance = mif_curve[mif_curve <= chance]
            if not below_chance.empty:
                k_collapse = int(below_chance.index.min())
                lines.append(f"  Removing the {k_collapse} most-important channel(s) brings mean "
                             f"accuracy down to chance ({chance:.3f}).")
            else:
                lines.append(f"  Even after removing all {n_channels} channels most-important-first, "
                             f"mean accuracy stayed above chance ({mif_curve.iloc[-1]:.4f}).")
        if not lif.empty:
            lif_curve = lif.groupby("k_removed")["accuracy"].mean()
            baseline_curve = lif_curve.iloc[0]
            drop_10pct = lif_curve[lif_curve < baseline_curve - 0.05]
            if not drop_10pct.empty:
                k_safe = int(drop_10pct.index.min()) - 1
                lines.append(f"  Up to {max(k_safe, 0)} of the least-important channel(s) could be "
                             f"dropped before mean accuracy fell more than 0.05 below baseline "
                             f"-- candidates for a reduced-channel edge deployment.")
            else:
                lines.append("  Removing least-important channels one by one never dropped mean "
                             "accuracy by more than 0.05 -- the model is fairly robust to losing "
                             "its weakest channels.")
        lines.append("")

    if group_agg is not None and not group_agg.empty:
        lines.append("-" * 70)
        lines.append("CHANNEL-GROUP (SCALP-REGION) ABLATION")
        lines.append("-" * 70)
        for grp, row in group_agg.iterrows():
            lines.append(f"      {grp:22s} mean_drop={row['mean']:.4f}  std={row['std']:.4f}")
        most_important_group = group_agg.index[0]
        lines.append(f"  -> Most relied-upon scalp region: {most_important_group}")
        lines.append("")

    if window_agg is not None and not window_agg.empty:
        lines.append("-" * 70)
        lines.append("TEMPORAL-WINDOW ABLATION")
        lines.append("-" * 70)
        for win, row in window_agg.iterrows():
            lines.append(f"      {win:22s} mean_drop={row['mean']:.4f}  std={row['std']:.4f}")
        most_important_window = window_agg["mean"].idxmax()
        lines.append(f"  -> Most information-carrying time window: {most_important_window}")
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
    p.add_argument("--skip-ablation", action="store_true", help="Skip Phase 2a model-component ablations")
    p.add_argument("--skip-channels", action="store_true", help="Skip Phase 2b single-channel occlusion importance")
    p.add_argument("--skip-progressive", action="store_true",
                  help="Skip Phase 2c progressive (cumulative) channel-removal curve")
    p.add_argument("--skip-groups", action="store_true", help="Skip Phase 2d channel-group (scalp-region) ablation")
    p.add_argument("--skip-temporal", action="store_true", help="Skip Phase 2e temporal-window ablation")
    p.add_argument("--temporal-windows", type=int, default=4,
                  help="Number of equal-length time windows to ablate in Phase 2e (default: 4)")
    p.add_argument("--ablation-subjects", type=str, default=None,
                  help="Comma-separated internal subject indices to use for Phase 2a "
                       "(default: all subjects)")
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
    per_subject, channel_rows, progressive_rows, channel_group_rows, temporal_window_rows = run_phase1(
        X, y, subject_ids, meta, device, train_cfg, model_cfg,
        channel_names, do_channels=not args.skip_channels,
        do_progressive=not args.skip_progressive,
        do_groups=not args.skip_groups,
        do_temporal=not args.skip_temporal,
        batch_size=train_cfg["batch_size"], n_steps_eval=train_cfg["n_steps_eval"],
        readout_mode=train_cfg["readout_mode"], dataset_key=params["DATASET_KEY"],
        n_temporal_windows=args.temporal_windows,
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
        ablation_results = run_ablations(params, ablation_subjects, meta, device, X, y, subject_ids)
        plot_ablation_comparison(baseline_mean, baseline_std, ablation_results, out_dir)
        pd.DataFrame(
            [{"name": "baseline", "mean_bal_acc": baseline_mean, "std_bal_acc": baseline_std,
              "n_subjects": len(per_subject)}] + ablation_results
        ).to_csv(out_dir / "ablation_summary.csv", index=False)
        print(f"Saved -> {out_dir / 'ablation_summary.csv'}")

    # ── Phase 2b: single-channel occlusion importance ───────────────────
    channel_agg = None
    if not args.skip_channels:
        channel_agg = plot_channel_importance(channel_rows, out_dir)

    # ── Phase 2c: progressive (cumulative) channel-removal curve ────────
    progressive_df = None
    if not args.skip_progressive:
        progressive_df = plot_progressive_channel_removal(progressive_rows, meta, out_dir)

    # ── Phase 2d: channel-group (scalp-region) ablation ──────────────────
    group_agg = None
    if not args.skip_groups:
        group_agg = plot_channel_group_importance(channel_group_rows, out_dir)

    # ── Phase 2e: temporal-window ablation ───────────────────────────────
    window_agg = None
    if not args.skip_temporal:
        window_agg = plot_temporal_window_importance(temporal_window_rows, out_dir)

    write_recommendations(subject_df, ablation_results, baseline_mean, channel_agg,
                           progressive_df, group_agg, window_agg, out_dir, meta)

    print("\nDone. All outputs written to:", out_dir.resolve())


if __name__ == "__main__":
    main()