# baseline.py
"""
Literal-EEGNet baselines for LOSO comparison against the tuned SpikingEEGNet
(Optuna-searched architecture).

This module now produces TWO baselines, both using the exact EEGNet paper
architecture for the SMR / BNCI2014_001 paradigm (Lawhern et al. 2018,
Table 2 + Sec 2.1.4/2.3), so the comparison against the tuned SNN isolates
architecture search as the factor, not activation type:

  1. EEGNetReLU  -- continuous (ReLU) activations, no timesteps.
                    -> results_baseline.json
  2. SpikingEEGNet (this study's actual SNN model, imported from
     spiking_eegnet.py) run with the SAME literal EEGNet architecture
     values, and default SNN dynamics (beta, spike_grad_slope) taken from
     SpikingEEGNet's own constructor defaults.
                    -> results_snn.json

Mirrors a single run of the Optuna study (pipeline.py -> experiment_loso_all)
as closely as possible so the comparison isolates architecture tuning.
Everything else -- LOSO protocol, dataset, preprocessing, optimizer, epochs,
batch size -- is held identical across all three runs (tuned SNN, literal
ReLU, literal SNN).

What's reused unchanged from the spiking pipeline:
    constants.py, cache_key.py, load_moabb_dataset.py, make_loader.py,
    spiking_eegnet.py (SpikingEEGNet architecture, unmodified)

What's different for the ReLU baseline (by necessity, since it has no
timesteps to aggregate):
    - No `num_steps` forward loop -- one standard forward pass per batch.
    - No `readout_mode` -- there's only one set of logits (from the classifier).
    - No BETA / SPIKE_GRAD_SLOPE / N_STEPS hyperparameters.

What's different for the literal-architecture SNN baseline vs. the tuned SNN
used elsewhere in the study:
    - Architecture (temporal_filters, depth_multiplier, pointwise_filters,
      temporal_kernel_size -> temporal_kernel_div, separable_kernel_size,
      pool1_size, pool2_size, dropout) is FIXED to the literal EEGNet paper
      config instead of pulled from the Optuna study.
    - BETA / SPIKE_GRAD_SLOPE are left at SpikingEEGNet's own constructor
      defaults (0.95 / 25.0) rather than tuned values, since those are SNN
      dynamics parameters the paper has no equivalent for.
    - Readout mode defaults to `spk_mean` (mean of per-timestep classifier
      logits across `num_steps`), matching this study's preferred readout
      mode. `mem_last` is intentionally not offered here since it has a
      known shape-mismatch bug in the wider codebase.

Both baselines pull architecture-irrelevant training hyperparameters
(lr, epochs, batch_size) from the same best_params source as ablation.py so
the optimizer settings stay comparable across all runs.

Usage (run from the repo root, e.g. inside baseline.slurm):
    python3 src/baseline.py
        # reads sqlite:///optuna_study.db automatically (same as ablation.py),
        # writes results/BNCI2014_001/results_baseline.json
        #    and results/BNCI2014_001/results_snn.json

    python3 src/baseline.py --params-json best_params.json
    python3 src/baseline.py --optuna-db sqlite:///optuna_study.db --study-name snn_eegnet_v4_200_20
    python3 src/baseline.py --num-steps 25 --readout-mode spk_mean
    python3 src/baseline.py --skip-ann      # only run the SNN baseline
    python3 src/baseline.py --skip-snn      # only run the ANN baseline
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score

from load_moabb_dataset import load_moabb_dataset
from make_loader import make_loader
from spiking_eegnet import SpikingEEGNet

# FIXED params held constant during the Optuna study (main.py FIXED dict) --
# same defaults as ablation.py's DEFAULT_FIXED.
DEFAULT_FIXED = dict(
    DATASET_KEY="BNCI2014_001",
    EPOCHS=50,
    BATCH_SIZE=32,
)


# ----------------------------------------------------------------------
# Model: EEGNet with ReLU activations (continuous, non-spiking)
# ----------------------------------------------------------------------
class EEGNetReLU(nn.Module):
    """
    Continuous-activation counterpart of SpikingEEGNet (see spiking_eegnet.py).

    Same block structure and same fixed kernel-shape constraints:
      - Temporal conv kernel:            (1, temporal_kernel_size)
      - Depthwise spatial conv kernel:   (num_channels, 1)
      - Separable depthwise conv kernel: (1, separable_kernel_size)
      - Pointwise conv kernel:           (1, 1)

    The only architectural difference vs. SpikingEEGNet is that each
    snn.Leaky() membrane-potential nonlinearity is replaced with a standard
    ReLU, and there is no timestep loop / spike aggregation -- a single
    forward pass produces the logits directly.
    """

    def __init__(
        self,
        num_classes: int,
        num_channels: int,
        num_samples: int,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        pointwise_filters: int = 16,
        temporal_kernel_size: int = 32,
        separable_kernel_size: int = 16,
        pool1_size: int = 4,
        pool2_size: int = 8,
        dropout: float = 0.25,
    ):
        """
        Defaults match the EEGNet paper's literal SMR (BNCI2014_001) config
        (Lawhern et al. 2018, Table 2 + Sec 2.1.4/2.3):
          - F1=8 temporal filters, D=2 depth multiplier -> F2=16 pointwise filters
          - temporal_kernel_size=32 (paper uses 32 samples for SMR specifically,
            since the data is high-passed at 4Hz -- NOT num_samples/2 like the
            other three paradigms)
          - separable_kernel_size=16 (500ms @ 32Hz after pool1)
          - pool1_size=4 (128Hz -> 32Hz), pool2_size=8
          - dropout=0.25 (paper's CROSS-SUBJECT value; LOSO is a cross-subject
            protocol, so this -- not the within-subject 0.5 -- is the right one)

        Unlike SpikingEEGNet's temporal_kernel_div (a ratio clamped to the
        [2, 8] HPO search space), this takes the kernel length directly in
        samples, since the literal paper baseline isn't tied to that search
        space and the paper itself specifies an absolute sample count.
        """
        super().__init__()
        self.num_classes = num_classes

        assert temporal_kernel_size >= 1, (
            f"temporal_kernel_size={temporal_kernel_size} is too small."
        )

        time_after_pool1 = num_samples // pool1_size
        assert separable_kernel_size <= time_after_pool1, (
            f"separable_kernel_size={separable_kernel_size} >= "
            f"time_after_pool1={time_after_pool1}. "
            f"Reduce separable_kernel_size or pool1_size."
        )

        time_after_pool2 = time_after_pool1 // pool2_size
        assert time_after_pool2 >= 1, (
            f"pool1_size={pool1_size} * pool2_size={pool2_size} = "
            f"{pool1_size * pool2_size} collapses time dimension to zero "
            f"for num_samples={num_samples}."
        )

        # Block 1 -- temporal
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, temporal_filters,
                      kernel_size=(1, temporal_kernel_size),
                      padding=(0, temporal_kernel_size // 2), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.act1 = nn.ReLU()

        # Block 1 -- depthwise spatial
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(temporal_filters, temporal_filters * depth_multiplier,
                      kernel_size=(num_channels, 1),
                      groups=temporal_filters, bias=False),
            nn.BatchNorm2d(temporal_filters * depth_multiplier),
        )
        self.act2 = nn.ReLU()

        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1_size))
        self.drop1 = nn.Dropout(dropout)

        # Block 2 -- separable
        self.separable_depthwise = nn.Conv2d(
            temporal_filters * depth_multiplier,
            temporal_filters * depth_multiplier,
            kernel_size=(1, separable_kernel_size),
            padding=(0, separable_kernel_size // 2),
            groups=temporal_filters * depth_multiplier, bias=False,
        )
        self.separable_pointwise = nn.Sequential(
            nn.Conv2d(temporal_filters * depth_multiplier, pointwise_filters,
                      kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(pointwise_filters),
        )
        self.act3 = nn.ReLU()

        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2_size))
        self.drop2 = nn.Dropout(dropout)

        flat_size = self._get_flat_size(num_channels, num_samples)
        self.classifier = nn.Linear(flat_size, num_classes, bias=True)

    def _get_flat_size(self, num_channels, num_samples):
        with torch.no_grad():
            x = torch.zeros(1, 1, num_channels, num_samples)
            x = self._features(x)
            return x.flatten(1).shape[1]

    def _features(self, x):
        x = self.act1(self.temporal_conv(x))
        x = self.act2(self.depthwise_conv(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.act3(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor):
        x = self._features(x)
        return self.classifier(x.flatten(1))


def build_model_baseline(meta, device, **model_kwargs):
    model = EEGNetReLU(
        num_classes=meta["n_classes"],
        num_channels=meta["n_channels"],
        num_samples=meta["n_samples"],
        **model_kwargs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    return model


# ----------------------------------------------------------------------
# Model: SpikingEEGNet (imported unmodified from spiking_eegnet.py),
# instantiated with the literal EEGNet paper architecture instead of a
# tuned Optuna config.
# ----------------------------------------------------------------------
def build_model_snn(meta, device, temporal_kernel_size: int, **model_kwargs):
    """
    SpikingEEGNet takes `temporal_kernel_div` (kernel = num_samples // div),
    not an absolute `temporal_kernel_size` -- see spiking_eegnet.py's
    docstring. The literal EEGNet paper config specifies an absolute sample
    count (32), so it's converted to the nearest equivalent div here, using
    this run's actual num_samples, then clamped to SpikingEEGNet's own
    supported search range [2, 8] (it clamps internally too, but we clamp
    here as well so the resulting kernel size is reported accurately).
    """
    num_samples = meta["n_samples"]
    temporal_kernel_div = round(num_samples / temporal_kernel_size)
    temporal_kernel_div = max(2, min(temporal_kernel_div, 8))
    effective_kernel_size = num_samples // temporal_kernel_div
    if effective_kernel_size != temporal_kernel_size:
        print(f"  [note] literal temporal_kernel_size={temporal_kernel_size} -> "
              f"temporal_kernel_div={temporal_kernel_div} -> "
              f"effective kernel size={effective_kernel_size} for "
              f"num_samples={num_samples} (SpikingEEGNet takes a divisor, "
              f"not an absolute kernel length)")

    model = SpikingEEGNet(
        num_classes=meta["n_classes"],
        num_channels=meta["n_channels"],
        num_samples=num_samples,
        temporal_kernel_div=temporal_kernel_div,
        **model_kwargs,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    return model


def snn_readout(spk_out: torch.Tensor, readout_mode: str) -> torch.Tensor:
    """
    Reduce SpikingEEGNet's per-timestep classifier output
    spk_out: (num_steps, batch, num_classes) into a single (batch, num_classes)
    logits tensor for loss/accuracy.

    Only spk-based readout modes are offered here. `mem_last` is deliberately
    left out -- it has a known shape-mismatch bug elsewhere in the codebase,
    and this baseline should stay on the readout mode (`spk_mean`) that this
    study's HPO analysis found preferable anyway.
    """
    if readout_mode == "spk_mean":
        return spk_out.mean(dim=0)
    elif readout_mode == "spk_last":
        return spk_out[-1]
    else:
        raise ValueError(
            f"Unsupported readout_mode={readout_mode!r} for this baseline. "
            f"Supported: 'spk_mean', 'spk_last'."
        )


# ----------------------------------------------------------------------
# Training / evaluation (single forward pass, no timesteps)
# ----------------------------------------------------------------------
def train_one_epoch_baseline(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_baseline(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return balanced_accuracy_score(labels, preds)


def run_training_baseline(model, train_loader, val_loader, epochs, lr, device, patience=None):
    """Mirrors run_training.py's eval_every_epoch=True path (bal_acc-monitored
    early stopping), just without num_steps / readout_mode."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "bal_acc": []}

    best_metric = None
    epochs_no_imp = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_baseline(model, train_loader, optimizer, criterion, device)
        history["loss"].append(loss)

        bal_acc = evaluate_baseline(model, val_loader, device)
        history["bal_acc"].append(bal_acc)
        print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}  bal_acc={bal_acc:.4f}", end="")

        if patience is not None:
            improved = best_metric is None or bal_acc > best_metric
            if improved:
                best_metric = bal_acc
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"  [early stop: no bal_acc improvement for {patience} epochs]")
                break
        print()

    return history


# ----------------------------------------------------------------------
# Training / evaluation for SpikingEEGNet (num_steps forward loop + readout)
# ----------------------------------------------------------------------
def train_one_epoch_snn(model, loader, optimizer, criterion, device, num_steps, readout_mode):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        spk_out, _mem_out = model(xb, num_steps=num_steps)
        logits = snn_readout(spk_out, readout_mode)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_snn(model, loader, device, num_steps, readout_mode):
    model.eval()
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        spk_out, _mem_out = model(xb, num_steps=num_steps)
        logits = snn_readout(spk_out, readout_mode)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(yb.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return balanced_accuracy_score(labels, preds)


def run_training_snn(model, train_loader, val_loader, epochs, lr, device,
                      num_steps, readout_mode, patience=None):
    """Mirrors run_training_baseline but with the num_steps forward loop and
    spike readout SpikingEEGNet requires."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {"loss": [], "bal_acc": []}

    best_metric = None
    epochs_no_imp = 0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch_snn(model, train_loader, optimizer, criterion, device,
                                    num_steps, readout_mode)
        history["loss"].append(loss)

        bal_acc = evaluate_snn(model, val_loader, device, num_steps, readout_mode)
        history["bal_acc"].append(bal_acc)
        print(f"  epoch {epoch:3d}/{epochs}  loss={loss:.4f}  bal_acc={bal_acc:.4f}", end="")

        if patience is not None:
            improved = best_metric is None or bal_acc > best_metric
            if improved:
                best_metric = bal_acc
                epochs_no_imp = 0
            else:
                epochs_no_imp += 1
            if epochs_no_imp >= patience:
                print(f"  [early stop: no bal_acc improvement for {patience} epochs]")
                break
        print()

    return history


# ----------------------------------------------------------------------
# True LOSO, mirroring experiment_loso.py
# ----------------------------------------------------------------------
def experiment_loso_baseline(X, y, subject_ids, meta, device, cfg,
                             test_subject_idx: int, model_kwargs: dict = None):
    print(f"\n=== LOSO baseline (hold out subject {test_subject_idx}) ===")
    test_mask = subject_ids == test_subject_idx
    train_mask = ~test_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"  Train: {X_tr.shape[0]} trials  |  Test: {X_te.shape[0]} trials")

    train_loader = make_loader(X_tr, y_tr, cfg["batch_size"])
    val_loader = make_loader(X_te, y_te, cfg["batch_size"], shuffle=False)

    model = build_model_baseline(meta, device, **(model_kwargs or {}))
    history = run_training_baseline(
        model, train_loader, val_loader,
        epochs=cfg["epochs"], lr=cfg["lr"], device=device,
        patience=cfg.get("patience"),
    )
    final_acc = history["bal_acc"][-1]
    return history, final_acc, model


def experiment_loso_all_baseline(X, y, subject_ids, meta, device, cfg, model_kwargs: dict = None):
    """True LOSO across every subject -- mirrors experiment_loso_all()."""
    subjects = sorted(set(int(s) for s in subject_ids))
    n_subjects = len(subjects)

    histories = {}
    accs = []

    for i, subj in enumerate(subjects):
        history, final_acc, _model = experiment_loso_baseline(
            X, y, subject_ids, meta, device, cfg,
            test_subject_idx=subj, model_kwargs=model_kwargs,
        )
        histories[subj] = history
        accs.append(final_acc)

        running_mean = sum(accs) / len(accs)
        print(f"  [LOSO {i + 1}/{n_subjects}] subject {subj}: "
              f"acc={final_acc:.4f}  running_mean={running_mean:.4f}")

    mean_acc = sum(accs) / len(accs)
    return histories, accs, mean_acc


# ----------------------------------------------------------------------
# True LOSO for SpikingEEGNet, mirroring experiment_loso_baseline above
# ----------------------------------------------------------------------
def experiment_loso_snn(X, y, subject_ids, meta, device, cfg,
                         test_subject_idx: int, model_kwargs: dict = None):
    print(f"\n=== LOSO SNN baseline (hold out subject {test_subject_idx}) ===")
    test_mask = subject_ids == test_subject_idx
    train_mask = ~test_mask

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"  Train: {X_tr.shape[0]} trials  |  Test: {X_te.shape[0]} trials")

    train_loader = make_loader(X_tr, y_tr, cfg["batch_size"])
    val_loader = make_loader(X_te, y_te, cfg["batch_size"], shuffle=False)

    model_kwargs = dict(model_kwargs or {})
    num_steps = model_kwargs.pop("num_steps")
    readout_mode = model_kwargs.pop("readout_mode")

    model = build_model_snn(meta, device, **model_kwargs)
    history = run_training_snn(
        model, train_loader, val_loader,
        epochs=cfg["epochs"], lr=cfg["lr"], device=device,
        num_steps=num_steps, readout_mode=readout_mode,
        patience=cfg.get("patience"),
    )
    final_acc = history["bal_acc"][-1]
    return history, final_acc, model


def experiment_loso_all_snn(X, y, subject_ids, meta, device, cfg, model_kwargs: dict = None):
    """True LOSO across every subject -- mirrors experiment_loso_all_baseline()."""
    subjects = sorted(set(int(s) for s in subject_ids))
    n_subjects = len(subjects)

    histories = {}
    accs = []

    for i, subj in enumerate(subjects):
        history, final_acc, _model = experiment_loso_snn(
            X, y, subject_ids, meta, device, cfg,
            test_subject_idx=subj, model_kwargs=model_kwargs,
        )
        histories[subj] = history
        accs.append(final_acc)

        running_mean = sum(accs) / len(accs)
        print(f"  [LOSO {i + 1}/{n_subjects}] subject {subj}: "
              f"acc={final_acc:.4f}  running_mean={running_mean:.4f}")

    mean_acc = sum(accs) / len(accs)
    return histories, accs, mean_acc


# ----------------------------------------------------------------------
# Best-params loading (same source/logic as ablation.py)
# ----------------------------------------------------------------------
def resolve_study_name(storage, study_name):
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


# Literal EEGNet architecture for the SMR / BNCI2014_001 paradigm, taken
# directly from the paper (Lawhern et al. 2018, Table 2 + Sec 2.1.4/2.3) --
# NOT pulled from the Optuna study, since that study tuned architecture for
# the spiking model's search space, not this baseline. See EEGNetReLU's
# docstring for the per-value justification.
LITERAL_EEGNET_SMR_CFG = {
    "temporal_filters": 8,          # F1
    "depth_multiplier": 2,          # D
    "pointwise_filters": 16,        # F2 = D * F1
    "temporal_kernel_size": 32,     # paper's SMR-specific value (4Hz high-pass)
    "separable_kernel_size": 16,    # 500ms @ 32Hz
    "pool1_size": 4,                # 128Hz -> 32Hz
    "pool2_size": 8,
    "dropout": 0.25,                # cross-subject value (LOSO = cross-subject)
}


# Training hyperparameters this baseline uses. LR/epochs/batch_size still
# come from the best_params source (Optuna study or --params-json) so the
# optimizer settings stay comparable to the spiking run -- only the
# ARCHITECTURE is fixed to the literal paper config above. Everything
# spiking-specific (BETA, SPIKE_GRAD_SLOPE, N_STEPS_TRAIN, N_STEPS_EVAL,
# READOUT_MODE) is dropped since it has no ReLU counterpart.
def make_cfgs_baseline(params):
    LR = 10 ** params["LR_EXP"]
    train_cfg = {
        "epochs": params["EPOCHS"], "batch_size": params["BATCH_SIZE"], "lr": LR,
        "patience": params.get("EARLY_STOPPING_PATIENCE"),
    }
    model_cfg = dict(LITERAL_EEGNET_SMR_CFG)
    return train_cfg, model_cfg


# Same literal architecture as LITERAL_EEGNET_SMR_CFG, but for SpikingEEGNet:
# BETA / SPIKE_GRAD_SLOPE are left at SpikingEEGNet's own constructor
# defaults (not tuned) since the paper has no equivalent for SNN dynamics.
# `num_steps` / `readout_mode` are training-loop settings, not passed to the
# model constructor -- they're popped off in experiment_loso_snn.
LITERAL_EEGNET_SNN_CFG = {
    "temporal_filters": 8,          # F1
    "depth_multiplier": 2,          # D
    "pointwise_filters": 16,        # F2 = D * F1
    "temporal_kernel_size": 32,     # converted to temporal_kernel_div at build time
    "separable_kernel_size": 16,    # 500ms @ 32Hz
    "pool1_size": 4,                # 128Hz -> 32Hz
    "pool2_size": 8,
    "dropout": 0.25,                # cross-subject value (LOSO = cross-subject)
    "num_steps": 25,                # default timestep count; override with --num-steps
    "readout_mode": "spk_mean",     # this study's preferred readout mode
}


def make_cfgs_snn(params, num_steps: int = None, readout_mode: str = None):
    LR = 10 ** params["LR_EXP"]
    train_cfg = {
        "epochs": params["EPOCHS"], "batch_size": params["BATCH_SIZE"], "lr": LR,
        "patience": params.get("EARLY_STOPPING_PATIENCE"),
    }
    model_cfg = dict(LITERAL_EEGNET_SNN_CFG)
    if num_steps is not None:
        model_cfg["num_steps"] = num_steps
    if readout_mode is not None:
        model_cfg["readout_mode"] = readout_mode
    return train_cfg, model_cfg


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--params-json", type=str, default=None,
                  help="JSON file of best hyperparameters. If omitted, best params "
                       "are read live from --optuna-db instead (same as ablation.py).")
    p.add_argument("--optuna-db", type=str, default="sqlite:///optuna_study.db",
                  help="Optuna storage URL (default: sqlite:///optuna_study.db)")
    p.add_argument("--study-name", type=str, default=None,
                  help="Optuna study name. Auto-detected if the DB has exactly one study.")
    p.add_argument("--dataset-key", type=str, default=None, help="Override DATASET_KEY")
    p.add_argument("--epochs", type=int, default=None, help="Override EPOCHS")
    p.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE")
    p.add_argument("--out-dir", type=str, default=None,
                  help="Output directory (default: results/<DATASET_KEY>)")
    p.add_argument("--num-steps", type=int, default=None,
                  help="SNN baseline timestep count (default: 25, see LITERAL_EEGNET_SNN_CFG)")
    p.add_argument("--readout-mode", type=str, default=None, choices=["spk_mean", "spk_last"],
                  help="SNN baseline readout mode (default: spk_mean)")
    p.add_argument("--skip-ann", action="store_true",
                  help="Skip the EEGNetReLU baseline, only run the SNN baseline")
    p.add_argument("--skip-snn", action="store_true",
                  help="Skip the SpikingEEGNet baseline, only run the ANN baseline")
    return p.parse_args()


def main():
    args = parse_args()

    params = load_best_params(args)
    print("Resolved training config (lr/epochs/batch_size shared across baselines):")
    for k, v in params.items():
        print(f"  {k:25s} = {v}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / params["DATASET_KEY"]
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    X, y, subject_ids, meta = load_moabb_dataset(params["DATASET_KEY"])
    print(f"Dataset meta: {meta}")

    source_params = {
        "params_json": args.params_json,
        "optuna_db": args.optuna_db if not args.params_json else None,
        "study_name": args.study_name if not args.params_json else None,
    }

    # --- ANN baseline (EEGNetReLU) -> results_baseline.json ---
    if not args.skip_ann:
        print("\n" + "=" * 60)
        print("Running literal EEGNet baseline: ReLU (non-spiking)")
        print("=" * 60)

        train_cfg, model_cfg = make_cfgs_baseline(params)
        histories, accs, mean_acc = experiment_loso_all_baseline(
            X, y, subject_ids, meta, device, train_cfg, model_kwargs=model_cfg,
        )

        results = {
            "dataset": params["DATASET_KEY"],
            "model_type": "EEGNetReLU (non-spiking baseline)",
            "model_cfg": model_cfg,
            "meta": {k: v for k, v in meta.items() if k != "subject_list"},
            "train_cfg": train_cfg,
            "source_params": source_params,
            "loso_all_subjects": {
                "per_subject_acc": dict(zip(sorted(histories.keys()), accs)),
                "mean_bal_acc": mean_acc,
                "histories": {str(k): v for k, v in histories.items()},
            },
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        results_path = out_dir / "results_baseline.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        print(f"Baseline (EEGNet-ReLU) mean balanced accuracy: {mean_acc:.4f}")

    # --- SNN baseline (SpikingEEGNet, literal architecture) -> results_snn.json ---
    if not args.skip_snn:
        print("\n" + "=" * 60)
        print("Running literal EEGNet baseline: SpikingEEGNet (spiking)")
        print("=" * 60)

        train_cfg, model_cfg = make_cfgs_snn(
            params, num_steps=args.num_steps, readout_mode=args.readout_mode,
        )
        print("SNN model_cfg:")
        for k, v in model_cfg.items():
            print(f"  {k:25s} = {v}")

        histories, accs, mean_acc = experiment_loso_all_snn(
            X, y, subject_ids, meta, device, train_cfg, model_kwargs=model_cfg,
        )

        results = {
            "dataset": params["DATASET_KEY"],
            "model_type": "SpikingEEGNet (spiking baseline, literal EEGNet architecture)",
            "model_cfg": model_cfg,
            "meta": {k: v for k, v in meta.items() if k != "subject_list"},
            "train_cfg": train_cfg,
            "source_params": source_params,
            "loso_all_subjects": {
                "per_subject_acc": dict(zip(sorted(histories.keys()), accs)),
                "mean_bal_acc": mean_acc,
                "histories": {str(k): v for k, v in histories.items()},
            },
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        }

        results_path = out_dir / "results_snn.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        print(f"Baseline (SpikingEEGNet) mean balanced accuracy: {mean_acc:.4f}")


if __name__ == "__main__":
    main()