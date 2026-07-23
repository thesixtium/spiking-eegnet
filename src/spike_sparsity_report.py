#!/usr/bin/env python3
"""
spike_sparsity_report.py

Standalone, single-file runtime-sparsity characterization for
SpikingEEGNet. Does NOT modify baseline.py or either
*_train_and_export.py script -- it only imports from baseline.py
(experiment_loso_snn, LITERAL_EEGNET_SNN_CFG, load_best_params).

How the SpikeSparsityTracker works:
SpikingEEGNet's LIF neurons (snn.Leaky() or similar, from snntorch) each
return a (spk, mem) pair every time they're called -- spk is a {0,1}
tensor marking which units fired that step. SpikingEEGNet.forward() calls
the SAME neuron module instance once per timestep inside its num_steps
loop (that's also what lets torch.onnx.export unroll it into a static
graph for export). That means a PyTorch forward hook registered on a
neuron module fires once per timestep, in order -- so hook call #k on a
given layer *is* timestep k. The tracker exploits this directly:

  1. On construction, it walks model.named_modules() and finds every
     submodule whose type matches a known snntorch neuron class (Leaky,
     Synaptic, Alpha, Lapicque, RLeaky, RSynaptic) -- this works against
     any SpikingEEGNet variant without needing to touch spiking_eegnet.py,
     since it never assumes a particular layer name or count.
  2. Entering the tracker as a context manager (`with tracker:`) registers
     a forward hook on each of those neuron modules. Each hook call reads
     off that step's spk tensor, sums it for a spike count and element
     count, and buckets both into a (layer, timestep) slot via a running
     call counter modulo num_steps.
  3. It also maintains a running "has this neuron ever fired" boolean mask
     per layer (OR'd across every call), so a dead-neuron fraction can be
     reported once accumulation is done.
  4. summary() turns the accumulated counts into per-layer and overall
     firing rates, a per-timestep firing-rate trend, the single busiest
     ("worst-case") timestep's rate, and dead-neuron fraction.
  5. Exiting the context manager removes the hooks, so the tracked model
     goes back to behaving exactly as it did before.

For each LOSO fold (or a chosen subset of subjects), this script:
  1. Trains SpikingEEGNet on the other subjects via experiment_loso_snn,
     completely unmodified -- same architecture/training pipeline your
     other scripts already use and that's already been validated.
  2. Runs a sparsity-characterization pass with SpikeSparsityTracker over
     that fold's held-out test subject (never trained on, so the numbers
     reflect generalization behavior rather than train-time activity).
  3. Aggregates per-layer / per-timestep firing-rate statistics across all
     folds run (mean +/- std across folds), and generates summary plots.

Why it retrains per fold rather than loading a checkpoint: none of the
three existing scripts persist a trained state_dict anywhere (they only
export to ONNX), so there's currently no saved checkpoint to characterize
sparsity on. If you add checkpoint saving later, swap the
experiment_loso_snn() call below for a plain torch.load() + model.eval().

All configuration is hardcoded in the CONFIG block at the top of main() --
edit the variables there directly rather than passing CLI flags.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from load_moabb_dataset import load_moabb_dataset
from make_loader import make_loader

from baseline import (
    LITERAL_EEGNET_SNN_CFG,
    load_best_params,
    make_cfgs_snn,
    experiment_loso_snn,
)

try:
    import snntorch as snn
    _SPIKING_NEURON_TYPES = tuple(
        cls for cls in vars(snn).values()
        if isinstance(cls, type) and issubclass(cls, torch.nn.Module)
        and cls.__name__ in {"Leaky", "Synaptic", "Alpha", "Lapicque",
                              "RLeaky", "RSynaptic"}
    )
except ImportError:
    _SPIKING_NEURON_TYPES = ()


# --------------------------------------------------------------------------- #
# Spike sparsity tracker
#
# Works unmodified against ANY SpikingEEGNet variant (Optuna-tuned or
# literal paper-config baseline) since it doesn't touch spiking_eegnet.py
# at all -- it just finds every snntorch spiking-neuron submodule via
# named_modules() and registers a forward hook on each one.
#
# Why this works without knowing the model's internal loop structure:
# SpikingEEGNet.forward() calls the SAME neuron module instance once per
# timestep inside its num_steps loop (that's what lets torch.onnx.export
# unroll it into a static graph). So hook call #k on a given module is
# exactly timestep k, and passing num_steps into the tracker lets it bucket
# spike counts by (layer, timestep) via a simple modulo counter, without
# needing any batch-boundary signal.
# --------------------------------------------------------------------------- #
class SpikeSparsityTracker:
    """Context manager: register hooks on __enter__, remove on __exit__.

    Accumulates over however many forward() calls happen while the context
    is active, so you can run it over an entire val/test loader and get one
    aggregate report -- call reset() between passes if you want separate
    reports instead of one pooled report.
    """

    def __init__(self, model: torch.nn.Module, num_steps: int,
                 neuron_types: tuple = _SPIKING_NEURON_TYPES):
        if not neuron_types:
            raise RuntimeError(
                "No snntorch neuron types resolved -- either snntorch isn't "
                "importable, or none of its exported names matched "
                "{'Leaky','Synaptic','Alpha','Lapicque','RLeaky','RSynaptic'}. "
                "Pass neuron_types=(YourNeuronClass,) explicitly if "
                "SpikingEEGNet uses a custom/renamed neuron."
            )
        self.model = model
        self.num_steps = num_steps
        self.layers = {
            name: module for name, module in model.named_modules()
            if isinstance(module, neuron_types)
        }
        if not self.layers:
            raise RuntimeError(
                "No spiking-neuron submodules found via named_modules() -- "
                "confirm `model` is the raw SpikingEEGNet (named_modules() "
                "recurses through wrappers too, so an InferenceWrapper "
                "works fine as long as its child names show up as expected)."
            )
        self._handles = []
        self.reset()

    def reset(self):
        self.spike_count = defaultdict(int)
        self.elem_count = defaultdict(int)
        self.per_step_spikes = defaultdict(lambda: [0] * self.num_steps)
        self.per_step_elems = defaultdict(lambda: [0] * self.num_steps)
        self._step_idx = defaultdict(int)
        self._ever_fired = {}

    def _make_hook(self, name):
        def hook(module, inputs, output):
            spk = output[0] if isinstance(output, tuple) else output
            spk = spk.detach()

            n_spikes = spk.sum().item()
            n_elems = spk.numel()
            self.spike_count[name] += n_spikes
            self.elem_count[name] += n_elems

            t = self._step_idx[name] % self.num_steps
            self.per_step_spikes[name][t] += n_spikes
            self.per_step_elems[name][t] += n_elems
            self._step_idx[name] += 1

            per_neuron_any = (spk > 0).flatten(start_dim=0, end_dim=0).any(dim=0) \
                if spk.dim() > 1 else (spk > 0)
            if name not in self._ever_fired:
                self._ever_fired[name] = per_neuron_any.clone()
            else:
                self._ever_fired[name] |= per_neuron_any
        return hook

    def __enter__(self):
        for name, module in self.layers.items():
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summary(self) -> dict:
        per_layer = {}
        for name in self.layers:
            elems = self.elem_count[name]
            step_rates = [
                (s / e if e else 0.0)
                for s, e in zip(self.per_step_spikes[name], self.per_step_elems[name])
            ]
            ever_fired = self._ever_fired.get(name)
            dead_frac = (
                float((~ever_fired).float().mean().item())
                if ever_fired is not None else None
            )
            per_layer[name] = {
                "firing_rate": (self.spike_count[name] / elems) if elems else 0.0,
                "total_spikes": int(self.spike_count[name]),
                "total_elements": int(elems),
                "per_timestep_rate": step_rates,
                "worst_timestep_rate": max(step_rates) if step_rates else 0.0,
                "dead_neuron_frac": dead_frac,
            }

        total_spikes = sum(self.spike_count.values())
        total_elems = sum(self.elem_count.values())
        overall_step_spikes = [0] * self.num_steps
        overall_step_elems = [0] * self.num_steps
        for name in self.layers:
            for t in range(self.num_steps):
                overall_step_spikes[t] += self.per_step_spikes[name][t]
                overall_step_elems[t] += self.per_step_elems[name][t]
        overall_step_rates = [
            (s / e if e else 0.0)
            for s, e in zip(overall_step_spikes, overall_step_elems)
        ]

        return {
            "per_layer": per_layer,
            "overall_firing_rate": (total_spikes / total_elems) if total_elems else 0.0,
            "overall_sparsity": 1 - (total_spikes / total_elems) if total_elems else 1.0,
            "overall_per_timestep_rate": overall_step_rates,
            "overall_worst_timestep_rate": max(overall_step_rates) if overall_step_rates else 0.0,
        }


# --------------------------------------------------------------------------- #
# Same fix as in *_train_and_export.py's reset_snn_state -- detach stale
# membrane/synaptic state left over from training before a clean
# characterization pass. Duplicated here (not imported) so this script has
# zero coupling to the export scripts.
# --------------------------------------------------------------------------- #
def reset_snn_state(model: torch.nn.Module):
    for module in model.modules():
        for state_attr in ("mem", "syn", "spk"):
            if hasattr(module, state_attr):
                val = getattr(module, state_attr)
                if isinstance(val, torch.Tensor):
                    setattr(module, state_attr, val.detach().clone())


def build_model_kwargs(model_kwargs_json: str, num_steps: int, readout_mode: str) -> dict:
    if model_kwargs_json:
        with open(model_kwargs_json) as f:
            cfg = json.load(f)
        print(f"Loaded architecture/model kwargs from {model_kwargs_json}")
    else:
        cfg = dict(LITERAL_EEGNET_SNN_CFG)
        print("Using LITERAL_EEGNET_SNN_CFG (literal EEGNet-paper architecture)")

    if num_steps is not None:
        cfg["num_steps"] = num_steps
    if readout_mode is not None:
        cfg["readout_mode"] = readout_mode
    return cfg


def characterize_fold(model, val_loader, num_steps, device) -> dict:
    model.eval()
    model.to(device)
    reset_snn_state(model)
    tracker = SpikeSparsityTracker(model, num_steps=num_steps)
    with tracker, torch.no_grad():
        for xb, _yb in val_loader:
            xb = xb.to(device)
            model(xb, num_steps=num_steps)
    return tracker.summary()


def aggregate_fold_reports(fold_reports: dict) -> dict:
    """fold_reports: {subject_id: summary_dict} -> mean/std across folds,
    per layer and overall. Assumes all folds share the same layer names and
    num_steps, which holds as long as architecture/config is fixed across
    folds (the normal LOSO setup -- only the held-out subject changes)."""
    subjects = sorted(fold_reports)
    layer_names = sorted(fold_reports[subjects[0]]["per_layer"])

    per_layer_agg = {}
    for name in layer_names:
        rates = np.array([fold_reports[s]["per_layer"][name]["firing_rate"] for s in subjects])
        worst = np.array([fold_reports[s]["per_layer"][name]["worst_timestep_rate"] for s in subjects])
        dead_vals = [
            fold_reports[s]["per_layer"][name]["dead_neuron_frac"]
            for s in subjects
            if fold_reports[s]["per_layer"][name]["dead_neuron_frac"] is not None
        ]
        dead = np.array(dead_vals)
        step_rates = np.array([fold_reports[s]["per_layer"][name]["per_timestep_rate"] for s in subjects])

        per_layer_agg[name] = {
            "firing_rate_mean": float(rates.mean()),
            "firing_rate_std": float(rates.std()),
            "worst_timestep_rate_mean": float(worst.mean()),
            "worst_timestep_rate_std": float(worst.std()),
            "dead_neuron_frac_mean": float(dead.mean()) if dead.size else None,
            "dead_neuron_frac_std": float(dead.std()) if dead.size else None,
            "per_timestep_rate_mean": step_rates.mean(axis=0).tolist(),
            "per_timestep_rate_std": step_rates.std(axis=0).tolist(),
        }

    overall_rates = np.array([fold_reports[s]["overall_firing_rate"] for s in subjects])
    overall_worst = np.array([fold_reports[s]["overall_worst_timestep_rate"] for s in subjects])
    overall_step_rates = np.array([fold_reports[s]["overall_per_timestep_rate"] for s in subjects])

    return {
        "subjects": subjects,
        "per_layer": per_layer_agg,
        "overall_firing_rate_mean": float(overall_rates.mean()),
        "overall_firing_rate_std": float(overall_rates.std()),
        "overall_worst_timestep_rate_mean": float(overall_worst.mean()),
        "overall_worst_timestep_rate_std": float(overall_worst.std()),
        "overall_per_timestep_rate_mean": overall_step_rates.mean(axis=0).tolist(),
        "overall_per_timestep_rate_std": overall_step_rates.std(axis=0).tolist(),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_per_layer_firing_rates(agg: dict, out_path: Path):
    names = list(agg["per_layer"])
    means = [agg["per_layer"][n]["firing_rate_mean"] for n in names]
    stds = [agg["per_layer"][n]["firing_rate_std"] for n in names]
    worst = [agg["per_layer"][n]["worst_timestep_rate_mean"] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, means, width=0.4, yerr=stds, capsize=3, label="mean firing rate")
    ax.bar(x + 0.2, worst, width=0.4, label="worst-timestep firing rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Firing rate (fraction of neurons spiking)")
    ax.set_title("Per-layer spike sparsity (mean across LOSO folds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_timestep_trend(agg: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, layer in agg["per_layer"].items():
        mean = np.array(layer["per_timestep_rate_mean"])
        std = np.array(layer["per_timestep_rate_std"])
        t = np.arange(len(mean))
        line, = ax.plot(t, mean, label=name)
        ax.fill_between(t, mean - std, mean + std, alpha=0.15, color=line.get_color())

    overall_mean = np.array(agg["overall_per_timestep_rate_mean"])
    ax.plot(np.arange(len(overall_mean)), overall_mean, color="black",
            linewidth=2.5, linestyle="--", label="overall (pooled)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Firing rate")
    ax.set_title("Firing rate over timesteps (mean +/- std across LOSO folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dead_neuron_fraction(agg: dict, out_path: Path):
    names = [n for n in agg["per_layer"] if agg["per_layer"][n]["dead_neuron_frac_mean"] is not None]
    if not names:
        print("No dead-neuron data available -- skipping dead-neuron plot.")
        return
    means = [agg["per_layer"][n]["dead_neuron_frac_mean"] for n in names]
    stds = [agg["per_layer"][n]["dead_neuron_frac_std"] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.3), 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=3, color="firebrick")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Fraction of neurons that never fired")
    ax.set_title("Dead-neuron fraction per layer (mean +/- std across LOSO folds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_fold_overall_rate(fold_reports: dict, out_path: Path):
    subjects = sorted(fold_reports)
    rates = [fold_reports[s]["overall_firing_rate"] for s in subjects]
    mean_rate = float(np.mean(rates))

    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.9), 5))
    ax.bar([str(s) for s in subjects], rates, color="steelblue")
    ax.axhline(mean_rate, color="black", linestyle="--", label=f"mean = {mean_rate:.4f}")
    ax.set_xlabel("Held-out subject (LOSO fold)")
    ax.set_ylabel("Overall firing rate")
    ax.set_title("Overall firing rate per LOSO fold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # CONFIG -- edit these directly instead of passing CLI flags.
    # ------------------------------------------------------------------- #
    PARAMS_JSON = None          # path to a JSON file of lr/epochs/batch_size,
                                # or None to read live from OPTUNA_DB instead
                                # (same source baseline.py itself uses)
    OPTUNA_DB = "sqlite:///optuna_study.db"
    STUDY_NAME = None
    DATASET_KEY = None          # None -> whatever load_best_params defaults to

    EPOCHS = None               # override EPOCHS -- lower this (e.g. 5) for
                                # faster characterization runs; None -> use
                                # whatever PARAMS_JSON/OPTUNA_DB already says
    BATCH_SIZE = None           # override batch size; None -> use params value

    MODEL_KWARGS_JSON = None    # path to a JSON dict of architecture/SNN-
                                # dynamics kwargs (see module docstring for
                                # the required key schema), or None to use
                                # LITERAL_EEGNET_SNN_CFG (the literal
                                # EEGNet-paper architecture). To characterize
                                # the Optuna-tuned model instead, point this
                                # at a JSON file using that same key schema.
    NUM_STEPS = None            # override num_steps; None -> use whatever
                                # MODEL_KWARGS_JSON/LITERAL_EEGNET_SNN_CFG says
    READOUT_MODE = None         # "spk_mean" | "spk_last" | None (no override)

    SUBJECTS = None             # list of subject IDs to run as LOSO folds,
                                # e.g. [1, 2, 3] for a quick subset, or None
                                # to run every subject in the dataset
    OUT_DIR = "spike_sparsity_report"   # where JSON reports + PNG plots go
    # ------------------------------------------------------------------- #

    # load_best_params (imported from baseline.py) expects an
    # argparse.Namespace with these specific attributes -- build one that
    # matches, so we can reuse it unmodified.
    import argparse
    lb_args = argparse.Namespace(
        params_json=PARAMS_JSON,
        optuna_db=OPTUNA_DB,
        study_name=STUDY_NAME,
        dataset_key=DATASET_KEY,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    params = load_best_params(lb_args)
    print("Training config (lr/epochs/batch_size):")
    for k, v in params.items():
        print(f"  {k:25s} = {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y, subject_ids, meta = load_moabb_dataset(params["DATASET_KEY"])
    print(f"Dataset meta: {meta}")

    model_kwargs_template = build_model_kwargs(MODEL_KWARGS_JSON, NUM_STEPS, READOUT_MODE)
    num_steps = model_kwargs_template["num_steps"]

    # Only need make_cfgs_snn for its LR/epoch/batch_size extraction --
    # its model_cfg output is ignored here since architecture comes from
    # model_kwargs_template above instead.
    train_cfg, _ignored_model_cfg = make_cfgs_snn(params)

    all_subjects = sorted(set(int(s) for s in subject_ids))
    subjects = SUBJECTS or all_subjects
    print(f"Running LOSO sparsity characterization for subjects: {subjects}")

    fold_reports = {}
    for i, subj in enumerate(subjects):
        print(f"\n[{i + 1}/{len(subjects)}] LOSO fold: held-out subject {subj}")
        model_kwargs = dict(model_kwargs_template)

        _history, _acc, model = experiment_loso_snn(
            X, y, subject_ids, meta, device, train_cfg,
            test_subject_idx=subj, model_kwargs=model_kwargs,
        )

        test_mask = subject_ids == subj
        val_loader = make_loader(X[test_mask], y[test_mask], train_cfg["batch_size"], shuffle=False)

        summary = characterize_fold(model, val_loader, num_steps, device)
        fold_reports[subj] = summary
        print(f"  overall firing rate: {summary['overall_firing_rate']:.4f}  "
              f"worst-timestep: {summary['overall_worst_timestep_rate']:.4f}")

    agg = aggregate_fold_reports(fold_reports)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "sparsity_per_fold.json", "w") as f:
        json.dump(fold_reports, f, indent=2)
    with open(out_dir / "sparsity_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    plot_per_layer_firing_rates(agg, out_dir / "per_layer_firing_rates.png")
    plot_per_timestep_trend(agg, out_dir / "per_timestep_trend.png")
    plot_dead_neuron_fraction(agg, out_dir / "dead_neuron_fraction.png")
    plot_per_fold_overall_rate(fold_reports, out_dir / "per_fold_overall_rate.png")

    print("\n" + "=" * 60)
    print(f"Mean overall firing rate across {len(subjects)} fold(s): "
          f"{agg['overall_firing_rate_mean']:.4f} +/- {agg['overall_firing_rate_std']:.4f}")
    print(f"Mean worst-timestep firing rate: "
          f"{agg['overall_worst_timestep_rate_mean']:.4f} +/- "
          f"{agg['overall_worst_timestep_rate_std']:.4f}")
    print(f"Wrote JSON reports and plots to {out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()