#!/usr/bin/env python3
"""
source ~/venvs/spiking-eegnet/bin/activate
cd /mnt/c/Users/ajrbe/Documents/Git/spiking-eegnet/src
python3 train_and_export.py
"""

import inspect
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from spiking_eegnet import SpikingEEGNet
from train_one_epoch import aggregate_logits, train_one_epoch


# --------------------------------------------------------------------------- #
# 1. Load best hyperparameters
# --------------------------------------------------------------------------- #
def load_best_params(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded trial #{data['trial_number']} "
          f"(value={data['value']:.4f}) from {path}")
    return data["params"]


def split_params(params: dict, extra_defaults: dict):
    """Split the flat Optuna params dict into (model_kwargs, run_cfg) by
    matching against SpikingEEGNet's actual constructor signature -- so this
    doesn't silently break if you add/rename search params later."""
    model_sig = inspect.signature(SpikingEEGNet.__init__)
    model_arg_names = set(model_sig.parameters) - {
        "self", "num_classes", "num_channels", "num_samples"
    }
    model_kwargs = {k: v for k, v in params.items() if k in model_arg_names}
    run_cfg = {k: v for k, v in params.items() if k not in model_arg_names}
    for k, v in extra_defaults.items():
        run_cfg.setdefault(k, v)
    return model_kwargs, run_cfg


# --------------------------------------------------------------------------- #
# 2. Data -- swap this out for your existing MOABB loading code if preferred
# --------------------------------------------------------------------------- #
def load_data(dataset_name: str = "BNCI2014_001"):
    """Loads BNCI2014_001 via MOABB using a fresh tempfile.mkdtemp() per
    process, so we never resolve stale SLURM scratch paths."""
    import numpy as np
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    scratch = tempfile.mkdtemp(prefix="moabb_")
    mne.set_config("MNE_DATA", scratch, set_env=False)

    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)
    X, y, meta = paradigm.get_data(dataset=dataset)

    classes = sorted(set(y))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[c] for c in y], dtype="int64")

    num_channels = X.shape[1]
    num_samples = X.shape[2]
    num_classes = len(classes)
    return X.astype("float32"), y_idx, num_channels, num_samples, num_classes


# --------------------------------------------------------------------------- #
# 3. Train
# --------------------------------------------------------------------------- #
def train_model(model, X, y, run_cfg, device):
    dataset = TensorDataset(
        torch.from_numpy(X).unsqueeze(1),  # (N, 1, C, T)
        torch.from_numpy(y),
    )
    loader = DataLoader(dataset, batch_size=run_cfg.get("batch_size", 32),
                         shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=run_cfg.get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss()

    epochs = run_cfg.get("epochs", 50)
    n_steps_train = run_cfg.get("n_steps_train", run_cfg.get("n_steps", 20))
    readout_mode = run_cfg.get("readout_mode", "spk_mean")

    model.to(device)
    for epoch in range(epochs):
        avg_loss = train_one_epoch(model, loader, optimizer, criterion, device,
                                    n_steps=n_steps_train, readout_mode=readout_mode)
        print(f"  epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
    return model


# --------------------------------------------------------------------------- #
# 4. ONNX export
# --------------------------------------------------------------------------- #
class InferenceWrapper(nn.Module):
    """Wraps SpikingEEGNet so forward() returns a single logits tensor.
    This is what makes the exported graph -- and the C onnx2c generates
    from it -- directly comparable to your non-spiking model's export."""

    def __init__(self, model: SpikingEEGNet, num_steps: int, readout_mode: str):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.readout_mode = readout_mode

    def forward(self, x):
        spk, mem = self.model(x, num_steps=self.num_steps)
        if self.readout_mode == "mem_last":
            return self.model.classifier(mem[-1].flatten(1))
        return aggregate_logits(spk, mem, self.readout_mode)


def export_onnx(model, num_channels, num_samples, num_steps, readout_mode,
                 onnx_path: str):
    model.eval()
    wrapper = InferenceWrapper(model, num_steps=num_steps, readout_mode=readout_mode)
    dummy = torch.zeros(1, 1, num_channels, num_samples)

    # model.forward() uses a plain Python `for` loop over num_steps, so
    # torch.onnx.export's tracer unrolls it into a static graph -- no
    # dynamic control-flow (Loop/If) ops end up in the exported model,
    # which is what lets onnx2c handle it at all.
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["eeg"],
        output_names=["logits"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {onnx_path}")


# --------------------------------------------------------------------------- #
# 5. onnx2c
# --------------------------------------------------------------------------- #
def convert_to_c(onnx_path: str, c_path: str):
    exe = shutil.which("onnx2c")
    if exe is None:
        print(
            "\nonnx2c not found on PATH.\n"
            "Install it with:\n"
            "  git clone https://github.com/kraiskil/onnx2c\n"
            "  cd onnx2c && mkdir build && cd build\n"
            "  cmake .. && make\n"
            "then add the resulting `onnx2c` binary to your PATH.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    result = subprocess.run([exe, onnx_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("onnx2c failed:\n" + result.stderr, file=sys.stderr)
        sys.exit(1)

    Path(c_path).write_text(result.stdout)
    print(f"Wrote generated C to {c_path}\n")
    print(result.stdout)


# --------------------------------------------------------------------------- #
def main():
    # -------------------- config -------------------- #
    BEST_PARAMS_PATH = "best_params.json"
    ONNX_OUT = "spiking_eegnet.onnx"
    C_OUT = "spiking_eegnet.c"
    EPOCHS_OVERRIDE = None  # set to an int to override epochs from best_params.json
    # -------------------------------------------------- #

    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = load_best_params(BEST_PARAMS_PATH)
        model_kwargs, run_cfg = split_params(
            params,
            extra_defaults={"epochs": 2, "batch_size": 32, "lr": 1e-3,
                             "n_steps_train": 20, "readout_mode": "spk_mean"},
        )
        if EPOCHS_OVERRIDE is not None:
            run_cfg["epochs"] = EPOCHS_OVERRIDE

        print("Model kwargs:", model_kwargs)
        print("Run config:  ", run_cfg)

        X, y, num_channels, num_samples, num_classes = load_data()
        print(f"Data: X={X.shape} y={y.shape} "
              f"channels={num_channels} samples={num_samples} classes={num_classes}")

        model = SpikingEEGNet(
            num_classes=num_classes,
            num_channels=num_channels,
            num_samples=num_samples,
            **model_kwargs,
        )

        train_model(model, X, y, run_cfg, device)

        model.cpu()
        export_onnx(
            model, num_channels, num_samples,
            num_steps=run_cfg.get("n_steps_eval", run_cfg.get("n_steps_train", 20)),
            readout_mode=run_cfg.get("readout_mode", "spk_mean"),
            onnx_path=ONNX_OUT,
        )

    convert_to_c(ONNX_OUT, C_OUT)


if __name__ == "__main__":
    main()