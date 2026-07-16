#!/usr/bin/env python3
"""
source ~/venvs/spiking-eegnet/bin/activate
cd /mnt/c/Users/ajrbe/Documents/Git/spiking-eegnet/src
python3 baseline_train_and_export.py

python3 baseline_train_and_export.py 2>&1 | tee output_baseline.log

This is the non-spiking counterpart of train_and_export.py. It follows the
EXACT SAME steps (load params -> load data -> train -> export to ONNX ->
convert to C via onnx2c -> patch known onnx2c bugs), but trains and exports
EEGNetReLU (the literal, continuous-activation EEGNet from baseline.py)
instead of SpikingEEGNet. The goal is to give the C-code characterization
tool a "normal" (non-spiking) EEGNet baseline to compare against, produced
by a conversion pipeline that's as close to identical as possible to the
spiking one -- same ONNX export settings, same onnx2c invocation, same
post-generation patching.

Differences vs. train_and_export.py (all necessary, not stylistic):
  - Model is EEGNetReLU (baseline.py), not SpikingEEGNet.
  - No num_steps forward loop -- one standard forward pass per batch.
  - No readout_mode / InferenceWrapper / reset_snn_state -- EEGNetReLU's
    forward() already returns a single (batch, num_classes) logits tensor,
    so it can be torch.onnx.export'd directly with no wrapping.
  - split_params() matches against EEGNetReLU's constructor signature
    instead of SpikingEEGNet's. In practice best_params.json was tuned for
    SpikingEEGNet's search space (e.g. `temporal_kernel_div`, `beta`,
    `spike_grad_slope`), so none of those keys match EEGNetReLU's
    signature (e.g. `temporal_kernel_size`) -- model_kwargs ends up empty
    and EEGNetReLU falls back to its own constructor defaults, which ARE
    the literal EEGNet paper architecture (see baseline.py's
    LITERAL_EEGNET_SMR_CFG / EEGNetReLU docstring). Only the
    architecture-irrelevant training hyperparameters (epochs, batch_size,
    lr) carry over from best_params.json, exactly like baseline.py does
    for its own ANN baseline.
"""

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Prevent any stray tensor from ever being printed in full (e.g. if a
# warning, assertion, or debug print anywhere in the stack happens to
# include a tensor repr) -- this is what was spamming the console with
# giant blocks of numbers like "Columns 1 to 8 ... [ torch.FloatTensor{...} ]".
torch.set_printoptions(threshold=100, edgeitems=3, linewidth=120)

from baseline import EEGNetReLU


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
    matching against EEGNetReLU's actual constructor signature -- so this
    doesn't silently break if you add/rename search params later.

    Note: best_params.json is tuned for SpikingEEGNet's search space, so
    most (likely all) of its keys won't match EEGNetReLU's signature. Any
    key that DOES match (e.g. `dropout`) is passed through; everything else
    -- including all SNN-only keys -- lands in run_cfg and is simply
    ignored by train_model/export_onnx below, exactly like train_cfg in
    baseline.py only pulls lr/epochs/batch_size out of params regardless of
    what else is in there.
    """
    model_sig = inspect.signature(EEGNetReLU.__init__)
    model_arg_names = set(model_sig.parameters) - {
        "self", "num_classes", "num_channels", "num_samples"
    }
    model_kwargs = {k: v for k, v in params.items() if k in model_arg_names}
    run_cfg = {k: v for k, v in params.items() if k not in model_arg_names}
    for k, v in extra_defaults.items():
        run_cfg.setdefault(k, v)
    return model_kwargs, run_cfg


# --------------------------------------------------------------------------- #
# 2. Data -- identical to train_and_export.py's load_data()
# --------------------------------------------------------------------------- #
def load_data(dataset_name: str = "BNCI2014_001"):
    """Loads BNCI2014_001 via MOABB using a fresh tempfile.mkdtemp() per
    process, so we never resolve stale SLURM scratch paths."""
    import numpy as np
    import mne
    import moabb
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    scratch = tempfile.mkdtemp(prefix="moabb_")

    # moabb's get_dataset_path() only ever copies MNE_DATA into the
    # dataset-specific MNE_DATASETS_BNCI_PATH key the first time that key is
    # unset, then freezes it permanently in ~/.mne/mne-python.json -- every
    # later call reads straight from that frozen key regardless of MNE_DATA.
    # moabb.set_download_dir() (and mne.set_config("MNE_DATA", ...)) only
    # ever touch the generic MNE_DATA key, so neither can override a
    # MNE_DATASETS_BNCI_PATH that's already stuck pointing at a deleted
    # scratch dir from an earlier session. Set both explicitly so this run's
    # fresh dir always wins.
    stale_bnci_path = mne.get_config("MNE_DATASETS_BNCI_PATH")
    if stale_bnci_path:
        print(f"Overriding frozen MNE_DATASETS_BNCI_PATH: {stale_bnci_path!r}")
    mne.set_config("MNE_DATA", scratch)
    mne.set_config("MNE_DATASETS_BNCI_PATH", scratch)
    moabb.set_download_dir(scratch)
    print(f"Using scratch data dir: {scratch}")

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
# 3. Train (single forward pass per batch -- no num_steps / readout_mode)
# --------------------------------------------------------------------------- #
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

    model.to(device)
    for epoch in range(epochs):
        avg_loss = train_one_epoch_baseline(model, loader, optimizer, criterion, device)
        print(f"  epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
    return model


# --------------------------------------------------------------------------- #
# 4. ONNX export
# --------------------------------------------------------------------------- #
# No InferenceWrapper / reset_snn_state needed here: EEGNetReLU has no
# snntorch hidden state (mem/syn/spk) to detach, and forward() already
# returns a single (batch, num_classes) logits tensor directly, so the
# model can be traced by torch.onnx.export as-is.
def export_onnx(model, num_channels, num_samples, onnx_path: str):
    model.eval()
    dummy = torch.zeros(1, 1, num_channels, num_samples)

    # Same exporter settings as train_and_export.py's export_onnx, so the
    # resulting graph (and the onnx2c-generated C) is as directly
    # comparable as possible to the spiking export.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            onnx_path,
            input_names=["eeg"],
            output_names=["logits"],
            opset_version=13,
            do_constant_folding=True,
            dynamo=False,  # legacy TorchScript-based exporter, matching the
                            # spiking export -- keeps both conversion
                            # pipelines on the same exporter path
        )
    print(f"Exported ONNX model to {onnx_path}")


# --------------------------------------------------------------------------- #
# 5. onnx2c -- identical to train_and_export.py (unchanged bug patches)
# --------------------------------------------------------------------------- #
def patch_onnx2c_bugs(c_path: str) -> dict:
    """Post-process onnx2c-generated C to work around known onnx2c codegen
    bugs. Safe to run on every export -- each regex is a no-op once its
    pattern no longer appears (e.g. if onnx2c fixes these upstream, or if
    this particular graph never triggers them in the first place)."""
    path = Path(c_path)
    src = path.read_text()
    original = src

    # --- Bug 1: malformed address-of on union-temp members -----------------
    # onnx2c emits `structvar.&member` instead of `&structvar.member` when
    # taking the address of a tensor living inside a tuN union temp.
    # Invalid C -> "expected identifier before '&' token" and a cascade of
    # "too few arguments" errors at every call site.
    n_addr = [0]

    def _fix_addr(m: re.Match) -> str:
        n_addr[0] += 1
        return f"&{m.group(1)}.{m.group(2)}"

    src = re.sub(r'([A-Za-z_]\w*)\.&([A-Za-z_]\w*)', _fix_addr, src)

    # --- Bug 2: un-dereferenced pointers in the Clip macro ------------------
    # In node__*_Clip functions, onnx2c dereferences min_val/max_val into
    # local scalars (minv/maxv) but leaves `input` AND `output` as raw
    # pointer params, then uses/assigns them directly in
    # `output = MAX(MIN(input, maxv), minv);` -> first a pointer/float
    # comparison error on `input`, and (once that's fixed) an
    # incompatible-assignment error on `output = <float>`.
    # Matches both the original unpatched line and the previously
    # half-patched one (input already dereferenced, output not).
    src, n_clip = re.subn(
        r'output = MAX\(\s*MIN\(\s*\*?input\s*,\s*maxv\s*\)\s*,\s*minv\s*\);',
        '*output = MAX( MIN( *input, maxv), minv);',
        src,
    )

    path.write_text(src)

    counts = {"address_of_fixes": n_addr[0], "clip_deref_fixes": n_clip}
    print(f"patch_onnx2c_bugs: {counts} in {c_path}")
    if src == original:
        print("  no changes made -- already patched, or onnx2c fixed these upstream")
    return counts


def patch_reducemean_bug(c_path: str) -> dict:
    """Bug 3: mismatched output-index rank in ReduceMean accumulation loops.

    onnx2c squeezes size-1 dims out of the *declared* output array (e.g.
    `float y[4]` for a [1,4]-shaped output) but doesn't correspondingly
    squeeze the index expression in the accumulation loop -- it still
    emits the full pre-squeeze index list, e.g. `y[i1][i2]+=x[i0][i1][i2];`
    where `i1` only ever ranges over 1 value. That's a rank mismatch against
    the 1-D declaration -> "subscripted value is neither array nor pointer
    nor vector".

    EEGNetReLU's graph has no ReduceMean node (that op only comes from
    SpikingEEGNet's spk_mean readout), so this is expected to be a no-op
    here -- kept in place unchanged so both conversion pipelines run the
    exact same patch steps, in case that ever changes.

    Fix: for each node__ReduceMean function, parse the declared output rank
    from the signature, find which loop indices have bound 1 (i.e. were the
    ones squeezed out of the declaration), and drop exactly those indices
    from the accumulation line's output subscript so its rank matches the
    declaration. Generalizes to any axes/shape combo, not just this one.
    """
    path = Path(c_path)
    src = path.read_text()

    func_start_re = re.compile(r'FUNC_PREFIX\s+void\s+node__ReduceMean\s*\(')
    param_re = re.compile(r'(const\s+)?float\s+(\w+)((?:\[\d+\])+)')
    loop_bound_re = re.compile(r'for \(unsigned (i\d+) = 0; \1<(\d+);')

    fixed = 0
    skipped = []
    out = []
    pos = 0
    for m in func_start_re.finditer(src):
        sig_start = m.start()
        paren_end = src.index(')', m.end())
        brace_start = src.index('{', paren_end)
        depth = 0
        brace_end = None
        for i in range(brace_start, len(src)):
            if src[i] == '{':
                depth += 1
            elif src[i] == '}':
                depth -= 1
                if depth == 0:
                    brace_end = i
                    break
        if brace_end is None:
            continue

        sig_text = src[sig_start:paren_end]
        body_text = src[brace_start:brace_end + 1]

        params = list(param_re.finditer(sig_text))
        outvar_match = None
        for p in reversed(params):
            if not p.group(1):  # not const -> it's the output param
                outvar_match = p
                break
        if outvar_match is None:
            continue
        outvar = outvar_match.group(2)
        out_rank = outvar_match.group(3).count('[')

        loop_bounds = {k: int(v) for k, v in loop_bound_re.findall(body_text)}

        accum_re = re.compile(re.escape(outvar) + r'((?:\[\w+\])+)\s*\+=')
        am = accum_re.search(body_text)
        if am is None:
            continue
        idxs = re.findall(r'\[(\w+)\]', am.group(1))
        if len(idxs) <= out_rank:
            continue  # already matches declared rank, nothing to fix

        kept = [ix for ix in idxs if loop_bounds.get(ix) != 1]
        if len(kept) != out_rank:
            skipped.append(outvar)
            continue

        new_sub = ''.join(f'[{ix}]' for ix in kept)
        old_sub = am.group(1)
        new_body_text = body_text[:am.start(1)] + new_sub + body_text[am.end(1):]

        out.append(src[pos:brace_start])
        out.append(new_body_text)
        pos = brace_end + 1
        fixed += 1
        print(f"  fixed {outvar}{old_sub} -> {outvar}{new_sub} "
              f"(declared rank {out_rank})")

    out.append(src[pos:])
    new_src = ''.join(out)
    if new_src != src:
        path.write_text(new_src)

    counts = {"reducemean_fixes": fixed, "reducemean_skipped": skipped}
    print(f"patch_reducemean_bug: {counts} in {c_path}")
    if fixed == 0 and not skipped:
        print("  no ReduceMean rank mismatches found -- "
              "already patched, or onnx2c fixed this upstream")
    return counts


def find_reduce_mean_context(c_path: str, context_lines: int = 15) -> None:
    """A third onnx2c bug ('subscripted value is neither array nor pointer
    nor vector' in a ReduceMean node) needs a look at the actual declaration
    of the output tensor before it can be auto-patched -- the fix depends on
    how many dims onnx2c actually gave it. This just prints the surrounding
    context so it can be diagnosed/patched by hand."""
    lines = Path(c_path).read_text().splitlines()
    for i, line in enumerate(lines):
        if "node__ReduceMean" in line and ("FUNC_PREFIX" in line or "{" in line):
            start = max(0, i - 2)
            end = min(len(lines), i + context_lines)
            print(f"\n--- context around line {i + 1} in {c_path} ---")
            for j in range(start, end):
                print(f"{j + 1:6}\t{lines[j]}")
            print("--- end context ---\n")


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
    # (Not dumping the full generated C source here -- it's already saved
    # to disk at c_path above; open that file to inspect it.)

    patch_onnx2c_bugs(c_path)
    rm_counts = patch_reducemean_bug(c_path)
    if rm_counts["reducemean_skipped"]:
        print("Could not auto-fix ReduceMean rank mismatch for: "
              f"{rm_counts['reducemean_skipped']} -- dumping context for "
              "manual patching:")
        find_reduce_mean_context(c_path)


# --------------------------------------------------------------------------- #
def main():
    # -------------------- config -------------------- #
    EPOCHS = 1  # <-- change this to set how many epochs to train for
    BEST_PARAMS_PATH = "best_params.json"
    ONNX_OUT = "baseline_eegnet.onnx"
    C_OUT = "baseline_eegnet.c"
    # -------------------------------------------------- #

    # Resolve to absolute paths up front so we can report exactly where
    # everything landed, regardless of what the current working directory
    # happens to be when this script is invoked.
    onnx_out_abs = str(Path(ONNX_OUT).resolve())
    c_out_abs = str(Path(C_OUT).resolve())

    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        params = load_best_params(BEST_PARAMS_PATH)
        model_kwargs, run_cfg = split_params(
            params,
            extra_defaults={"epochs": EPOCHS, "batch_size": 32, "lr": 1e-3},
        )
        run_cfg["epochs"] = EPOCHS

        print("Model kwargs:", model_kwargs)
        print("Run config:  ", run_cfg)

        X, y, num_channels, num_samples, num_classes = load_data()
        print(f"Data: X={X.shape} y={y.shape} "
              f"channels={num_channels} samples={num_samples} classes={num_classes}")

        model = EEGNetReLU(
            num_classes=num_classes,
            num_channels=num_channels,
            num_samples=num_samples,
            **model_kwargs,
        )

        train_model(model, X, y, run_cfg, device)

        model.cpu()
        export_onnx(
            model, num_channels, num_samples,
            onnx_path=onnx_out_abs,
        )

    convert_to_c(onnx_out_abs, c_out_abs)

    # -------------------- summary -------------------- #
    print("\n" + "=" * 60)
    print("DONE. Output files were saved to:")
    print(f"  ONNX model : {onnx_out_abs}")
    print(f"  C source   : {c_out_abs}")
    print(f"  (current working directory was: {Path.cwd()})")
    print("=" * 60)


if __name__ == "__main__":
    main()