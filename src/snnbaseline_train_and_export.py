#!/usr/bin/env python3
"""
source ~/venvs/spiking-eegnet/bin/activate
cd /mnt/c/Users/ajrbe/Documents/Git/spiking-eegnet/src
python3 snnbaseline_train_and_export.py

python3 snnbaseline_train_and_export.py 2>&1 | tee output_snnbaseline.log

This is the "straight architecture conversion" baseline: SpikingEEGNet (the
spiking architecture), configured ENTIRELY from the literal EEGNet-paper
config (LITERAL_EEGNET_SMR_CFG) -- no best_params.json, no Optuna tuning
anywhere in this script, for architecture OR for training hyperparameters.
It sits between the other two scripts:

  - train_and_export.py           : SpikingEEGNet + Optuna-tuned everything
  - baseline_train_and_export.py  : EEGNetReLU  (ANN) + literal paper architecture
                                     (training hyperparams still pulled from
                                     best_params.json)
  - snnbaseline_train_and_export.py (this file):
                                     SpikingEEGNet + literal paper architecture
                                     AND fixed/untuned training hyperparams --
                                     best_params.json is never read.

The goal is to isolate the effect of the ANN->SNN conversion itself (LIF
neurons, num_steps unrolling, spike readout) with NOTHING borrowed from the
tuned run -- this is what "just straight architecture conversion" means:
take EEGNetReLU's paper config, swap the activations/pooling for their
spiking equivalents, and train it with plain, untuned defaults.

Differences vs. train_and_export.py (all necessary, not stylistic):
  - No load_best_params() / best_params.json anywhere in this script.
    model_kwargs is always LITERAL_EEGNET_SMR_CFG, imported unchanged from
    baseline.py (same architecture-fixing rationale as
    baseline_train_and_export.py's split_params: several of
    LITERAL_EEGNET_SMR_CFG's field names -- temporal_filters,
    depth_multiplier, pointwise_filters, separable_kernel_size, pool1_size,
    pool2_size, dropout -- collide with SpikingEEGNet's constructor
    argument names, so pulling from a tuned params dict would silently feed
    it the wrong architecture).
  - run_cfg (epochs, batch_size, lr, n_steps_train, n_steps_eval,
    readout_mode) is a fixed dict of untuned defaults declared in main(),
    not read from any JSON file -- there is no Optuna trial backing this
    run at all.

Everything else -- training loop (num_steps unrolling, readout_mode),
InferenceWrapper, reset_snn_state, ONNX export settings, onnx2c invocation,
and post-generation bug patches -- is identical to train_and_export.py, so
the only thing that differs between that script's output and this one's is
that every knob here (architecture AND training) comes from fixed/paper
defaults instead of Optuna.
"""

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

from spiking_eegnet import SpikingEEGNet
from train_one_epoch import aggregate_logits, train_one_epoch
from baseline import LITERAL_EEGNET_SMR_CFG


# --------------------------------------------------------------------------- #
# 1. Data -- identical to train_and_export.py's load_data()
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
# 2. Train -- identical to train_and_export.py's train_model()
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
# 3. ONNX export -- identical to train_and_export.py (SpikingEEGNet still
#    needs InferenceWrapper / reset_snn_state, unlike the EEGNetReLU
#    baseline, since it still carries snntorch hidden state and returns
#    (spk, mem) instead of a single logits tensor)
# --------------------------------------------------------------------------- #
class InferenceWrapper(nn.Module):
    """Wraps SpikingEEGNet so forward() returns a single logits tensor.
    This is what makes the exported graph -- and the C onnx2c generates
    from it -- directly comparable to the other two scripts' exports."""

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


def reset_snn_state(model: nn.Module):
    """Detach stale membrane/synaptic state left over from training.

    snntorch neurons (Leaky, Synaptic, etc.) keep their hidden state as a
    plain instance attribute (e.g. `mem`, `syn`) that persists across
    forward calls. reset_mem() unconditionally assumes that attribute is
    already a real tensor (it calls `torch.zeros_like(self.mem, ...)`
    directly, with no None/isinstance check), so we can't just null it out
    -- that raises `AttributeError: 'NoneType' object has no attribute
    'device'` on the next forward call.

    The actual problem is that after training, `self.mem` is still
    attached to the training-time autograd graph (requires_grad=True).
    During ONNX tracing, `zeros_like` captures that stale, grad-requiring
    tensor as a closure variable and fails with:
        "Cannot insert a Tensor that requires grad as a constant."

    Detaching (and cloning, to fully drop the graph reference) keeps the
    attribute a real tensor -- same shape/dtype/device -- but removes it
    from the autograd graph and sets requires_grad=False, which is all
    reset_mem() needs to build a fresh state on the next call.
    """
    for module in model.modules():
        for state_attr in ("mem", "syn", "spk"):
            if hasattr(module, state_attr):
                val = getattr(module, state_attr)
                if isinstance(val, torch.Tensor):
                    setattr(module, state_attr, val.detach().clone())


def export_onnx(model, num_channels, num_samples, num_steps, readout_mode,
                 onnx_path: str):
    model.eval()
    reset_snn_state(model)
    wrapper = InferenceWrapper(model, num_steps=num_steps, readout_mode=readout_mode)
    dummy = torch.zeros(1, 1, num_channels, num_samples)

    # model.forward() uses a plain Python `for` loop over num_steps, so
    # torch.onnx.export's tracer unrolls it into a static graph -- no
    # dynamic control-flow (Loop/If) ops end up in the exported model,
    # which is what lets onnx2c handle it at all.
    wrapper.eval()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_path,
            input_names=["eeg"],
            output_names=["logits"],
            opset_version=13,
            do_constant_folding=True,
            dynamo=False,  # legacy TorchScript-based exporter -- the newer
                            # dynamo exporter emits axes-as-input ops that fail
                            # to downconvert to opset 13 and produce a malformed
                            # graph onnx2c can't parse
        )
    print(f"Exported ONNX model to {onnx_path}")


# --------------------------------------------------------------------------- #
# 4. onnx2c -- identical to train_and_export.py (unchanged bug patches)
# --------------------------------------------------------------------------- #
def patch_onnx2c_bugs(c_path: str) -> dict:
    """Post-process onnx2c-generated C to work around known onnx2c codegen
    bugs that surface on deeply unrolled graphs (e.g. LIF neurons unrolled
    over many timesteps, as here). Safe to run on every export -- each regex
    is a no-op once its pattern no longer appears (e.g. if onnx2c fixes these
    upstream)."""
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
    # No best_params.json anywhere in this script -- this is the "straight
    # architecture conversion" baseline, so every knob below is either the
    # literal EEGNet paper architecture (LITERAL_EEGNET_SMR_CFG) or a plain,
    # untuned training default. Nothing here comes from Optuna.
    EPOCHS = 1  # <-- change this to set how many epochs to train for
    ONNX_OUT = "snnbaseline_eegnet.onnx"
    C_OUT = "snnbaseline_eegnet.c"

    # model_kwargs is always the literal paper config, unchanged from
    # baseline.py -- never derived from a tuned params file.
    model_kwargs = dict(LITERAL_EEGNET_SMR_CFG)

    # run_cfg: fixed, untuned training hyperparameters. n_steps_train /
    # readout_mode have no EEGNet-paper equivalent (the paper's EEGNet isn't
    # spiking), so these are plain reasonable defaults, not tuned values.
    run_cfg = {
        "epochs": EPOCHS,
        "batch_size": 32,
        "lr": 1e-3,
        "n_steps_train": 20,
        "n_steps_eval": 20,
        "readout_mode": "spk_mean",
    }
    # -------------------------------------------------- #

    # Resolve to absolute paths up front so we can report exactly where
    # everything landed, regardless of what the current working directory
    # happens to be when this script is invoked.
    onnx_out_abs = str(Path(ONNX_OUT).resolve())
    c_out_abs = str(Path(C_OUT).resolve())

    if True:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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