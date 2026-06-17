#!/usr/bin/env python3
# src/quantization_testing/run_quantization.py
"""
Entry point for the fixed-point FPGA quantization analysis.

Workflow
--------
1. Train the SpikingEEGNet model once in full float32 precision using the
   existing LOSO pipeline (or load a previously saved checkpoint).
2. Run the post-training fixed-point quantization sweep over
   Q<integer_bits>.64 → Q<integer_bits>.0.
3. Save all results (CSV + PNG) under results/<DATASET_KEY>/quantization/.

Usage
-----
Run from the src/ directory:
    python -m quantization_testing.run_quantization
    python -m quantization_testing.run_quantization --load-checkpoint results/BNCI2014_001/model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ── Add src/ to path so sibling project modules are importable ──────────────
# __file__ is src/quantization_testing/run_quantization.py
# parent       is src/quantization_testing/
# parent.parent is src/   ← where bandpass_filter.py etc. live
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ── Project imports (from src/) ──────────────────────────────────────────────
from zscore_normalize import zscore_normalize
from bandpass_filter import bandpass_filter
from load_moabb_dataset import load_moabb_dataset
from make_loader import make_loader
from experiment_loso import experiment_loso
from build_model import build_model

# ── Quantization imports (relative — within quantization_testing/) ───────────
from quantization_testing.sweep import run_quantization_sweep
from quantization_testing.config import QuantConfig


# ============================================================================
# CONFIG — edit this block to change experiment settings
# ============================================================================

DATASET_KEY        = "BNCI2014_001"
TEST_SUBJECT_IDX   = 0
EPOCHS             = 100
BATCH_SIZE         = 32
N_STEPS_TRAIN      = 4
N_STEPS_EVAL       = 20
LR                 = 3e-4
FLOW               = 4.0
FHIGH              = 40.0
RUN_ZSCORE         = True
RUN_BANDPASS       = True
NORM_AXIS          = (1, 2, 3)

MODEL_CFG = dict(
    temporal_filters      = 8,
    depth_multiplier      = 2,
    pointwise_filters     = 16,
    temporal_kernel_div   = 2,
    separable_kernel_size = 16,
    pool1_size            = 4,
    pool2_size            = 4,
    dropout               = 0.5,
    beta                  = 0.95,
    spike_grad_slope      = 25.0,
)

QUANT_CFG = QuantConfig(
    signed              = True,
    overflow            = "saturate",
    rounding            = "nearest",
    safety_margin_bits  = 1,
    frac_bits_start     = 64,
    frac_bits_end       = 0,
    frac_bits_step      = 1,
    n_steps_eval        = N_STEPS_EVAL,
)

# ============================================================================


def main(checkpoint_path: str | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path("results") / DATASET_KEY
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "model.pt"

    # ── Load data ────────────────────────────────────────────────────────────
    X, y, subject_ids, meta = load_moabb_dataset(DATASET_KEY)
    print(f"Dataset meta: {meta}")

    if RUN_ZSCORE:
        X = zscore_normalize(X, axis=NORM_AXIS)
    if RUN_BANDPASS:
        print(f"Applying bandpass filter: {FLOW}–{FHIGH} Hz")
        X = bandpass_filter(X, sfreq=meta["sfreq"], flow=FLOW, fhigh=FHIGH)

    test_mask  = subject_ids == TEST_SUBJECT_IDX
    train_mask = ~test_mask
    train_loader = make_loader(X[train_mask], y[train_mask], BATCH_SIZE)
    val_loader   = make_loader(X[test_mask],  y[test_mask],  BATCH_SIZE, shuffle=False)

    # ── Train or load model ───────────────────────────────────────────────────
    if checkpoint_path is not None:
        print(f"\nLoading checkpoint from {checkpoint_path} …")
        model = build_model(meta, device, **MODEL_CFG)
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print("  Checkpoint loaded.")
    else:
        print("\nTraining model (float32) …")
        train_cfg = dict(
            epochs         = EPOCHS,
            batch_size     = BATCH_SIZE,
            lr             = LR,
            n_steps_train  = N_STEPS_TRAIN,
            n_steps_eval   = N_STEPS_EVAL,
            patience       = None,
            trial          = None,
        )
        _history, _acc, model = experiment_loso(
            X, y, subject_ids, meta, device, train_cfg,
            test_subject_idx = TEST_SUBJECT_IDX,
            model_kwargs     = MODEL_CFG,
        )
        print(f"\nTraining complete.  Final balanced accuracy: {_acc:.4f}")

        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved → {checkpoint_file}")

    # ── Quantization sweep ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Starting fixed-point quantization sweep")
    print("="*60)

    quant_output = output_dir / "quantization"
    df = run_quantization_sweep(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = device,
        output_dir   = quant_output,
        cfg          = QUANT_CFG,
    )

    print(f"\nAll done.  Results saved to {quant_output}/")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-point FPGA quantization sweep for SpikingEEGNet."
    )
    parser.add_argument(
        "--load-checkpoint",
        metavar="PATH",
        default=None,
        help="Path to a saved model checkpoint (.pt). Skips training if provided.",
    )
    args = parser.parse_args()
    main(checkpoint_path=args.load_checkpoint)