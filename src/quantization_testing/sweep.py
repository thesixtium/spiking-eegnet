# quantization/sweep.py
"""
End-to-end quantization sweep orchestrator.

Ties together:
  1. Auto integer-bit selection
  2. Float32 baseline evaluation
  3. Fractional-bit sweep
  4. CSV export
  5. Plot generation

This module is the single entry point for the quantization analysis.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import torch

from .config import QuantConfig
from .quantizer import FixedPointQuantizer, determine_integer_bits
from .evaluator import evaluate_quantized
from .plotting import generate_all_plots


def run_quantization_sweep(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path = "results/quantization",
    cfg: QuantConfig | None = None,
) -> pd.DataFrame:
    """
    Run the complete fixed-point quantization sweep.

    Steps
    -----
    1. Determine integer-bit width from training data + model parameters.
    2. Evaluate the model at full float32 precision as a baseline.
    3. For each fractional-bit value in cfg.frac_bits_range:
       a. Build a FixedPointQuantizer for Q<integer_bits>.<frac_bits>.
       b. Evaluate accuracy and loss using fully fixed-point inference.
       c. Record results.
    4. Export results to CSV.
    5. Generate and save all plots.

    Parameters
    ----------
    model        : SpikingEEGNet  Trained model (not modified).
    train_loader : DataLoader     Used only to scan input magnitudes.
    val_loader   : DataLoader     Used for all evaluations.
    device       : torch.device
    output_dir   : Path           Where to save CSV + PNG results.
    cfg          : QuantConfig    Sweep configuration. Uses defaults if None.

    Returns
    -------
    pd.DataFrame with columns:
        fractional_bits, integer_bits, format, balanced_accuracy, loss, eval_time_s
    """
    if cfg is None:
        cfg = QuantConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # ── Step 1: Auto integer-bit selection ───────────────────────────────────
    integer_bits = determine_integer_bits(train_loader, model, cfg, device)
    format_str = f"Q{integer_bits}"

    print(f"\n{'='*60}")
    print(f"  Fixed-Point Sweep: {format_str}.<F>  (F = {cfg.frac_bits_start} → {cfg.frac_bits_end})")
    print(f"  Overflow mode  : {cfg.overflow}")
    print(f"  Rounding mode  : {cfg.rounding}")
    print(f"  Signed         : {cfg.signed}")
    print(f"  SNN time steps : {cfg.n_steps_eval}")
    print(f"  Output dir     : {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 2: Float32 baseline ──────────────────────────────────────────────
    print("Evaluating float32 baseline …")
    float_acc, float_loss = _evaluate_float(model, val_loader, device, cfg.n_steps_eval)
    print(f"  Float32 baseline → acc={float_acc:.4f}  loss={float_loss:.4f}\n")

    # ── Step 3: Fractional-bit sweep ─────────────────────────────────────────
    rows = []
    frac_range = cfg.frac_bits_range
    n_total = len(frac_range)

    for idx, frac_bits in enumerate(frac_range):
        q = FixedPointQuantizer(integer_bits=integer_bits, frac_bits=frac_bits, cfg=cfg)
        fmt_label = f"{format_str}.{frac_bits}"

        t0 = time.perf_counter()
        acc, loss = evaluate_quantized(model, val_loader, q, device, cfg.n_steps_eval)
        elapsed = time.perf_counter() - t0

        rows.append({
            "fractional_bits":    frac_bits,
            "integer_bits":       integer_bits,
            "format":             fmt_label,
            "balanced_accuracy":  round(acc, 6),
            "loss":               round(loss, 6),
            "eval_time_s":        round(elapsed, 2),
        })

        pct = (idx + 1) / n_total * 100
        print(
            f"  [{idx+1:3d}/{n_total}]  {fmt_label:12s}  "
            f"acc={acc:.4f}  loss={loss:.4f}  "
            f"({elapsed:.1f}s)  [{pct:.0f}%]"
        )

    df = pd.DataFrame(rows)

    # ── Step 4: Export CSV ────────────────────────────────────────────────────
    csv_path = output_dir / "quantization_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Sweep] Results saved → {csv_path}")

    # Also save float baseline row for completeness
    baseline_row = pd.DataFrame([{
        "fractional_bits":    "float32",
        "integer_bits":       "N/A",
        "format":             "float32",
        "balanced_accuracy":  round(float_acc, 6),
        "loss":               round(float_loss, 6),
        "eval_time_s":        0.0,
    }])
    baseline_csv = output_dir / "float_baseline.csv"
    baseline_row.to_csv(baseline_csv, index=False)
    print(f"[Sweep] Baseline saved → {baseline_csv}")

    # ── Step 5: Generate plots ────────────────────────────────────────────────
    print("\n[Sweep] Generating plots …")
    generate_all_plots(
        df=df,
        output_dir=output_dir,
        float_acc=float_acc,
        float_loss=float_loss,
    )

    print(f"\n[Sweep] Complete.  Results in {output_dir}/")
    _print_summary(df, float_acc)

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _evaluate_float(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_steps: int,
) -> tuple[float, float]:
    """
    Standard (float32) evaluation.  Mirrors the existing ``evaluate`` function
    but does not depend on it being importable (avoids circular imports).
    """
    from sklearn.metrics import balanced_accuracy_score
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            spk = model(xb, num_steps=n_steps)
            logits = spk.mean(0)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    return (
        float(balanced_accuracy_score(all_labels, all_preds)),
        float(total_loss / max(n_batches, 1)),
    )


def _print_summary(df: pd.DataFrame, float_acc: float) -> None:
    """Print a compact summary table of the sweep results."""
    print("\n" + "="*55)
    print(f"  {'F-bits':>8}  {'Accuracy':>10}  {'Loss':>10}  {'Drop pp':>8}")
    print("-"*55)

    # Show every 4th row + first and last for brevity
    show_idx = sorted(set(
        [0] +
        list(range(0, len(df), max(1, len(df) // 16))) +
        [len(df) - 1]
    ))
    for i in show_idx:
        row = df.iloc[i]
        drop = (float_acc - row["balanced_accuracy"]) * 100
        print(
            f"  {int(row['fractional_bits']):>8}  "
            f"{row['balanced_accuracy']*100:>9.2f}%  "
            f"{row['loss']:>10.4f}  "
            f"{drop:>+7.2f}pp"
        )
    print("="*55)

    # Find the minimum fractional bits where accuracy drops < 5 pp from float
    threshold_pp = 5.0
    acceptable = df[
        (float_acc - df["balanced_accuracy"]) * 100 <= threshold_pp
    ]
    if not acceptable.empty:
        min_bits = acceptable["fractional_bits"].min()
        print(f"\n  Minimum fractional bits for <{threshold_pp}pp accuracy drop: {min_bits}")
    else:
        print(f"\n  No configuration achieved <{threshold_pp}pp accuracy drop.")
