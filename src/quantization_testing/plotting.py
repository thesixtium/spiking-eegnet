# quantization/plotting.py
"""
Publication-quality plots for the fixed-point quantization sweep.

All figures are saved to ``output_dir`` with 300 dpi and tight bounding boxes.

Figures produced
----------------
1. accuracy_vs_fractional_bits.png  — Balanced accuracy vs. fractional bits
2. loss_vs_fractional_bits.png      — Cross-entropy loss vs. fractional bits
3. accuracy_drop_vs_fractional_bits.png — Accuracy degradation relative to float
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
_FIGSIZE = (8, 5)
_DPI = 300
_LINE_KW = dict(linewidth=2.0, marker="o", markersize=4)
_GRID_KW = dict(alpha=0.35, linestyle="--", linewidth=0.8)
_SPINE_ALPHA = 0.4


def _apply_style(ax: plt.Axes) -> None:
    """Apply a clean, publication-ready style to an axes object."""
    ax.grid(True, **_GRID_KW)
    for spine in ax.spines.values():
        spine.set_alpha(_SPINE_ALPHA)
    ax.tick_params(labelsize=10)


def plot_accuracy(df: pd.DataFrame, output_dir: Path, float_acc: float | None = None) -> Path:
    """
    Balanced accuracy vs. fractional bits.

    Parameters
    ----------
    df          : DataFrame with columns ['fractional_bits', 'balanced_accuracy']
    output_dir  : Directory where the figure will be saved.
    float_acc   : Full-precision (float32) accuracy — drawn as a dashed baseline.

    Returns
    -------
    Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    ax.plot(
        df["fractional_bits"], df["balanced_accuracy"] * 100,
        color="#2176AE", label="Fixed-point model",
        **_LINE_KW,
    )

    if float_acc is not None:
        ax.axhline(
            float_acc * 100, color="#E84855",
            linestyle="--", linewidth=1.5,
            label=f"Float32 baseline ({float_acc*100:.1f}%)",
        )

    ax.set_xlabel("Fractional Bits (F)", fontsize=12)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=12)
    ax.set_title("Fixed-Point Precision vs. Balanced Accuracy", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    _apply_style(ax)
    fig.tight_layout()

    out = output_dir / "accuracy_vs_fractional_bits.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out}")
    return out


def plot_loss(df: pd.DataFrame, output_dir: Path, float_loss: float | None = None) -> Path:
    """
    Cross-entropy loss vs. fractional bits.

    Parameters
    ----------
    df          : DataFrame with columns ['fractional_bits', 'loss']
    output_dir  : Directory where the figure will be saved.
    float_loss  : Full-precision (float32) loss — drawn as a dashed baseline.

    Returns
    -------
    Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    ax.plot(
        df["fractional_bits"], df["loss"],
        color="#F77F00", label="Fixed-point model",
        **_LINE_KW,
    )

    if float_loss is not None:
        ax.axhline(
            float_loss, color="#6A0572",
            linestyle="--", linewidth=1.5,
            label=f"Float32 baseline ({float_loss:.4f})",
        )

    ax.set_xlabel("Fractional Bits (F)", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title("Fixed-Point Precision vs. Cross-Entropy Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    _apply_style(ax)
    fig.tight_layout()

    out = output_dir / "loss_vs_fractional_bits.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out}")
    return out


def plot_accuracy_drop(df: pd.DataFrame, output_dir: Path, float_acc: float) -> Path:
    """
    Accuracy degradation (percentage points) relative to floating-point baseline.

    Parameters
    ----------
    df         : DataFrame with columns ['fractional_bits', 'balanced_accuracy']
    output_dir : Directory where the figure will be saved.
    float_acc  : Full-precision (float32) accuracy used as the reference.

    Returns
    -------
    Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    drop = (float_acc - df["balanced_accuracy"]) * 100  # percentage points

    ax.plot(
        df["fractional_bits"], drop,
        color="#C1121F", label="Accuracy drop vs. float32",
        **_LINE_KW,
    )
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.8, alpha=0.6)
    ax.axhline(5, color="#E84855", linestyle=":", linewidth=1.2, label="5 pp threshold")

    ax.set_xlabel("Fractional Bits (F)", fontsize=12)
    ax.set_ylabel("Accuracy Drop (percentage points)", fontsize=12)
    ax.set_title("Accuracy Degradation vs. Fractional Bits", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.invert_yaxis()   # larger drop is "worse" → points downward
    _apply_style(ax)
    fig.tight_layout()

    out = output_dir / "accuracy_drop_vs_fractional_bits.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out}")
    return out


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    float_acc: float | None = None,
    float_loss: float | None = None,
) -> None:
    """
    Convenience wrapper — generate all quantization plots in one call.

    Parameters
    ----------
    df          : Full results DataFrame (fractional_bits, balanced_accuracy, loss, …)
    output_dir  : Directory where figures are saved (created if absent).
    float_acc   : Full-precision balanced accuracy (optional baseline line).
    float_loss  : Full-precision loss (optional baseline line).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy(df, output_dir, float_acc=float_acc)
    plot_loss(df, output_dir, float_loss=float_loss)

    if float_acc is not None:
        plot_accuracy_drop(df, output_dir, float_acc=float_acc)
