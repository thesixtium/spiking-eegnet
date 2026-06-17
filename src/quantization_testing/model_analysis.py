# quantization/model_analysis.py
"""
Model architecture visualisation and weight distribution analysis for SpikingEEGNet.

Produces
--------
1. model_architecture.txt      — Pretty-printed layer tree with shapes & param counts
2. layer_weight_stats.csv      — Per-layer: shape, numel, min, max, mean, std, |max|
3. weight_outlier_report.txt   — Top-N heaviest tensors + per-layer outlier counts
4. weight_distributions.png    — Grid of per-layer weight histograms
5. weight_heatmaps.png         — Per-layer |weight| heatmaps (conv kernels flattened)
6. parameter_magnitude_bar.png — Bar chart: max |weight| ranked across all tensors
7. outlier_fraction_bar.png    — Bar chart: fraction of weights > k*std per layer

All outputs land in ``output_dir``.

Usage
-----
    from quantization.model_analysis import analyse_model

    analyse_model(model, output_dir=Path("results/BNCI2014_001/quantization"))

This is called automatically by ``run_quantization_sweep`` before the sweep
begins when ``cfg.run_model_analysis=True`` (default True).  It can also be
called standalone.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_DPI = 150
_OUTLIER_SIGMA = 3.0        # flag weights more than this many std from mean
_TOP_N_OUTLIERS = 20        # number of heaviest individual weights to report
_HIST_BINS = 60


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Collect per-layer tensor statistics
# ─────────────────────────────────────────────────────────────────────────────

def _collect_tensor_stats(model: nn.Module) -> pd.DataFrame:
    """
    Walk every named parameter and buffer; compute descriptive statistics.

    Returns a DataFrame with one row per tensor, columns:
        name, tensor_type, shape, numel,
        min, max, abs_max, mean, std,
        outlier_count, outlier_fraction,
        q01, q25, median, q75, q99
    """
    rows = []

    def _stats(name: str, tensor_type: str, t: torch.Tensor):
        data = t.detach().float().cpu()
        flat = data.flatten()
        n = flat.numel()
        if n == 0:
            return

        mn   = flat.min().item()
        mx   = flat.max().item()
        amx  = flat.abs().max().item()
        mean = flat.mean().item()
        std  = flat.std().item() if n > 1 else 0.0

        # outliers: |x - mean| > k*std
        if std > 0:
            outlier_mask = (flat - mean).abs() > _OUTLIER_SIGMA * std
            out_count = outlier_mask.sum().item()
        else:
            out_count = 0

        qs = torch.quantile(flat, torch.tensor([0.01, 0.25, 0.50, 0.75, 0.99]))

        rows.append({
            "name":             name,
            "tensor_type":      tensor_type,
            "shape":            str(tuple(data.shape)),
            "numel":            n,
            "min":              round(mn,   6),
            "max":              round(mx,   6),
            "abs_max":          round(amx,  6),
            "mean":             round(mean, 6),
            "std":              round(std,  6),
            "outlier_count":    int(out_count),
            "outlier_fraction": round(out_count / n, 6),
            "q01":              round(qs[0].item(), 6),
            "q25":              round(qs[1].item(), 6),
            "median":           round(qs[2].item(), 6),
            "q75":              round(qs[3].item(), 6),
            "q99":              round(qs[4].item(), 6),
        })

    for name, param in model.named_parameters():
        _stats(name, "parameter", param)

    for name, buf in model.named_buffers():
        # Skip num_batches_tracked — it's a scalar counter, not a real weight
        if "num_batches_tracked" in name:
            continue
        _stats(name, "buffer", buf)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_max", ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Architecture summary (text)
# ─────────────────────────────────────────────────────────────────────────────

def _architecture_summary(model: nn.Module) -> str:
    """
    Return a multi-line string: the full module tree with
    per-parameter shapes and total counts per block.
    """
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Model: {model.__class__.__name__}")
    lines.append(f"{'='*70}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines.append(f"  Total parameters  : {total_params:,}")
    lines.append(f"  Trainable params  : {trainable:,}")
    lines.append(f"  Non-trainable     : {total_params - trainable:,}")
    lines.append("")

    def _walk(module: nn.Module, prefix: str = "", depth: int = 0) -> None:
        indent = "  " + "  " * depth
        children = list(module.named_children())

        if not children:
            # Leaf: print parameters inline
            params_info = []
            for pname, p in module.named_parameters(recurse=False):
                params_info.append(f"{pname}:{list(p.shape)}  numel={p.numel():,}")
            for bname, b in module.named_buffers(recurse=False):
                if "num_batches_tracked" not in bname:
                    params_info.append(f"{bname}:{list(b.shape)}  [buffer]")
            label = f"{prefix}  ({module.__class__.__name__})"
            if params_info:
                lines.append(f"{indent}{label}")
                for pi in params_info:
                    lines.append(f"{indent}    └─ {pi}")
            else:
                lines.append(f"{indent}{label}  [no params]")
        else:
            block_params = sum(p.numel() for p in module.parameters())
            lines.append(f"{indent}{prefix}  ({module.__class__.__name__})  [{block_params:,} params]")
            for child_name, child in children:
                _walk(child, prefix=child_name, depth=depth + 1)

    _walk(model, prefix=model.__class__.__name__, depth=0)
    lines.append(f"{'='*70}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Outlier report (text)
# ─────────────────────────────────────────────────────────────────────────────

def _outlier_report(df: pd.DataFrame, model: nn.Module) -> str:
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  Weight Outlier Report  (threshold = {_OUTLIER_SIGMA}σ)")
    lines.append(f"{'='*70}")
    lines.append("")

    # ── Top N individual values ───────────────────────────────────────────────
    lines.append(f"  Top {_TOP_N_OUTLIERS} tensors by maximum |weight|:")
    lines.append(f"  {'Rank':>4}  {'Tensor name':<45}  {'|max|':>10}  {'std':>8}  {'shape'}")
    lines.append("  " + "-" * 82)
    for i, row in df.head(_TOP_N_OUTLIERS).iterrows():
        lines.append(
            f"  {i+1:>4}  {row['name']:<45}  {row['abs_max']:>10.5f}  "
            f"{row['std']:>8.5f}  {row['shape']}"
        )
    lines.append("")

    # ── Layers with high outlier fraction ────────────────────────────────────
    high_out = df[df["outlier_fraction"] > 0.01].sort_values("outlier_fraction", ascending=False)
    lines.append(f"  Layers where > 1% of weights are >{_OUTLIER_SIGMA}σ outliers:")
    if high_out.empty:
        lines.append("    (none)")
    else:
        lines.append(f"  {'Tensor name':<45}  {'out%':>6}  {'out_count':>10}  {'abs_max':>10}")
        lines.append("  " + "-" * 76)
        for _, row in high_out.iterrows():
            lines.append(
                f"  {row['name']:<45}  {row['outlier_fraction']*100:>5.2f}%  "
                f"{row['outlier_count']:>10,}  {row['abs_max']:>10.5f}"
            )
    lines.append("")

    # ── BN running stats ─────────────────────────────────────────────────────
    bn_rows = df[df["name"].str.contains("running_")]
    if not bn_rows.empty:
        lines.append("  BatchNorm running statistics (can be large → drives integer bit width):")
        lines.append(f"  {'Tensor name':<45}  {'min':>10}  {'max':>10}  {'abs_max':>10}")
        lines.append("  " + "-" * 80)
        for _, row in bn_rows.iterrows():
            lines.append(
                f"  {row['name']:<45}  {row['min']:>10.5f}  "
                f"{row['max']:>10.5f}  {row['abs_max']:>10.5f}"
            )
        lines.append("")

    # ── Suspected culprits ────────────────────────────────────────────────────
    abs_max_overall = df["abs_max"].max()
    culprits = df[df["abs_max"] >= 0.5 * abs_max_overall]
    lines.append(f"  Likely drivers of large integer-bit requirement")
    lines.append(f"  (tensors with |max| >= 50% of global max = {abs_max_overall:.5f}):")
    if culprits.empty:
        lines.append("    (none)")
    else:
        for _, row in culprits.iterrows():
            pct = row["abs_max"] / abs_max_overall * 100
            lines.append(f"    • {row['name']}  |max|={row['abs_max']:.5f}  ({pct:.1f}% of global max)")
    lines.append("")
    lines.append(f"{'='*70}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Plots
# ─────────────────────────────────────────────────────────────────────────────

def _plot_weight_distributions(df: pd.DataFrame, model: nn.Module, output_dir: Path) -> Path:
    """Grid of per-layer weight histograms, coloured by |max|."""
    # Only parameters (not buffers) for histograms
    param_names = [r["name"] for _, r in df.iterrows() if r["tensor_type"] == "parameter"]
    n = len(param_names)
    if n == 0:
        return output_dir / "weight_distributions.png"

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    # colour map: red = large abs_max
    abs_maxes = [df.loc[df["name"] == nm, "abs_max"].values[0] for nm in param_names]
    norm = mcolors.Normalize(vmin=0, vmax=max(abs_maxes) or 1)
    cmap = plt.cm.RdYlGn_r

    for ax, name, amx in zip(axes, param_names, abs_maxes):
        # fetch the actual tensor
        tensor = dict(model.named_parameters())[name].detach().float().cpu().flatten().numpy()
        color = cmap(norm(amx))
        ax.hist(tensor, bins=_HIST_BINS, color=color, edgecolor="none", alpha=0.85)
        row = df.loc[df["name"] == name].iloc[0]
        ax.set_title(
            f"{name}\nshape={row['shape']}  |max|={amx:.4f}  std={row['std']:.4f}",
            fontsize=7, pad=3,
        )
        ax.axvline(0, color="black", linewidth=0.7, linestyle="--")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("value", fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.grid(alpha=0.3, linewidth=0.5)

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:n], label="|max weight| (red = large)", shrink=0.6)

    fig.suptitle("Per-Layer Weight Distributions", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = output_dir / "weight_distributions.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {out}")
    return out


def _plot_magnitude_bar(df: pd.DataFrame, output_dir: Path) -> Path:
    """Horizontal bar chart: max |weight| per tensor, ranked."""
    plot_df = df.sort_values("abs_max", ascending=True).tail(40)  # top 40
    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.32)))

    colors = plt.cm.RdYlGn_r(
        np.linspace(0, 1, len(plot_df))
    )
    bars = ax.barh(plot_df["name"], plot_df["abs_max"], color=colors, edgecolor="none")

    ax.set_xlabel("|max weight|", fontsize=11)
    ax.set_title("Maximum Absolute Weight per Tensor\n(top 40, ranked)", fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3, linewidth=0.7)

    # Annotate with tensor type
    for bar, (_, row) in zip(bars, plot_df.iterrows()):
        label = "buf" if row["tensor_type"] == "buffer" else ""
        if label:
            ax.text(
                bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=6, color="white", fontweight="bold",
            )

    fig.tight_layout()
    out = output_dir / "parameter_magnitude_bar.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {out}")
    return out


def _plot_outlier_fraction(df: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart: fraction of outlier weights per layer."""
    plot_df = df[df["tensor_type"] == "parameter"].sort_values("outlier_fraction", ascending=False)
    if plot_df.empty:
        return output_dir / "outlier_fraction_bar.png"

    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.38)))

    threshold_color = "#E84855"
    bar_color = "#2176AE"
    colors = [threshold_color if v > 0.05 else bar_color for v in plot_df["outlier_fraction"]]
    ax.barh(plot_df["name"], plot_df["outlier_fraction"] * 100, color=colors, edgecolor="none")
    ax.axvline(1.0, color="grey", linestyle=":", linewidth=1.2, label="1% threshold")
    ax.axvline(5.0, color=threshold_color, linestyle="--", linewidth=1.2, label="5% threshold")

    ax.set_xlabel(f"Outlier fraction (%) — weights > {_OUTLIER_SIGMA}σ from layer mean", fontsize=10)
    ax.set_title("Per-Layer Outlier Weight Fraction", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3, linewidth=0.7)

    fig.tight_layout()
    out = output_dir / "outlier_fraction_bar.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {out}")
    return out


def _plot_weight_heatmaps(df: pd.DataFrame, model: nn.Module, output_dir: Path) -> Path:
    """
    For each Conv2d weight tensor, show a 2-D heatmap of absolute values.
    Rows = output filters, columns = flattened (in_ch × kH × kW).
    Skips non-conv parameters (bias, BN, linear).
    """
    conv_params = {
        name: param
        for name, param in model.named_parameters()
        if "weight" in name and param.ndim == 4  # Conv2d weights are 4-D
    }
    if not conv_params:
        return output_dir / "weight_heatmaps.png"

    n = len(conv_params)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, param) in zip(axes, conv_params.items()):
        data = param.detach().float().cpu().abs()
        # Reshape to (out_filters, in*kH*kW)
        mat = data.view(data.shape[0], -1).numpy()
        im = ax.imshow(mat, aspect="auto", cmap="hot", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="|weight|", shrink=0.8)
        row = df.loc[df["name"] == name].iloc[0] if name in df["name"].values else None
        title_extra = f"\n|max|={row['abs_max']:.4f}  std={row['std']:.4f}" if row is not None else ""
        ax.set_title(f"{name}\nshape={tuple(data.shape)}{title_extra}", fontsize=8)
        ax.set_xlabel("in_ch × kH × kW (flattened)", fontsize=8)
        ax.set_ylabel("out filters", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle("Conv2d |Weight| Heatmaps\n(bright = large magnitude)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "weight_heatmaps.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {out}")
    return out


def _plot_layer_stats_table(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Render the stats DataFrame as a matplotlib table image for quick at-a-glance review.
    Shows only parameter rows (not buffers) for readability.
    """
    show_cols = ["name", "shape", "numel", "min", "max", "abs_max", "mean", "std",
                 "outlier_fraction", "q01", "median", "q99"]
    plot_df = df[df["tensor_type"] == "parameter"][show_cols].copy()
    # Shorten names for display
    plot_df["name"] = plot_df["name"].apply(lambda s: s[-38:] if len(s) > 38 else s)

    n_rows = len(plot_df)
    fig_h = max(2.5, 0.38 * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.axis("off")

    col_labels = show_cols
    cell_text  = plot_df.values.tolist()
    cell_text  = [[str(v) for v in row] for row in cell_text]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Colour header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2176AE")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight rows with large abs_max (top quartile)
    abs_maxes = [float(df[df["tensor_type"] == "parameter"].iloc[i]["abs_max"])
                 for i in range(n_rows)]
    if abs_maxes:
        threshold = np.percentile(abs_maxes, 75)
        for i, amx in enumerate(abs_maxes):
            if amx >= threshold:
                for j in range(len(col_labels)):
                    tbl[(i + 1, j)].set_facecolor("#FFF3CD")  # light yellow

    fig.suptitle("Per-Layer Weight Statistics (parameters only)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = output_dir / "layer_stats_table.png"
    fig.savefig(out, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Analysis] Saved → {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyse_model(
    model: nn.Module,
    output_dir: Path | str = "results/quantization/model_analysis",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full model weight analysis.  Call this before the quantization sweep.

    Parameters
    ----------
    model      : SpikingEEGNet (or any nn.Module)
    output_dir : Where to write all output files.
    verbose    : Print the architecture summary and outlier report to stdout.

    Returns
    -------
    pd.DataFrame — the per-layer stats table (same as layer_weight_stats.csv).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    print("\n" + "="*60)
    print("  Model Weight Analysis")
    print("="*60)

    # ── 1. Architecture summary ───────────────────────────────────────────────
    arch_str = _architecture_summary(model)
    arch_path = output_dir / "model_architecture.txt"
    arch_path.write_text(arch_str)
    print(f"[Analysis] Architecture summary → {arch_path}")
    if verbose:
        print("\n" + arch_str)

    # ── 2. Collect per-tensor stats ───────────────────────────────────────────
    df = _collect_tensor_stats(model)
    csv_path = output_dir / "layer_weight_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Analysis] Layer stats CSV     → {csv_path}")

    # ── 3. Outlier report ─────────────────────────────────────────────────────
    report_str = _outlier_report(df, model)
    report_path = output_dir / "weight_outlier_report.txt"
    report_path.write_text(report_str)
    print(f"[Analysis] Outlier report      → {report_path}")
    if verbose:
        print("\n" + report_str)

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n[Analysis] Generating plots …")
    _plot_weight_distributions(df, model, output_dir)
    _plot_magnitude_bar(df, output_dir)
    _plot_outlier_fraction(df, output_dir)
    _plot_weight_heatmaps(df, model, output_dir)
    _plot_layer_stats_table(df, output_dir)

    print(f"\n[Analysis] Complete.  All outputs in {output_dir}/")
    print("="*60 + "\n")

    return df