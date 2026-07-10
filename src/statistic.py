"""
Publication-ready analysis of Optuna/SNN-EEGNet hyperparameter trial results.

Reads a trials.csv produced by pipeline.py and produces figures suitable for
direct inclusion in a research paper (expanded variable names, proper title
casing, 300 DPI output, consistent serif typography).

Outputs:
  1. pearson_correlation_matrix.png     - correlation heatmap (numeric vars + target)
  2. scatter_numeric_vs_target.png      - target vs each numeric hyperparameter
  3. boxplot_categorical_vs_target.png  - target by categorical/boolean params
                                           (only params with >=2 observed levels)
  4. optimization_progress.png          - target per trial + running best (convergence)
  5. feature_importance.png             - Random Forest importance of each param
  6. top_trials_param_distributions.png - where top-N trials sit vs full search range
  7. target_distribution.png            - NEW: histogram/KDE of the target across all trials
  8. importance_vs_correlation.png      - NEW: RF importance vs |Pearson r|, flags
                                           non-linear/interaction-driven parameters
  9. summary_stats.csv                  - describe() of all columns
 10. recommendations.txt                - plain-text written summary

If a `results_baseline.json` and/or `results_snn.json` file is found
alongside the trials csv, four additional baseline-comparison figures are
produced (each baseline found is overlaid on the same figures --
`results_baseline.json` is treated as the primary "Baseline" and
`results_snn.json`, if present, is plotted alongside it as an "Untuned SNN"
baseline, i.e. a plain SpikingEEGNet run without the Optuna study):
 11. baseline_vs_trials_summary.png      - NEW: baseline vs median/mean/top-N/best
 12. baseline_vs_trials_distribution.png - NEW: target histogram with baseline marked,
                                            % of trials that beat the baseline
 13. top_trials_vs_baseline.png          - NEW: each top-N trial vs baseline, %-delta labels
 14. baseline_vs_best_param_diff.png     - NEW: baseline vs best-trial hyperparameters

Usage:
    python statistic.py [csv_path] [out_dir] [top_n]
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


# ----------------------------------------------------------------------
# Publication-style global plot settings
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


# ----------------------------------------------------------------------
# Known search-space bounds (from main.py / spiking_eegnet.py docstrings)
# ----------------------------------------------------------------------
SEARCH_SPACE = {
    "lr_exp":                (-4.5, -2.0),
    "dropout":               (0.1, 0.75),
    "beta":                  (0.5, 0.99),
    "spike_grad_slope":      (5.0, 100.0),
    "temporal_filters":      (4, 32),
    "depth_multiplier":      (1, 4),
    "pointwise_filters":     (8, 64),
    "temporal_kernel_div":   (2, 8),
    "separable_kernel_size": (4, 32),
    "pool1_size":            (2, 8),
    "pool2_size":            (2, 8),
}

# Columns that are categorical/boolean by construction (not numeric ranges)
CATEGORICAL_HINTS = {"run_zscore", "norm_axis", "readout_mode"}

# ----------------------------------------------------------------------
# Human-readable names for every variable that can appear in trials.csv.
# Anything not listed falls back to an automatic "snake_case -> Title Case"
# conversion (see `pretty()` below), so new/unknown columns still render
# reasonably instead of erroring out.
# ----------------------------------------------------------------------
PRETTY_NAMES = {
    "mean_bal_acc":          "Mean Balanced Accuracy",
    "acc_loso":              "Leave-One-Subject-Out Accuracy",
    "trial_number":          "Trial Number",
    "lr":                    "Learning Rate",
    "lr_exp":                "Learning Rate Exponent (log10)",
    "dropout":                "Dropout Rate",
    "beta":                  "LIF Decay Rate (Beta)",
    "spike_grad_slope":      "Spike Gradient Slope",
    "temporal_filters":      "Number of Temporal Filters",
    "depth_multiplier":      "Depth Multiplier",
    "pointwise_filters":     "Number of Pointwise Filters",
    "temporal_kernel_div":   "Temporal Kernel Divisor",
    "separable_kernel_size": "Separable Kernel Size",
    "pool1_size":            "Pooling Layer 1 Size",
    "pool2_size":            "Pooling Layer 2 Size",
    "n_steps_train":         "Number of Training Time Steps",
    "n_steps_eval":          "Number of Evaluation Time Steps",
    "run_zscore":            "Z-Score Normalization Enabled",
    "norm_axis":             "Normalization Axis",
    "readout_mode":          "Readout Mode",
}

# Categorical level labels that deserve expansion (applied after pretty())
LEVEL_NAMES = {
    "spk_last": "Spike (Last Step)",
    "spk_mean": "Spike (Mean)",
    "spk_sum":  "Spike (Sum)",
    "mem_last": "Membrane (Last Step)",
    "True":     "True",
    "False":    "False",
}


def pretty(col: str) -> str:
    """Expand a raw column name into a publication-quality label."""
    if col in PRETTY_NAMES:
        return PRETTY_NAMES[col]
    return col.replace("_", " ").strip().title()


def pretty_level(val: str) -> str:
    val = str(val)
    return LEVEL_NAMES.get(val, val.replace("_", " ").title())


def identify_column_types(df_analysis, target):
    numeric_cols = []
    categorical_cols = []
    for col in df_analysis.columns:
        if col == target:
            continue
        if col in CATEGORICAL_HINTS:
            categorical_cols.append(col)
            continue
        coerced = pd.to_numeric(df_analysis[col], errors="coerce")
        if coerced.notna().sum() >= 0.9 * len(df_analysis) and df_analysis[col].dtype != bool:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    for col in list(numeric_cols):
        unique_vals = set(df_analysis[col].dropna().unique())
        if unique_vals.issubset({True, False, "True", "False", 0, 1}) and df_analysis[col].dtype == object:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    constant_cols = [c for c in numeric_cols
                      if pd.to_numeric(df_analysis[c], errors="coerce").nunique(dropna=True) <= 1]
    numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    return numeric_cols, categorical_cols, constant_cols


def plot_correlation_matrix(df_analysis, numeric_cols, target, out_dir, method="pearson"):
    corr_cols = numeric_cols + [target]
    corr_df = df_analysis[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = corr_df.corr(method=method)

    labels = [pretty(c) for c in corr_matrix.columns]
    plt.figure(figsize=(max(9, 0.55 * len(corr_cols)), max(7, 0.55 * len(corr_cols))))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8, "label": f"{method.title()} Correlation Coefficient"},
                annot_kws={"size": 7}, xticklabels=labels, yticklabels=labels)
    plt.title(f"{method.title()} Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fname = f"{method.lower()}_correlation_matrix.png"
    path = os.path.join(out_dir, fname)
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved correlation matrix -> {path}")

    target_corr = corr_matrix[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
    return target_corr


def plot_scatter_numeric(df_analysis, numeric_cols, target, target_corr, out_dir):
    n = len(numeric_cols)
    if n == 0:
        print("No numeric variables for scatter plots.")
        return
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        x = pd.to_numeric(df_analysis[col], errors="coerce")
        y = pd.to_numeric(df_analysis[target], errors="coerce")
        ax.scatter(x, y, alpha=0.7, s=22, color="steelblue", edgecolor="white", linewidth=0.3)

        valid = x.notna() & y.notna()
        if valid.sum() >= 3 and x[valid].nunique() > 1:
            z = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(x[valid].min(), x[valid].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color="firebrick", linewidth=1.3, linestyle="--", alpha=0.8)

        r = target_corr.get(col, np.nan)
        ax.set_title(f"{pretty(col)} (r = {r:.2f})", fontsize=9.5)
        ax.set_xlabel(pretty(col), fontsize=8.5)
        ax.set_ylabel(pretty(target), fontsize=8.5)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Hyperparameters vs. {pretty(target)}", y=1.01, fontsize=14)
    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_numeric_vs_target.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved scatter plots -> {path}")


def plot_boxplot_categorical(df_analysis, categorical_cols, target, out_dir):
    """Only plots categorical/boolean params that actually vary (>=2 observed levels).
    Single-level (constant) params are skipped -- they carry no information for a
    'X vs target' comparison and just waste space in a paper figure."""
    plot_cols = [c for c in categorical_cols if df_analysis[c].astype(str).nunique(dropna=True) >= 2]
    skipped = [c for c in categorical_cols if c not in plot_cols]
    if skipped:
        print(f"Skipping single-level categorical columns (no variation): {skipped}")
    if not plot_cols:
        print("No categorical/boolean variables with multiple levels found for box plots.")
        return

    n = len(plot_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        data = df_analysis[[col, target]].copy()
        data[col] = data[col].astype(str).map(pretty_level)
        data[target] = pd.to_numeric(data[target], errors="coerce")

        sns.boxplot(data=data, x=col, y=target, ax=ax, color="#4C72B0")
        sns.stripplot(
            data=data, x=col, y=target,
            ax=ax, color="black", alpha=0.5, size=4
        )

        # Compute statistics
        stats = data.groupby(col)[target].agg(["mean", "median"])

        # Keep x tick labels simple
        categories = list(stats.index)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45)

        # Add mean/median beneath each category
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin

        for j, category in enumerate(categories):
            ax.text(
                j,
                ymin + 0.02 * yrange,  # slightly above bottom of plot
                f"mean={stats.loc[category, 'mean']:.3f}\n"
                f"median={stats.loc[category, 'median']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8)
            )

        ax.set_title(f"{pretty(col)} vs. {pretty(target)}", fontsize=10.5)
        ax.set_xlabel(pretty(col))
        ax.set_ylabel(pretty(target))

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "boxplot_categorical_vs_target.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved box plots -> {path}")


def plot_optimization_progress(df, target, out_dir):
    data = df.copy()
    if "trial_number" in data.columns and pd.to_numeric(data["trial_number"], errors="coerce").notna().all():
        data = data.sort_values("trial_number")
        x = pd.to_numeric(data["trial_number"], errors="coerce")
        xlabel = pretty("trial_number")
    else:
        data = data.reset_index(drop=True)
        x = data.index
        xlabel = "Trial Order (Row Index)"

    y = pd.to_numeric(data[target], errors="coerce")
    running_best = y.cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, alpha=0.55, s=28, label=pretty(target), color="steelblue", edgecolor="white", linewidth=0.3)
    ax.plot(x, running_best, color="darkorange", linewidth=2.2, label="Running Best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(pretty(target))
    ax.set_title(f"Optimization Progress: {pretty(target)} per Trial and Running Best")
    ax.legend()
    ax.grid(alpha=0.3)

    best_idx = y.idxmax()
    ax.annotate(
        f"Best = {y[best_idx]:.4f}",
        xy=(x[best_idx], y[best_idx]),
        xytext=(10, -15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="grey"),
        fontsize=9,
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "optimization_progress.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved optimization progress plot -> {path}")

    return running_best.iloc[-1], y.idxmax()


def compute_feature_importance(df_analysis, numeric_cols, categorical_cols, target, out_dir):
    feature_cols = numeric_cols + categorical_cols
    if len(feature_cols) == 0:
        print("No features available for importance analysis.")
        return None

    X = df_analysis[feature_cols].copy()
    for col in categorical_cols:
        X[col] = X[col].astype(str)
    X = pd.get_dummies(X, columns=[c for c in categorical_cols if c in X.columns], drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df_analysis[target], errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()
    X, y = X[valid], y[valid]

    if len(X) < 8:
        print(f"Only {len(X)} valid rows; skipping feature importance (need >= 8).")
        return None

    rf = RandomForestRegressor(n_estimators=300, random_state=0, max_depth=None)
    rf.fit(X, y)

    result = permutation_importance(rf, X, y, n_repeats=30, random_state=0, n_jobs=-1)
    importance = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    importance_std = pd.Series(result.importances_std, index=X.columns)

    def pretty_feature(name):
        for col in categorical_cols:
            prefix = f"{col}_"
            if name.startswith(prefix):
                return f"{pretty(col)}: {pretty_level(name[len(prefix):])}"
        return pretty(name)

    display_labels = [pretty_feature(c) for c in importance.index]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.38 * len(importance))))
    ax.barh(display_labels, importance.values, xerr=importance_std[importance.index].values, color="seagreen")
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (Drop in R\u00b2 When Shuffled)")
    ax.set_title(f"Random Forest Feature Importance for {pretty(target)}")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved feature importance plot -> {path}")

    r2 = rf.score(X, y)
    print(f"RandomForest in-sample R^2 on {target}: {r2:.3f} (n={len(X)})")

    return importance


def plot_top_trials_param_distributions(df_analysis, numeric_cols, target, top_n, out_dir):
    y = pd.to_numeric(df_analysis[target], errors="coerce")
    top_idx = y.nlargest(min(top_n, len(y))).index

    plot_cols = [c for c in numeric_cols if c in SEARCH_SPACE]
    if not plot_cols:
        print("No recognized search-space columns for top-trial distribution plots.")
        return top_idx

    ncols = 4
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        vals = pd.to_numeric(df_analysis[col], errors="coerce")
        top_vals = vals.loc[top_idx]

        sns.histplot(vals, ax=ax, color="lightgrey", label="All Trials",
                      bins=12, kde=False, stat="count")
        sns.histplot(top_vals, ax=ax, color="crimson", label=f"Top {len(top_idx)}",
                      bins=12, kde=False, stat="count", alpha=0.7)

        lo, hi = SEARCH_SPACE[col]
        ax.axvline(lo, color="black", linestyle=":", linewidth=1)
        ax.axvline(hi, color="black", linestyle=":", linewidth=1)

        ax.set_title(pretty(col), fontsize=9.5)
        ax.set_xlabel(pretty(col), fontsize=8)
        ax.set_ylabel("Count")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(plot_cols), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Parameter Distributions: All Trials vs. Top {len(top_idx)} by {pretty(target)}", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "top_trials_param_distributions.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved top-trial distribution plot -> {path}")

    return top_idx


def plot_target_distribution(df_analysis, target, out_dir):
    """NEW: distribution of the target metric across all trials.
    Useful in a paper to characterize search variance / how 'lucky' the
    best trial is relative to the overall spread."""
    y = pd.to_numeric(df_analysis[target], errors="coerce").dropna()
    if y.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(y, bins=15, kde=True, color="steelblue", ax=ax, edgecolor="white")
    ax.axvline(y.mean(), color="darkorange", linestyle="--", linewidth=2, label=f"Mean = {y.mean():.4f}")
    ax.axvline(y.median(), color="seagreen", linestyle=":", linewidth=2, label=f"Median = {y.median():.4f}")
    ax.axvline(y.max(), color="firebrick", linestyle="-", linewidth=1.5, label=f"Best = {y.max():.4f}")
    ax.set_xlabel(pretty(target))
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {pretty(target)} Across All Trials (n = {len(y)})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "target_distribution.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved target distribution plot -> {path}")


def plot_importance_vs_correlation(importance, target_corr, target, out_dir):
    """NEW: compares Random Forest permutation importance against linear
    |Pearson r| for each numeric parameter. Points far above the diagonal
    trend indicate parameters with a non-linear / interaction-driven effect
    that a simple correlation table would miss -- a useful figure to justify
    using a non-linear importance method in the paper's methodology."""
    if importance is None or target_corr is None:
        return

    common = [c for c in target_corr.index if c in importance.index]
    if len(common) < 3:
        print("Not enough overlapping parameters for importance-vs-correlation plot.")
        return

    abs_r = target_corr[common].abs()
    imp = importance[common]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.scatter(abs_r, imp, s=60, color="seagreen", edgecolor="black", linewidth=0.5, zorder=3)

    # Pad the axes so adjustText has room to push labels without them
    # falling outside the plot area.
    x_pad = 0.12 * (abs_r.max() - abs_r.min() + 1e-9)
    y_pad = 0.12 * (imp.max() - imp.min() + 1e-9)
    ax.set_xlim(abs_r.min() - x_pad, abs_r.max() + x_pad)
    ax.set_ylim(imp.min() - y_pad, imp.max() + y_pad)

    texts = [
        ax.text(abs_r[c], imp[c], pretty(c), fontsize=8.5)
        for c in common
    ]

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, ax=ax,
            x=list(abs_r.values), y=list(imp.values),
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.6, alpha=0.7),
            expand_text=(1.15, 1.3), expand_points=(1.3, 1.5),
            force_text=(0.6, 0.8), force_points=(0.4, 0.6),
        )
    except ImportError:
        print("adjustText not installed (pip install adjustText) -- "
              "falling back to simple offset labels, which may overlap.")
        for t, c in zip(texts, common):
            t.set_position((0, 0))
            ax.annotate(pretty(c), (abs_r[c], imp[c]), fontsize=8,
                        xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel("Absolute Pearson Correlation Coefficient |r|")
    ax.set_ylabel("Random Forest Permutation Importance")
    ax.set_title(f"Feature Importance vs. Linear Correlation for {pretty(target)}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "importance_vs_correlation.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved importance-vs-correlation plot -> {path}")


# ----------------------------------------------------------------------
# Baseline comparison (results_baseline.json, results_snn.json)
# ----------------------------------------------------------------------

# Every baseline JSON we know how to look for, in the same folder as
# trials.csv, paired with the label it should carry in plots/reports.
BASELINE_SPECS = [
    ("results_baseline.json", "Baseline"),
    ("results_snn.json",      "Untuned SNN"),
]


def load_baseline(csv_path, target, filename="results_baseline.json", label="Baseline"):
    """Look for `filename` in the same folder as trials.csv (i.e. alongside
    the other per-trial result files) and pull out its target metric plus
    any recognized hyperparameters.

    The JSON's exact shape isn't assumed -- this searches recursively for
    a key matching `target`, falling back to a few common alias names
    (value/score/result/accuracy/...) if `target` itself isn't present.
    Returns None (and prints why) if no baseline file / no numeric metric
    is found, so downstream plotting functions can just no-op.
    """
    baseline_dir = os.path.dirname(os.path.abspath(csv_path)) or "."
    baseline_path = os.path.join(baseline_dir, filename)
    if not os.path.exists(baseline_path):
        print(f"No baseline file found at {baseline_path} -- skipping '{label}' baseline comparison.")
        return None

    with open(baseline_path, "r") as f:
        data = json.load(f)

    def find_value(d, key):
        if isinstance(d, dict):
            if key in d and isinstance(d[key], (int, float)) and not isinstance(d[key], bool):
                return d[key]
            for v in d.values():
                found = find_value(v, key)
                if found is not None:
                    return found
        return None

    value = find_value(data, target)
    if value is None:
        for alt in ("value", "score", "objective_value", "result", "acc", "accuracy", "mean_bal_acc"):
            value = find_value(data, alt)
            if value is not None:
                break

    if value is None:
        print(f"Could not find a numeric '{target}' (or common alias) in {baseline_path} -- "
              f"skipping '{label}' baseline comparison.")
        return None

    # Best-effort: also collect any recognized search-space hyperparameters
    # so we can plot a baseline-vs-best parameter diff.
    params = {}
    def collect_params(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in SEARCH_SPACE and isinstance(v, (int, float)) and not isinstance(v, bool):
                    params.setdefault(k, v)
                elif isinstance(v, dict):
                    collect_params(v)
    collect_params(data)

    print(f"Loaded '{label}' baseline from {baseline_path}: {pretty(target)} = {value:.4f}"
          + (f" (+{len(params)} recognized hyperparameters)" if params else ""))
    return {"path": baseline_path, "value": value, "params": params, "raw": data, "label": label}


def load_all_baselines(csv_path, target):
    """Load every baseline JSON listed in BASELINE_SPECS that's present
    alongside trials.csv (e.g. `results_baseline.json` and, if present,
    `results_snn.json` for an untuned/no-Optuna SNN run). Returns a list
    (possibly empty) so all baseline plots can iterate over however many
    baselines were actually found.
    """
    baselines = []
    for filename, label in BASELINE_SPECS:
        b = load_baseline(csv_path, target, filename=filename, label=label)
        if b is not None:
            baselines.append(b)
    return baselines


def plot_baseline_summary(df_analysis, target, baselines, top_n, out_dir):
    """NEW: single bar chart -- every known baseline (e.g. the non-spiking
    baseline plus an untuned/no-Optuna SNN baseline, if present) vs.
    median/mean/top-N/best trial. Trial-summary bars show their %-change
    vs. the primary (first) baseline."""
    if not baselines:
        return
    y = pd.to_numeric(df_analysis[target], errors="coerce").dropna()
    if y.empty:
        return

    primary_val = baselines[0]["value"]
    n_top = min(top_n, len(y))
    top_avg = y.nlargest(n_top).mean()

    baseline_labels = [b.get("label", "Baseline") for b in baselines]
    baseline_values = [b["value"] for b in baselines]
    baseline_palette = ["#7f7f7f", "#9467bd", "#8c564b", "#e377c2"]
    baseline_colors = [baseline_palette[i % len(baseline_palette)] for i in range(len(baselines))]

    labels = baseline_labels + ["Median\nTrial", "Mean\nTrial", f"Top-{n_top}\nAvg", "Best\nTrial"]
    values = baseline_values + [y.median(), y.mean(), top_avg, y.max()]
    colors = baseline_colors + ["#4C72B0", "#4C72B0", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(labels)), 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(primary_val, color="#7f7f7f", linestyle="--", linewidth=1, alpha=0.7, zorder=0)

    for bar, val, label in zip(bars, values, labels):
        text = f"{val:.4f}"
        if primary_val != 0:
            if label not in baseline_labels:
                pct = 100 * (val - primary_val) / primary_val
                text += f"\n({pct:+.1f}%)"
            elif label != baseline_labels[0]:
                pct = 100 * (val - primary_val) / primary_val
                text += f"\n({pct:+.1f}% vs {baseline_labels[0]})"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), text,
                ha="center", va="bottom", fontsize=8.5)

    ax.set_ylabel(pretty(target))
    ax.set_title(f"{pretty(target)}: Baseline vs. HPO Trial Summary")
    ax.set_ylim(0, max(values) * 1.2)
    plt.tight_layout()
    path = os.path.join(out_dir, "baseline_vs_trials_summary.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved baseline summary comparison -> {path}")


def plot_baseline_distribution(df_analysis, target, baselines, out_dir):
    """NEW: target histogram/KDE across all trials with every known baseline
    marked, reporting what fraction of trials actually beat each one."""
    if not baselines:
        return
    y = pd.to_numeric(df_analysis[target], errors="coerce").dropna()
    if y.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.histplot(y, bins=15, kde=True, color="steelblue", ax=ax, edgecolor="white")

    baseline_palette = ["black", "#9467bd", "#8c564b", "#e377c2"]
    beat_summaries = []
    for i, b in enumerate(baselines):
        base_val = b["value"]
        label = b.get("label", "Baseline")
        color = baseline_palette[i % len(baseline_palette)]
        n_beat = int((y > base_val).sum())
        pct_beat = 100 * n_beat / len(y)
        ax.axvline(base_val, color=color, linewidth=2.2, label=f"{label} = {base_val:.4f}")
        beat_summaries.append(f"{n_beat}/{len(y)} ({pct_beat:.1f}%) beat {label}")

    ax.axvline(y.mean(), color="darkorange", linestyle="--", linewidth=1.6, label=f"Trial Mean = {y.mean():.4f}")
    ax.axvline(y.max(), color="firebrick", linestyle=":", linewidth=1.6, label=f"Best Trial = {y.max():.4f}")

    ax.set_xlabel(pretty(target))
    ax.set_ylabel("Count")
    ax.set_title(f"{pretty(target)} Distribution vs. Baseline(s)\n" + "; ".join(beat_summaries))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "baseline_vs_trials_distribution.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved baseline distribution comparison -> {path}")


def plot_top_trials_vs_baseline(df, target, baselines, top_idx, out_dir):
    """NEW: per-trial bar chart of the top-N trials against the primary
    baseline (bar color/%-delta), with every other known baseline (e.g.
    an untuned/no-Optuna SNN baseline) drawn as an additional reference
    line so all baselines are visible on the same figure."""
    if not baselines:
        return
    primary = baselines[0]
    base_val = primary["value"]
    primary_label = primary.get("label", "Baseline")

    top = df.loc[top_idx].copy()
    top[target] = pd.to_numeric(top[target], errors="coerce")
    top = top.sort_values(target, ascending=False)

    if "trial_number" in top.columns:
        xlabels = [f"Trial {int(t)}" for t in top["trial_number"]]
    else:
        xlabels = [f"Trial {i}" for i in top.index]

    vals = top[target].values
    colors = ["seagreen" if v > base_val else "firebrick" for v in vals]

    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(vals)), 5.5))
    bars = ax.bar(xlabels, vals, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axhline(base_val, color="black", linestyle="--", linewidth=2, label=f"{primary_label} = {base_val:.4f}")

    extra_palette = ["#9467bd", "#8c564b", "#e377c2"]
    for i, b in enumerate(baselines[1:]):
        ax.axhline(b["value"], color=extra_palette[i % len(extra_palette)], linestyle=":",
                   linewidth=2, label=f"{b.get('label', 'Baseline')} = {b['value']:.4f}")

    for bar, v in zip(bars, vals):
        pct = 100 * (v - base_val) / base_val if base_val != 0 else np.nan
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{pct:+.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel(pretty(target))
    ax.set_title(f"Top {len(vals)} Trials vs. Baseline ({pretty(target)})")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(out_dir, "top_trials_vs_baseline.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved top-trials-vs-baseline plot -> {path}")


def plot_baseline_param_diff(df_analysis, numeric_cols, target, baselines, out_dir):
    """NEW: every baseline's hyperparameters vs. the best-trial's, normalized
    to each parameter's search-space range so differing units/scales sit on
    one comparable [0, 1] axis. Highlights exactly which settings the best
    trial changed relative to each baseline (and by how much)."""
    baselines_with_params = [b for b in (baselines or []) if b.get("params")]
    if not baselines_with_params:
        if baselines:
            print("No baseline JSON has recognized hyperparameters -- skipping param-diff plot.")
        return

    plot_cols = [c for c in numeric_cols if c in SEARCH_SPACE
                 and any(c in b["params"] for b in baselines_with_params)]
    if not plot_cols:
        print("No overlap between baseline hyperparameters and trial columns -- skipping param-diff plot.")
        return

    y = pd.to_numeric(df_analysis[target], errors="coerce")
    best_idx = y.idxmax()

    x = np.arange(len(plot_cols))
    n_groups = len(baselines_with_params) + 1  # one bar per baseline, plus Best Trial
    width = 0.8 / n_groups
    palette = ["#7f7f7f", "#9467bd", "#8c564b", "#e377c2"]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(plot_cols)), 5.5))

    def group_offset(group_idx):
        return (group_idx - (n_groups - 1) / 2) * width

    for i, b in enumerate(baselines_with_params):
        offset = group_offset(i)
        positions, norms, raws = [], [], []
        for j, c in enumerate(plot_cols):
            if c not in b["params"]:
                continue
            lo, hi = SEARCH_SPACE[c]
            rng = (hi - lo) if hi != lo else 1
            val = b["params"][c]
            positions.append(x[j] + offset)
            norms.append((val - lo) / rng)
            raws.append(val)
        bars = ax.bar(positions, norms, width, label=b.get("label", f"Baseline {i + 1}"),
                       color=palette[i % len(palette)], edgecolor="black")
        for bar, raw in zip(bars, raws):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{raw:.3g}",
                    ha="center", va="bottom", fontsize=7.5)

    # Best-trial bar group (always the last group)
    offset = group_offset(n_groups - 1)
    best_positions, best_norms, best_raws = [], [], []
    for j, c in enumerate(plot_cols):
        lo, hi = SEARCH_SPACE[c]
        rng = (hi - lo) if hi != lo else 1
        t_val = pd.to_numeric(pd.Series([df_analysis.loc[best_idx, c]]), errors="coerce").iloc[0]
        best_positions.append(x[j] + offset)
        best_norms.append((t_val - lo) / rng)
        best_raws.append(t_val)
    bars_t = ax.bar(best_positions, best_norms, width, label="Best Trial", color="#C44E52", edgecolor="black")
    for bar, raw in zip(bars_t, best_raws):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{raw:.3g}",
                ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty(c) for c in plot_cols], rotation=45, ha="right")
    ax.set_ylabel("Normalized Position in Search Range [0, 1]")
    ax.set_title("Baseline(s) vs. Best Trial: Key Hyperparameter Differences")
    ax.legend()
    ax.set_ylim(-0.05, 1.2)
    plt.tight_layout()
    path = os.path.join(out_dir, "baseline_vs_best_param_diff.png")
    plt.savefig(path, **SAVE_KW)
    plt.close()
    print(f"Saved baseline-vs-best parameter diff plot -> {path}")


def write_recommendations(df, df_analysis, target, numeric_cols, categorical_cols, constant_cols,
                          target_corr, importance, top_idx, out_dir, top_n, baselines=None):
    lines = []
    y = pd.to_numeric(df_analysis[target], errors="coerce")

    lines.append("=" * 70)
    lines.append("HYPERPARAMETER TRIAL ANALYSIS - RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total trials analyzed: {len(df)}")
    lines.append(f"Best {pretty(target)}: {y.max():.4f}  (trial index {y.idxmax()})")
    lines.append(f"Median {pretty(target)}: {y.median():.4f}")
    lines.append(f"Worst {pretty(target)}: {y.min():.4f}")
    lines.append("")

    if baselines:
        lines.append("-" * 70)
        lines.append("BASELINE COMPARISON")
        lines.append("-" * 70)
        for baseline in baselines:
            base_val = baseline["value"]
            label = baseline.get("label", "Baseline")
            n_beat = int((y > base_val).sum())
            lines.append(f"[{label}]  (source: {baseline['path']})")
            lines.append(f"  {label} {pretty(target)}: {base_val:.4f}")
            lines.append(f"  Best trial vs. {label}:   {y.max():.4f}  "
                          f"({100 * (y.max() - base_val) / base_val:+.1f}%)")
            lines.append(f"  Median trial vs. {label}: {y.median():.4f}  "
                          f"({100 * (y.median() - base_val) / base_val:+.1f}%)")
            lines.append(f"  Trials that beat {label}: {n_beat}/{len(y)} ({100 * n_beat / len(y):.1f}%)")
            if baseline.get("params"):
                best_idx_local = y.idxmax()
                lines.append(f"  {label} vs. best-trial hyperparameters that differ:")
                for col, b_val in baseline["params"].items():
                    if col not in df_analysis.columns:
                        continue
                    t_val = pd.to_numeric(pd.Series([df_analysis.loc[best_idx_local, col]]), errors="coerce").iloc[0]
                    if pd.notna(t_val) and t_val != b_val:
                        lines.append(f"      {pretty(col):35s} {label}={b_val:g}  best_trial={t_val:g}")
            lines.append("")

        if len(baselines) > 1:
            primary = baselines[0]
            primary_label = primary.get("label", "Baseline")
            lines.append(f"Baseline-to-baseline comparison (vs. {primary_label}):")
            for b in baselines[1:]:
                delta = (100 * (b["value"] - primary["value"]) / primary["value"]
                         if primary["value"] != 0 else float("nan"))
                lines.append(f"  {b.get('label', 'Baseline')}: {b['value']:.4f}  "
                              f"vs. {primary_label}: {primary['value']:.4f}  ({delta:+.1f}%)")
            lines.append("")

    lines.append("-" * 70)
    lines.append(f"TOP {min(top_n, len(df))} TRIALS")
    lines.append("-" * 70)
    cols_to_show = (["trial_number"] if "trial_number" in df.columns else []) + [target] + numeric_cols + categorical_cols
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    top_table = df.loc[top_idx, cols_to_show].sort_values(target, ascending=False)
    lines.append(top_table.to_string(index=False))
    lines.append("")

    lines.append("-" * 70)
    lines.append(f"CORRELATION WITH {pretty(target)} (sorted by |r|)")
    lines.append("-" * 70)
    for var, r in target_corr.items():
        direction = "higher is better" if r > 0 else "lower is better"
        strength = "strong" if abs(r) >= 0.4 else ("moderate" if abs(r) >= 0.2 else "weak")
        lines.append(f"  {pretty(var):35s} r = {r:+.3f}  ({strength}, {direction})")
    lines.append("")

    if importance is not None:
        lines.append("-" * 70)
        lines.append("RANDOM FOREST PERMUTATION IMPORTANCE (non-linear effects)")
        lines.append("-" * 70)
        for var, imp in importance.head(10).items():
            lines.append(f"  {pretty(var):35s} importance = {imp:+.4f}")
        lines.append("")
        lines.append("Note: importance can pick up non-linear / interaction effects that")
        lines.append("the linear correlation above misses. Compare both rankings -- params")
        lines.append("that rank high in importance but low in |r| likely have a 'sweet spot'")
        lines.append("in the middle of their range rather than a monotonic relationship.")
        lines.append("")

    lines.append("-" * 70)
    lines.append(f"SUGGESTED SEARCH SPACE FOR A FOLLOW-UP STUDY")
    lines.append(f"(based on the {len(top_idx)} top trials' parameter values)")
    lines.append("-" * 70)
    for col in numeric_cols:
        if col not in SEARCH_SPACE:
            continue
        lo_search, hi_search = SEARCH_SPACE[col]
        top_vals = pd.to_numeric(df_analysis.loc[top_idx, col], errors="coerce").dropna()
        if top_vals.empty:
            continue
        t_lo, t_hi = top_vals.min(), top_vals.max()

        is_int = col in {"temporal_filters", "depth_multiplier", "pointwise_filters",
                          "temporal_kernel_div", "separable_kernel_size",
                          "pool1_size", "pool2_size"}

        if is_int:
            sug_lo, sug_hi = int(np.floor(t_lo)), int(np.ceil(t_hi))
        else:
            pad = 0.1 * (hi_search - lo_search)
            sug_lo, sug_hi = max(lo_search, t_lo - pad), min(hi_search, t_hi + pad)

        line = f"  {pretty(col):35s} top trials span [{t_lo:.4g}, {t_hi:.4g}]" \
               f" -> try [{sug_lo:.4g}, {sug_hi:.4g}]  (full range was [{lo_search:.4g}, {hi_search:.4g}])"

        rng = hi_search - lo_search
        near_lo = abs(t_lo - lo_search) < 0.05 * rng
        near_hi = abs(t_hi - hi_search) < 0.05 * rng
        if near_lo and near_hi:
            line += "   [FLAG: top trials span nearly the FULL range -- param may not matter much]"
        elif near_lo:
            line += "   [FLAG: top trials cluster near the LOWER bound -- consider extending range downward]"
        elif near_hi:
            line += "   [FLAG: top trials cluster near the UPPER bound -- consider extending range upward]"

        lines.append(line)
    lines.append("")

    if categorical_cols:
        lines.append("-" * 70)
        lines.append("CATEGORICAL / BOOLEAN PARAMETERS")
        lines.append("-" * 70)
        for col in categorical_cols:
            data = df_analysis[[col, target]].copy()
            data[col] = data[col].astype(str)
            data[target] = pd.to_numeric(data[target], errors="coerce")
            grouped = data.groupby(col)[target].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
            lines.append(f"  {pretty(col)}:")
            for level, row in grouped.iterrows():
                lines.append(f"      {pretty_level(level):20s} mean={row['mean']:.4f}  median={row['median']:.4f}  n={int(row['count'])}")
            best_level = grouped.index[0]
            lines.append(f"      -> best on average: {pretty(col)} = {pretty_level(best_level)}")
            lines.append("")

    if constant_cols:
        lines.append("-" * 70)
        lines.append("CONSTANT COLUMNS (no variance across trials, excluded from analysis)")
        lines.append("-" * 70)
        for col in constant_cols:
            val = df_analysis[col].dropna().unique()
            lines.append(f"  {pretty(col)} = {val[0] if len(val) else 'NaN'}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("GENERAL IDEAS TO PUSH ACCURACY HIGHER")
    lines.append("-" * 70)
    lines.append("  1. Re-run the HPO study with the narrowed ranges above -- this")
    lines.append("     concentrates the search budget near where good trials already live.")
    lines.append("  2. If Z-Score Normalization Enabled shows a")
    lines.append("     clear winner above but was rarely sampled in this run, increase")
    lines.append("     TPESampler n_startup_trials / prior_weight so both branches get")
    lines.append("     explored.")
    lines.append("  3. If Number of Training Time Steps was constant, consider trying more")
    lines.append("     steps for the current best config -- accuracy may not have converged.")
    lines.append("  4. Look at LIF Decay Rate (Beta) and Spike Gradient Slope together")
    lines.append("     (SNN-specific dynamics): a poor combination can silently cripple")
    lines.append("     gradient flow through LIF layers even when other hyperparameters")
    lines.append("     look fine.")
    lines.append("  5. Consider averaging accuracy across MULTIPLE held-out subjects (not")
    lines.append("     just subject 0) for the top few configs -- a config that wins on one")
    lines.append("     held-out subject may not generalize; this also reduces noise in the")
    lines.append("     objective that the sampler is optimizing against.")
    lines.append("  6. If Pooling Layer 1 Size * Pooling Layer 2 Size is large relative to")
    lines.append("     num_samples, the separable conv may be operating on very few time")
    lines.append("     steps -- check the 'time_after_pool2' assertion margin for the")
    lines.append("     current best trial.")
    lines.append("")

    path = os.path.join(out_dir, "recommendations.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved recommendations -> {path}")
    print("\n" + "\n".join(lines[:30]) + "\n  ... (see recommendations.txt for full report)")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ajrbe\Documents\Git\spiking-eegnet\results\BNCI2014_001\trials.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "trial_analysis_output"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    target = "mean_bal_acc"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    # lr_exp is dropped: it's just log10(lr), the actual learning rate (lr) is
    # what's passed into the optimizer, so we report a single "Learning Rate"
    # variable instead of two redundant features.
    drop_cols = ["timestamp", "lr_exp"]
    df_analysis = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols, categorical_cols, constant_cols = identify_column_types(df_analysis, target)
    numeric_cols = [c for c in numeric_cols if c != "trial_number"]

    print(f"Loaded {len(df)} trials from {csv_path}")
    print(f"Target variable: {pretty(target)}")
    print(f"Numeric variables ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical/boolean variables ({len(categorical_cols)}): {categorical_cols}")
    if constant_cols:
        print(f"Constant variables ({len(constant_cols)}, excluded): {constant_cols}")
    print()

    # 1. Correlation matrix (Pearson)
    target_corr = plot_correlation_matrix(df_analysis, numeric_cols, target, out_dir, method="pearson")
    print(f"\nCorrelation with {pretty(target)} (sorted by |r|):")
    for var, r in target_corr.items():
        print(f"  {pretty(var):35s} r = {r:+.3f}")

    # 2. Scatter plots
    plot_scatter_numeric(df_analysis, numeric_cols, target, target_corr, out_dir)

    # 3. Box plots (multi-level categorical params only)
    plot_boxplot_categorical(df_analysis, categorical_cols, target, out_dir)

    # 4. Optimization progress
    final_best, best_idx = plot_optimization_progress(df, target, out_dir)
    print(f"\nFinal running best {pretty(target)}: {final_best:.4f}")

    # 5. Feature importance
    importance = compute_feature_importance(df_analysis, numeric_cols, categorical_cols, target, out_dir)

    # 6. Top-trial param distributions
    top_idx = plot_top_trials_param_distributions(df_analysis, numeric_cols, target, top_n, out_dir)

    # 7. NEW: target distribution
    plot_target_distribution(df_analysis, target, out_dir)

    # 8. NEW: importance vs correlation
    plot_importance_vs_correlation(importance, target_corr, target, out_dir)

    # NEW: baseline comparison (results_baseline.json and, if present,
    # results_snn.json for an untuned/no-Optuna SNN baseline), alongside the csv
    baselines = load_all_baselines(csv_path, target)
    plot_baseline_summary(df_analysis, target, baselines, top_n, out_dir)
    plot_baseline_distribution(df_analysis, target, baselines, out_dir)
    plot_top_trials_vs_baseline(df_analysis, target, baselines, top_idx, out_dir)
    plot_baseline_param_diff(df_analysis, numeric_cols, target, baselines, out_dir)

    # 9. Summary stats
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    df_analysis.describe(include="all").to_csv(summary_path)
    print(f"\nSaved summary statistics -> {summary_path}")

    print(f"\nBest trial(s) by {pretty(target)}:")
    top = df.nlargest(min(3, len(df)), target)
    display_cols = ([c for c in ["trial_number"] if c in df.columns]
                     + [target]
                     + [c for c in numeric_cols if c in df.columns][:8])
    print(top[display_cols].to_string(index=False))

    # 10. Recommendations
    write_recommendations(df, df_analysis, target, numeric_cols, categorical_cols, constant_cols,
                          target_corr, importance, top_idx, out_dir, top_n, baselines=baselines)


if __name__ == "__main__":
    main()