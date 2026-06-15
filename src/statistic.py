#!/usr/bin/env python3
"""
Expanded analysis of Optuna/SNN-EEGNet hyperparameter trial results.

Reads a trials.csv produced by pipeline.py and produces:

  1. correlation_matrix.png        - correlation heatmap (numeric vars + target)
  2. scatter_numeric_vs_target.png - acc_loso vs each numeric hyperparameter
  3. boxplot_categorical_vs_target.png - acc_loso by categorical/boolean params
  4. optimization_progress.png     - acc_loso per trial + running best (convergence)
  5. feature_importance.png        - Random Forest importance of each param for acc_loso
  6. top_trials_param_distributions.png - where top-N trials sit vs full search range
  7. pairplot_top_params.png        - pairwise relationships of the most important params,
                                       colored by acc_loso
  8. summary_stats.csv              - describe() of all columns
  9. recommendations.txt            - plain-text written summary of:
                                       - best trial(s) and their configs
                                       - which params correlate most with acc_loso
                                       - suggested narrowed search ranges for a follow-up study
                                       - flags for params that look saturated at a range edge

Usage:
    python statistic.py [csv_path] [out_dir] [top_n]

    csv_path : path to trials.csv (default: "trials.csv")
    out_dir  : output directory   (default: "trial_analysis_output")
    top_n    : number of top trials to use for "top trials" analyses (default: 10)
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
# Known search-space bounds (from main.py / spiking_eegnet.py docstrings)
# Used to (a) flag if top trials cluster at an edge of the searched range,
# and (b) sanity-check suggested narrowed ranges.
# ----------------------------------------------------------------------
SEARCH_SPACE = {
    "flow":                  (1.0, 40.0),
    "fhigh":                 (8.0, 120.0),
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
CATEGORICAL_HINTS = {"run_zscore", "run_bandpass", "norm_axis"}


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

    # Boolean-like columns -> categorical
    for col in list(numeric_cols):
        unique_vals = set(df_analysis[col].dropna().unique())
        if unique_vals.issubset({True, False, "True", "False", 0, 1}) and df_analysis[col].dtype == object:
            numeric_cols.remove(col)
            categorical_cols.append(col)

    # Drop constant numeric columns (zero variance -> undefined correlation)
    constant_cols = [c for c in numeric_cols
                      if pd.to_numeric(df_analysis[c], errors="coerce").nunique(dropna=True) <= 1]

    numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    return numeric_cols, categorical_cols, constant_cols


def plot_correlation_matrix(df_analysis, numeric_cols, target, out_dir):
    corr_cols = numeric_cols + [target]
    corr_df = df_analysis[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(max(8, 0.5 * len(corr_cols)), max(6, 0.5 * len(corr_cols))))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=150)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        x = pd.to_numeric(df_analysis[col], errors="coerce")
        y = pd.to_numeric(df_analysis[target], errors="coerce")
        ax.scatter(x, y, alpha=0.7, s=20)

        # trend line if enough points and non-degenerate x
        valid = x.notna() & y.notna()
        if valid.sum() >= 3 and x[valid].nunique() > 1:
            z = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(x[valid].min(), x[valid].max(), 50)
            ax.plot(xs, np.poly1d(z)(xs), color="red", linewidth=1, linestyle="--", alpha=0.7)

        r = target_corr.get(col, np.nan)
        ax.set_title(f"{col} (r={r:.2f})", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel(target, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_numeric_vs_target.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved scatter plots -> {path}")


def plot_boxplot_categorical(df_analysis, categorical_cols, target, out_dir):
    if not categorical_cols:
        print("No categorical/boolean variables found for box plots.")
        return
    n = len(categorical_cols)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1) if n > 1 else np.array([axes])

    for i, col in enumerate(categorical_cols):
        ax = axes[i]
        data = df_analysis[[col, target]].copy()
        data[col] = data[col].astype(str)
        data[target] = pd.to_numeric(data[target], errors="coerce")
        sns.boxplot(data=data, x=col, y=target, ax=ax)
        sns.stripplot(data=data, x=col, y=target, ax=ax, color="black", alpha=0.5, size=4)
        ax.set_title(f"{col} vs {target}", fontsize=10)
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "boxplot_categorical_vs_target.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved box plots -> {path}")


def plot_optimization_progress(df, target, out_dir):
    """
    Plot acc_loso per trial (in trial order) plus the running best.
    Uses trial_number if present and ordered; otherwise row order.
    """
    data = df.copy()
    if "trial_number" in data.columns and pd.to_numeric(data["trial_number"], errors="coerce").notna().all():
        data = data.sort_values("trial_number")
        x = pd.to_numeric(data["trial_number"], errors="coerce")
        xlabel = "Trial number"
    else:
        data = data.reset_index(drop=True)
        x = data.index
        xlabel = "Trial order (row index)"

    y = pd.to_numeric(data[target], errors="coerce")
    running_best = y.cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x, y, alpha=0.5, s=25, label=target, color="steelblue")
    ax.plot(x, running_best, color="darkorange", linewidth=2, label="Running best")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(target)
    ax.set_title("Optimization progress: accuracy per trial and running best")
    ax.legend()
    ax.grid(alpha=0.3)

    # annotate the overall best point
    best_idx = y.idxmax()
    ax.annotate(
        f"best={y[best_idx]:.4f}",
        xy=(x[best_idx], y[best_idx]),
        xytext=(10, -15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="grey"),
        fontsize=9,
    )

    plt.tight_layout()
    path = os.path.join(out_dir, "optimization_progress.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved optimization progress plot -> {path}")

    return running_best.iloc[-1], y.idxmax()


def compute_feature_importance(df_analysis, numeric_cols, categorical_cols, target, out_dir):
    """
    Fit a small Random Forest regressor on all params -> acc_loso and
    report permutation importance. Robust to small N (this is exploratory,
    not a rigorous model).
    """
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

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importance))))
    order = importance.index
    ax.barh(order, importance[order], xerr=importance_std[order], color="seagreen")
    ax.invert_yaxis()
    ax.set_xlabel("Permutation importance (drop in R² when shuffled)")
    ax.set_title(f"Random Forest feature importance for {target}")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot -> {path}")

    r2 = rf.score(X, y)
    print(f"RandomForest in-sample R^2 on acc_loso: {r2:.3f} (n={len(X)})")

    return importance


def plot_top_trials_param_distributions(df_analysis, numeric_cols, target, top_n, out_dir):
    """
    For each numeric search-space param, show the full distribution of
    trialled values vs the distribution restricted to the top-N trials
    by acc_loso, with the searched range marked.
    """
    y = pd.to_numeric(df_analysis[target], errors="coerce")
    top_idx = y.nlargest(min(top_n, len(y))).index

    plot_cols = [c for c in numeric_cols if c in SEARCH_SPACE]
    if not plot_cols:
        print("No recognized search-space columns for top-trial distribution plots.")
        return top_idx

    ncols = 4
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        vals = pd.to_numeric(df_analysis[col], errors="coerce")
        top_vals = vals.loc[top_idx]

        sns.histplot(vals, ax=ax, color="lightgrey", label="All trials",
                      bins=12, kde=False, stat="count")
        sns.histplot(top_vals, ax=ax, color="crimson", label=f"Top {len(top_idx)}",
                      bins=12, kde=False, stat="count", alpha=0.7)

        lo, hi = SEARCH_SPACE[col]
        ax.axvline(lo, color="black", linestyle=":", linewidth=1)
        ax.axvline(hi, color="black", linestyle=":", linewidth=1)

        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(plot_cols), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Param distributions: all trials vs top-{len(top_idx)} by {target}", y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, "top_trials_param_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved top-trial distribution plot -> {path}")

    return top_idx


def plot_pairplot_top_params(df_analysis, importance, target, out_dir, max_params=4):
    """Pairplot of the most important numeric params, colored by acc_loso."""
    if importance is None:
        print("Skipping pairplot (no importance ranking available).")
        return

    # restrict to numeric (non-dummy) columns that exist directly in df_analysis
    candidates = [c for c in importance.index if c in df_analysis.columns
                   and pd.to_numeric(df_analysis[c], errors="coerce").notna().all()]
    top_params = candidates[:max_params]
    if len(top_params) < 2:
        print("Not enough numeric top params for pairplot.")
        return

    plot_df = df_analysis[top_params + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(plot_df) < 5:
        print("Not enough valid rows for pairplot.")
        return

    g = sns.pairplot(
        plot_df, vars=top_params, hue=None,
        plot_kws={"alpha": 0.7},
        diag_kind="kde",
    )
    # color points by target value using a scatter overlay on off-diagonals
    norm = plt.Normalize(plot_df[target].min(), plot_df[target].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    for i, yvar in enumerate(top_params):
        for j, xvar in enumerate(top_params):
            ax = g.axes[i, j]
            if i != j:
                for coll in list(ax.collections):
                    coll.remove()
                ax.scatter(plot_df[xvar], plot_df[yvar],
                           c=plot_df[target], cmap="viridis", alpha=0.8, s=25)

    g.fig.colorbar(sm, ax=g.axes, label=target, shrink=0.6)
    g.fig.suptitle(f"Top-{len(top_params)} important params, colored by {target}", y=1.02)

    path = os.path.join(out_dir, "pairplot_top_params.png")
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved pairplot of top params -> {path}")


def write_recommendations(df, df_analysis, target, numeric_cols, categorical_cols, constant_cols,
                          target_corr, importance, top_idx, out_dir, top_n):
    """
    Write a plain-text summary: best trials, strongest correlates,
    suggested narrowed ranges for a follow-up study, and edge-of-range flags.
    """
    lines = []
    y = pd.to_numeric(df_analysis[target], errors="coerce")

    lines.append("=" * 70)
    lines.append("HYPERPARAMETER TRIAL ANALYSIS - RECOMMENDATIONS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total trials analyzed: {len(df)}")
    lines.append(f"Best {target}: {y.max():.4f}  (trial index {y.idxmax()})")
    lines.append(f"Median {target}: {y.median():.4f}")
    lines.append(f"Worst {target}: {y.min():.4f}")
    lines.append("")

    # ---- Best trials table ----
    lines.append("-" * 70)
    lines.append(f"TOP {min(top_n, len(df))} TRIALS")
    lines.append("-" * 70)
    cols_to_show = (["trial_number"] if "trial_number" in df.columns else []) + [target] + numeric_cols + categorical_cols
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    top_table = df.loc[top_idx, cols_to_show].sort_values(target, ascending=False)
    lines.append(top_table.to_string(index=False))
    lines.append("")

    # ---- Correlation ranking ----
    lines.append("-" * 70)
    lines.append(f"CORRELATION WITH {target} (sorted by |r|)")
    lines.append("-" * 70)
    for var, r in target_corr.items():
        direction = "higher is better" if r > 0 else "lower is better"
        strength = "strong" if abs(r) >= 0.4 else ("moderate" if abs(r) >= 0.2 else "weak")
        lines.append(f"  {var:25s} r = {r:+.3f}  ({strength}, {direction})")
    lines.append("")

    # ---- Feature importance ----
    if importance is not None:
        lines.append("-" * 70)
        lines.append("RANDOM FOREST PERMUTATION IMPORTANCE (non-linear effects)")
        lines.append("-" * 70)
        for var, imp in importance.head(10).items():
            lines.append(f"  {var:25s} importance = {imp:+.4f}")
        lines.append("")
        lines.append("Note: importance can pick up non-linear / interaction effects that")
        lines.append("the linear correlation above misses. Compare both rankings -- params")
        lines.append("that rank high in importance but low in |r| likely have a 'sweet spot'")
        lines.append("in the middle of their range rather than a monotonic relationship.")
        lines.append("")

    # ---- Suggested narrowed ranges + edge flags ----
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

        line = f"  {col:25s} top trials span [{t_lo:.4g}, {t_hi:.4g}]" \
               f" -> try [{sug_lo:.4g}, {sug_hi:.4g}]  (full range was [{lo_search:.4g}, {hi_search:.4g}])"

        # Edge-of-range flags
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

    # ---- Categorical recommendations ----
    if categorical_cols:
        lines.append("-" * 70)
        lines.append("CATEGORICAL / BOOLEAN PARAMETERS")
        lines.append("-" * 70)
        for col in categorical_cols:
            data = df_analysis[[col, target]].copy()
            data[col] = data[col].astype(str)
            data[target] = pd.to_numeric(data[target], errors="coerce")
            grouped = data.groupby(col)[target].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
            lines.append(f"  {col}:")
            for level, row in grouped.iterrows():
                lines.append(f"      {level:10s} mean={row['mean']:.4f}  median={row['median']:.4f}  n={int(row['count'])}")
            best_level = grouped.index[0]
            lines.append(f"      -> best on average: {col} = {best_level}")
            lines.append("")

    # ---- Constant columns note ----
    if constant_cols:
        lines.append("-" * 70)
        lines.append("CONSTANT COLUMNS (no variance across trials, excluded from analysis)")
        lines.append("-" * 70)
        for col in constant_cols:
            val = df_analysis[col].dropna().unique()
            lines.append(f"  {col} = {val[0] if len(val) else 'NaN'}")
        lines.append("")

    # ---- General improvement ideas ----
    lines.append("-" * 70)
    lines.append("GENERAL IDEAS TO PUSH ACCURACY HIGHER")
    lines.append("-" * 70)
    lines.append("  1. Re-run the HPO study with the narrowed ranges above -- this")
    lines.append("     concentrates the search budget near where good trials already live.")
    lines.append("  2. If `run_bandpass` or `run_zscore` shows a clear winner above but")
    lines.append("     was rarely sampled in this run (see prior conversation re: Optuna")
    lines.append("     diversity), increase TPESampler n_startup_trials / prior_weight so")
    lines.append("     both branches (and their conditional flow/fhigh params) get explored.")
    lines.append("  3. If `epochs` was constant, consider trying more epochs for the")
    lines.append("     current best config -- LOSO accuracy may not have fully converged.")
    lines.append("  4. Look at `beta` and `spike_grad_slope` together (SNN-specific dynamics):")
    lines.append("     a poor combination can silently cripple gradient flow through LIF")
    lines.append("     layers even when other hyperparameters look fine.")
    lines.append("  5. Consider averaging LOSO accuracy across MULTIPLE held-out subjects")
    lines.append("     (not just subject 0) for the top few configs -- a config that wins")
    lines.append("     on one held-out subject may not generalize; this also reduces noise")
    lines.append("     in the objective that the sampler is optimizing against.")
    lines.append("  6. If pool1_size * pool2_size is large relative to num_samples, the")
    lines.append("     separable conv may be operating on very few time steps -- check the")
    lines.append("     'time_after_pool2' assertion margin for the current best trial.")
    lines.append("")

    path = os.path.join(out_dir, "recommendations.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved recommendations -> {path}")
    print("\n" + "\n".join(lines[:30]) + "\n  ... (see recommendations.txt for full report)")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\ajrbe\Documents\Git\spiking-eegnet\results_noresampler\BNCI2014_001\trials.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "trial_analysis_output"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    target = "acc_loso"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    drop_cols = ["timestamp"]
    df_analysis = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols, categorical_cols, constant_cols = identify_column_types(df_analysis, target)
    # keep trial_number out of "numeric features" for modeling/plots but keep in df for display
    numeric_cols = [c for c in numeric_cols if c != "trial_number"]

    print(f"Loaded {len(df)} trials from {csv_path}")
    print(f"Target variable: {target}")
    print(f"Numeric variables ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical/boolean variables ({len(categorical_cols)}): {categorical_cols}")
    if constant_cols:
        print(f"Constant variables ({len(constant_cols)}, excluded): {constant_cols}")
    print()

    # 1. Correlation matrix
    target_corr = plot_correlation_matrix(df_analysis, numeric_cols, target, out_dir)
    print(f"\nCorrelation with {target} (sorted by |r|):")
    for var, r in target_corr.items():
        print(f"  {var:25s} r = {r:+.3f}")

    # 2. Scatter plots
    plot_scatter_numeric(df_analysis, numeric_cols, target, target_corr, out_dir)

    # 3. Box plots
    plot_boxplot_categorical(df_analysis, categorical_cols, target, out_dir)

    # 4. Optimization progress
    final_best, best_idx = plot_optimization_progress(df, target, out_dir)
    print(f"\nFinal running best {target}: {final_best:.4f}")

    # 5. Feature importance
    importance = compute_feature_importance(df_analysis, numeric_cols, categorical_cols, target, out_dir)

    # 6. Top-trial param distributions
    top_idx = plot_top_trials_param_distributions(df_analysis, numeric_cols, target, top_n, out_dir)

    # 7. Pairplot of top important params
    plot_pairplot_top_params(df_analysis, importance, target, out_dir)

    # 8. Summary stats
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    df_analysis.describe(include="all").to_csv(summary_path)
    print(f"\nSaved summary statistics -> {summary_path}")

    print(f"\nBest trial(s) by {target}:")
    top = df.nlargest(min(3, len(df)), target)
    display_cols = ([c for c in ["trial_number"] if c in df.columns]
                     + [target]
                     + [c for c in numeric_cols if c in df.columns][:8])
    print(top[display_cols].to_string(index=False))

    # 9. Recommendations
    write_recommendations(df, df_analysis, target, numeric_cols, categorical_cols, constant_cols,
                          target_corr, importance, top_idx, out_dir, top_n)


if __name__ == "__main__":
    main()