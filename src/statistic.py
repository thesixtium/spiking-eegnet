import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    csv_path = r"/results_noresampler\BNCI2014_001\trials.csv"
    out_dir = "trial_analysis_output"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    target = "acc_loso"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}")

    # Drop columns not useful for analysis
    drop_cols = ["timestamp", "trial_number"]
    df_analysis = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Identify column types
    numeric_cols = []
    categorical_cols = []
    for col in df_analysis.columns:
        if col == target:
            continue
        coerced = pd.to_numeric(df_analysis[col], errors="coerce")
        # If mostly numeric (allow some NaN from bools/strings), treat as numeric
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
    if constant_cols:
        print(f"Dropping constant columns (no variance): {constant_cols}")
        numeric_cols = [c for c in numeric_cols if c not in constant_cols]

    print(f"Loaded {len(df)} trials from {csv_path}")
    print(f"Target variable: {target}")
    print(f"Numeric variables ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical/boolean variables ({len(categorical_cols)}): {categorical_cols}")
    print()

    # ----------------------------------------------------------------
    # Correlation matrix (numeric vars + target)
    # ----------------------------------------------------------------
    corr_cols = numeric_cols + [target]
    corr_df = df_analysis[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(max(8, 0.5 * len(corr_cols)), max(6, 0.5 * len(corr_cols))))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, annot_kws={"size": 7})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    corr_path = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(corr_path, dpi=150)
    plt.close()
    print(f"Saved correlation matrix -> {corr_path}")

    # Print sorted correlation with target
    target_corr = corr_matrix[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
    print(f"\nCorrelation with {target} (sorted by |r|):")
    for var, r in target_corr.items():
        print(f"  {var:25s} r = {r:+.3f}")

    # ----------------------------------------------------------------
    # Scatter plots: numeric variable vs target
    # ----------------------------------------------------------------
    n = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        x = pd.to_numeric(df_analysis[col], errors="coerce")
        y = pd.to_numeric(df_analysis[target], errors="coerce")
        ax.scatter(x, y, alpha=0.7, s=20)
        r = target_corr.get(col, np.nan)
        ax.set_title(f"{col} (r={r:.2f})", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel(target, fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "scatter_numeric_vs_target.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"\nSaved scatter plots -> {scatter_path}")

    # ----------------------------------------------------------------
    # Box plots: categorical/boolean variable vs target
    # ----------------------------------------------------------------
    if categorical_cols:
        n = len(categorical_cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)

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
        box_path = os.path.join(out_dir, "boxplot_categorical_vs_target.png")
        plt.savefig(box_path, dpi=150)
        plt.close()
        print(f"Saved box plots -> {box_path}")
    else:
        print("No categorical/boolean variables found for box plots.")

    # ----------------------------------------------------------------
    # Summary stats
    # ----------------------------------------------------------------
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    df_analysis.describe(include="all").to_csv(summary_path)
    print(f"\nSaved summary statistics -> {summary_path}")

    print(f"\nBest trial(s) by {target}:")
    top = df.nlargest(3, target)
    print(top[["trial_number", target] + [c for c in numeric_cols if c in df.columns][:8]].to_string(index=False))


if __name__ == "__main__":
    main()