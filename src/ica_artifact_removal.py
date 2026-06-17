"""
ica_artifact_removal.py
-----------------------
ICA artifact removal — two clean, non-overlapping figures.

Figure 1: Intro banner  +  IC traces  +  Kurtosis bar
Figure 2: Channel artifact bar  +  Before/After trials  +  Summary

Run:
    python ica_artifact_removal.py [--subject 0] [--n-examples 3]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from load_moabb_dataset import load_moabb_dataset

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
GREEN  = "#3fb950"
RED    = "#f85149"
BLUE   = "#79c0ff"
AMBER  = "#d29922"
ORANGE = "#e3b341"


# ── Tiny helpers ──────────────────────────────────────────────────────────────

def _kurtosis(sig):
    s = sig - sig.mean()
    v = np.mean(s ** 2)
    return 0.0 if v < 1e-12 else float(np.mean(s ** 4) / v ** 2 - 3.0)


def _changed_mask(raw, clean, thr=0.05):
    d = np.abs(raw - clean)
    return d > thr * (np.max(np.abs(raw)) + 1e-12)


def _shade(ax, mask, t, color=AMBER, alpha=0.30):
    in_r = False
    for i, v in enumerate(mask):
        if v and not in_r:
            start, in_r = t[i], True
        elif not v and in_r:
            ax.axvspan(start, t[i - 1], color=color, alpha=alpha, lw=0)
            in_r = False
    if in_r:
        ax.axvspan(start, t[-1], color=color, alpha=alpha, lw=0)


def _style(ax, xlabel="", ylabel=""):
    """Bare dark style — no title logic here."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER); sp.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(color=BORDER, linewidth=0.35, alpha=0.9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=MUTED, labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED, labelpad=3)


def _title(ax, t, color=TEXT, size=9):
    """Set title with enough pad that it never overlaps the axes content."""
    ax.set_title(t, fontsize=size, fontweight="bold", color=color,
                 loc="left", pad=8)


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def figure1(sources, kurtosis_vals, bad_idx, sfreq, n_ic_show=6):
    n_ic      = sources.shape[0]
    n_ic_show = min(n_ic_show, n_ic)
    show_samp = min(int(5 * sfreq), sources.shape[1])
    t         = np.arange(show_samp) / sfreq

    # Layout:  banner  |  IC rows  |  kurtosis
    # Give kurtosis plenty of height so IC labels don't pile up
    IC_H   = 1.5          # height units per IC row
    KUR_H  = max(n_ic * 0.45, 5)   # scales with number of ICs
    BAN_H  = 1.6

    fig = plt.figure(figsize=(15, BAN_H + n_ic_show * IC_H + KUR_H + 1.5),
                     facecolor=BG)

    gs = gridspec.GridSpec(
        n_ic_show + 2, 1,          # banner + ICs + kurtosis
        figure=fig,
        height_ratios=[BAN_H] + [IC_H] * n_ic_show + [KUR_H],
        hspace=0.0,                # we control gaps manually via subplots_adjust
        left=0.13, right=0.97,
        top=0.96, bottom=0.06,
    )

    # ── Banner ────────────────────────────────────────────────────────────────
    ax_ban = fig.add_subplot(gs[0])
    ax_ban.set_facecolor("#1c2128")
    for sp in ax_ban.spines.values():
        sp.set_color(BORDER)
    ax_ban.set_xticks([]); ax_ban.set_yticks([])

    ax_ban.text(0.012, 0.80, "What is ICA artifact removal?",
                transform=ax_ban.transAxes, fontsize=11, fontweight="bold",
                color=TEXT, va="top")
    ax_ban.text(0.012, 0.52,
                "EEG picks up brain signals AND noise (blinks, muscle, "
                "electrical interference).  ICA splits them into separate "
                "components so we can remove just the noise.",
                transform=ax_ban.transAxes, fontsize=8.5, color=MUTED, va="top")
    ax_ban.text(0.012, 0.18,
                f"Result:  {len(bad_idx)} artifact component(s) found out of "
                f"{n_ic}.  They are zeroed out and the signal is rebuilt.",
                transform=ax_ban.transAxes, fontsize=8.5, color=ORANGE, va="top")

    # ── IC traces ─────────────────────────────────────────────────────────────
    for row in range(n_ic_show):
        is_bad = row in bad_idx
        col    = RED if is_bad else GREEN
        label  = "ARTIFACT" if is_bad else "brain"

        ax = fig.add_subplot(gs[1 + row])
        ax.plot(t, sources[row, :show_samp], color=col, lw=0.75)
        ax.set_facecolor(PANEL)
        ax.set_yticks([])
        ax.set_xticks([])      # only bottom row gets ticks
        for sp in ax.spines.values():
            sp.set_color(RED if is_bad else BORDER)
            sp.set_linewidth(2.0 if is_bad else 0.6)
        ax.grid(color=BORDER, lw=0.3, alpha=0.7)

        # ylabel — two short lines, fixed labelpad
        ax.set_ylabel(f"IC {row}\n{label}", fontsize=7.5, rotation=0,
                      labelpad=48, va="center", color=col,
                      fontweight="bold" if is_bad else "normal")

        # Section heading above very first IC only
        if row == 0:
            ax.set_title(
                "Step 1 — Signal components (ICs)     "
                "[ Red border = artifact IC, will be removed ]",
                fontsize=9, fontweight="bold", color=TEXT, loc="left", pad=6)

    # Bottom time axis on last IC
    ax_last = fig.add_subplot(gs[n_ic_show])   # reuse last IC axes reference
    # Actually just re-enable x ticks on the last IC subplot
    fig.axes[n_ic_show].set_xticks(
        np.arange(0, show_samp / sfreq + 0.1, 1.0))
    fig.axes[n_ic_show].tick_params(bottom=True, colors=MUTED, labelsize=7)
    fig.axes[n_ic_show].set_xlabel("Time (s)", fontsize=8, color=MUTED)

    # ── Kurtosis ──────────────────────────────────────────────────────────────
    ax_k = fig.add_subplot(gs[n_ic_show + 1])
    bar_cols = [RED if i in bad_idx else GREEN for i in range(n_ic)]
    ax_k.barh(range(n_ic), np.abs(kurtosis_vals),
              color=bar_cols, height=0.55, zorder=2)
    ax_k.axvline(3.0, color=ORANGE, ls="--", lw=1.4, zorder=3,
                 label="Threshold = 3")
    _style(ax_k, xlabel="|Excess Kurtosis|")
    _title(ax_k,
           "Step 2 — Kurtosis: how 'spiky' is each component?\n"
           "Brain signals are smooth.  Blinks/muscle are spiky.  "
           "Score > 3  →  flagged as artifact.")

    ax_k.set_yticks(range(n_ic))
    ax_k.set_yticklabels([f"IC {i}" for i in range(n_ic)],
                          fontsize=7.5, color=TEXT)
    ax_k.invert_yaxis()
    ax_k.legend(fontsize=8, labelcolor=TEXT,
                 facecolor=PANEL, edgecolor=BORDER, loc="lower right")

    x_max = max(np.abs(kurtosis_vals).max() * 1.08, 4.0)
    ax_k.set_xlim(0, x_max + 0.5)
    for i, kv in enumerate(kurtosis_vals):
        tag   = "  ARTIFACT" if i in bad_idx else "  ok"
        color = RED         if i in bad_idx else GREEN
        ax_k.text(np.abs(kv) + 0.1, i, tag,
                  va="center", fontsize=7, color=color)

    # Increase gap between ICs and kurtosis section
    gs.update(hspace=0.45)
    fig.patch.set_facecolor(BG)
    return fig


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def figure2(X_raw, X_clean, ch_names, sfreq, n_examples=3):
    n_trials, n_ch, n_samp = X_raw.shape
    n_examples = min(n_examples, n_trials)
    t = np.arange(n_samp) / sfreq

    TRIAL_H = 3.5
    CHAN_H  = 3.0
    SUM_H   = 0.75

    fig = plt.figure(
        figsize=(15, CHAN_H + n_examples * TRIAL_H + SUM_H + 1.5),
        facecolor=BG)

    gs = gridspec.GridSpec(
        n_examples + 2, 1,
        figure=fig,
        height_ratios=[CHAN_H] + [TRIAL_H] * n_examples + [SUM_H],
        hspace=0.55,
        left=0.08, right=0.97,
        top=0.97, bottom=0.04,
    )

    # ── Channel artifact bar ──────────────────────────────────────────────────
    ax_ch = fig.add_subplot(gs[0])
    diff  = np.abs(X_raw - X_clean).mean(axis=(0, 2))
    p33, p66 = np.percentile(diff, 33), np.percentile(diff, 66)
    bcols = [RED if v >= p66 else AMBER if v >= p33 else GREEN for v in diff]

    ax_ch.bar(range(n_ch), diff, color=bcols, width=0.75, zorder=2)
    ax_ch.set_xticks(range(n_ch))
    ax_ch.set_xticklabels(ch_names, rotation=50, fontsize=7,
                           ha="right", color=TEXT)
    _style(ax_ch, ylabel="Avg. change (µV)")
    _title(ax_ch,
           "Step 3 — How much artifact was removed from each electrode?\n"
           "Taller bar = more noise removed.   Red = most,   Green = least.")

    patches = [
        mpatches.Patch(color=RED,   label="Most artifact removed"),
        mpatches.Patch(color=AMBER, label="Moderate"),
        mpatches.Patch(color=GREEN, label="Least artifact removed"),
    ]
    ax_ch.legend(handles=patches, fontsize=8, labelcolor=TEXT,
                  facecolor=PANEL, edgecolor=BORDER, loc="upper right")

    # ── Before / After trials ─────────────────────────────────────────────────
    rep_ch = int(np.argmax(X_raw.var(axis=(0, 2))))
    idxs   = np.linspace(0, n_trials - 1, n_examples, dtype=int)

    legend_patches = [
        mpatches.Patch(color=BLUE,  label="Before cleaning (original)"),
        mpatches.Patch(color=GREEN, label="After cleaning"),
        mpatches.Patch(color=AMBER, alpha=0.5, label="Region where artifact was removed"),
    ]

    for i, ti in enumerate(idxs):
        ax = fig.add_subplot(gs[1 + i])
        raw = X_raw[ti, rep_ch, :]
        cln = X_clean[ti, rep_ch, :]
        mask = _changed_mask(raw, cln)

        _shade(ax, mask, t, color=AMBER, alpha=0.28)
        ax.plot(t, raw, color=BLUE,  lw=0.9, alpha=0.9)
        ax.plot(t, cln, color=GREEN, lw=0.9, alpha=0.95)

        pct_var = 100.0 * (1.0 - np.var(cln) / (np.var(raw) + 1e-12))
        ms_chg  = mask.sum() / sfreq * 1000

        _style(ax,
               xlabel="Time (s)" if i == n_examples - 1 else "",
               ylabel="µV")
        _title(ax,
               f"Trial {ti}   —   electrode {ch_names[rep_ch]}   "
               f"|   {pct_var:.1f}% variance reduced   "
               f"|   ~{ms_chg:.0f} ms affected")

        ax.legend(handles=legend_patches, fontsize=7.5, labelcolor=TEXT,
                   facecolor=PANEL, edgecolor=BORDER, loc="upper right",
                   framealpha=0.85)

    # ── Summary ───────────────────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[n_examples + 1])
    ax_s.set_facecolor("#1c2128")
    for sp in ax_s.spines.values():
        sp.set_color(BORDER)
    ax_s.set_xticks([]); ax_s.set_yticks([])

    vb  = float(np.var(X_raw))
    va  = float(np.var(X_clean))
    pct = 100.0 * (1.0 - va / (vb + 1e-12))
    if pct > 30:
        col, icon, msg = RED,   "⚠", \
            "Over 30% removed — check flagged ICs, may be too aggressive."
    elif pct < 0.1:
        col, icon, msg = AMBER, "ℹ", \
            "Under 0.1% removed — thresholds may be too conservative."
    else:
        col, icon, msg = GREEN, "✓", \
            "Removal looks reasonable — noise cleaned, brain signal preserved."

    ax_s.text(0.012, 0.5,
              f"{icon}  {pct:.1f}% of total signal variance removed.   {msg}",
              transform=ax_s.transAxes, fontsize=9.5, color=col,
              va="center", fontweight="bold")

    fig.patch.set_facecolor(BG)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main(subject_idx=0, n_examples=3):
    print("Loading BNCI2014_001 ...")
    X, y, subject_ids, meta = load_moabb_dataset("BNCI2014_001")
    sfreq    = meta["sfreq"]
    ch_names = meta.get("ch_names",
                        [f"CH{i}" for i in range(meta["n_channels"])])

    mask   = subject_ids == subject_idx
    X_subj = X[mask][:, 0, :, :]
    n_trials, n_ch, n_samp = X_subj.shape
    print(f"Subject {subject_idx}: {n_trials} trials, "
          f"{n_ch} ch, {n_samp} samp @ {sfreq} Hz")

    from sklearn.decomposition import FastICA
    n_comp = min(n_ch, 20)
    print(f"Fitting FastICA ({n_comp} components) ...")
    ica     = FastICA(n_components=n_comp, random_state=42,
                      max_iter=500, tol=1e-4)
    sources = ica.fit_transform(X_subj.reshape(n_ch, -1).T).T
    mixing  = ica.mixing_

    kurtosis_vals = np.array([_kurtosis(sources[i]) for i in range(n_comp)])
    FRONTAL = {"Fp1","Fp2","AF3","AF4","F7","F3","Fz","F4","F8",
               "AF7","AF8","FP1","FP2"}
    frontal_idx = [j for j, n in enumerate(ch_names) if n in FRONTAL]

    bad_idx = []
    for i in range(n_comp):
        kur_bad = abs(kurtosis_vals[i]) > 3.0
        frontal_bad = False
        if frontal_idx:
            sp = np.abs(mixing[:, i])
            frontal_bad = sp[frontal_idx].mean() / (sp.mean() + 1e-12) > 0.8
        if kur_bad or frontal_bad:
            bad_idx.append(i)

    print(f"Flagged {len(bad_idx)} artifact IC(s): {bad_idx}")

    src_clean = sources.copy()
    src_clean[bad_idx] = 0.0
    X_clean = ((mixing @ src_clean).T
               .T.reshape(n_trials, n_ch, n_samp).astype(np.float32))

    vb  = float(np.var(X_subj))
    va  = float(np.var(X_clean))
    pct = 100.0 * (1.0 - va / (vb + 1e-12))
    print(f"Variance before={vb:.3f}  after={va:.3f}  removed={pct:.2f}%")

    print("Drawing figures ...")
    f1 = figure1(sources, kurtosis_vals, bad_idx, sfreq, n_ic_show=6)
    f1.canvas.manager.set_window_title("Figure 1 — ICA Components & Kurtosis")

    f2 = figure2(X_subj, X_clean, ch_names, sfreq, n_examples=n_examples)
    f2.canvas.manager.set_window_title("Figure 2 — Before vs After Cleaning")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject",    type=int, default=0)
    parser.add_argument("--n-examples", type=int, default=3)
    args = parser.parse_args()
    main(subject_idx=args.subject, n_examples=args.n_examples)