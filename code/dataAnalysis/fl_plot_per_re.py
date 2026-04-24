"""
Two-mode heatmap visualisation for per-Re generated data.

Mode 1 — Overall heatmap (FL averaged over all Re levels, same layout as fl_plot.py):
  3-panel figure per file: heatmap | FL vs X (one curve/Z) | FL vs Z (one curve/X)

Mode 2 — Per-Re heatmap:
  One figure per file with a 2×5 grid of subplots, one subplot per Re level.
  Each subplot is a pcolormesh of FL(X,Z) for that Re slice only.

Output
------
  figures/per_re/overall/{label}_overall.png
  figures/per_re/per_re/{label}_per_re.png
  figures/per_re/summary/ALL_overall_summary.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DATA_DIR = "data/processed/2026-04-22"
FIG_ROOT  = os.path.join(DATA_DIR, "figures", "per_re")
DIR_OVERALL = os.path.join(FIG_ROOT, "overall")
DIR_PER_RE  = os.path.join(FIG_ROOT, "per_re")
DIR_SUMMARY = os.path.join(FIG_ROOT, "summary")
for d in [DIR_OVERALL, DIR_PER_RE, DIR_SUMMARY]:
    os.makedirs(d, exist_ok=True)

gen_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith("_generated_per_re.xlsx"))


# ── shared helpers ────────────────────────────────────────────────────────────

def load(fname):
    df    = pd.read_excel(os.path.join(DATA_DIR, fname))
    label = fname.replace("_generated_per_re.xlsx", "")
    col_sp = df.columns[3]          # Re column
    return df, label, col_sp


def pivot_mean(df):
    """Mean FL over all Re levels → (Z × X) matrix."""
    pt = df.pivot_table(index="Z", columns="X", values="FL", aggfunc="mean")
    return pt.sort_index().sort_index(axis=1)


def pivot_for_re(df, col_sp, sp):
    """FL matrix for a single Re level."""
    sub = df[df[col_sp] == sp]
    pt  = sub.pivot_table(index="Z", columns="X", values="FL", aggfunc="mean")
    return pt.sort_index().sort_index(axis=1)


def sym_norm(FL_mat):
    """Symmetric diverging normalisation centred at 0."""
    v = np.nanmax(np.abs(FL_mat))
    if v == 0:
        v = 1.0
    return mcolors.TwoSlopeNorm(vmin=-v, vcenter=0, vmax=v)


def draw_heatmap(ax, pt, title, cmap="RdBu_r"):
    X_vals = pt.columns.values.astype(float)
    Z_vals = pt.index.values.astype(float)
    FL_mat = pt.values
    norm   = sym_norm(FL_mat)
    im     = ax.pcolormesh(X_vals, Z_vals, FL_mat, cmap=cmap, norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label="FL", shrink=0.85)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(title, fontsize=9)
    return im


# ── Mode 1: overall heatmap (3-panel, mean over Re) ──────────────────────────

def plot_overall(df, label, col_sp):
    pt     = pivot_mean(df)
    X_vals = pt.columns.values.astype(float)
    Z_vals = pt.index.values.astype(float)
    FL_mat = pt.values

    norm = sym_norm(FL_mat)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"FL (per-Re generated, mean over Re) — {label}",
                 fontsize=13, fontweight="bold")

    # panel 1: heatmap
    ax = axes[0]
    im = ax.pcolormesh(X_vals, Z_vals, FL_mat, cmap="RdBu_r", norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label="FL")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title("FL heatmap  mean(Re)")

    # panel 2: FL vs X per Z
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(Z_vals)))
    for i, z in enumerate(Z_vals):
        row = df[df["Z"] == z].groupby("X")["FL"].mean().reset_index().sort_values("X")
        ax.plot(row["X"], row["FL"], marker="o", ms=2, color=colors[i], label=f"Z={z:.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("X"); ax.set_ylabel("FL")
    ax.set_title("FL vs X  (one line per Z, mean Re)")
    ax.legend(fontsize=6, ncol=2)

    # panel 3: FL vs Z per X
    ax = axes[2]
    colors = plt.cm.plasma(np.linspace(0, 1, len(X_vals)))
    for i, x in enumerate(X_vals):
        col = df[df["X"] == x].groupby("Z")["FL"].mean().reset_index().sort_values("Z")
        ax.plot(col["Z"], col["FL"], marker="s", ms=2, color=colors[i], label=f"X={x:.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Z"); ax.set_ylabel("FL")
    ax.set_title("FL vs Z  (one line per X, mean Re)")
    ax.legend(fontsize=4, ncol=3)

    plt.tight_layout()
    out = os.path.join(DIR_OVERALL, f"{label}_overall.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved overall : {out}")


# ── Mode 2: per-Re heatmap (2 × 5 grid, one subplot per Re level) ────────────

def plot_per_re(df, label, col_sp):
    sp_levels = sorted(df[col_sp].unique())
    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(f"FL per Re level — {label}", fontsize=13, fontweight="bold")

    # compute a shared colour scale across all Re levels for comparability
    fl_global_max = np.nanmax(np.abs(df["FL"].values))
    shared_norm   = mcolors.TwoSlopeNorm(vmin=-fl_global_max, vcenter=0, vmax=fl_global_max)

    for idx, sp in enumerate(sp_levels):
        row_i, col_i = divmod(idx, ncols)
        ax = axes[row_i][col_i]

        pt     = pivot_for_re(df, col_sp, sp)
        X_vals = pt.columns.values.astype(float)
        Z_vals = pt.index.values.astype(float)
        FL_mat = pt.values

        im = ax.pcolormesh(X_vals, Z_vals, FL_mat,
                           cmap="RdBu_r", norm=shared_norm, shading="nearest")
        plt.colorbar(im, ax=ax, label="FL", shrink=0.85)
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Z", fontsize=8)
        ax.set_title(f"Re = {sp}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7)

    # hide any spare axes
    for idx in range(len(sp_levels), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    out = os.path.join(DIR_PER_RE, f"{label}_per_re.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved per-Re  : {out}")


# ── Summary: overall heatmaps for all files on one canvas ────────────────────

def plot_summary(file_data):
    n     = len(file_data)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, (df, label, col_sp) in enumerate(file_data):
        pt     = pivot_mean(df)
        X_vals = pt.columns.values.astype(float)
        Z_vals = pt.index.values.astype(float)
        FL_mat = pt.values
        norm   = sym_norm(FL_mat)

        ax = axes[idx]
        im = ax.pcolormesh(X_vals, Z_vals, FL_mat, cmap="RdBu_r", norm=norm, shading="nearest")
        plt.colorbar(im, ax=ax, label="FL")
        ax.set_xlabel("X"); ax.set_ylabel("Z")
        ax.set_title(label, fontsize=10, fontweight="bold")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("FL heatmaps — per-Re generated data (mean over Re), all cross-sections",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(DIR_SUMMARY, "ALL_per_re_summary.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved summary : {out}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not gen_files:
        print("No *_generated_per_re.xlsx files found. Run generate_data_per_re.py first.")
        return

    file_data = []
    for fname in gen_files:
        df, label, col_sp = load(fname)
        print(f"\n[{label}]")
        plot_overall(df, label, col_sp)
        plot_per_re(df, label, col_sp)
        file_data.append((df, label, col_sp))

    print("\n[Summary]")
    plot_summary(file_data)
    print("\nDone.")


if __name__ == "__main__":
    main()
