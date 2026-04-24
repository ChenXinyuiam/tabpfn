import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

DATA_DIR = "data/processed/2026-04-22"
OUT_DIR = "data/processed/2026-04-22/figures"
os.makedirs(OUT_DIR, exist_ok=True)

fl_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith("-FL.xlsx")])


def load_file(fname):
    df = pd.read_excel(os.path.join(DATA_DIR, fname))
    df = df[["X", "Z", "FL"]].dropna()
    label = fname.replace("-FL.xlsx", "")
    return df, label


def pivot_fl(df):
    """Pivot X x Z → FL matrix, sorted by X and Z."""
    pt = df.pivot_table(index="Z", columns="X", values="FL", aggfunc="mean")
    pt = pt.sort_index(ascending=True)
    pt = pt.sort_index(axis=1, ascending=True)
    return pt


# ── per-file plots ──────────────────────────────────────────────────────────
for fname in fl_files:
    df, label = load_file(fname)
    pt = pivot_fl(df)

    X_vals = pt.columns.values.astype(float)
    Z_vals = pt.index.values.astype(float)
    FL_mat = pt.values

    fl_abs_max = np.nanmax(np.abs(FL_mat))
    norm = mcolors.TwoSlopeNorm(vmin=-fl_abs_max, vcenter=0, vmax=fl_abs_max)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"FL distribution — {label}", fontsize=14, fontweight="bold")

    # ── 1: heatmap ──────────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.pcolormesh(X_vals, Z_vals, FL_mat, cmap=cmap, norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label="FL")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("FL heatmap (X, Z)")

    # ── 2: FL vs X, one curve per Z ─────────────────────────────────────────
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(Z_vals)))
    for i, z in enumerate(Z_vals):
        row = df[df["Z"] == z].sort_values("X")
        ax.plot(row["X"], row["FL"], marker="o", ms=3, color=colors[i], label=f"Z={z:.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("X")
    ax.set_ylabel("FL")
    ax.set_title("FL vs X  (each line = one Z)")
    ax.legend(fontsize=7, ncol=2, loc="best")

    # ── 3: FL vs Z, one curve per X ─────────────────────────────────────────
    ax = axes[2]
    colors = plt.cm.plasma(np.linspace(0, 1, len(X_vals)))
    for i, x in enumerate(X_vals):
        col = df[df["X"] == x].sort_values("Z")
        ax.plot(col["Z"], col["FL"], marker="s", ms=3, color=colors[i], label=f"X={x:.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Z")
    ax.set_ylabel("FL")
    ax.set_title("FL vs Z  (each line = one X)")
    ax.legend(fontsize=5, ncol=3, loc="best")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"{label}_FL.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── summary: all files in one grid ─────────────────────────────────────────
n = len(fl_files)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for idx, fname in enumerate(fl_files):
    df, label = load_file(fname)
    pt = pivot_fl(df)

    X_vals = pt.columns.values.astype(float)
    Z_vals = pt.index.values.astype(float)
    FL_mat = pt.values

    fl_abs_max = np.nanmax(np.abs(FL_mat))
    norm = mcolors.TwoSlopeNorm(vmin=-fl_abs_max, vcenter=0, vmax=fl_abs_max)

    ax = axes[idx]
    im = ax.pcolormesh(X_vals, Z_vals, FL_mat, cmap="RdBu_r", norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label="FL")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title(label, fontsize=10, fontweight="bold")

for ax in axes[n:]:
    ax.set_visible(False)

fig.suptitle("FL heatmaps — all cross-sections  (X, Z) → FL", fontsize=14, fontweight="bold")
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "ALL_FL_summary.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"Saved: {out_path}")


# ── continuity check: gradient magnitude ────────────────────────────────────
print("\n-- Continuity check (gradient magnitude of FL) --")
for fname in fl_files:
    df, label = load_file(fname)
    pt = pivot_fl(df)
    FL_mat = pt.values

    # finite differences (central where possible)
    grad_x = np.gradient(FL_mat, axis=1)   # dFL/dX  (axis=1 = columns = X)
    grad_z = np.gradient(FL_mat, axis=0)   # dFL/dZ  (axis=0 = rows    = Z)
    grad_mag = np.sqrt(grad_x**2 + grad_z**2)

    print(f"{label:20s}  max|grad_FL|={np.nanmax(grad_mag):.5f}  mean|grad_FL|={np.nanmean(grad_mag):.5f}")
