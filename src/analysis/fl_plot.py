"""Create FL heatmaps and continuity summaries for processed FL files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import PUBLISHED_PROCESSED_ROOT
from utils.paths import ensure_dir, select_batch_dir
from utils.plotting import gradient_summary, load_fl_excel, pivot_fl, symmetric_norm


def plot_file(df, label: str, output_dir: Path, output_label_suffix: str) -> None:
    pt = pivot_fl(df)
    x_vals = pt.columns.values.astype(float)
    z_vals = pt.index.values.astype(float)
    fl_mat = pt.values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"FL distribution - {label}", fontsize=14, fontweight="bold")

    ax = axes[0]
    im = ax.pcolormesh(x_vals, z_vals, fl_mat, cmap="RdBu_r", norm=symmetric_norm(fl_mat), shading="nearest")
    plt.colorbar(im, ax=ax, label="FL")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("FL heatmap (X, Z)")

    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_vals)))
    for i, z in enumerate(z_vals):
        row = df[df["Z"] == z].sort_values("X")
        ax.plot(row["X"], row["FL"], marker="o", ms=3, color=colors[i], label=f"Z={z:.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("X")
    ax.set_ylabel("FL")
    ax.set_title("FL vs X (each line = one Z)")
    ax.legend(fontsize=7, ncol=2, loc="best")

    ax = axes[2]
    colors = plt.cm.plasma(np.linspace(0, 1, len(x_vals)))
    for i, x in enumerate(x_vals):
        col = df[df["X"] == x].sort_values("Z")
        ax.plot(col["Z"], col["FL"], marker="s", ms=3, color=colors[i], label=f"X={x:.0f}")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Z")
    ax.set_ylabel("FL")
    ax.set_title("FL vs Z (each line = one X)")
    ax.legend(fontsize=5, ncol=3, loc="best")

    plt.tight_layout()
    out_path = output_dir / f"{label}{output_label_suffix}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary(file_data: list[tuple[object, str]], output_dir: Path, filename: str) -> None:
    if not file_data:
        return

    ncols = 4
    nrows = (len(file_data) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for idx, (df, label) in enumerate(file_data):
        pt = pivot_fl(df)
        fl_mat = pt.values
        ax = axes[idx]
        im = ax.pcolormesh(
            pt.columns.values.astype(float),
            pt.index.values.astype(float),
            fl_mat,
            cmap="RdBu_r",
            norm=symmetric_norm(fl_mat),
            shading="nearest",
        )
        plt.colorbar(im, ax=ax, label="FL")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(label, fontsize=10, fontweight="bold")

    for ax in axes[len(file_data):]:
        ax.set_visible(False)

    fig.suptitle("FL heatmaps - all cross-sections (X, Z) -> FL", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FL heatmaps for processed files.")
    parser.add_argument("--batch", default=None, help="Batch under data/processed. Defaults to the latest batch.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*-FL.xlsx")
    parser.add_argument("--suffix", default="-FL.xlsx")
    parser.add_argument("--summary-name", default="ALL_FL_summary.png")
    parser.add_argument("--output-label-suffix", default="_FL")
    args = parser.parse_args()

    data_dir = args.data_dir or select_batch_dir(PUBLISHED_PROCESSED_ROOT, args.batch)
    output_dir = ensure_dir(args.out_dir or (data_dir / "figures" / "original"))
    files = sorted(path for path in data_dir.glob(args.pattern) if not path.name.startswith("~$"))
    if not files:
        print(f"No {args.pattern} files found in {data_dir}")
        return

    file_data = []
    for path in files:
        df, label = load_fl_excel(path, args.suffix)
        plot_file(df, label, output_dir, args.output_label_suffix)
        file_data.append((df, label))

    plot_summary(file_data, output_dir, args.summary_name)

    print("\n-- Continuity check (gradient magnitude of FL) --")
    for df, label in file_data:
        max_grad, mean_grad = gradient_summary(pivot_fl(df).values)
        print(f"{label:20s}  max|grad_FL|={max_grad:.5f}  mean|grad_FL|={mean_grad:.5f}")


if __name__ == "__main__":
    main()
