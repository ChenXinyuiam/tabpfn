"""Draw FD heatmaps by triangle position for the 2026-05-21 batch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

from src.config import RAW_DIR, RESULTS_DIR


X_COL = "X [ m ]"
Z_COL = "Z [ m ]"
RE_COL = "Re"
VX_COL = "dean_velocity_x"
VY_COL = "dean_velocity_y"
FD_MAG_COL = "FD_magnitude"


def safe_stem(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", Path(name).stem)


def discover_group_dirs(batch_root: Path, group_names: list[str]) -> list[Path]:
    groups = set(group_names)
    dirs = [path for path in batch_root.rglob("*") if path.is_dir() and path.name in groups]
    return sorted(dirs, key=lambda path: int(path.name))


def add_fd_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in [RE_COL, X_COL, Z_COL, VX_COL, VY_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FD_MAG_COL] = np.sqrt(df[VX_COL] ** 2 + df[VY_COL] ** 2)
    return df.dropna(subset=[RE_COL, X_COL, Z_COL, FD_MAG_COL])


def draw_panel(ax, sub: pd.DataFrame, value_col: str, title: str, vmin: float, vmax: float) -> None:
    x_um = sub[X_COL].to_numpy(dtype=float) * 1e6
    z_um = sub[Z_COL].to_numpy(dtype=float) * 1e6
    values = sub[value_col].to_numpy(dtype=float)

    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (um)", fontsize=8)
    ax.set_ylabel("Z (um)", fontsize=8)
    ax.tick_params(labelsize=7)

    if len(sub) < 3 or np.nanmax(values) == np.nanmin(values):
        sc = ax.scatter(x_um, z_um, c=values, s=6, cmap="viridis", vmin=vmin, vmax=vmax)
        return sc

    try:
        triang = mtri.Triangulation(x_um, z_um)
        return ax.tricontourf(triang, values, levels=40, cmap="viridis", vmin=vmin, vmax=vmax)
    except Exception:
        return ax.scatter(x_um, z_um, c=values, s=6, cmap="viridis", vmin=vmin, vmax=vmax)


def plot_triangle_heatmap(
    df: pd.DataFrame,
    source_file: Path,
    group: str,
    out_dir: Path,
    value_col: str,
) -> dict:
    re_values = sorted(int(value) for value in df[RE_COL].dropna().unique())
    n_cols = 5
    n_rows = int(np.ceil(len(re_values) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, max(3.2, 3.1 * n_rows)), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)

    vmin = float(df[value_col].quantile(0.01))
    vmax = float(df[value_col].quantile(0.99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(df[value_col].min())
        vmax = float(df[value_col].max())

    mappable = None
    for ax, re_value in zip(axes_arr, re_values):
        sub = df[df[RE_COL] == re_value]
        mappable = draw_panel(ax, sub, value_col, f"Re={re_value}", vmin, vmax)

    for ax in axes_arr[len(re_values):]:
        ax.axis("off")

    fig.suptitle(f"group {group} | {source_file.stem} | {value_col}", fontsize=14)
    if mappable is not None:
        fig.colorbar(mappable, ax=axes_arr.tolist(), shrink=0.82, label=value_col)

    out_path = out_dir / f"{safe_stem(source_file.name)}_{value_col}_heatmap.png"
    fig.savefig(out_path, dpi=260)
    plt.close(fig)

    return {
        "group": group,
        "file": source_file.name,
        "plot": out_path.as_posix(),
        "rows": int(len(df)),
        "re_count": len(re_values),
        "value_col": value_col,
        "value_min": float(df[value_col].min()),
        "value_max": float(df[value_col].max()),
        "value_mean": float(df[value_col].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FD heatmaps by triangle position.")
    parser.add_argument("--batch", default="2026-05-21")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR / "fd_heatmaps" / "2026-05-21")
    parser.add_argument("--groups", default="10,20,30")
    parser.add_argument(
        "--value",
        choices=[FD_MAG_COL, VX_COL, VY_COL],
        default=FD_MAG_COL,
        help="FD value to draw. Default is velocity magnitude.",
    )
    args = parser.parse_args()

    batch_root = args.data_root or (RAW_DIR / args.batch)
    group_names = [item.strip() for item in args.groups.split(",") if item.strip()]
    group_dirs = discover_group_dirs(batch_root, group_names)
    missing = sorted(set(group_names) - {path.name for path in group_dirs})
    if missing:
        raise RuntimeError(f"Missing group directories under {batch_root}: {missing}")

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for group_dir in group_dirs:
        group_out_dir = args.out_dir / f"group_{group_dir.name}"
        group_out_dir.mkdir(parents=True, exist_ok=True)
        files = [path for path in sorted(group_dir.glob("*.xlsx")) if not path.name.startswith("~$")]
        print(f"group {group_dir.name}: plotting {len(files)} triangle files")
        for path in files:
            df = add_fd_columns(pd.read_excel(path, engine="openpyxl"))
            row = plot_triangle_heatmap(df, path, group_dir.name, group_out_dir, args.value)
            rows.append(row)
            print(f"  saved {row['plot']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / f"heatmap_summary_{args.value}.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved heatmaps to {args.out_dir}")


if __name__ == "__main__":
    main()
