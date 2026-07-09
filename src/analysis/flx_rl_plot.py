"""
输入: 单个表格(含有X,Z,RL,FLx的列)
绘制 FLx-RL 曲线 by fixed (X, Z) position.

默认文件是2026-05-31
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import RAW_DIR, RESULTS_DIR
from utils.paths import ensure_dir, iter_excel_files_for_batch, select_batch_dir
from utils.validation import check_columns

COL_X = "X"
COL_Z = "Z"
COL_FLX = "FLx"
DEFAULT_ROUND_DECIMALS = 10
N_SUBPLOTS_PER_FIG = 10


def infer_rl_col(df: pd.DataFrame, explicit_col: str | None = None) -> str:
    if explicit_col:
        check_columns(df, [explicit_col])
        return explicit_col

    if "RL" in df.columns:
        return "RL"

    numeric_candidates = []
    for column in df.columns:
        if column in {COL_X, COL_Z, COL_FLX}:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            numeric_candidates.append(column)

    if len(numeric_candidates) != 1:
        raise ValueError(
            "Could not infer RL column. "
            f"Numeric candidates={numeric_candidates}; pass --rl-col explicitly."
        )
    return numeric_candidates[0]


def load_data(path: Path, rl_col: str | None) -> tuple[pd.DataFrame, str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    check_columns(df, [COL_X, COL_Z, COL_FLX])
    resolved_rl_col = infer_rl_col(df, rl_col)
    df = df[[COL_X, COL_Z, resolved_rl_col, COL_FLX]].copy()
    df = df.rename(columns={resolved_rl_col: "RL"})

    for column in [COL_X, COL_Z, "RL", COL_FLX]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[COL_X, COL_Z, "RL", COL_FLX]).reset_index(drop=True)
    return df, resolved_rl_col


def add_rounded_position(df: pd.DataFrame, round_decimals: int) -> pd.DataFrame:
    df = df.copy()
    df["X_round"] = df[COL_X].round(round_decimals)
    df["Z_round"] = df[COL_Z].round(round_decimals)
    return df


def positions_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["X_round", "Z_round"], as_index=False)
        .agg(
            n_points=("RL", "size"),
            n_rl=("RL", "nunique"),
            rl_min=("RL", "min"),
            rl_max=("RL", "max"),
            flx_min=(COL_FLX, "min"),
            flx_max=(COL_FLX, "max"),
        )
        .sort_values(["n_rl", "X_round", "Z_round"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return summary


def choose_position(
    df: pd.DataFrame,
    x_value: float | None,
    z_value: float | None,
    round_decimals: int,
) -> tuple[float, float, str]:
    if (x_value is None) != (z_value is None):
        raise ValueError("Please provide both --x and --z, or omit both.")

    summary = positions_summary(df)
    if summary.empty:
        raise ValueError("No valid positions found after dropping missing values.")

    if x_value is None and z_value is None:
        chosen = summary.iloc[0]
        return float(chosen["X_round"]), float(chosen["Z_round"]), "default_complete_position"

    x_round = round(float(x_value), round_decimals)
    z_round = round(float(z_value), round_decimals)
    exact = summary[(summary["X_round"] == x_round) & (summary["Z_round"] == z_round)]
    if not exact.empty:
        return x_round, z_round, "requested_position"

    distances = (summary["X_round"] - x_round).pow(2) + (summary["Z_round"] - z_round).pow(2)
    nearest_idx = distances.idxmin()
    nearest = summary.loc[nearest_idx]
    return float(nearest["X_round"]), float(nearest["Z_round"]), "nearest_available_position"


def selected_curve(df: pd.DataFrame, x_round: float, z_round: float) -> pd.DataFrame:
    group = df[(df["X_round"] == x_round) & (df["Z_round"] == z_round)].copy()
    if group.empty:
        raise ValueError(f"No rows found for X={x_round}, Z={z_round}")

    curve = group[[COL_X, COL_Z, "X_round", "Z_round", "RL", COL_FLX]].sort_values(
        ["RL", COL_FLX],
        kind="mergesort",
    )
    return curve


def position_groups(df: pd.DataFrame) -> list[pd.DataFrame]:
    groups = []
    for (_, _), group in df.groupby(["X_round", "Z_round"], sort=True):
        groups.append(group.sort_values(["RL", COL_FLX], kind="mergesort").reset_index(drop=True))
    return groups


def display_coord(value: float, scale: float) -> str:
    return f"{value * scale:g}"


def plot_position_pages(
    groups: list[pd.DataFrame],
    output_dir: Path,
    file_label: str,
    display_scale: float,
    n_subplots_per_fig: int = N_SUBPLOTS_PER_FIG,
) -> None:
    if not groups:
        print("No positions to plot.")
        return

    n_figs = math.ceil(len(groups) / n_subplots_per_fig)
    print(f"Positions to plot: {len(groups)}")
    print(f"Figures to write: {n_figs}")

    for fig_idx in range(n_figs):
        start = fig_idx * n_subplots_per_fig
        end = min((fig_idx + 1) * n_subplots_per_fig, len(groups))
        chunk = groups[start:end]

        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18))
        axes = axes.flatten()

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(chunk):
                ax.axis("off")
                continue

            group = chunk[ax_idx]
            x_value = float(group["X_round"].iloc[0])
            z_value = float(group["Z_round"].iloc[0])
            ax.scatter(group["RL"], group[COL_FLX], s=28)
            ax.plot(group["RL"], group[COL_FLX], linewidth=1.4)
            ax.set_title(
                f"X={display_coord(x_value, display_scale)}, Z={display_coord(z_value, display_scale)}"
            )
            ax.set_xlabel("RL")
            ax.set_ylabel("FLx")

        fig.suptitle(f"RL-FLx scatter by position | Figure {fig_idx + 1}/{n_figs}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = output_dir / f"{file_label}_flx_rl_scatter_group_{fig_idx + 1:02d}.png"
        fig.savefig(save_path, dpi=220)
        plt.close(fig)
        print(f"Saved: {save_path}")


def plot_curve(curve: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["RL"], curve[COL_FLX], marker="o", linewidth=1.8, markersize=5)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.65)
    ax.set_xlabel("RL")
    ax.set_ylabel("FLx")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def process_file(
    input_file: Path,
    output_dir: Path,
    rl_col: str | None,
    x_value: float | None,
    z_value: float | None,
    round_decimals: int,
    display_scale: float,
) -> None:
    df, source_rl_col = load_data(input_file, rl_col)
    df = add_rounded_position(df, round_decimals)
    ensure_dir(output_dir)

    summary = positions_summary(df)
    positions_path = output_dir / "available_positions.csv"
    summary.to_csv(positions_path, index=False, encoding="utf-8-sig")

    file_label = input_file.stem
    if x_value is None and z_value is None:
        all_points = df[[COL_X, COL_Z, "X_round", "Z_round", "RL", COL_FLX]].sort_values(
            ["X_round", "Z_round", "RL", COL_FLX],
            kind="mergesort",
        )
        all_points_path = output_dir / f"{file_label}_all_positions_FLx_RL_points.csv"
        all_points.to_csv(all_points_path, index=False, encoding="utf-8-sig")
        plot_position_pages(position_groups(df), output_dir, file_label, display_scale)

        print(f"Input: {input_file}")
        print(f"Rows used: {len(df)}")
        print(f"RL column: {source_rl_col!r} -> RL")
        print("Position mode: all_positions_raw_points")
        print(f"Saved positions: {positions_path}")
        print(f"Saved all curve points: {all_points_path}")
        return

    x_round, z_round, mode = choose_position(df, x_value, z_value, round_decimals)
    curve = selected_curve(df, x_round, z_round)

    coord_label = f"X_{x_round:.10g}_Z_{z_round:.10g}"
    curve_path = output_dir / f"{file_label}_{coord_label}_FLx_RL_points.csv"
    figure_path = output_dir / f"{file_label}_{coord_label}_FLx_RL.png"
    curve.to_csv(curve_path, index=False, encoding="utf-8-sig")

    title = f"FLx-RL | {file_label} | X={x_round:.10g}, Z={z_round:.10g}"
    plot_curve(curve, figure_path, title)

    print(f"Input: {input_file}")
    print(f"Rows used: {len(df)}")
    print(f"RL column: {source_rl_col!r} -> RL")
    print(f"Position mode: {mode}")
    print(f"Selected X={x_round:.10g}, Z={z_round:.10g}, points={len(curve)}")
    print(f"Saved positions: {positions_path}")
    print(f"Saved curve data: {curve_path}")
    print(f"Saved figure: {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FLx-RL curve for a fixed X and Z.")
    parser.add_argument("--batch", default="2026-5-31", help="Batch under data/rawdata.")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*.xlsx")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rl-col", default=None, help="RL column name. Defaults to automatic inference.")
    parser.add_argument("--x", type=float, default=None, help="Fixed X value.")
    parser.add_argument("--z", type=float, default=None, help="Fixed Z value.")
    parser.add_argument("--round-decimals", type=int, default=DEFAULT_ROUND_DECIMALS)
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1_000_000.0,
        help="Scale used only for X/Z subplot titles. Default converts meter-like values to micrometers.",
    )
    args = parser.parse_args()

    if args.input_file:
        input_files = [args.input_file]
        base_output_dir = args.output_dir or (RESULTS_DIR / "FLx_RL" / args.input_file.stem)
    else:
        input_dir = args.input_dir or select_batch_dir(RAW_DIR, args.batch)
        input_files = [
            path
            for path in iter_excel_files_for_batch(input_dir)
            if path.match(args.pattern) and not path.name.startswith("~$")
        ]
        if not input_files:
            raise FileNotFoundError(f"No files matching {args.pattern} under {input_dir}")
        base_output_dir = args.output_dir or (RESULTS_DIR / "FLx_RL" / input_dir.name)

    for input_file in input_files:
        output_dir = base_output_dir if len(input_files) == 1 else base_output_dir / input_file.stem
        process_file(
            input_file=input_file,
            output_dir=output_dir,
            rl_col=args.rl_col,
            x_value=args.x,
            z_value=args.z,
            round_decimals=args.round_decimals,
            display_scale=args.display_scale,
        )


if __name__ == "__main__":
    main()
