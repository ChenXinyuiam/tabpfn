"""Plot Re-FL curves for positions that contain every target Re value."""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ANALYSIS_DIR, PUBLISHED_PROCESSED_DIR, RAW_DIR, RE_COL, TARGET_RE_LIST
from utils.paths import ensure_dir, select_batch_dir
from utils.validation import check_columns

COL_X = "X"
COL_Z = "Z"
COL_FL = "FL"
N_SUBPLOTS_PER_FIG = 10


def load_data(path: Path, re_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_excel(path, engine="openpyxl")
    check_columns(df, [COL_X, COL_Z, re_col, COL_FL])
    return df


def preprocess(df: pd.DataFrame, re_col: str, round_decimals: int) -> pd.DataFrame:
    df = df.copy()
    for column in [COL_X, COL_Z, re_col, COL_FL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[COL_X, COL_Z, re_col, COL_FL]).reset_index(drop=True)
    df["X_round"] = df[COL_X].round(round_decimals)
    df["Z_round"] = df[COL_Z].round(round_decimals)
    return df


def find_valid_positions(df: pd.DataFrame, re_col: str, target_re_list: list[int]) -> list[pd.DataFrame]:
    valid_groups = []
    for (x_value, z_value), group in df.groupby(["X_round", "Z_round"], sort=True):
        by_re = group.groupby(re_col, as_index=False)[COL_FL].mean()
        re_values = sorted(by_re[re_col].astype(int).tolist())
        if re_values == target_re_list:
            by_re["X_round"] = x_value
            by_re["Z_round"] = z_value
            valid_groups.append(by_re)
    return valid_groups


def plot_groups(
    valid_groups: list[pd.DataFrame],
    output_dir: Path,
    re_col: str,
    target_re_list: list[int],
    n_subplots_per_fig: int = N_SUBPLOTS_PER_FIG,
) -> None:
    ensure_dir(output_dir)

    if not valid_groups:
        print("No positions contain the full Re sequence.")
        return

    n_figs = math.ceil(len(valid_groups) / n_subplots_per_fig)
    print(f"Valid positions: {len(valid_groups)}")
    print(f"Figures to write: {n_figs}")

    for fig_idx in range(n_figs):
        start = fig_idx * n_subplots_per_fig
        end = min((fig_idx + 1) * n_subplots_per_fig, len(valid_groups))
        chunk = valid_groups[start:end]

        fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(12, 18))
        axes = axes.flatten()

        for ax_idx, ax in enumerate(axes):
            if ax_idx >= len(chunk):
                ax.axis("off")
                continue

            group = chunk[ax_idx].sort_values(by=re_col)
            x_value = group["X_round"].iloc[0]
            z_value = group["Z_round"].iloc[0]
            ax.scatter(group[re_col], group[COL_FL])
            ax.plot(group[re_col], group[COL_FL])
            ax.set_title(f"X={x_value}, Z={z_value}")
            ax.set_xlabel("Re")
            ax.set_ylabel("FL")
            ax.set_xticks(target_re_list)

        fig.suptitle(f"Re-FL scatter by position | Figure {fig_idx + 1}/{n_figs}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = output_dir / f"re_fl_scatter_group_{fig_idx + 1:02d}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {save_path}")


def write_positions(valid_groups: list[pd.DataFrame], output_dir: Path) -> None:
    if not valid_groups:
        return
    rows = [
        {"X_round": group["X_round"].iloc[0], "Z_round": group["Z_round"].iloc[0]}
        for group in valid_groups
    ]
    out_path = output_dir / "valid_positions.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved positions: {out_path}")


def process_file(input_file: Path, output_dir: Path, re_col: str, round_decimals: int) -> None:
    print(f"\n[{input_file.name}]")
    df = load_data(input_file, re_col)
    df = preprocess(df, re_col, round_decimals)
    valid_groups = find_valid_positions(df, re_col, TARGET_RE_LIST)
    plot_groups(valid_groups, output_dir, re_col, TARGET_RE_LIST)
    write_positions(valid_groups, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Re-FL curves for complete positions.")
    parser.add_argument("--batch", default=None, help="Batch under data/rawdata. Defaults to the latest batch.")
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--pattern", default="*FL*.xlsx")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR / "ReFixed" / "ReFixed830-342-FL")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--re-col", default=RE_COL)
    parser.add_argument("--round-decimals", type=int, default=6)
    args = parser.parse_args()

    if args.input_file:
        process_file(args.input_file, args.output_dir, args.re_col, args.round_decimals)
        return

    input_dir = args.input_dir or select_batch_dir(RAW_DIR, args.batch)
    output_root = args.output_root or (input_dir / "figures" / "fl_re")
    files = sorted(path for path in input_dir.glob(args.pattern) if not path.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} under {input_dir}")
    for input_file in files:
        output_dir = output_root / input_file.stem
        process_file(input_file, output_dir, args.re_col, args.round_decimals)


if __name__ == "__main__":
    main()
