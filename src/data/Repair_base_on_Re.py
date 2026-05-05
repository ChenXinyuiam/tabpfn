"""Repair bad FL values in original processed files by Re-FL regression.

The script reads every ``*-FL.xlsx`` file from ``data/processed/Origin``.
For each file it:

1. Groups rows by spatial position ``(X, Z)``.
2. Aggregates duplicate rows at the same ``雷诺数`` with the mean FL value.
3. Fits a simple linear regression ``FL ~ 雷诺数`` inside each position group.
4. Flags large robust residuals as bad values.
5. Replaces flagged FL values with the regression prediction.

Outputs are written to ``data/processed/ReFixed`` by default. The Excel files
keep the original schema with repaired ``FL`` values; a separate CSV records
the original values, predictions, and repaired rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA_DIR, RE_COL, TARGET_RE_LIST
from utils.paths import ensure_dir
from utils.validation import check_columns

COL_X = "X"
COL_Z = "Z"
COL_FL = "FL"


@dataclass(frozen=True)
class RepairStats:
    file: str
    rows_in: int
    rows_out: int
    groups_total: int
    groups_complete: int
    groups_repaired: int
    values_repaired: int


def parse_re_values(value: str | None) -> list[int]:
    '''获取RE列表,默认是[6,12,...,60]'''
    if not value:
        return TARGET_RE_LIST
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def linear_predict(x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    if len(np.unique(x_train)) < 2:
        return np.full_like(x_pred, float(np.mean(y_train)), dtype=float)
    slope, intercept = np.polyfit(x_train, y_train, deg=1)
    return slope * x_pred + intercept


def robust_outlier_mask(residuals: np.ndarray, z_threshold: float) -> np.ndarray:
    if len(residuals) < 3:
        return np.zeros(len(residuals), dtype=bool)

    center = np.median(residuals)
    abs_dev = np.abs(residuals - center)
    mad = np.median(abs_dev)
    if mad > 1e-12:
        robust_z = 0.6745 * abs_dev / mad
        return robust_z > z_threshold

    std = np.std(residuals)
    if std <= 1e-12:
        return np.zeros(len(residuals), dtype=bool)
    return abs_dev / std > z_threshold


def aggregate_position(group: pd.DataFrame, re_col: str) -> pd.DataFrame:
    rows = []
    for _, re_group in group.groupby(re_col, sort=True):
        row = re_group.iloc[0].copy()
        row[COL_FL] = re_group[COL_FL].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(re_col).reset_index(drop=True)


def repair_position_group(
    group: pd.DataFrame,
    re_col: str,
    target_re_values: list[int],
    z_threshold: float,
    min_points: int,
) -> tuple[pd.DataFrame, list[dict], bool]:
    aggregated = aggregate_position(group, re_col)
    aggregated[re_col] = pd.to_numeric(aggregated[re_col], errors="coerce")
    aggregated[COL_FL] = pd.to_numeric(aggregated[COL_FL], errors="coerce")
    aggregated = aggregated.dropna(subset=[re_col, COL_FL]).reset_index(drop=True)

    aggregated["FL_original"] = aggregated[COL_FL]
    aggregated["FL_regression_pred"] = np.nan
    aggregated["FL_repair_residual"] = np.nan
    aggregated["FL_was_repaired"] = False

    re_values = sorted(aggregated[re_col].astype(int).tolist())
    is_complete = re_values == target_re_values
    if not is_complete or len(aggregated) < min_points:
        return aggregated, [], False

    x = aggregated[re_col].to_numpy(dtype=float)
    y = aggregated[COL_FL].to_numpy(dtype=float)
    pred = linear_predict(x, y, x)
    residuals = y - pred
    outlier_mask = robust_outlier_mask(residuals, z_threshold=z_threshold)

    if outlier_mask.any() and (~outlier_mask).sum() >= 2:
        pred = linear_predict(x[~outlier_mask], y[~outlier_mask], x)
        residuals = y - pred

    aggregated["FL_regression_pred"] = pred
    aggregated["FL_repair_residual"] = residuals

    repair_rows = []
    if outlier_mask.any():
        aggregated.loc[outlier_mask, COL_FL] = pred[outlier_mask]
        aggregated.loc[outlier_mask, "FL_was_repaired"] = True
        for _, row in aggregated.loc[outlier_mask].iterrows():
            repair_rows.append(
                {
                    "X": row[COL_X],
                    "Z": row[COL_Z],
                    re_col: row[re_col],
                    "FL_original": row["FL_original"],
                    "FL_repaired": row[COL_FL],
                    "FL_repair_residual": row["FL_repair_residual"],
                }
            )

    return aggregated, repair_rows, True


def repair_file(
    input_path: Path,
    output_path: Path,
    re_col: str,
    target_re_values: list[int],
    round_decimals: int,
    z_threshold: float,
    min_points: int,
    keep_diagnostics: bool,
) -> tuple[RepairStats, pd.DataFrame]:
    df = pd.read_excel(input_path, engine="openpyxl")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    check_columns(df, [COL_X, COL_Z, re_col, COL_FL])

    for column in [COL_X, COL_Z, re_col, COL_FL]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[COL_X, COL_Z, re_col, COL_FL]).reset_index(drop=True)

    df["X_round"] = df[COL_X].round(round_decimals)
    df["Z_round"] = df[COL_Z].round(round_decimals)

    repaired_groups = []
    repair_rows = []
    groups_total = 0
    groups_complete = 0
    groups_repaired = 0

    for (_, _), group in df.groupby(["X_round", "Z_round"], sort=True):
        groups_total += 1
        repaired, rows, is_complete = repair_position_group(
            group=group,
            re_col=re_col,
            target_re_values=target_re_values,
            z_threshold=z_threshold,
            min_points=min_points,
        )
        groups_complete += int(is_complete)
        groups_repaired += int(bool(rows))
        for row in rows:
            row["file"] = input_path.name
        repair_rows.extend(rows)
        repaired_groups.append(repaired)

    out_df = pd.concat(repaired_groups, ignore_index=True) if repaired_groups else pd.DataFrame()
    drop_columns = ["X_round", "Z_round"]
    if not keep_diagnostics:
        drop_columns.extend(["FL_original", "FL_regression_pred", "FL_repair_residual", "FL_was_repaired"])
    out_df = out_df.drop(columns=drop_columns, errors="ignore")
    out_df.to_excel(output_path, index=False, engine="openpyxl")

    stats = RepairStats(
        file=input_path.name,
        rows_in=len(df),
        rows_out=len(out_df),
        groups_total=groups_total,
        groups_complete=groups_complete,
        groups_repaired=groups_repaired,
        values_repaired=len(repair_rows),
    )
    return stats, pd.DataFrame(repair_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair bad FL values in Origin processed files.")
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "processed" / "Origin")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "processed" / "ReFixed")
    parser.add_argument("--pattern", default="*-FL.xlsx")
    parser.add_argument("--re-col", default=RE_COL)
    parser.add_argument("--target-re", default=None, help="Comma-separated Re values. Defaults to 6,12,...,60.")
    parser.add_argument("--round-decimals", type=int, default=6)
    parser.add_argument("--z-threshold", type=float, default=3.5)
    parser.add_argument("--min-points", type=int, default=6)
    parser.add_argument("--keep-diagnostics", action="store_true")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    ensure_dir(args.output_dir)
    target_re_values = parse_re_values(args.target_re)
    files = sorted(path for path in args.input_dir.glob(args.pattern) if not path.name.startswith("~$"))
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} under {args.input_dir}")

    stats_rows = []
    repair_reports = []

    for input_path in files:
        output_path = args.output_dir / input_path.name
        print(f"Repairing {input_path.name} -> {output_path}")
        stats, report = repair_file(
            input_path=input_path,
            output_path=output_path,
            re_col=args.re_col,
            target_re_values=target_re_values,
            round_decimals=args.round_decimals,
            z_threshold=args.z_threshold,
            min_points=args.min_points,
            keep_diagnostics=args.keep_diagnostics,
        )
        stats_rows.append(stats.__dict__)
        if not report.empty:
            repair_reports.append(report)
        print(
            f"  complete groups={stats.groups_complete}/{stats.groups_total}, "
            f"repaired values={stats.values_repaired}"
        )

    summary = pd.DataFrame(stats_rows)
    summary_path = args.output_dir / "_repair_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    report = pd.concat(repair_reports, ignore_index=True) if repair_reports else pd.DataFrame()
    report_path = args.output_dir / "_repair_values.csv"
    report.to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"Saved summary: {summary_path}")
    print(f"Saved repaired-value report: {report_path}")


if __name__ == "__main__":
    main()
