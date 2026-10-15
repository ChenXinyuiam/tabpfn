"""Prepare FD raw data with per-column unit normalization."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ANGLE_A_COL,
    ANGLE_B_COL,
    ANGLE_C_COL,
    AREA_COL,
    BASE_COL,
    HEIGHT_COL,
    INTERIM_DIR,
    PERIMETER_COL,
    RAW_DIR,
    RE_COL,
    X_COL,
    Z_COL,
)
from src.data.calculate import (
    ANGLE_SUM_COL,
    AREA_CALC_COL,
    LEFT_SIDE_COL,
    RIGHT_SIDE_COL,
    VALID_GEOMETRY_COL,
    _sin_deg,
)
from src.data.units import normalize_units
from utils.paths import ensure_dir, iter_excel_files_for_batch

FD_OUTPUT_ROOT = INTERIM_DIR / "fd"
FD_TARGET_COLS = ["dean_velocity_x", "dean_velocity_y"]

FD_COLUMN_ALIASES = {
    "Re": RE_COL,
    "X [ m ]": X_COL,
    "Y [ m ]": "Y",
    "Z [ m ]": Z_COL,
    "底边": BASE_COL,
    "角度A": ANGLE_A_COL,
    "角度B": ANGLE_B_COL,
    "角度C": ANGLE_C_COL,
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in FD_COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    return df.rename(columns=rename_map)


def find_fd_group_dir(batch: str, radius: str) -> Path:
    batch_root = RAW_DIR / batch
    direct = batch_root / "FD数据" / radius
    if direct.is_dir():
        return direct
    matches = sorted(path for path in batch_root.glob(f"FD*/{radius}") if path.is_dir())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous FD radius {radius} under {batch_root}: {matches}")
    raise FileNotFoundError(f"FD data directory not found for batch={batch}, radius={radius}")


def add_fd_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = [
        BASE_COL,
        HEIGHT_COL,
        AREA_COL,
        PERIMETER_COL,
        ANGLE_A_COL,
        ANGLE_B_COL,
        ANGLE_C_COL,
        X_COL,
        Z_COL,
        RE_COL,
        *FD_TARGET_COLS,
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    required = {BASE_COL, HEIGHT_COL, ANGLE_A_COL, ANGLE_B_COL, ANGLE_C_COL}
    if required.issubset(df.columns):
        df[AREA_CALC_COL] = df[BASE_COL] * df[HEIGHT_COL] / 2.0
        df[ANGLE_SUM_COL] = df[ANGLE_A_COL] + df[ANGLE_B_COL] + df[ANGLE_C_COL]
        sin_a = _sin_deg(df[ANGLE_A_COL])
        valid = (
            (df[BASE_COL] > 0)
            & (df[HEIGHT_COL] > 0)
            & (np.abs(df[ANGLE_SUM_COL] - 180.0) < 1.0)
            & (np.abs(sin_a) > 1e-15)
        )
        df[VALID_GEOMETRY_COL] = valid
        df[RIGHT_SIDE_COL] = np.nan
        df[LEFT_SIDE_COL] = np.nan
        df.loc[valid, RIGHT_SIDE_COL] = (
            df.loc[valid, BASE_COL] * _sin_deg(df.loc[valid, ANGLE_B_COL]) / sin_a.loc[valid]
        )
        df.loc[valid, LEFT_SIDE_COL] = (
            df.loc[valid, BASE_COL] * _sin_deg(df.loc[valid, ANGLE_C_COL]) / sin_a.loc[valid]
        )

    if all(column in df.columns for column in FD_TARGET_COLS):
        df["FD_magnitude"] = np.sqrt(df[FD_TARGET_COLS[0]] ** 2 + df[FD_TARGET_COLS[1]] ** 2)
    return df


def prepare_fd_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)
    df = df.dropna(axis=1, how="all")
    df, unit_report = normalize_units(df)

    missing_targets = [column for column in FD_TARGET_COLS if column not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing FD target columns: {missing_targets}")

    for column in FD_TARGET_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=FD_TARGET_COLS).reset_index(drop=True)
    df = add_fd_features(df)
    return df, unit_report


def process_group(input_dir: Path, output_root: Path, batch: str, radius: str) -> pd.DataFrame:
    files = iter_excel_files_for_batch(input_dir)
    if not files:
        raise RuntimeError(f"No .xlsx files found in {input_dir}")

    dst_root = ensure_dir(output_root / batch / radius)
    manifest_rows: list[dict[str, object]] = []
    unit_reports: list[pd.DataFrame] = []

    for source_path in files:
        output_name = re.sub(r"\s+", "_", source_path.name)
        output_path = dst_root / output_name

        raw_df = pd.read_excel(source_path, engine="openpyxl")
        processed, unit_report = prepare_fd_frame(raw_df)
        processed["source_file"] = source_path.name
        processed["radius_group"] = radius
        processed.to_excel(output_path, index=False, engine="openpyxl")

        if not unit_report.empty:
            unit_report.insert(0, "source_file", source_path.name)
            unit_reports.append(unit_report)

        manifest_rows.append(
            {
                "source_file": source_path.as_posix(),
                "output_file": output_path.as_posix(),
                "rows_in": int(len(raw_df)),
                "rows_out": int(len(processed)),
                "columns_out": int(len(processed.columns)),
                "fd_x_min": float(np.nanmin(processed[FD_TARGET_COLS[0]])) if len(processed) else np.nan,
                "fd_x_max": float(np.nanmax(processed[FD_TARGET_COLS[0]])) if len(processed) else np.nan,
                "fd_y_min": float(np.nanmin(processed[FD_TARGET_COLS[1]])) if len(processed) else np.nan,
                "fd_y_max": float(np.nanmax(processed[FD_TARGET_COLS[1]])) if len(processed) else np.nan,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(dst_root / "manifest_fd.csv", index=False, encoding="utf-8-sig")
    if unit_reports:
        pd.concat(unit_reports, ignore_index=True).to_csv(
            dst_root / "unit_report_fd.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FD data with safe unit normalization.")
    parser.add_argument("--batch", default="2026-05-25")
    parser.add_argument("--radius", default="10")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=FD_OUTPUT_ROOT)
    args = parser.parse_args()

    input_dir = args.input_dir or find_fd_group_dir(args.batch, args.radius)
    manifest = process_group(input_dir, args.output_root, args.batch, args.radius)
    print(f"Prepared FD files: {len(manifest)}")
    print(f"Rows written: {int(manifest['rows_out'].sum())}")
    print(f"Output root: {args.output_root / args.batch / args.radius}")


if __name__ == "__main__":
    main()
