"""Prepare FL raw data with per-column unit normalization."""

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
    BASE_COL,
    D_BASE_COL,
    D_LEFT_COL,
    D_RIGHT_COL,
    INTERIM_DIR,
    RAW_DIR,
)
from src.data.calculate import compute_geometry_features
from src.data.clean import impute_centroid_distance, impute_distance_cols
from src.data.units import normalize_units
from utils.paths import ensure_dir, iter_excel_files_for_batch, select_batch_dir

FL_OUTPUT_ROOT = INTERIM_DIR / "fl"
FL_TARGET_COL = "FL"

FL_COLUMN_ALIASES = {
    "BC": BASE_COL,
    "角度A": ANGLE_A_COL,
    "角度B": ANGLE_B_COL,
    "角度C": ANGLE_C_COL,
    "点到BC的垂直距离": D_BASE_COL,
    "到BC垂直距离": D_BASE_COL,
    "点到AB的垂直距离": D_LEFT_COL,
    "到AB垂直距离": D_LEFT_COL,
    "点到AC的垂直距离": D_RIGHT_COL,
    "到AC垂直距离": D_RIGHT_COL,
    "Fly": FL_TARGET_COL,
    "FLy": FL_TARGET_COL,
}


def normalized_output_name(filename: str) -> str:
    match = re.search(r"(\d+-\d+)", filename)
    if not match:
        return filename
    suffix = "-sym.xlsx" if "反" in filename else "-FL.xlsx"
    return f"{match.group(1)}{suffix}"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        source: target
        for source, target in FL_COLUMN_ALIASES.items()
        if source in df.columns and target not in df.columns
    }
    return df.rename(columns=rename_map)


def prepare_fl_frame(df: pd.DataFrame, *, max_abs_fl: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = normalize_columns(df)
    df = df.dropna(axis=1, how="all")

    skeleton_mask = df.isnull().sum(axis=1) > 8
    if skeleton_mask.any():
        df = df[~skeleton_mask].reset_index(drop=True)

    df, unit_report = normalize_units(df)

    dist_cols = [column for column in [D_BASE_COL, D_LEFT_COL, D_RIGHT_COL] if column in df.columns]
    if len(dist_cols) == 3 and df[dist_cols].isnull().any().any():
        df = impute_distance_cols(df, dist_cols)
    df = impute_centroid_distance(df)

    if FL_TARGET_COL not in df.columns:
        raise RuntimeError(f"Missing required FL target column: {FL_TARGET_COL}")
    df[FL_TARGET_COL] = pd.to_numeric(df[FL_TARGET_COL], errors="coerce")
    if max_abs_fl is not None:
        df = df[df[FL_TARGET_COL].abs() <= max_abs_fl].reset_index(drop=True)
    df = df.dropna(subset=[FL_TARGET_COL]).reset_index(drop=True)

    df = compute_geometry_features(df)
    return df, unit_report


def process_batch(batch: str | None, output_root: Path, max_abs_fl: float | None) -> pd.DataFrame:
    batch_dir = select_batch_dir(RAW_DIR, batch)
    batch_name = batch_dir.name if batch_dir != RAW_DIR else "root"
    files = iter_excel_files_for_batch(batch_dir)
    if not files:
        raise RuntimeError(f"No .xlsx files found in {batch_dir}")

    dst_root = ensure_dir(output_root / batch_name)
    manifest_rows: list[dict[str, object]] = []
    unit_reports: list[pd.DataFrame] = []

    for source_path in files:
        relative_parent = source_path.relative_to(batch_dir).parent
        output_name = normalized_output_name(source_path.name)
        output_path = ensure_dir(dst_root / relative_parent) / output_name

        raw_df = pd.read_excel(source_path, engine="openpyxl")
        processed, unit_report = prepare_fl_frame(raw_df, max_abs_fl=max_abs_fl)
        processed["source_file"] = source_path.name
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
                "fl_min": float(np.nanmin(processed[FL_TARGET_COL])) if len(processed) else np.nan,
                "fl_max": float(np.nanmax(processed[FL_TARGET_COL])) if len(processed) else np.nan,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(dst_root / "manifest_fl.csv", index=False, encoding="utf-8-sig")
    if unit_reports:
        pd.concat(unit_reports, ignore_index=True).to_csv(
            dst_root / "unit_report_fl.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FL data with safe unit normalization.")
    parser.add_argument("--batch", default="2026-05-01")
    parser.add_argument("--output-root", type=Path, default=FL_OUTPUT_ROOT)
    parser.add_argument("--max-abs-fl", type=float, default=None)
    args = parser.parse_args()

    manifest = process_batch(args.batch, args.output_root, args.max_abs_fl)
    print(f"Prepared FL files: {len(manifest)}")
    print(f"Rows written: {int(manifest['rows_out'].sum())}")
    print(f"Output root: {args.output_root / args.batch}")


if __name__ == "__main__":
    main()

