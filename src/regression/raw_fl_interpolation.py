"""Interpolate raw FL/FLX workbooks independently within each Re slice."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, QhullError, cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

COORD_COLUMNS = ("x", "z")
GROUP_COLUMN = "re"
TARGET_CANDIDATES = ("fl", "flx")


def midpoint_axis(values: pd.Series) -> np.ndarray:
    unique = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique().astype(float))
    if len(unique) < 2:
        return unique
    return np.sort(np.unique(np.concatenate([unique, (unique[:-1] + unique[1:]) / 2.0])))


def inside_convex_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    try:
        hull = ConvexHull(hull_points)
    except (QhullError, ValueError):
        return np.ones(len(points), dtype=bool)
    a, b = hull.equations[:, :-1], hull.equations[:, -1]
    return np.all((points @ a.T + b) <= 1e-9, axis=1)


def interpolate_values(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(source_points).all(axis=1) & np.isfinite(source_values)
    source_points, source_values = source_points[valid], source_values[valid]
    if len(source_points) == 0:
        return np.full(len(target_points), np.nan), np.full(len(target_points), "missing", dtype=object)

    unique_points, inverse = np.unique(source_points, axis=0, return_inverse=True)
    if len(unique_points) != len(source_points):
        sums = np.bincount(inverse, weights=source_values)
        counts = np.bincount(inverse)
        source_points, source_values = unique_points, sums / counts

    result = np.full(len(target_points), np.nan)
    methods = np.full(len(target_points), "missing", dtype=object)
    for method in ("cubic", "linear", "nearest"):
        missing = ~np.isfinite(result)
        if not missing.any():
            break
        if method != "nearest" and len(source_points) < 4:
            continue
        try:
            values = np.asarray(
                griddata(source_points, source_values, target_points[missing], method=method),
                dtype=float,
            )
        except (QhullError, ValueError):
            continue
        fill = np.isfinite(values)
        indices = np.flatnonzero(missing)[fill]
        result[indices] = values[fill]
        methods[indices] = method
    return result, methods


def target_column(df: pd.DataFrame) -> str:
    lower_to_original = {str(column).lower(): str(column) for column in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    raise ValueError(f"No target column found. Expected one of {TARGET_CANDIDATES}")


def aggregate_positions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicated = int(df.duplicated([GROUP_COLUMN, *COORD_COLUMNS]).sum())
    if duplicated == 0:
        return df.reset_index(drop=True), 0

    agg: dict[str, str] = {}
    for column in df.columns:
        if column in {GROUP_COLUMN, *COORD_COLUMNS}:
            continue
        agg[column] = "mean" if pd.api.types.is_numeric_dtype(df[column]) else "first"
    grouped = df.groupby([GROUP_COLUMN, *COORD_COLUMNS], as_index=False).agg(agg)
    return grouped[df.columns], duplicated


def validation_metrics(df: pd.DataFrame, value_column: str) -> dict[str, object]:
    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    methods_all: list[str] = []

    for _, group in df.groupby(GROUP_COLUMN, sort=True):
        group = group.dropna(subset=[*COORD_COLUMNS, value_column]).reset_index(drop=True)
        if len(group) < 5:
            continue
        points = group[list(COORD_COLUMNS)].to_numpy(dtype=float)
        values = group[value_column].to_numpy(dtype=float)
        n_splits = min(5, len(group))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_index, test_index in kfold.split(points):
            predicted, methods = interpolate_values(points[train_index], values[train_index], points[test_index])
            valid = np.isfinite(predicted)
            y_true_all.extend(values[test_index][valid].tolist())
            y_pred_all.extend(predicted[valid].tolist())
            methods_all.extend(methods[valid].tolist())

    if not y_true_all:
        return {
            "validation_points": 0,
            "validation_mae": None,
            "validation_rmse": None,
            "validation_r2": None,
            "validation_method_counts": {},
        }

    y_true = np.asarray(y_true_all, dtype=float)
    y_pred = np.asarray(y_pred_all, dtype=float)
    return {
        "validation_points": int(len(y_true)),
        "validation_mae": float(mean_absolute_error(y_true, y_pred)),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "validation_method_counts": pd.Series(methods_all).value_counts().to_dict(),
    }


def interpolate_file(input_path: Path, output_path: Path) -> dict[str, object]:
    source = pd.read_excel(input_path, engine="openpyxl")
    value_column = target_column(source)
    required = [GROUP_COLUMN, *COORD_COLUMNS, value_column]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{input_path.name} missing required columns: {missing}")

    valid = source.dropna(subset=required).copy()
    valid = valid[(valid[GROUP_COLUMN] > 0)].reset_index(drop=True)
    valid, duplicate_positions = aggregate_positions(valid)
    numeric_columns = [column for column in valid.columns if pd.api.types.is_numeric_dtype(valid[column])]
    reports: list[dict[str, object]] = []
    chunks: list[pd.DataFrame] = []

    for re_value, group in valid.groupby(GROUP_COLUMN, sort=True):
        group = group.dropna(subset=[*COORD_COLUMNS, value_column]).reset_index(drop=True)
        source_points = group[list(COORD_COLUMNS)].to_numpy(dtype=float)
        gx, gz = np.meshgrid(midpoint_axis(group["x"]), midpoint_axis(group["z"]))
        candidates = np.column_stack([gx.ravel(), gz.ravel()])
        dense_points = candidates[inside_convex_hull(candidates, source_points)]

        chunk = pd.DataFrame({GROUP_COLUMN: re_value, "x": dense_points[:, 0], "z": dense_points[:, 1]})
        method_column = f"{value_column}_fill_method"
        target_methods = np.full(len(chunk), "missing", dtype=object)

        for column in valid.columns:
            if column in {GROUP_COLUMN, *COORD_COLUMNS}:
                continue
            if column not in numeric_columns:
                chunk[column] = group[column].dropna().iloc[0] if group[column].notna().any() else np.nan
                continue
            series = pd.to_numeric(group[column], errors="coerce")
            if series.nunique(dropna=True) <= 1:
                chunk[column] = series.dropna().iloc[0] if series.notna().any() else np.nan
                if column == value_column:
                    target_methods[:] = "constant"
                continue
            values, methods = interpolate_values(source_points, series.to_numpy(dtype=float), dense_points)
            chunk[column] = values
            if column == value_column:
                target_methods = methods

        original_lookup = {
            (round(float(row["x"]), 12), round(float(row["z"]), 12)): row
            for _, row in group.iterrows()
        }
        generated = np.ones(len(chunk), dtype=bool)
        for index, point in enumerate(dense_points):
            key = (round(float(point[0]), 12), round(float(point[1]), 12))
            if key not in original_lookup:
                continue
            row = original_lookup[key]
            for column in valid.columns:
                chunk.loc[index, column] = row[column]
            generated[index] = False
            target_methods[index] = "original"

        nearest = cKDTree(source_points).query(dense_points)[0]
        chunk["generated"] = generated
        chunk[method_column] = target_methods
        chunk["nearest_original_distance"] = nearest
        chunks.append(chunk)

        generated_values = pd.to_numeric(chunk.loc[chunk["generated"], value_column], errors="coerce")
        original_values = pd.to_numeric(group[value_column], errors="coerce")
        original_min = float(original_values.min())
        original_max = float(original_values.max())
        overshoot = int(((generated_values < original_min) | (generated_values > original_max)).sum())
        reports.append({
            "re": float(re_value),
            "original_rows": int(len(group)),
            "output_rows": int(len(chunk)),
            "rows_added": int(chunk["generated"].sum()),
            "method_counts": chunk[method_column].value_counts().to_dict(),
            "target_original_min": original_min,
            "target_original_max": original_max,
            "target_output_min": float(pd.to_numeric(chunk[value_column], errors="coerce").min()),
            "target_output_max": float(pd.to_numeric(chunk[value_column], errors="coerce").max()),
            "generated_overshoot_rows": overshoot,
            "generated_nearest_distance_mean": float(nearest[generated].mean()) if generated.any() else 0.0,
            "generated_nearest_distance_max": float(nearest[generated].max()) if generated.any() else 0.0,
        })

    result = pd.concat(chunks, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False, engine="openpyxl")

    generated_target = pd.to_numeric(result.loc[result["generated"], value_column], errors="coerce")
    original_target = pd.to_numeric(valid[value_column], errors="coerce")
    original_min = float(original_target.min())
    original_max = float(original_target.max())
    generated_overshoot_rows = int(((generated_target < original_min) | (generated_target > original_max)).sum())
    method_column = f"{value_column}_fill_method"
    report = {
        "source_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "target_column": value_column,
        "rows_in": int(len(source)),
        "valid_rows_in": int(len(valid)),
        "invalid_rows_excluded": int(len(source) - len(valid)),
        "duplicate_position_rows_aggregated": duplicate_positions,
        "rows_out": int(len(result)),
        "rows_added": int(result["generated"].sum()),
        "original_rows_out": int((~result["generated"]).sum()),
        "re_count": int(result[GROUP_COLUMN].nunique()),
        "method_counts": result[method_column].value_counts().to_dict(),
        "target_original_min": original_min,
        "target_original_mean": float(original_target.mean()),
        "target_original_max": original_max,
        "target_output_min": float(pd.to_numeric(result[value_column], errors="coerce").min()),
        "target_output_mean": float(pd.to_numeric(result[value_column], errors="coerce").mean()),
        "target_output_max": float(pd.to_numeric(result[value_column], errors="coerce").max()),
        "generated_overshoot_rows": generated_overshoot_rows,
        "generated_target_nan_rows": int(generated_target.isna().sum()),
        "per_re": reports,
    }
    report.update(validation_metrics(valid, value_column))
    return report


def run_batch(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    paths = sorted(path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if not paths:
        raise ValueError(f"No .xlsx files found under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        output_name = f"{path.stem}-interpolated.xlsx"
        print(f"[interpolate {index + 1}/{len(paths)}] {path.name}", flush=True)
        reports.append(interpolate_file(path, output_dir / output_name))

    summary = pd.DataFrame(reports)
    summary.to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (output_dir / "run_manifest.json").write_text(
        json.dumps({
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_dir": input_dir.as_posix(),
            "output_dir": output_dir.as_posix(),
            "design": "each workbook interpolated independently; each Re slice interpolated independently",
            "file_count": len(paths),
            "reports": reports,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate raw FL/FLX workbooks independently.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/rawdata/2026-07-10"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpolated_2026-07-10"))
    args = parser.parse_args()
    summary = run_batch(args.input_dir, args.output_dir)
    print(summary[[
        "source_file", "target_column", "rows_in", "rows_out", "rows_added",
        "generated_target_nan_rows", "generated_overshoot_rows",
        "validation_mae", "validation_rmse", "validation_r2",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
