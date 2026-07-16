"""Interpolate raw FL/FLX workbooks with TabPFN regression."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR
from src.experiments.interpolate_raw_flx_files import (
    COORD_COLUMNS,
    GROUP_COLUMN,
    aggregate_positions,
    inside_convex_hull,
    interpolate_values,
    midpoint_axis,
    target_column,
)
from src.experiments.predict_fl_at_fd_points_tabpfn import fit_predict_tabpfn, resolve_device


def build_dense_frame(source: pd.DataFrame, value_column: str) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    numeric_columns = [column for column in source.columns if pd.api.types.is_numeric_dtype(source[column])]
    chunks: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []

    for re_value, group in source.groupby(GROUP_COLUMN, sort=True):
        group = group.dropna(subset=[*COORD_COLUMNS, value_column]).reset_index(drop=True)
        source_points = group[list(COORD_COLUMNS)].to_numpy(dtype=float)
        gx, gz = np.meshgrid(midpoint_axis(group["x"]), midpoint_axis(group["z"]))
        candidates = np.column_stack([gx.ravel(), gz.ravel()])
        dense_points = candidates[inside_convex_hull(candidates, source_points)]

        chunk = pd.DataFrame({GROUP_COLUMN: re_value, "x": dense_points[:, 0], "z": dense_points[:, 1]})
        for column in source.columns:
            if column in {GROUP_COLUMN, *COORD_COLUMNS}:
                continue
            if column == value_column:
                chunk[column] = np.nan
                continue
            if column not in numeric_columns:
                chunk[column] = group[column].dropna().iloc[0] if group[column].notna().any() else np.nan
                continue

            series = pd.to_numeric(group[column], errors="coerce")
            if series.nunique(dropna=True) <= 1:
                chunk[column] = series.dropna().iloc[0] if series.notna().any() else np.nan
            else:
                values, _ = interpolate_values(source_points, series.to_numpy(dtype=float), dense_points)
                chunk[column] = values

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
            for column in source.columns:
                chunk.loc[index, column] = row[column]
            generated[index] = False

        chunk["generated"] = generated
        chunks.append(chunk)
        reports.append({
            "re": float(re_value),
            "original_rows": int(len(group)),
            "output_rows": int(len(chunk)),
            "rows_added": int(generated.sum()),
        })

    return pd.concat(chunks, ignore_index=True), reports


def select_features(train_df: pd.DataFrame, predict_df: pd.DataFrame, value_column: str) -> list[str]:
    excluded = {value_column, "generated", f"{value_column}_fill_method"}
    features = [
        column for column in train_df.columns
        if column not in excluded
        and column in predict_df.columns
        and pd.api.types.is_numeric_dtype(train_df[column])
        and train_df[column].notna().any()
        and predict_df[column].notna().any()
    ]
    missing = sorted({GROUP_COLUMN, *COORD_COLUMNS}.difference(features))
    if missing:
        raise ValueError(f"Missing required TabPFN features: {missing}")
    return features


def tabpfn_predict(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    value_column: str,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
) -> tuple[np.ndarray, list[str]]:
    features = select_features(train_df, predict_df, value_column)
    train = train_df.dropna(subset=[value_column]).reset_index(drop=True)
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train = preprocessor.fit_transform(train[features])
    x_predict = preprocessor.transform(predict_df[features])
    predictions = fit_predict_tabpfn(
        x_train,
        train[value_column].to_numpy(dtype=float),
        x_predict,
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        chunk_size=chunk_size,
    )
    return predictions, features


def validation_metrics(
    source: pd.DataFrame,
    value_column: str,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
    folds: int,
) -> dict[str, object]:
    source = source.dropna(subset=[value_column]).reset_index(drop=True)
    if len(source) < 5 or folds < 2:
        return {"validation_points": 0, "validation_mae": None, "validation_rmse": None, "validation_r2": None}

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    n_splits = min(folds, len(source))
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for index, (train_index, test_index) in enumerate(kfold.split(source)):
        train = source.iloc[train_index].reset_index(drop=True)
        test = source.iloc[test_index].reset_index(drop=True)
        predicted, _ = tabpfn_predict(
            train,
            test,
            value_column,
            model_path=model_path,
            device=device,
            n_estimators=n_estimators,
            random_state=random_state + index,
            chunk_size=chunk_size,
        )
        y_true_all.extend(test[value_column].to_numpy(dtype=float).tolist())
        y_pred_all.extend(predicted.tolist())

    y_true = np.asarray(y_true_all, dtype=float)
    y_pred = np.asarray(y_pred_all, dtype=float)
    return {
        "validation_points": int(len(y_true)),
        "validation_mae": float(mean_absolute_error(y_true, y_pred)),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "validation_r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
    validation_folds: int,
) -> dict[str, object]:
    source = pd.read_excel(input_path, engine="openpyxl")
    value_column = target_column(source)
    required = [GROUP_COLUMN, *COORD_COLUMNS, value_column]
    missing = [column for column in required if column not in source.columns]
    if missing:
        raise ValueError(f"{input_path.name} missing required columns: {missing}")

    valid = source.dropna(subset=required).copy()
    valid = valid[valid[GROUP_COLUMN] > 0].reset_index(drop=True)
    valid, duplicate_positions = aggregate_positions(valid)
    dense, per_re_reports = build_dense_frame(valid, value_column)
    generated_mask = dense["generated"].to_numpy(dtype=bool)
    predictions, features = tabpfn_predict(
        valid,
        dense.loc[generated_mask].reset_index(drop=True),
        value_column,
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        chunk_size=chunk_size,
    )
    dense.loc[generated_mask, value_column] = predictions
    dense.loc[generated_mask, f"{value_column}_fill_method"] = "tabpfn"
    dense.loc[~dense["generated"], f"{value_column}_fill_method"] = "original"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dense.to_excel(output_path, index=False, engine="openpyxl")

    original_target = pd.to_numeric(valid[value_column], errors="coerce")
    generated_target = pd.to_numeric(dense.loc[dense["generated"], value_column], errors="coerce")
    original_min = float(original_target.min())
    original_max = float(original_target.max())
    report = {
        "source_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "target_column": value_column,
        "rows_in": int(len(source)),
        "valid_rows_in": int(len(valid)),
        "invalid_rows_excluded": int(len(source) - len(valid)),
        "duplicate_position_rows_aggregated": duplicate_positions,
        "rows_out": int(len(dense)),
        "rows_added": int(dense["generated"].sum()),
        "original_rows_out": int((~dense["generated"]).sum()),
        "re_count": int(dense[GROUP_COLUMN].nunique()),
        "method_counts": dense[f"{value_column}_fill_method"].value_counts().to_dict(),
        "target_original_min": original_min,
        "target_original_mean": float(original_target.mean()),
        "target_original_max": original_max,
        "target_output_min": float(pd.to_numeric(dense[value_column], errors="coerce").min()),
        "target_output_mean": float(pd.to_numeric(dense[value_column], errors="coerce").mean()),
        "target_output_max": float(pd.to_numeric(dense[value_column], errors="coerce").max()),
        "generated_overshoot_rows": int(((generated_target < original_min) | (generated_target > original_max)).sum()),
        "generated_target_nan_rows": int(generated_target.isna().sum()),
        "features": features,
        "per_re": per_re_reports,
    }
    report.update(validation_metrics(
        valid,
        value_column,
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        chunk_size=chunk_size,
        folds=validation_folds,
    ))
    return report


def run_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
    validation_folds: int,
) -> pd.DataFrame:
    paths = sorted(path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if not paths:
        raise ValueError(f"No .xlsx files found under {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    reports: list[dict[str, object]] = []
    for index, path in enumerate(paths):
        print(f"[tabpfn interpolate {index + 1}/{len(paths)}] {path.name}", flush=True)
        output_path = output_dir / f"{path.stem}-tabpfn-interpolated.xlsx"
        reports.append(process_file(
            path,
            output_path,
            model_path=model_path,
            device=resolved_device,
            n_estimators=n_estimators,
            random_state=random_state + index,
            chunk_size=chunk_size,
            validation_folds=validation_folds,
        ))
        pd.DataFrame(reports).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "design": "each workbook independently densified; generated target values predicted with TabPFN",
        "model_path": model_path.as_posix(),
        "device": resolved_device,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "chunk_size": chunk_size,
        "validation_folds": validation_folds,
        "file_count": len(paths),
        "reports": reports,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pd.DataFrame(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate raw FL/FLX workbooks with TabPFN.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/rawdata/2026-07-10"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpolated_2026-07-10-tabpfn"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--validation-folds", type=int, default=3)
    args = parser.parse_args()
    summary = run_batch(
        args.input_dir,
        args.output_dir,
        model_path=args.model_path,
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
        validation_folds=args.validation_folds,
    )
    print(summary[[
        "source_file", "target_column", "rows_in", "rows_out", "rows_added",
        "generated_target_nan_rows", "generated_overshoot_rows",
        "validation_mae", "validation_rmse", "validation_r2",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
