"""Augment FL samples, fit TabPFN, and predict FL at every FD location."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, QhullError
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR
from src.data.template_schema import (
    read_data_workbook,
    standardize_template_frame,
    validate_template_frame,
)

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")


DEFAULT_OUTPUT_DIR = Path("results/fl_at_fd_points_tabpfn")
MODEL_FEATURE_CANDIDATES = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "re", "x", "z", "x/H", "z/H",
    "prep_dist_to_ab", "prep_dist_to_ac",
]


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
    if not len(source_points):
        return np.full(len(target_points), np.nan), np.full(len(target_points), "missing", dtype=object)

    unique_points, inverse = np.unique(source_points, axis=0, return_inverse=True)
    if len(unique_points) != len(source_points):
        sums = np.bincount(inverse, weights=source_values)
        counts = np.bincount(inverse)
        source_points, source_values = unique_points, sums / counts

    result = np.full(len(target_points), np.nan)
    methods = np.full(len(target_points), "missing", dtype=object)
    for method in ["cubic", "linear", "nearest"]:
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


def augment_fl_points(fl_df: pd.DataFrame) -> pd.DataFrame:
    """Add midpoint-grid samples independently within each Reynolds-number slice."""
    chunks: list[pd.DataFrame] = []
    interpolated_columns = [
        column for column in ["fl", "prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc",
                              "line_dist_to_ab", "line_dist_to_ac"]
        if column in fl_df.columns
    ]
    constants = [
        column for column in MODEL_FEATURE_CANDIDATES
        if column not in {"re", "x", "z", "x/H", "z/H", *interpolated_columns}
        and column in fl_df.columns
    ]

    for re_value, group in fl_df.groupby("re", sort=True):
        group = group.dropna(subset=["x", "z", "fl"]).copy()
        if group.empty:
            continue
        original_points = group[["x", "z"]].to_numpy(dtype=float)
        gx, gz = np.meshgrid(midpoint_axis(group["x"]), midpoint_axis(group["z"]))
        candidates = np.column_stack([gx.ravel(), gz.ravel()])
        dense_points = candidates[inside_convex_hull(candidates, original_points)]

        chunk = pd.DataFrame({"re": re_value, "x": dense_points[:, 0], "z": dense_points[:, 1]})
        for column in constants:
            chunk[column] = group[column].dropna().iloc[0] if group[column].notna().any() else np.nan
        method_for_fl = np.full(len(chunk), "missing", dtype=object)
        for column in interpolated_columns:
            values, methods = interpolate_values(
                original_points,
                pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=float),
                dense_points,
            )
            chunk[column] = values
            if column == "fl":
                method_for_fl = methods

        original_values = group.groupby(["x", "z"], as_index=False)["fl"].mean()
        original_lookup = {
            (round(float(row.x), 12), round(float(row.z), 12)): float(row.fl)
            for row in original_values.itertuples(index=False)
        }
        generated = np.ones(len(chunk), dtype=bool)
        for index, point in enumerate(dense_points):
            key = (round(float(point[0]), 12), round(float(point[1]), 12))
            if key in original_lookup:
                chunk.loc[index, "fl"] = original_lookup[key]
                method_for_fl[index] = "original"
                generated[index] = False
        chunk["generated"] = generated
        chunk["fl_fill_method"] = method_for_fl
        if "h" in chunk:
            chunk["x/H"] = chunk["x"] / chunk["h"]
            chunk["z/H"] = chunk["z"] / chunk["h"]
        chunks.append(chunk)

    if not chunks:
        raise ValueError("No valid FL rows are available for augmentation")
    return pd.concat(chunks, ignore_index=True)


def select_model_features(fl_df: pd.DataFrame, fd_df: pd.DataFrame) -> list[str]:
    features = [
        column for column in MODEL_FEATURE_CANDIDATES
        if column in fl_df.columns and column in fd_df.columns
        and fl_df[column].notna().any() and fd_df[column].notna().any()
    ]
    required = {"re", "x", "z"}
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"FL and FD do not share required model features: {missing}")
    return features


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "prefer-cuda"}:
        return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def fit_predict_tabpfn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
) -> np.ndarray:
    if chunk_size < 1 or n_estimators < 1:
        raise ValueError("chunk_size and n_estimators must be positive")
    if len(x_train) == 0 or len(y_train) != len(x_train):
        raise ValueError("Training features and targets must be nonempty and aligned")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all() or not np.isfinite(x_predict).all():
        raise ValueError("Model inputs must be finite after preprocessing")
    if len(x_predict) == 0:
        return np.empty(0, dtype=float)
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        ignore_pretraining_limits=True,
        random_state=random_state,
        show_progress_bar=True,
    )
    model.fit(x_train, y_train)
    predictions = np.empty(len(x_predict), dtype=float)
    for start in range(0, len(x_predict), chunk_size):
        chunk = x_predict[start:start + chunk_size]
        current = np.asarray(model.predict(chunk), dtype=float)
        if current.shape != (len(chunk),) or not np.isfinite(current).all():
            raise ValueError("Regressor returned nonfinite or misaligned predictions")
        predictions[start:start + len(chunk)] = current
    return predictions


def run(
    fl_path: Path,
    fd_path: Path,
    output_dir: Path,
    *,
    model_path: Path,
    fl_length_unit: str,
    fd_length_unit: str,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fl_df, fl_transition = standardize_template_frame(
        read_data_workbook(fl_path), kind="fl", source_length_unit=fl_length_unit,
    )
    fd_df, fd_transition = standardize_template_frame(
        read_data_workbook(fd_path), kind="fd", source_length_unit=fd_length_unit,
    )
    validation_errors = validate_template_frame(fl_df, kind="fl") + validate_template_frame(fd_df, kind="fd")
    if validation_errors:
        raise ValueError("Template validation failed: " + "; ".join(validation_errors))

    augmented = augment_fl_points(fl_df)
    features = select_model_features(augmented, fd_df)
    train = augmented.dropna(subset=["fl"])
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train = preprocessor.fit_transform(train[features])
    x_predict = preprocessor.transform(fd_df[features])
    resolved_device = resolve_device(device)
    predictions = fit_predict_tabpfn(
        x_train, train["fl"].to_numpy(dtype=float), x_predict,
        model_path=model_path, device=resolved_device, n_estimators=n_estimators,
        random_state=random_state, chunk_size=chunk_size,
    )

    from .fd_to_fl import nearest_fd_distance_by_re

    distances = nearest_fd_distance_by_re(augmented, fd_df)
    result = fd_df.copy()
    result["fl_pred"] = predictions
    result["fl_pred_ok"] = np.isfinite(predictions)
    result["nearest_augmented_fl_distance_um"] = distances
    result["fl_source_file"] = fl_path.name
    result["fd_source_file"] = fd_path.name

    augmented_path = output_dir / f"{fl_path.stem}_augmented.xlsx"
    prediction_path = output_dir / f"{fd_path.stem}_with_fl_pred.xlsx"
    augmented.to_excel(augmented_path, index=False, engine="openpyxl")
    result.to_excel(prediction_path, index=False, engine="openpyxl")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fl_file": fl_path.as_posix(), "fd_file": fd_path.as_posix(),
        "augmented_file": augmented_path.as_posix(), "prediction_file": prediction_path.as_posix(),
        "fl_transition": fl_transition, "fd_transition": fd_transition,
        "fl_rows_original": int(len(fl_df)), "fl_rows_after_augmentation": int(len(augmented)),
        "fl_rows_added": int(augmented["generated"].sum()),
        "fd_rows": int(len(fd_df)), "fd_rows_predicted": int(result["fl_pred_ok"].sum()),
        "features": features, "model_path": model_path.as_posix(),
        "device": resolved_device, "n_estimators": n_estimators, "random_state": random_state,
        "nearest_augmented_fl_distance_um_mean": float(distances[:, 0].mean()),
        "nearest_augmented_fl_distance_um_max": float(distances[:, 0].max()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transition current templates, augment FL, fit TabPFN, and predict FL at all FD points."
    )
    parser.add_argument("--fl", type=Path, required=True, help="FL workbook in current or historical format")
    parser.add_argument("--fd", type=Path, required=True, help="FD workbook in current or historical format")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--fl-length-unit", choices=["auto", "m", "um"], default="auto")
    parser.add_argument("--fd-length-unit", choices=["auto", "m", "um"], default="auto")
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {args.model_path}")
    report = run(
        args.fl, args.fd, args.output_dir, model_path=args.model_path,
        fl_length_unit=args.fl_length_unit, fd_length_unit=args.fd_length_unit,
        device=args.device, n_estimators=args.n_estimators,
        random_state=args.random_state, chunk_size=args.chunk_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
