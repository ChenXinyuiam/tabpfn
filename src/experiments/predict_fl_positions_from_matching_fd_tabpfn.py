"""Fit each FD workbook and predict u at every position in its matching FL workbook."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR
from src.data.template_schema import (
    read_data_workbook,
    standardize_template_frame,
    validate_template_frame,
)
from src.experiments.predict_fd_u_from_interpolated_fl_tabpfn import discover_pairs
from src.experiments.predict_fl_at_fd_points_tabpfn import fit_predict_tabpfn, resolve_device


MODEL_FEATURE_CANDIDATES = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "re", "x", "z", "x/H", "z/H",
    "prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc",
    "line_dist_to_ab", "line_dist_to_ac",
]
REQUIRED_GEOMETRY = ["bc", "h", "angle_a", "angle_b", "angle_c", "re", "x", "z"]


def valid_rows(df: pd.DataFrame, target: str | None = None) -> pd.Series:
    required = [*REQUIRED_GEOMETRY, *([target] if target else [])]
    mask = df[required].notna().all(axis=1)
    mask &= (df["bc"] > 0) & (df["h"] > 0) & (df["re"] > 0)
    return mask


def select_features(fd_df: pd.DataFrame, fl_df: pd.DataFrame) -> list[str]:
    features = [
        column for column in MODEL_FEATURE_CANDIDATES
        if column in fd_df.columns and column in fl_df.columns
        and fd_df[column].notna().any() and fl_df[column].notna().any()
    ]
    missing = sorted({"re", "x", "z"}.difference(features))
    if missing:
        raise ValueError(f"FD and FL do not share required features: {missing}")
    return features


def nearest_fd_distance_by_re(fd_df: pd.DataFrame, fl_df: pd.DataFrame) -> np.ndarray:
    distances = np.full(len(fl_df), np.nan, dtype=float)
    for re_value, group in fl_df.groupby("re", sort=False):
        source = fd_df[fd_df["re"] == re_value]
        if source.empty:
            continue
        nearest = NearestNeighbors(n_neighbors=1).fit(source[["x", "z"]].to_numpy(dtype=float))
        current, _ = nearest.kneighbors(group[["x", "z"]].to_numpy(dtype=float))
        distances[group.index] = current[:, 0]
    return distances


def predict_pair(
    key: str,
    fl_path: Path,
    fd_path: Path,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
    max_train_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fd_df, fd_transition = standardize_template_frame(
        read_data_workbook(fd_path),
        kind="fd",
        source_length_unit="auto",
        target_column="u",
    )
    fl_df, fl_transition = standardize_template_frame(
        read_data_workbook(fl_path),
        kind="fl",
        source_length_unit="auto",
    )

    fd_mask = valid_rows(fd_df, "u")
    fl_mask = valid_rows(fl_df)
    train_full = fd_df.loc[fd_mask].reset_index(drop=True)
    prediction_input = fl_df.loc[fl_mask].copy()
    fd_errors = validate_template_frame(train_full, kind="fd", target_column="u")
    fl_errors = validate_template_frame(prediction_input, kind="fl")
    if fd_errors or fl_errors:
        raise ValueError(f"{key} validation failed: {'; '.join(fd_errors + fl_errors)}")

    features = select_features(train_full, prediction_input)
    train = train_full
    train_rows_available = len(train_full)
    if max_train_rows is not None and len(train) > max_train_rows:
        train = train.sample(n=max_train_rows, random_state=random_state).reset_index(drop=True)

    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train = preprocessor.fit_transform(train[features])
    x_predict = preprocessor.transform(prediction_input[features])
    predictions = fit_predict_tabpfn(
        x_train,
        train["u"].to_numpy(dtype=float),
        x_predict,
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    prediction_input["u_pred"] = predictions
    prediction_input["nearest_fd_distance_um"] = nearest_fd_distance_by_re(train_full, prediction_input)
    result = fl_df.copy()
    result["u_pred"] = np.nan
    result["nearest_fd_distance_um"] = np.nan
    result["prediction_status"] = "invalid_input"
    result.loc[fl_mask, "u_pred"] = prediction_input["u_pred"].to_numpy(dtype=float)
    result.loc[fl_mask, "nearest_fd_distance_um"] = prediction_input["nearest_fd_distance_um"].to_numpy(dtype=float)
    result.loc[fl_mask, "prediction_status"] = "predicted"
    result["fd_training_file"] = fd_path.name
    result["fl_source_file"] = fl_path.name

    report = {
        "pair_key": key,
        "fd_training_file": fd_path.as_posix(),
        "fl_prediction_file": fl_path.as_posix(),
        "fd_rows_available": int(train_rows_available),
        "fd_rows_used": int(len(train)),
        "fl_rows": int(len(result)),
        "fl_rows_predicted": int(result["u_pred"].notna().sum()),
        "fl_invalid_rows": int((~fl_mask).sum()),
        "features": features,
        "u_pred_min": float(np.nanmin(predictions)),
        "u_pred_mean": float(np.nanmean(predictions)),
        "u_pred_max": float(np.nanmax(predictions)),
        "nearest_fd_distance_um_mean": float(prediction_input["nearest_fd_distance_um"].mean()),
        "nearest_fd_distance_um_max": float(prediction_input["nearest_fd_distance_um"].max()),
        "fd_transition": fd_transition,
        "fl_transition": fl_transition,
    }
    return result, report


def run_batch(
    batch_dir: Path,
    output_dir: Path,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
    max_train_rows: int | None,
    only: set[str] | None = None,
) -> pd.DataFrame:
    pairs = discover_pairs(batch_dir)
    if only:
        unknown = sorted(only.difference(pairs))
        if unknown:
            raise ValueError(f"Unknown --only keys: {unknown}")
    keys = sorted(only or set(pairs))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    reports: list[dict[str, object]] = []

    for index, key in enumerate(keys):
        fl_path, fd_path = pairs[key]
        print(f"[predict {index + 1}/{len(keys)}] {key}: {fd_path.name} -> {fl_path.name}", flush=True)
        result, report = predict_pair(
            key,
            fl_path,
            fd_path,
            model_path=model_path,
            device=resolved_device,
            n_estimators=n_estimators,
            random_state=random_state + index,
            chunk_size=chunk_size,
            max_train_rows=max_train_rows,
        )
        output_path = output_dir / f"{key}-FL-with-u-pred.xlsx"
        result.to_excel(output_path, index=False, engine="openpyxl")
        report["output_file"] = output_path.as_posix()
        reports.append(report)
        pd.DataFrame(reports).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_dir": batch_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "design": "fit each FD file independently and predict only its matching FL file",
        "target": "u",
        "preserved_fl_column": "fl",
        "model_path": model_path.as_posix(),
        "device": resolved_device,
        "n_estimators": n_estimators,
        "max_train_rows": max_train_rows,
        "pair_count": len(keys),
        "reports": reports,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return pd.DataFrame(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit matching FD files and predict u at all FL positions.")
    parser.add_argument("--batch-dir", type=Path, default=Path("data/rawdata/2026-06-20"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_to_matching_fl_2026-06-20"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-train-rows", type=int, default=10000)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {args.model_path}")
    summary = run_batch(
        args.batch_dir,
        args.output_dir,
        model_path=args.model_path,
        device=args.device,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
        max_train_rows=args.max_train_rows,
        only=set(args.only) if args.only else None,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
