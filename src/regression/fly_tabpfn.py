"""Predict horizontal and vertical FLy neighbor midpoints with TabPFN.

New coordinates are restricted to one midpoint between neighboring original
points on the same horizontal or vertical line.  TabPFN predicts only the
``fly`` target; the remaining numeric features at a midpoint are the linear
midpoints of their two source rows.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.regression.fly_midpoints import (
    build_interpolated_frame,
    normalize_source,
)

TARGET_COLUMN = "fly"
EXCLUDED_FEATURES = {TARGET_COLUMN, "generated", "interpolation_direction", "source_pair_count"}


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    return "cpu"


def select_features(train: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
    features = [
        column
        for column in train.columns
        if column not in EXCLUDED_FEATURES
        and column in candidates.columns
        and pd.api.types.is_numeric_dtype(train[column])
        and train[column].notna().any()
        and candidates[column].notna().any()
    ]
    for coordinate in ("re", "x", "z"):
        if coordinate not in features:
            raise ValueError(f"Required TabPFN feature is unavailable: {coordinate}")
    return features


def predict_with_tabpfn(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
) -> tuple[np.ndarray, list[str]]:
    from tabpfn import TabPFNRegressor

    train = train.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    features = select_features(train, candidates)
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train = preprocessor.fit_transform(train[features])
    x_predict = preprocessor.transform(candidates[features])
    model = TabPFNRegressor(
        model_path=str(model_path),
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        ignore_pretraining_limits=True,
    )
    model.fit(x_train, train[TARGET_COLUMN].to_numpy(dtype=float))
    predictions = []
    for start in range(0, len(x_predict), chunk_size):
        predictions.append(np.asarray(model.predict(x_predict[start : start + chunk_size]), dtype=float))
    return np.concatenate(predictions), features


def process_file(
    input_path: Path,
    output_path: Path,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
) -> dict[str, object]:
    source_file = pd.read_excel(input_path, sheet_name=0, engine="openpyxl")
    source, duplicate_rows = normalize_source(source_file, input_path)
    result, counts = build_interpolated_frame(source)
    generated_mask = result["generated"].to_numpy(dtype=bool)
    candidates = result.loc[generated_mask].reset_index(drop=True)
    predictions, features = predict_with_tabpfn(
        source,
        candidates,
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        chunk_size=chunk_size,
    )
    result.loc[generated_mask, TARGET_COLUMN] = predictions
    result["fly_fill_method"] = np.where(result["generated"], "tabpfn_v3", "original")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False, sheet_name="FLy_tabpfn_midpoints", engine="openpyxl")

    source_target = pd.to_numeric(source[TARGET_COLUMN], errors="coerce")
    generated_target = pd.to_numeric(result.loc[generated_mask, TARGET_COLUMN], errors="coerce")
    return {
        "source_file": input_path.as_posix(),
        "output_file": output_path.as_posix(),
        "source_rows_total": int(len(source_file)),
        "valid_original_rows": int(len(source)),
        "invalid_rows_excluded": int(len(source_file) - len(source)),
        "duplicate_position_rows_aggregated": duplicate_rows,
        "rows_out": int(len(result)),
        "rows_added": int(generated_mask.sum()),
        "horizontal_rows_added": int((result["interpolation_direction"] == "horizontal").sum()),
        "vertical_rows_added": int((result["interpolation_direction"] == "vertical").sum()),
        "both_direction_rows_added": int((result["source_pair_count"] > 1).sum()),
        "generated_fly_nan_rows": int(generated_target.isna().sum()),
        "generated_fly_outside_source_range": int(
            ((generated_target < source_target.min()) | (generated_target > source_target.max())).sum()
        ),
        "source_fly_min": float(source_target.min()),
        "source_fly_max": float(source_target.max()),
        "generated_fly_min": float(generated_target.min()),
        "generated_fly_max": float(generated_target.max()),
        "features": features,
        **counts,
    }


def run_batch(
    input_dir: Path,
    output_dir: Path,
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
) -> pd.DataFrame:
    paths = sorted(path for path in input_dir.glob("*.xlsx") if not path.name.startswith("~$"))
    if not paths:
        raise ValueError(f"No .xlsx files found under {input_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, object]] = []
    for index, path in enumerate(paths, start=1):
        print(f"[TabPFN neighbor midpoint {index}/{len(paths)}] {path.name}", flush=True)
        output_path = output_dir / f"{path.stem}-tabpfn-neighbor-midpoints.xlsx"
        reports.append(
            process_file(
                path,
                output_path,
                model_path=model_path,
                device=device,
                n_estimators=n_estimators,
                random_state=random_state + index - 1,
                chunk_size=chunk_size,
            )
        )
    return pd.DataFrame(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FLy neighbor midpoints with TabPFN.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/rawdata/2026-09-26/FLy"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/interpolated_2026-09-26-FLy-tabpfn-v3-neighbor-midpoints"))
    parser.add_argument("--model-path", type=Path, default=Path("models/tabpfn_3/tabpfn-v3-regressor-v3_default.ckpt"))
    parser.add_argument("--device", default="prefer-cuda", choices=("prefer-cuda", "cuda", "cpu"))
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    device = resolve_device(args.device)
    summary = run_batch(
        args.input_dir,
        args.output_dir,
        model_path=args.model_path,
        device=device,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
    )
    print(
        summary[
            [
                "source_file",
                "valid_original_rows",
                "rows_out",
                "rows_added",
                "generated_fly_nan_rows",
                "generated_fly_outside_source_range",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
