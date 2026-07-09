"""Predict FD ``u`` using FL interpolated onto every FD sample location."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR
from src.data.template_schema import (
    read_data_workbook,
    standardize_template_frame,
    validate_template_frame,
)
from src.experiments.predict_fl_at_fd_points_tabpfn import (
    augment_fl_points,
    fit_predict_tabpfn,
    interpolate_values,
    resolve_device,
)


U_FEATURE_CANDIDATES = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "re", "x", "z", "x/H", "z/H",
    "prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc",
    "line_dist_to_ab", "line_dist_to_ac", "fl_interp",
]


def pair_key(path: Path, kind: str) -> str:
    stem = path.stem
    if kind == "fl":
        return re.sub(r"-FL$", "", stem, flags=re.IGNORECASE)
    return re.sub(r"-FD.*$", "", stem, flags=re.IGNORECASE)


def family_key(key: str) -> str:
    match = re.match(r"^(.*)-(10|20|30)$", key)
    if not match:
        raise ValueError(f"Cannot parse radius suffix from pair key: {key}")
    return match.group(1)


def radius_from_key(key: str) -> int:
    match = re.search(r"-(10|20|30)$", key)
    if not match:
        raise ValueError(f"Cannot parse radius suffix from pair key: {key}")
    return int(match.group(1))


def discover_pairs(batch_dir: Path) -> dict[str, tuple[Path, Path]]:
    fl_paths = {
        pair_key(path, "fl"): path
        for path in (batch_dir / "FL").glob("*.xlsx")
        if not path.name.startswith("~$")
    }
    fd_paths = {
        pair_key(path, "fd"): path
        for path in (batch_dir / "FD").glob("*.xlsx")
        if not path.name.startswith("~$")
    }
    fl_only = sorted(set(fl_paths).difference(fd_paths))
    fd_only = sorted(set(fd_paths).difference(fl_paths))
    if fl_only or fd_only:
        raise ValueError(f"Unpaired files: fl_only={fl_only}, fd_only={fd_only}")
    if not fl_paths:
        raise ValueError(f"No FL/FD workbook pairs found under {batch_dir}")
    return {key: (fl_paths[key], fd_paths[key]) for key in sorted(fl_paths)}


def map_augmented_fl_to_fd(augmented_fl: pd.DataFrame, fd_df: pd.DataFrame) -> pd.DataFrame:
    out = fd_df.copy()
    out["fl_interp"] = np.nan
    out["fl_interp_method"] = "missing_re"
    out["nearest_augmented_fl_distance_um"] = np.nan

    for re_value, fd_group in out.groupby("re", sort=True):
        fl_group = augmented_fl[augmented_fl["re"] == re_value].dropna(subset=["x", "z", "fl"])
        if fl_group.empty:
            continue
        source_points = fl_group[["x", "z"]].to_numpy(dtype=float)
        target_points = fd_group[["x", "z"]].to_numpy(dtype=float)
        values, methods = interpolate_values(
            source_points,
            fl_group["fl"].to_numpy(dtype=float),
            target_points,
        )
        nearest = NearestNeighbors(n_neighbors=1).fit(source_points)
        distances, _ = nearest.kneighbors(target_points)
        out.loc[fd_group.index, "fl_interp"] = values
        out.loc[fd_group.index, "fl_interp_method"] = methods
        out.loc[fd_group.index, "nearest_augmented_fl_distance_um"] = distances[:, 0]

    if out["fl_interp"].isna().any():
        missing = int(out["fl_interp"].isna().sum())
        raise ValueError(f"FL interpolation left {missing} FD rows without fl_interp")
    return out


def prepare_pair(key: str, fl_path: Path, fd_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    fl_df, fl_report = standardize_template_frame(
        read_data_workbook(fl_path), kind="fl", source_length_unit="auto",
    )
    fd_df, fd_report = standardize_template_frame(
        read_data_workbook(fd_path), kind="fd", source_length_unit="auto", target_column="u",
    )
    fl_rows_before_clean = len(fl_df)
    fd_rows_before_clean = len(fd_df)
    required_geometry = ["bc", "h", "angle_a", "angle_b", "angle_c", "re", "x", "z"]
    fl_df = fl_df.dropna(subset=[*required_geometry, "fl"])
    fl_df = fl_df[(fl_df["bc"] > 0) & (fl_df["h"] > 0) & (fl_df["re"] > 0)].reset_index(drop=True)
    fd_df = fd_df.dropna(subset=[*required_geometry, "u"])
    fd_df = fd_df[(fd_df["bc"] > 0) & (fd_df["h"] > 0) & (fd_df["re"] > 0)].reset_index(drop=True)
    errors = validate_template_frame(fl_df, kind="fl")
    errors += validate_template_frame(fd_df, kind="fd", target_column="u")
    if errors:
        raise ValueError(f"{key} validation failed: {'; '.join(errors)}")
    augmented = augment_fl_points(fl_df)
    mapped_fd = map_augmented_fl_to_fd(augmented, fd_df)
    mapped_fd["pair_key"] = key
    mapped_fd["family_key"] = family_key(key)
    mapped_fd["radius"] = radius_from_key(key)
    report = {
        "pair_key": key,
        "fl_source": fl_path.as_posix(),
        "fd_source": fd_path.as_posix(),
        "fl_rows": int(len(fl_df)),
        "fl_invalid_rows_dropped": int(fl_rows_before_clean - len(fl_df)),
        "fl_augmented_rows": int(len(augmented)),
        "fl_added_rows": int(augmented["generated"].sum()),
        "fd_rows": int(len(mapped_fd)),
        "fd_invalid_rows_dropped": int(fd_rows_before_clean - len(fd_df)),
        "fl_transition": fl_report,
        "fd_transition": fd_report,
        "fl_interp_method_counts": mapped_fd["fl_interp_method"].value_counts().to_dict(),
    }
    return augmented, mapped_fd, report


def select_u_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[str]:
    features = [
        column for column in U_FEATURE_CANDIDATES
        if column in train_df.columns and column in test_df.columns
        and train_df[column].notna().any() and test_df[column].notna().any()
    ]
    required = {"re", "x", "z", "fl_interp"}
    missing = sorted(required.difference(features))
    if missing:
        raise ValueError(f"Missing required u prediction features: {missing}")
    return features


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan,
    }


def predict_one_pair(
    key: str,
    prepared: dict[str, pd.DataFrame],
    *,
    model_path: Path,
    device: str,
    n_estimators: int,
    random_state: int,
    chunk_size: int,
    max_train_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    test_df = prepared[key]
    family = family_key(key)
    train_keys = sorted(other for other in prepared if family_key(other) == family and other != key)
    if not train_keys:
        raise ValueError(f"{key} has no other-radius training file in family {family}")
    train_df = pd.concat([prepared[other] for other in train_keys], ignore_index=True)
    features = select_u_features(train_df, test_df)
    train_valid = train_df.dropna(subset=["u"])
    train_rows_available = len(train_valid)
    if max_train_rows is not None and len(train_valid) > max_train_rows:
        train_valid = train_valid.sample(n=max_train_rows, random_state=random_state).reset_index(drop=True)

    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train = preprocessor.fit_transform(train_valid[features])
    x_test = preprocessor.transform(test_df[features])
    predictions = fit_predict_tabpfn(
        x_train,
        train_valid["u"].to_numpy(dtype=float),
        x_test,
        model_path=model_path,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    out = test_df.copy()
    out["u_pred"] = predictions
    out["u_residual"] = out["u"] - out["u_pred"]
    out["u_abs_error"] = out["u_residual"].abs()
    out["u_train_files"] = ",".join(train_keys)
    metrics = regression_metrics(out["u"].to_numpy(dtype=float), predictions)
    nonzero = out["u"].to_numpy(dtype=float) != 0
    nonzero_metrics = regression_metrics(out.loc[nonzero, "u"].to_numpy(dtype=float), predictions[nonzero])
    metrics.update({f"nonzero_{name}": value for name, value in nonzero_metrics.items()})
    metrics.update({
        "pair_key": key,
        "family_key": family,
        "test_radius": radius_from_key(key),
        "train_keys": ",".join(train_keys),
        "train_rows": int(len(train_valid)),
        "train_rows_available": int(train_rows_available),
        "test_rows": int(len(out)),
        "features": ",".join(features),
    })
    return out, metrics


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
    output_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir = output_dir / "fl_augmented"
    prediction_dir = output_dir / "fd_u_predictions"
    augmented_dir.mkdir(exist_ok=True)
    prediction_dir.mkdir(exist_ok=True)

    prepared: dict[str, pd.DataFrame] = {}
    preparation_reports: list[dict[str, object]] = []
    for key, (fl_path, fd_path) in pairs.items():
        print(f"[prepare] {key}", flush=True)
        augmented, mapped_fd, report = prepare_pair(key, fl_path, fd_path)
        prepared[key] = mapped_fd
        preparation_reports.append(report)
        augmented.to_excel(augmented_dir / f"{key}-FL-augmented.xlsx", index=False, engine="openpyxl")

    resolved_device = resolve_device(device)
    metrics_rows: list[dict[str, object]] = []
    prediction_keys = sorted(only or set(pairs))
    for index, key in enumerate(prediction_keys):
        print(f"[predict {index + 1}/{len(prediction_keys)}] {key}", flush=True)
        result, metrics = predict_one_pair(
            key,
            prepared,
            model_path=model_path,
            device=resolved_device,
            n_estimators=n_estimators,
            random_state=random_state + index,
            chunk_size=chunk_size,
            max_train_rows=max_train_rows,
        )
        result.to_excel(prediction_dir / f"{key}-FD-u-pred.xlsx", index=False, engine="openpyxl")
        metrics_rows.append(metrics)
        pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(metrics_rows)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_dir": batch_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "prediction_design": "leave-one-radius-file-out within the same geometry and direction family",
        "target": "u",
        "interpolated_feature": "fl_interp",
        "model_path": model_path.as_posix(),
        "device": resolved_device,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "max_train_rows": max_train_rows,
        "pair_count": len(pairs),
        "predicted_pair_count": len(prediction_keys),
        "preparation": preparation_reports,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate FL and predict FD u with TabPFN.")
    parser.add_argument("--batch-dir", type=Path, default=Path("data/rawdata/2026-06-20"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_u_from_fl_2026-06-20"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--max-train-rows", type=int, default=10000)
    parser.add_argument("--only", nargs="*", default=None, help="Optional pair keys, e.g. 750-320-10")
    args = parser.parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"TabPFN checkpoint not found: {args.model_path}")
    metrics = run_batch(
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
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
