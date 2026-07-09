"""Predict FD components for 2026-05-06 generated FL grids using TabPFN."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR


TARGET_COLS = ["dean_velocity_x", "dean_velocity_y"]
FEATURE_COLS = [
    "Re",
    "X [ m ]",
    "Z [ m ]",
    "底边",
    "高",
    "角度A",
    "角度B",
    "角度C",
    "面积",
    "周长",
    "x/H",
    "z/H",
    "dist_AB_horizontal",
    "dist_AC_horizontal",
]
DISTANCE_COLS = ["dist_AB_horizontal", "dist_AC_horizontal"]


def geometry_key(path_or_name: Path | str) -> str:
    stem = Path(path_or_name).stem
    match = re.search(r"(\d+-\d+)", stem)
    if not match:
        return stem
    key = match.group(1)
    return "830-3416" if key == "830-341" else key


def to_meters(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.abs().median()
    if pd.notna(median) and median > 10:
        return values * 1e-6
    return values


def area_to_m2(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.abs().median()
    if pd.notna(median) and median > 1:
        return values * 1e-12
    return values


def load_training(train_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    for path in sorted(train_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df["source_file"] = path.name
        parts.append(df)
    if not parts:
        raise RuntimeError(f"No training .xlsx files found under {train_dir}")

    df = pd.concat(parts, ignore_index=True)
    missing = [col for col in FEATURE_COLS + TARGET_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Training data is missing required columns: {missing}")

    numeric = df[FEATURE_COLS + TARGET_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric[TARGET_COLS].notna().all(axis=1)
    if not valid.any():
        raise RuntimeError("Training data has no rows with both FD targets.")
    numeric = numeric.loc[valid].reset_index(drop=True)
    return numeric[FEATURE_COLS], numeric[TARGET_COLS]


def build_distance_lookup(distance_dir: Path) -> dict[str, tuple[NearestNeighbors, pd.DataFrame]]:
    lookup: dict[str, tuple[NearestNeighbors, pd.DataFrame]] = {}
    for path in sorted(distance_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, usecols=["point_x", "point_y", *DISTANCE_COLS], engine="openpyxl")
        for col in ["point_x", "point_y", *DISTANCE_COLS]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["point_x", "point_y", *DISTANCE_COLS]).drop_duplicates(["point_x", "point_y"])
        if df.empty:
            continue
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(df[["point_x", "point_y"]].to_numpy(dtype=float))
        lookup[geometry_key(path)] = (nn, df.reset_index(drop=True))
    return lookup


def standardize_generated(
    df: pd.DataFrame,
    source_path: Path,
    distance_lookup: dict[str, tuple[NearestNeighbors, pd.DataFrame]],
) -> pd.DataFrame:
    required = ["雷诺数", "X", "Z", "三角形底边", "高", "角度1", "角度2", "角度3", "面积", "周长"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"{source_path.name} is missing required generated columns: {missing}")

    out = pd.DataFrame(index=df.index)
    out["Re"] = pd.to_numeric(df["雷诺数"], errors="coerce")
    out["X [ m ]"] = to_meters(df["X"])
    out["Z [ m ]"] = to_meters(df["Z"])
    out["底边"] = to_meters(df["三角形底边"])
    out["高"] = to_meters(df["高"])
    out["角度A"] = pd.to_numeric(df["角度1"], errors="coerce")
    out["角度B"] = pd.to_numeric(df["角度2"], errors="coerce")
    out["角度C"] = pd.to_numeric(df["角度3"], errors="coerce")
    out["面积"] = area_to_m2(df["面积"])
    out["周长"] = to_meters(df["周长"])
    eps = 1e-12
    out["x/H"] = out["X [ m ]"] / np.maximum(out["高"], eps)
    out["z/H"] = out["Z [ m ]"] / np.maximum(out["高"], eps)

    key = geometry_key(source_path)
    if key not in distance_lookup:
        raise RuntimeError(f"No 2026-05-24 distance workbook found for {source_path.name} (key={key}).")
    nn, dist_df = distance_lookup[key]
    distances, indices = nn.kneighbors(out[["X [ m ]", "Z [ m ]"]].to_numpy(dtype=float))
    matched = dist_df.iloc[indices[:, 0]].reset_index(drop=True)
    out["dist_AB_horizontal"] = matched["dist_AB_horizontal"].to_numpy(dtype=float)
    out["dist_AC_horizontal"] = matched["dist_AC_horizontal"].to_numpy(dtype=float)
    out["distance_nearest_um"] = distances[:, 0] * 1e6
    return out


def output_name(source_path: Path) -> str:
    return source_path.name.replace("-FL_generated.xlsx", "-FD_generated.xlsx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FD components for 2026-05-06 generated FL files using TabPFN.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/rawdata/2026-05-25/FD数据/10"))
    parser.add_argument("--generated-dir", type=Path, default=Path("data/processed/2026-05-06"))
    parser.add_argument("--distance-dir", type=Path, default=Path("data/rawdata/2026-05-24"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/fd_prediction_2026-05-06_generated_from_2026-05-25_radius10_tabpfn"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    from tabpfn import TabPFNRegressor

    generated_paths = sorted(path for path in args.generated_dir.glob("*_generated.xlsx") if not path.name.startswith("~$"))
    if not generated_paths:
        raise RuntimeError(f"No *_generated.xlsx files found under {args.generated_dir}")
    if not args.model_path.exists():
        raise RuntimeError(f"TabPFN model checkpoint not found: {args.model_path}")

    x_train, y_train = load_training(args.train_dir)
    distance_lookup = build_distance_lookup(args.distance_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train_arr = preprocessor.fit_transform(x_train[FEATURE_COLS])

    models = {}
    for target in TARGET_COLS:
        model = TabPFNRegressor(
            n_estimators=args.n_estimators,
            model_path=args.model_path,
            device=args.device,
            ignore_pretraining_limits=True,
            random_state=args.random_state,
            show_progress_bar=True,
        )
        model.fit(x_train_arr, y_train[target].to_numpy(dtype=float))
        models[target] = model

    log_rows: list[dict[str, object]] = []
    for source_path in generated_paths:
        original = pd.read_excel(source_path, engine="openpyxl")
        x_pred_df = standardize_generated(original, source_path, distance_lookup)
        x_pred_arr = preprocessor.transform(x_pred_df[FEATURE_COLS])

        result = original.copy()
        result["dean_velocity_x_pred"] = np.asarray(models["dean_velocity_x"].predict(x_pred_arr), dtype=float)
        result["dean_velocity_y_pred"] = np.asarray(models["dean_velocity_y"].predict(x_pred_arr), dtype=float)

        out_path = args.out_dir / output_name(source_path)
        result.to_excel(out_path, index=False, engine="openpyxl")
        log_rows.append(
            {
                "source_file": source_path.as_posix(),
                "output_file": out_path.as_posix(),
                "rows": int(len(result)),
                "geometry_key": geometry_key(source_path),
                "nearest_distance_um_mean": float(x_pred_df["distance_nearest_um"].mean()),
                "nearest_distance_um_p95": float(x_pred_df["distance_nearest_um"].quantile(0.95)),
                "nearest_distance_um_max": float(x_pred_df["distance_nearest_um"].max()),
            }
        )

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(args.out_dir / "prediction_log.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "train_dir": args.train_dir.as_posix(),
        "generated_dir": args.generated_dir.as_posix(),
        "distance_dir": args.distance_dir.as_posix(),
        "model_path": args.model_path.as_posix(),
        "n_estimators": args.n_estimators,
        "device": args.device,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "training_rows": int(len(x_train)),
        "generated_files": len(generated_paths),
        "generated_rows": int(log_df["rows"].sum()),
    }
    (args.out_dir / "prediction_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(log_df.to_string(index=False))
    print(f"Saved predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
