"""Predict FD for generated grids using one same-geometry FD workbook per file."""

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


def resolve_tabpfn_device(device: str) -> str:
    """Prefer CUDA when requested and available, otherwise fall back to CPU."""
    if device not in {"prefer-cuda", "auto"}:
        return device

    try:
        import torch
    except Exception as exc:
        print(f"[device] torch import failed ({exc}); using cpu", flush=True)
        return "cpu"

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA available: {device_name}; using cuda", flush=True)
        return "cuda"

    print("[device] CUDA is not available; using cpu", flush=True)
    return "cpu"


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


def generated_output_name(source_path: Path) -> str:
    return source_path.name.replace("-FL_generated.xlsx", "-FD_generated.xlsx")


def find_training_file(train_dir: Path, key: str) -> Path | None:
    candidate = train_dir / f"{key}.xlsx"
    if candidate.exists():
        return candidate
    return None


def load_training_file(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_excel(path, engine="openpyxl")
    missing = [col for col in FEATURE_COLS + TARGET_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name} is missing required training columns: {missing}")
    numeric = df[FEATURE_COLS + TARGET_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric[TARGET_COLS].notna().all(axis=1)
    if not valid.any():
        raise RuntimeError(f"{path.name} has no valid rows with both FD targets.")
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
    distance, indices = nn.kneighbors(out[["X [ m ]", "Z [ m ]"]].to_numpy(dtype=float))
    matched = dist_df.iloc[indices[:, 0]].reset_index(drop=True)
    out["dist_AB_horizontal"] = matched["dist_AB_horizontal"].to_numpy(dtype=float)
    out["dist_AC_horizontal"] = matched["dist_AC_horizontal"].to_numpy(dtype=float)
    out["distance_nearest_um"] = distance[:, 0] * 1e6
    return out


def fit_predict_one_target(
    x_train_arr: np.ndarray,
    y_train: np.ndarray,
    x_pred_arr: np.ndarray,
    *,
    model_path: Path,
    n_estimators: int,
    device: str,
    random_state: int,
) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(
        n_estimators=n_estimators,
        model_path=model_path,
        device=device,
        ignore_pretraining_limits=True,
        random_state=random_state,
        show_progress_bar=True,
    )
    model.fit(x_train_arr, y_train)
    return np.asarray(model.predict(x_pred_arr), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict generated FD files with same-geometry TabPFN training files.")
    parser.add_argument("--train-dir", type=Path, default=Path("data/rawdata/2026-05-25/FD数据/10"))
    parser.add_argument("--generated-dir", type=Path, default=Path("data/processed/2026-05-23"))
    parser.add_argument("--distance-dir", type=Path, default=Path("data/rawdata/2026-05-24"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/fd_prediction_2026-05-23_generated_per_geometry_tabpfn"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument(
        "--device",
        default="prefer-cuda",
        help="TabPFN device. Default prefer-cuda uses cuda when available, otherwise cpu. You may also pass cuda, cpu, or auto.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise RuntimeError(f"TabPFN model checkpoint not found: {args.model_path}")
    generated_paths = sorted(path for path in args.generated_dir.glob("*_generated.xlsx") if not path.name.startswith("~$"))
    if not generated_paths:
        raise RuntimeError(f"No *_generated.xlsx files found under {args.generated_dir}")

    resolved_device = resolve_tabpfn_device(args.device)
    distance_lookup = build_distance_lookup(args.distance_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log_rows: list[dict[str, object]] = []
    for generated_path in generated_paths:
        key = geometry_key(generated_path)
        train_path = find_training_file(args.train_dir, key)
        if train_path is None:
            log_rows.append(
                {
                    "source_file": generated_path.as_posix(),
                    "status": "skipped_no_training_file",
                    "geometry_key": key,
                }
            )
            continue

        out_path = args.out_dir / generated_output_name(generated_path)
        if args.skip_existing and out_path.exists():
            log_rows.append(
                {
                    "source_file": generated_path.as_posix(),
                    "training_file": train_path.as_posix(),
                    "output_file": out_path.as_posix(),
                    "status": "skipped_existing",
                    "geometry_key": key,
                }
            )
            continue

        print(f"\n=== {generated_path.name} <= {train_path.name} ===", flush=True)
        x_train, y_train = load_training_file(train_path)
        generated_df = pd.read_excel(generated_path, engine="openpyxl")
        x_pred_df = standardize_generated(generated_df, generated_path, distance_lookup)

        preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
        x_train_arr = preprocessor.fit_transform(x_train[FEATURE_COLS])
        x_pred_arr = preprocessor.transform(x_pred_df[FEATURE_COLS])

        result = generated_df.copy()
        result["dean_velocity_x_pred"] = fit_predict_one_target(
            x_train_arr,
            y_train["dean_velocity_x"].to_numpy(dtype=float),
            x_pred_arr,
            model_path=args.model_path,
            n_estimators=args.n_estimators,
            device=resolved_device,
            random_state=args.random_state,
        )
        result["dean_velocity_y_pred"] = fit_predict_one_target(
            x_train_arr,
            y_train["dean_velocity_y"].to_numpy(dtype=float),
            x_pred_arr,
            model_path=args.model_path,
            n_estimators=args.n_estimators,
            device=resolved_device,
            random_state=args.random_state + 1,
        )

        result.to_excel(out_path, index=False, engine="openpyxl")
        log_rows.append(
            {
                "source_file": generated_path.as_posix(),
                "training_file": train_path.as_posix(),
                "output_file": out_path.as_posix(),
                "status": "written",
                "geometry_key": key,
                "training_rows": int(len(x_train)),
                "prediction_rows": int(len(result)),
                "nearest_distance_um_mean": float(x_pred_df["distance_nearest_um"].mean()),
                "nearest_distance_um_p95": float(x_pred_df["distance_nearest_um"].quantile(0.95)),
                "nearest_distance_um_max": float(x_pred_df["distance_nearest_um"].max()),
            }
        )
        pd.DataFrame(log_rows).to_csv(args.out_dir / "prediction_log.csv", index=False, encoding="utf-8-sig")

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(args.out_dir / "prediction_log.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "train_dir": args.train_dir.as_posix(),
        "generated_dir": args.generated_dir.as_posix(),
        "distance_dir": args.distance_dir.as_posix(),
        "out_dir": args.out_dir.as_posix(),
        "model_path": args.model_path.as_posix(),
        "n_estimators": args.n_estimators,
        "requested_device": args.device,
        "resolved_device": resolved_device,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "generated_files": len(generated_paths),
        "written_files": int((log_df.get("status") == "written").sum()) if not log_df.empty else 0,
    }
    (args.out_dir / "prediction_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nPrediction log:")
    print(log_df.to_string(index=False))
    print(f"Saved predictions to {args.out_dir}")


if __name__ == "__main__":
    main()
