"""Run FL spatial extrapolation on a single processed Excel file."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src.config import FL_CONFIG, MODEL_DIR, RE_COL, RESULTS_DIR
from utils import (
    add_engineered_features,
    available_feature_columns,
    check_columns,
    inverse_y,
    mape,
    rmse,
    smape,
    transform_y,
)


SPLIT_MODES = ("x_left", "x_right", "z_bottom", "z_top")
LOCAL_REGRESSOR_CKPT = MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    check_columns(df, ["X", "Z", RE_COL, "FL"])

    df = df.copy()
    for col in df.columns:
        if col == "geometry_valid":
            continue
        df[col] = pd.to_numeric(df[col], errors="ignore")

    try:
        df = add_engineered_features(df)
    except KeyError as exc:
        print(f"[warn] engineered features skipped for missing column: {exc}")
    df["__row_id__"] = np.arange(len(df))
    return df


def choose_features(df: pd.DataFrame) -> list[str]:
    configured = FL_CONFIG["features"]
    present = [col for col in configured if col in df.columns]
    if not present:
        present = available_feature_columns(df, include_engineered=True)

    usable = []
    for col in present:
        if col == "FL":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        usable.append(col)

    if not usable:
        raise RuntimeError("No usable feature columns found.")
    return usable


def split_spatial(df: pd.DataFrame, mode: str, test_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    if mode == "x_left":
        threshold = float(df["X"].quantile(test_ratio))
        test_df = df[df["X"] <= threshold].copy()
        train_df = df[df["X"] > threshold].copy()
    elif mode == "x_right":
        threshold = float(df["X"].quantile(1 - test_ratio))
        test_df = df[df["X"] >= threshold].copy()
        train_df = df[df["X"] < threshold].copy()
    elif mode == "z_bottom":
        threshold = float(df["Z"].quantile(test_ratio))
        test_df = df[df["Z"] <= threshold].copy()
        train_df = df[df["Z"] > threshold].copy()
    elif mode == "z_top":
        threshold = float(df["Z"].quantile(1 - test_ratio))
        test_df = df[df["Z"] >= threshold].copy()
        train_df = df[df["Z"] < threshold].copy()
    else:
        raise ValueError(f"Unsupported split mode: {mode}")

    if train_df.empty or test_df.empty:
        raise RuntimeError(f"Spatial split failed for {mode}: train/test is empty.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), threshold


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    max_time: int,
    transform_mode: str,
) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    train_x = train_df[features].apply(pd.to_numeric, errors="coerce")
    test_x = test_df[features].apply(pd.to_numeric, errors="coerce")

    medians = train_x.median(axis=0, numeric_only=True).fillna(0.0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_x.values)
    x_test = scaler.transform(test_x.values)

    y_train = pd.to_numeric(train_df["FL"], errors="coerce").values.astype(float)

    _ = max_time
    model = TabPFNRegressor(
        ignore_pretraining_limits=True,
        model_path=str(LOCAL_REGRESSOR_CKPT),
    )
    model.fit(x_train, transform_y(y_train, transform_mode))
    pred = inverse_y(model.predict(x_test), transform_mode)
    return np.asarray(pred, dtype=float)


def save_scatter_plot(pred_df: pd.DataFrame, save_path: Path, title: str) -> None:
    if pred_df.empty:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["FL_true"], pred_df["FL_pred"], alpha=0.75)

    min_v = min(pred_df["FL_true"].min(), pred_df["FL_pred"].min())
    max_v = max(pred_df["FL_true"].max(), pred_df["FL_pred"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("FL true")
    plt.ylabel("FL pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def run_one_mode(
    df: pd.DataFrame,
    features: list[str],
    mode: str,
    test_ratio: float,
    max_time: int,
    transform_mode: str,
    out_dir: Path,
) -> dict:
    train_df, test_df, threshold = split_spatial(df, mode, test_ratio)
    pred = fit_predict(train_df, test_df, features, max_time, transform_mode)

    pred_df = test_df[["__row_id__", "X", "Z", RE_COL, "FL"]].copy()
    pred_df = pred_df.rename(columns={"FL": "FL_true"})
    pred_df["FL_pred"] = pred
    pred_df["abs_error"] = np.abs(pred_df["FL_true"] - pred_df["FL_pred"])
    pred_df["split_mode"] = mode
    pred_df["threshold"] = threshold

    pred_df.to_csv(out_dir / f"pred_{mode}.csv", index=False, encoding="utf-8-sig")
    save_scatter_plot(pred_df, out_dir / f"scatter_{mode}.png", f"Spatial Extrapolation - {mode}")

    return {
        "split_mode": mode,
        "threshold": threshold,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "mae": float(mean_absolute_error(pred_df["FL_true"], pred_df["FL_pred"])),
        "rmse": rmse(pred_df["FL_true"].values, pred_df["FL_pred"].values),
        "r2": float(r2_score(pred_df["FL_true"], pred_df["FL_pred"])) if len(pred_df) >= 2 else float("nan"),
        "mape": mape(pred_df["FL_true"].values, pred_df["FL_pred"].values),
        "smape": smape(pred_df["FL_true"].values, pred_df["FL_pred"].values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FL spatial extrapolation on one Excel file.")
    parser.add_argument("--input", type=Path, required=True, help="Input Excel file.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Spatial holdout ratio.")
    parser.add_argument("--max-time", type=int, default=60, help="AutoTabPFNRegressor time budget.")
    parser.add_argument("--results-dir", type=Path, default=None, help="Output directory.")
    args = parser.parse_args()

    import torch

    _ = torch.__version__
    input_path = args.input.resolve()
    out_dir = args.results_dir or (RESULTS_DIR / "spatial_extrapolation" / input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(input_path)
    features = choose_features(df)

    summary_rows = []
    for mode in SPLIT_MODES:
        print(f"\n=== {mode} ===")
        row = run_one_mode(
            df=df,
            features=features,
            mode=mode,
            test_ratio=args.test_ratio,
            max_time=args.max_time,
            transform_mode=FL_CONFIG["transform"],
            out_dir=out_dir,
        )
        print(
            f"train={row['train_rows']} test={row['test_rows']} "
            f"MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} R2={row['r2']:.6f}"
        )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
