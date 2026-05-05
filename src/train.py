"""
Training entry points for FD and FL regression via TabPFN cross-validation.

Usage (run from project root):
    python -m src.train --target FD
    python -m src.train --target FL
    python -m src.train --target both
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .config import (
    MODEL_DIR, TRIANGLE_GROUPS_ROOT, RESULTS_DIR,
    FD_CONFIG, FL_CONFIG, N_SPLITS, TABPFN_IGNORE_PRETRAINING_LIMITS,
)
from utils import (
    setup_offline_tabpfn,
    mape, smape,
    transform_y, inverse_y,
    discover_leaves, load_fold_df, check_columns,
)


def run_experiment(config: dict) -> float:
    """
    Run K-fold cross-validation for a single target using AutoTabPFNRegressor.

    config keys:
        target      - column name to predict ("FD" or "FL")
        transform   - "identity" | "log1p" | "sqrt"
        features    - list of feature column names
        n_splits    - number of CV folds
        max_time    - seconds budget for AutoTabPFNRegressor
        data_root   - Path to the triangle_groups root directory

    Returns the mean R² across all folds.
    """
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

    data_root = Path(config["data_root"])
    n_splits  = config["n_splits"]
    features  = config["features"]
    target    = config["target"]
    add_features = config.get("add_features", True)

    leaves = discover_leaves(data_root, n_splits)
    print(f"[{target}] {len(leaves)} leaves found")

    r2_scores = []

    for k in range(1, n_splits + 1):
        train_parts, test_parts = [], []

        for leaf in leaves:
            bucket = f"{leaf['shape']}/{leaf['R']}/{leaf['Re']}"
            if k in leaf["fold_paths"]:
                test_parts.append(load_fold_df(leaf["fold_paths"][k], bucket, add_features=add_features))
            for j in range(1, n_splits + 1):
                if j != k and j in leaf["fold_paths"]:
                    train_parts.append(load_fold_df(leaf["fold_paths"][j], bucket, add_features=add_features))

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df  = pd.concat(test_parts,  ignore_index=True)

        check_columns(train_df, features + [target])
        check_columns(test_df,  features + [target])

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(train_df[features].values)
        X_te = scaler.transform(test_df[features].values)

        y_tr = train_df[target].values
        y_te = test_df[target].values

        model = AutoTabPFNRegressor(
            max_time=config["max_time"],
            ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS,
        )
        model.fit(X_tr, transform_y(y_tr, config["transform"]))
        pred = inverse_y(model.predict(X_te), config["transform"])

        r2 = r2_score(y_te, pred)
        print(f"  fold{k}  R²={r2:.4f}  MAPE={mape(y_te, pred):.2f}%")
        r2_scores.append(r2)

    mean_r2 = float(np.mean(r2_scores))
    print(f"\n[{target}] mean R² = {mean_r2:.4f}")
    return mean_r2


def run_cv_with_metrics(config: dict, results_dir: Path | None = None) -> pd.DataFrame:
    """
    Full cross-validation that saves per-fold and bucket-level metrics to CSV.

    Returns the fold-level metrics DataFrame.
    """
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

    data_root  = Path(config["data_root"])
    n_splits   = config["n_splits"]
    features   = config["features"]
    target     = config["target"]
    add_features = config.get("add_features", True)
    results_dir = results_dir or (RESULTS_DIR / f"cv_{target}")
    results_dir.mkdir(parents=True, exist_ok=True)

    leaves = discover_leaves(data_root, n_splits)
    print(f"[{target}] {len(leaves)} leaves found; running {n_splits}-fold CV")

    fold_metrics   = []
    bucket_metrics = []

    for k in range(1, n_splits + 1):
        train_parts, test_parts = [], []

        for leaf in leaves:
            bucket = f"{leaf['shape']}/{leaf['R']}/{leaf['Re']}"
            if k in leaf["fold_paths"]:
                test_parts.append(load_fold_df(leaf["fold_paths"][k], bucket, add_features=add_features))
            for j in range(1, n_splits + 1):
                if j != k and j in leaf["fold_paths"]:
                    train_parts.append(load_fold_df(leaf["fold_paths"][j], bucket, add_features=add_features))

        if not train_parts or not test_parts:
            print(f"  [warn] fold{k}: empty split, skipping")
            continue

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df  = pd.concat(test_parts,  ignore_index=True)

        check_columns(train_df, features + [target])
        check_columns(test_df,  features + [target])

        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(train_df[features].values)
        X_te = scaler.transform(test_df[features].values)

        y_tr = train_df[target].values
        y_te = test_df[target].values

        model = AutoTabPFNRegressor(
            max_time=config["max_time"],
            ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS,
        )
        model.fit(X_tr, transform_y(y_tr, config["transform"]))
        pred = inverse_y(model.predict(X_te), config["transform"])

        fold_metrics.append({
            "target": target, "fold": f"fold{k}", "rows": len(y_te),
            "mse":   mean_squared_error(y_te, pred),
            "mae":   mean_absolute_error(y_te, pred),
            "r2":    r2_score(y_te, pred),
            "mape":  mape(y_te, pred),
            "smape": smape(y_te, pred),
        })

        gb = test_df.copy()
        gb[f"{target}_pred"] = pred
        for bucket_id, sub in gb.groupby("__bucket__"):
            yt_b = sub[target].values
            yp_b = sub[f"{target}_pred"].values
            bucket_metrics.append({
                "target": target, "fold": f"fold{k}", "bucket": bucket_id,
                "rows": len(yt_b),
                "mse":   mean_squared_error(yt_b, yp_b),
                "mae":   mean_absolute_error(yt_b, yp_b),
                "r2":    r2_score(yt_b, yp_b) if len(yt_b) > 1 else float("nan"),
                "mape":  mape(yt_b, yp_b),
                "smape": smape(yt_b, yp_b),
            })

    df_folds   = pd.DataFrame(fold_metrics)
    df_buckets = pd.DataFrame(bucket_metrics)

    df_folds.to_csv(results_dir / "fold_global_metrics.csv",  index=False, encoding="utf-8-sig")
    df_buckets.to_csv(results_dir / "fold_bucket_metrics.csv", index=False, encoding="utf-8-sig")

    if not df_folds.empty:
        N = df_folds["rows"].sum()
        w = df_folds["rows"] / N
        summary = pd.DataFrame([{
            "target": target, "rows_total": int(N),
            "mse_w":   float(np.average(df_folds["mse"],   weights=w)),
            "mae_w":   float(np.average(df_folds["mae"],   weights=w)),
            "r2_w":    float(np.average(df_folds["r2"],    weights=w)),
            "mape_w":  float(np.average(df_folds["mape"],  weights=w)),
            "smape_w": float(np.average(df_folds["smape"], weights=w)),
        }])
        summary.to_csv(results_dir / "global_weighted_summary.csv", index=False, encoding="utf-8-sig")
        print(f"[{target}] mean R² (weighted) = {summary['r2_w'].iloc[0]:.4f}")

    print(f"[{target}] results saved to {results_dir}")
    return df_folds


def main():
    parser = argparse.ArgumentParser(description="Train FD/FL models with TabPFN CV")
    parser.add_argument("--target", choices=["FD", "FL", "both"], default="both")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    setup_offline_tabpfn(args.model_dir)

    configs = []
    if args.target in ("FD", "both"):
        cfg = dict(FD_CONFIG, data_root=TRIANGLE_GROUPS_ROOT)
        configs.append(cfg)
    if args.target in ("FL", "both"):
        cfg = dict(FL_CONFIG, data_root=TRIANGLE_GROUPS_ROOT)
        configs.append(cfg)

    for cfg in configs:
        run_cv_with_metrics(cfg)


if __name__ == "__main__":
    main()
