"""5-fold FL extrapolation with whole-triangle files as the split unit."""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR, PROCESSED_DIR, RE_COL, RESULTS_DIR
from utils import mape, rmse, smape


LOCAL_REGRESSOR_CKPT = MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt"
TARGET_COL = "FL"
DROP_COLS = {TARGET_COL, "X", "Z", "generated"}


def is_triangle_file(path: Path) -> bool:
    return path.suffix.lower() == ".xlsx" and (
        path.name.endswith("-FL.xlsx") or path.name.endswith("-sym.xlsx")
    )


def latest_unique_files(root: Path) -> list[Path]:
    latest: dict[str, Path] = {}
    for path in sorted(root.rglob("*.xlsx")):
        if not is_triangle_file(path):
            continue
        current = latest.get(path.name)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest[path.name] = path
    return sorted(latest.values(), key=lambda path: path.name)


def collect_files(root: Path, pattern: str | None, recursive: bool) -> list[Path]:
    if not pattern:
        return latest_unique_files(root)

    globber = root.rglob if recursive else root.glob
    return sorted(
        path
        for path in globber(pattern)
        if path.suffix.lower() == ".xlsx" and not path.name.startswith("~$")
    )


def paired_triangle_group(name: str) -> str:
    stem = Path(name).stem.removesuffix("_generated")
    return re.sub(r"-(FL|sym)$", "", stem)


def load_files(paths: list[Path]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    file_dfs: dict[str, pd.DataFrame] = {}
    common_cols: set[str] | None = None

    for path in paths:
        df = pd.read_excel(path, engine="openpyxl")
        df["__source_file__"] = path.name
        file_dfs[path.name] = df

        cols = set(df.columns) - {"__source_file__"}
        common_cols = cols if common_cols is None else (common_cols & cols)

    if common_cols is None:
        raise RuntimeError("No triangle files found.")

    feature_cols = sorted(common_cols - DROP_COLS)
    if not feature_cols:
        raise RuntimeError("No common feature columns remain after dropping target/X/Z.")

    return file_dfs, feature_cols


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    test_x = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    medians = train_x.median(axis=0, numeric_only=True).fillna(0.0)
    train_x = train_x.fillna(medians)
    test_x = test_x.fillna(medians)

    y_train = pd.to_numeric(train_df[TARGET_COL], errors="coerce").values.astype(float)
    y_test = pd.to_numeric(test_df[TARGET_COL], errors="coerce").values.astype(float)

    # Normalize every numeric model feature. This explicitly includes physical
    # size columns such as triangle base and height when they are present.
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(train_x.values)
    x_test = scaler.transform(test_x.values)
    return x_train, y_train, x_test, y_test


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int,
) -> tuple[np.ndarray, str]:
    x_train, y_train, x_test, _ = prepare_xy(train_df, test_df, feature_cols)
    try:
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            ignore_pretraining_limits=True,
            model_path=str(LOCAL_REGRESSOR_CKPT),
            n_estimators=n_estimators,
        )
        backend = "TabPFNRegressor"
    except ModuleNotFoundError:
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=max(100, n_estimators),
            random_state=42,
            n_jobs=-1,
        )
        backend = "ExtraTreesRegressor_fallback_tabpfn_missing"

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return np.asarray(pred, dtype=float), backend


def main() -> None:
    parser = argparse.ArgumentParser(description="5-fold triangle-level FL extrapolation.")
    parser.add_argument("--data-root", type=Path, default=PROCESSED_DIR, help="Root directory to scan for feature Excel files.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "triangle_5fold_cv", help="Output directory.")
    parser.add_argument("--pattern", default=None, help="Optional glob pattern, e.g. '*_generated.xlsx'.")
    parser.add_argument("--recursive", action="store_true", help="Use recursive globbing with --pattern.")
    parser.add_argument(
        "--paired-fl-sym",
        action="store_true",
        help="Keep xxx-xxx-FL and xxx-xxx-sym files in the same train/test fold.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1)
    args = parser.parse_args()

    files = collect_files(args.data_root, args.pattern, args.recursive)
    if len(files) < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} files, found {len(files)}")

    file_dfs, feature_cols = load_files(files)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    file_names = [path.name for path in files]
    group_to_files: dict[str, list[str]] = {}
    if args.paired_fl_sym:
        for name in file_names:
            group_to_files.setdefault(paired_triangle_group(name), []).append(name)
        split_units = sorted(group_to_files)
    else:
        split_units = file_names

    if len(split_units) < args.n_splits:
        raise RuntimeError(f"Need at least {args.n_splits} split units, found {len(split_units)}")

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    fold_rows: list[dict] = []
    file_rows: list[dict] = []
    pred_parts: list[pd.DataFrame] = []
    model_backends: set[str] = set()

    print(f"Using {len(files)} unique triangle files")
    if args.paired_fl_sym:
        print(f"Paired FL/sym split units ({len(split_units)}): {split_units}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(split_units), start=1):
        train_units = [split_units[i] for i in train_idx]
        test_units = [split_units[i] for i in test_idx]
        if args.paired_fl_sym:
            train_files = [name for unit in train_units for name in group_to_files[unit]]
            test_files = [name for unit in test_units for name in group_to_files[unit]]
        else:
            train_files = train_units
            test_files = test_units

        print(f"\n=== fold{fold_idx} ===")
        if args.paired_fl_sym:
            print("test groups:", ", ".join(test_units))
        print("test files:", ", ".join(test_files))

        train_df = pd.concat([file_dfs[name] for name in train_files], ignore_index=True)
        test_df = pd.concat([file_dfs[name] for name in test_files], ignore_index=True)

        pred, backend = fit_predict(train_df, test_df, feature_cols, args.n_estimators)
        model_backends.add(backend)
        pred_df = test_df.copy()
        pred_df["FL_true"] = pd.to_numeric(pred_df[TARGET_COL], errors="coerce").values.astype(float)
        pred_df["FL_pred"] = pred
        pred_df["abs_error"] = np.abs(pred_df["FL_true"] - pred_df["FL_pred"])
        pred_df["fold"] = f"fold{fold_idx}"
        pred_parts.append(pred_df)

        row = {
            "fold": f"fold{fold_idx}",
            "train_files": " | ".join(train_files),
            "test_files": " | ".join(test_files),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "mae": float(mean_absolute_error(pred_df["FL_true"], pred_df["FL_pred"])),
            "rmse": rmse(pred_df["FL_true"].values, pred_df["FL_pred"].values),
            "r2": float(r2_score(pred_df["FL_true"], pred_df["FL_pred"])),
            "mape": mape(pred_df["FL_true"].values, pred_df["FL_pred"].values),
            "smape": smape(pred_df["FL_true"].values, pred_df["FL_pred"].values),
        }
        print(
            f"train_rows={row['train_rows']} test_rows={row['test_rows']} "
            f"MAE={row['mae']:.6f} RMSE={row['rmse']:.6f} R2={row['r2']:.6f}"
        )
        fold_rows.append(row)

        for name in test_files:
            sub = pred_df[pred_df["__source_file__"] == name].copy()
            file_rows.append(
                {
                    "fold": f"fold{fold_idx}",
                    "file": name,
                    "rows": int(len(sub)),
                    "mae": float(mean_absolute_error(sub["FL_true"], sub["FL_pred"])),
                    "rmse": rmse(sub["FL_true"].values, sub["FL_pred"].values),
                    "r2": float(r2_score(sub["FL_true"], sub["FL_pred"])) if len(sub) >= 2 else float("nan"),
                    "mape": mape(sub["FL_true"].values, sub["FL_pred"].values),
                    "smape": smape(sub["FL_true"].values, sub["FL_pred"].values),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    file_df = pd.DataFrame(file_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True)

    total_rows = fold_df["test_rows"].sum()
    weights = fold_df["test_rows"] / total_rows
    summary_df = pd.DataFrame(
        [
            {
                "n_files": len(files),
                "n_split_units": len(split_units),
                "paired_fl_sym": bool(args.paired_fl_sym),
                "n_splits": args.n_splits,
                "n_estimators": args.n_estimators,
                "model_backend": " | ".join(sorted(model_backends)),
                "feature_count": len(feature_cols),
                "rows_total": int(total_rows),
                "mae_w": float(np.average(fold_df["mae"], weights=weights)),
                "rmse_w": float(np.average(fold_df["rmse"], weights=weights)),
                "r2_w": float(np.average(fold_df["r2"], weights=weights)),
                "mape_w": float(np.average(fold_df["mape"], weights=weights)),
                "smape_w": float(np.average(fold_df["smape"], weights=weights)),
            }
        ]
    )

    normalized_feature_rows = [
        {
            "feature": feature,
            "normalized_by": "MinMaxScaler fit on each fold's training split",
            "is_size_feature": any(token in feature for token in ["底", "高", "楂", "base", "height"]),
        }
        for feature in feature_cols
    ]
    preprocessing_df = pd.DataFrame(normalized_feature_rows)

    fold_df.to_csv(args.results_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    file_df.to_csv(args.results_dir / "file_metrics.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(args.results_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.results_dir / "summary.csv", index=False, encoding="utf-8-sig")
    preprocessing_df.to_csv(args.results_dir / "preprocessing_log.csv", index=False, encoding="utf-8-sig")

    print("\nWeighted summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved results to {args.results_dir}")


if __name__ == "__main__":
    main()
