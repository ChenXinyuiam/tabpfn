"""Independent 5-fold CV for the 2026-05-21 FD velocity-field batch."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from src.config import MODEL_DIR, RAW_DIR, RESULTS_DIR


TARGET_COLS = ["dean_velocity_x", "dean_velocity_y"]
BASE_FEATURE_COLS = ["Re", "X [ m ]", "Y [ m ]", "Z [ m ]"]
LOCAL_REGRESSOR_CKPT = MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def parse_file_features(path: Path) -> dict[str, float | int | str]:
    match = re.search(r"(\d+)-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse base/height from filename: {path.name}")
    return {
        "source_file": path.name,
        "base_mm": float(match.group(1)),
        "height_mm": float(match.group(2)),
        "is_reverse": int("反" in path.stem or "郡" in path.stem),
    }


def discover_group_dirs(batch_root: Path, group_names: list[str]) -> list[Path]:
    group_set = set(group_names)
    dirs = [path for path in batch_root.rglob("*") if path.is_dir() and path.name in group_set]
    return sorted(dirs, key=lambda path: int(path.name))


def load_group_files(group_dir: Path) -> dict[str, pd.DataFrame]:
    file_dfs: dict[str, pd.DataFrame] = {}
    for path in sorted(group_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        for key, value in parse_file_features(path).items():
            df[key] = value
        file_dfs[path.name] = df
    if not file_dfs:
        raise RuntimeError(f"No .xlsx files found in {group_dir}")
    return file_dfs


def build_model(backend: str, n_estimators: int):
    if backend == "tabpfn":
        from tabpfn import TabPFNRegressor

        kwargs = {"ignore_pretraining_limits": True, "n_estimators": n_estimators}
        if LOCAL_REGRESSOR_CKPT.exists():
            kwargs["model_path"] = str(LOCAL_REGRESSOR_CKPT)
        return TabPFNRegressor(**kwargs), "TabPFNRegressor"

    return (
        ExtraTreesRegressor(
            n_estimators=max(100, n_estimators),
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        ),
        "ExtraTreesRegressor",
    )


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    train[feature_cols + [target_col]] = train[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
    test[feature_cols + [target_col]] = test[feature_cols + [target_col]].apply(pd.to_numeric, errors="coerce")
    train = train.dropna(subset=[target_col]).reset_index(drop=True)

    x_train_df = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_test_df = test[feature_cols].replace([np.inf, -np.inf], np.nan)
    medians = x_train_df.median(axis=0, numeric_only=True).fillna(0.0)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_df.fillna(medians).to_numpy(dtype=float))
    x_test = scaler.transform(x_test_df.fillna(medians).to_numpy(dtype=float))
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, test


def evaluate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    backend: str,
    n_estimators: int,
) -> tuple[dict, pd.DataFrame, str]:
    x_train, y_train, x_test, y_test, test_clean = prepare_xy(train_df, test_df, feature_cols, target_col)
    model, backend_name = build_model(backend, n_estimators)
    model.fit(x_train, y_train)
    pred = np.asarray(model.predict(x_test), dtype=float)

    valid = np.isfinite(y_test) & np.isfinite(pred)
    pred_df = test_clean[["source_file", "Re", "X [ m ]", "Y [ m ]", "Z [ m ]", "base_mm", "height_mm", "is_reverse"]].copy()
    pred_df["target"] = target_col
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = pred
    pred_df["abs_error"] = np.where(valid, np.abs(y_test - pred), np.nan)

    if not valid.any():
        metrics = {"rows": int(len(y_test)), "rows_eval": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan}
    else:
        yt = y_test[valid]
        yp = pred[valid]
        metrics = {
            "rows": int(len(y_test)),
            "rows_eval": int(valid.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": rmse(yt, yp),
            "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
        }
    return metrics, pred_df, backend_name


def run_group_cv(
    group_dir: Path,
    results_dir: Path,
    n_splits: int,
    random_state: int,
    backend: str,
    n_estimators: int,
) -> pd.DataFrame:
    file_dfs = load_group_files(group_dir)
    file_names = sorted(file_dfs)
    if len(file_names) < n_splits:
        raise RuntimeError(f"Group {group_dir.name} needs at least {n_splits} files, found {len(file_names)}")

    out_dir = results_dir / f"group_{group_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict] = []
    assignment_rows: list[dict] = []
    pred_parts: list[pd.DataFrame] = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, test_idx) in enumerate(kf.split(file_names), start=1):
        train_files = [file_names[i] for i in train_idx]
        test_files = [file_names[i] for i in test_idx]
        train_df = pd.concat([file_dfs[name] for name in train_files], ignore_index=True)
        test_df = pd.concat([file_dfs[name] for name in test_files], ignore_index=True)

        for target_col in TARGET_COLS:
            metrics, pred_df, backend_name = evaluate_fold(
                train_df=train_df,
                test_df=test_df,
                feature_cols=BASE_FEATURE_COLS,
                target_col=target_col,
                backend=backend,
                n_estimators=n_estimators,
            )
            pred_df["group"] = group_dir.name
            pred_df["fold"] = f"fold{fold}"
            pred_parts.append(pred_df)
            fold_rows.append(
                {
                    "group": group_dir.name,
                    "target": target_col,
                    "fold": f"fold{fold}",
                    "train_files": " | ".join(train_files),
                    "test_files": " | ".join(test_files),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_df)),
                    "backend": backend_name,
                    **metrics,
                }
            )

        for name in train_files:
            assignment_rows.append({"group": group_dir.name, "fold": f"fold{fold}", "role": "train", "file": name})
        for name in test_files:
            assignment_rows.append({"group": group_dir.name, "fold": f"fold{fold}", "role": "test", "file": name})

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True)
    assignment_df = pd.DataFrame(assignment_rows)

    summary_rows = []
    for target_col, target_df in fold_df.groupby("target", sort=True):
        valid = target_df[target_df["rows_eval"] > 0]
        weights = valid["rows_eval"] / valid["rows_eval"].sum()
        summary_rows.append(
            {
                "group": group_dir.name,
                "target": target_col,
                "n_files": len(file_names),
                "n_splits": n_splits,
                "rows_eval_total": int(valid["rows_eval"].sum()),
                "mae_w": float(np.average(valid["mae"], weights=weights)),
                "rmse_w": float(np.average(valid["rmse"], weights=weights)),
                "r2_w": float(np.average(valid["r2"], weights=weights)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(out_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent 5-fold CV for nested FD groups.")
    parser.add_argument("--batch", default="2026-05-21")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "fd_2026-05-21_group_5fold")
    parser.add_argument("--groups", default="10,20,30")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--backend", choices=["extra-trees", "tabpfn"], default="extra-trees")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    data_root = args.data_root or (RAW_DIR / args.batch)
    group_names = [item.strip() for item in args.groups.split(",") if item.strip()]
    group_dirs = discover_group_dirs(data_root, group_names)
    found = {path.name for path in group_dirs}
    missing = sorted(set(group_names) - found)
    if missing:
        raise RuntimeError(f"Missing group directories under {data_root}: {missing}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group_dir in group_dirs:
        print(f"\n=== group {group_dir.name}: independent {args.n_splits}-fold CV ===")
        summary = run_group_cv(
            group_dir=group_dir,
            results_dir=args.results_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
            backend=args.backend,
            n_estimators=args.n_estimators,
        )
        print(summary.to_string(index=False))
        summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.results_dir / "summary_all_groups.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved all results to {args.results_dir}")


if __name__ == "__main__":
    main()
