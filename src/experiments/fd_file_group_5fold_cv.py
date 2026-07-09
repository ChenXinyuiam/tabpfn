"""File-level 5-fold CV for FD velocity-field workbooks.

Whole .xlsx files are kept intact in train/test splits. Reflected files such as
750-400.xlsx and 750-400反.xlsx are grouped into the same split unit.
"""

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

from src.config import RAW_DIR, RESULTS_DIR


TARGET_COLS = ["dean_velocity_x", "dean_velocity_y"]
EXCLUDED_FEATURE_COLS = ["Y [ m ]", *TARGET_COLS]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def discover_group_dirs(batch_root: Path, groups: list[str]) -> list[Path]:
    group_set = set(groups)
    dirs = [path for path in batch_root.rglob("*") if path.is_dir() and path.name in group_set]
    return sorted(dirs, key=lambda path: int(path.name))


def paired_triangle_group(filename: str) -> str:
    stem = Path(filename).stem
    stem = stem.replace("反", "")
    match = re.search(r"(\d+-\d+)", stem)
    return match.group(1) if match else stem


def load_group_files(group_dir: Path) -> dict[str, pd.DataFrame]:
    file_dfs: dict[str, pd.DataFrame] = {}
    for path in sorted(group_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df["source_file"] = path.name
        file_dfs[path.name] = df
    if not file_dfs:
        raise RuntimeError(f"No .xlsx files found in {group_dir}")
    return file_dfs


def select_feature_cols(df: pd.DataFrame) -> list[str]:
    missing_targets = [col for col in TARGET_COLS if col not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")

    feature_cols = [col for col in df.columns if col not in EXCLUDED_FEATURE_COLS + ["source_file"]]
    numeric_features: list[str] = []
    for col in feature_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            numeric_features.append(col)

    if not numeric_features:
        raise RuntimeError("No numeric feature columns found")
    return numeric_features


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()
    use_cols = feature_cols + TARGET_COLS
    train[use_cols] = train[use_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    test[use_cols] = test[use_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train = train.dropna(subset=TARGET_COLS).reset_index(drop=True)
    test = test.dropna(subset=TARGET_COLS).reset_index(drop=True)

    x_train_df = train[feature_cols]
    x_test_df = test[feature_cols]
    medians = x_train_df.median(axis=0, numeric_only=True).fillna(0.0)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train_df.fillna(medians).to_numpy(dtype=float))
    x_test = scaler.transform(x_test_df.fillna(medians).to_numpy(dtype=float))
    y_train = train[TARGET_COLS].to_numpy(dtype=float)
    y_test = test[TARGET_COLS].to_numpy(dtype=float)
    return x_train, y_train, x_test, y_test, test


def build_model(n_estimators: int, random_state: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )


def run_group_cv(
    group_dir: Path,
    results_dir: Path,
    n_splits: int,
    random_state: int,
    n_estimators: int,
) -> pd.DataFrame:
    file_dfs = load_group_files(group_dir)
    all_df = pd.concat(file_dfs.values(), ignore_index=True)
    feature_cols = select_feature_cols(all_df)

    group_to_files: dict[str, list[str]] = {}
    for file_name in sorted(file_dfs):
        group_to_files.setdefault(paired_triangle_group(file_name), []).append(file_name)
    split_units = sorted(group_to_files)
    if len(split_units) < n_splits:
        raise RuntimeError(f"Need at least {n_splits} split units, found {len(split_units)}")

    out_dir = results_dir / f"radius_{group_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    pred_parts: list[pd.DataFrame] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(split_units), start=1):
        train_units = [split_units[i] for i in train_idx]
        test_units = [split_units[i] for i in test_idx]
        train_files = [name for unit in train_units for name in group_to_files[unit]]
        test_files = [name for unit in test_units for name in group_to_files[unit]]

        train_df = pd.concat([file_dfs[name] for name in train_files], ignore_index=True)
        test_df = pd.concat([file_dfs[name] for name in test_files], ignore_index=True)
        x_train, y_train, x_test, y_test, test_clean = prepare_xy(train_df, test_df, feature_cols)

        model = build_model(n_estimators=n_estimators, random_state=random_state + fold)
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_test), dtype=float)

        for target_idx, target_col in enumerate(TARGET_COLS):
            yt = y_test[:, target_idx]
            yp = pred[:, target_idx]
            valid = np.isfinite(yt) & np.isfinite(yp)
            fold_rows.append(
                {
                    "radius": group_dir.name,
                    "fold": f"fold{fold}",
                    "target": target_col,
                    "train_units": " | ".join(train_units),
                    "test_units": " | ".join(test_units),
                    "train_files": " | ".join(train_files),
                    "test_files": " | ".join(test_files),
                    "train_rows": int(len(train_df)),
                    "test_rows": int(len(test_clean)),
                    "rows_eval": int(valid.sum()),
                    "mae": float(mean_absolute_error(yt[valid], yp[valid])) if valid.any() else np.nan,
                    "rmse": rmse(yt[valid], yp[valid]) if valid.any() else np.nan,
                    "r2": float(r2_score(yt[valid], yp[valid])) if valid.sum() >= 2 else np.nan,
                }
            )

        pred_df = test_clean[["source_file"]].copy()
        pred_df.insert(0, "fold", f"fold{fold}")
        pred_df.insert(0, "radius", group_dir.name)
        for col in ["Re", "X [ m ]", "Y [ m ]", "Z [ m ]"]:
            if col in test_clean.columns and col not in pred_df.columns:
                pred_df[col] = test_clean[col].to_numpy()
        for target_idx, target_col in enumerate(TARGET_COLS):
            pred_df[f"{target_col}_true"] = y_test[:, target_idx]
            pred_df[f"{target_col}_pred"] = pred[:, target_idx]
            pred_df[f"{target_col}_abs_error"] = np.abs(y_test[:, target_idx] - pred[:, target_idx])
        pred_parts.append(pred_df)

        for unit in train_units:
            for name in group_to_files[unit]:
                assignment_rows.append({"radius": group_dir.name, "fold": f"fold{fold}", "role": "train", "unit": unit, "file": name})
        for unit in test_units:
            for name in group_to_files[unit]:
                assignment_rows.append({"radius": group_dir.name, "fold": f"fold{fold}", "role": "test", "unit": unit, "file": name})

    fold_df = pd.DataFrame(fold_rows)
    assignment_df = pd.DataFrame(assignment_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True)

    summary_rows = []
    for target_col, target_df in fold_df.groupby("target", sort=True):
        valid = target_df[target_df["rows_eval"] > 0]
        weights = valid["rows_eval"] / valid["rows_eval"].sum()
        summary_rows.append(
            {
                "radius": group_dir.name,
                "target": target_col,
                "n_files": int(len(file_dfs)),
                "n_split_units": int(len(split_units)),
                "n_rows": int(sum(len(df) for df in file_dfs.values())),
                "n_splits": n_splits,
                "feature_count": len(feature_cols),
                "features": " | ".join(feature_cols),
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
    parser = argparse.ArgumentParser(description="Run file-level 5-fold CV for FD radius folders.")
    parser.add_argument("--batch", default="2026-05-25")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "fd_file_group_5fold")
    parser.add_argument("--groups", default="10")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    data_root = args.data_root or (RAW_DIR / args.batch)
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    group_dirs = discover_group_dirs(data_root, groups)
    found = {path.name for path in group_dirs}
    missing = sorted(set(groups) - found)
    if missing:
        raise RuntimeError(f"Missing group directories under {data_root}: {missing}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group_dir in group_dirs:
        print(f"\n=== radius {group_dir.name}: file-level {args.n_splits}-fold CV ===")
        summary = run_group_cv(
            group_dir=group_dir,
            results_dir=args.results_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        print(
            summary[
                ["radius", "target", "n_files", "n_split_units", "n_rows", "feature_count", "mae_w", "rmse_w", "r2_w"]
            ].to_string(index=False)
        )
        summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.results_dir / "summary_all_groups.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved all results to {args.results_dir}")


if __name__ == "__main__":
    main()
