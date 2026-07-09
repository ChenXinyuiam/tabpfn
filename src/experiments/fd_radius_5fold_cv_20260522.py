"""Row-level 5-fold CV for the 2026-05-22 FD velocity-field batch.

Each curvature-radius folder (10, 20, 30) is evaluated independently.
"""

from __future__ import annotations

import argparse
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


def load_group_data(group_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(group_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df["source_file"] = path.name
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No .xlsx files found in {group_dir}")

    return pd.concat(parts, ignore_index=True)


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


def prepare_arrays(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    use_cols = feature_cols + TARGET_COLS
    numeric = df[use_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = numeric[TARGET_COLS].notna().all(axis=1)
    clean = df.loc[valid].reset_index(drop=True).copy()
    numeric = numeric.loc[valid].reset_index(drop=True)
    clean[use_cols] = numeric
    return clean, clean[feature_cols], clean[TARGET_COLS]


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
    df = load_group_data(group_dir)
    feature_cols = select_feature_cols(df)
    clean_df, x_df, y_df = prepare_arrays(df, feature_cols)

    if len(x_df) < n_splits:
        raise RuntimeError(f"Group {group_dir.name} needs at least {n_splits} rows, found {len(x_df)}")

    out_dir = results_dir / f"radius_{group_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    pred_parts: list[pd.DataFrame] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (train_idx, test_idx) in enumerate(kf.split(x_df), start=1):
        x_train_df = x_df.iloc[train_idx]
        x_test_df = x_df.iloc[test_idx]
        y_train = y_df.iloc[train_idx].to_numpy(dtype=float)
        y_test = y_df.iloc[test_idx].to_numpy(dtype=float)

        medians = x_train_df.median(axis=0, numeric_only=True).fillna(0.0)
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train_df.fillna(medians).to_numpy(dtype=float))
        x_test = scaler.transform(x_test_df.fillna(medians).to_numpy(dtype=float))

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
                    "train_rows": int(len(train_idx)),
                    "test_rows": int(len(test_idx)),
                    "rows_eval": int(valid.sum()),
                    "mae": float(mean_absolute_error(yt[valid], yp[valid])) if valid.any() else np.nan,
                    "rmse": rmse(yt[valid], yp[valid]) if valid.any() else np.nan,
                    "r2": float(r2_score(yt[valid], yp[valid])) if valid.sum() >= 2 else np.nan,
                }
            )

        pred_df = pd.DataFrame(
            {
                "radius": group_dir.name,
                "fold": f"fold{fold}",
                "source_file": clean_df.loc[test_idx, "source_file"].to_numpy(),
            }
        )
        for col in ["Re", "X [ m ]", "Y [ m ]", "Z [ m ]"]:
            if col in clean_df.columns:
                pred_df[col] = clean_df.loc[test_idx, col].to_numpy()
        for target_idx, target_col in enumerate(TARGET_COLS):
            pred_df[f"{target_col}_true"] = y_test[:, target_idx]
            pred_df[f"{target_col}_pred"] = pred[:, target_idx]
            pred_df[f"{target_col}_abs_error"] = np.abs(y_test[:, target_idx] - pred[:, target_idx])
        pred_parts.append(pred_df)

    fold_df = pd.DataFrame(fold_rows)
    summary_rows = []
    for target_col, target_df in fold_df.groupby("target", sort=True):
        valid = target_df[target_df["rows_eval"] > 0]
        weights = valid["rows_eval"] / valid["rows_eval"].sum()
        summary_rows.append(
            {
                "radius": group_dir.name,
                "target": target_col,
                "n_files": int(df["source_file"].nunique()),
                "n_rows": int(len(x_df)),
                "n_splits": n_splits,
                "feature_count": len(feature_cols),
                "features": " | ".join(feature_cols),
                "mae_w": float(np.average(valid["mae"], weights=weights)),
                "rmse_w": float(np.average(valid["rmse"], weights=weights)),
                "r2_w": float(np.average(valid["r2"], weights=weights)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    pred_df = pd.concat(pred_parts, ignore_index=True)

    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run row-level 5-fold CV for FD radius folders.")
    parser.add_argument("--batch", default="2026-05-22")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR / "fd_2026-05-22_radius_5fold")
    parser.add_argument("--groups", default="10,20,30")
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
        print(f"\n=== radius {group_dir.name}: row-level {args.n_splits}-fold CV ===")
        summary = run_group_cv(
            group_dir=group_dir,
            results_dir=args.results_dir,
            n_splits=args.n_splits,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
        print(summary[["radius", "target", "n_files", "n_rows", "feature_count", "mae_w", "rmse_w", "r2_w"]].to_string(index=False))
        summaries.append(summary)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.results_dir / "summary_all_radii.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved all results to {args.results_dir}")


if __name__ == "__main__":
    main()
