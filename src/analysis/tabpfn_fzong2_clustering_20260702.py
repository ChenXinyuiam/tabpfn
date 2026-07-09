"""Fzong2 clustering for the 2026-07-02 batch.

The raw batch has one workbook per chip and ten Reynolds numbers per workbook.
This script keeps raw Excel files unchanged and clusters the existing Fzong2
values for each source_file + re experiment separately. An optional TabPFN
auxiliary mode can add out-of-fold predictions, but it is not required for
ordinary Fzong2 clustering because Fzong2 is already present in the data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR


os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

TARGET_COL = "Fzong2"
RE_COL = "re"
SPATIAL_COLS = ["x", "z"]
FEATURE_COLS = [
    "re",
    "x",
    "z",
    "bc",
    "ab",
    "ac",
    "h",
    "angle_a",
    "angle_b",
    "angle_c",
    "perimeter",
    "area",
    "R",
    "prep_dist_to_ab",
    "prep_dist_to_ac",
    "prep_dist_to_bc",
    "line_dist_to_ab",
    "line_dist_to_ac",
]
FORCE_COMPONENT_COLS = ["fl", "FD2"]


@dataclass(frozen=True)
class FileMeta:
    source_file: str
    chip_id: str
    geometry: str
    radius_or_case: str
    orientation: str


def parse_file_meta(path: Path) -> FileMeta:
    stem = path.stem.replace("-Fzong", "")
    numbers = re.findall(r"\d+", stem)
    geometry = f"{numbers[0]}-{numbers[1]}" if len(numbers) >= 2 else stem
    radius_or_case = numbers[2] if len(numbers) >= 3 else ""
    orientation = "reflected" if "反" in stem else "original"
    return FileMeta(
        source_file=path.name,
        chip_id=stem,
        geometry=geometry,
        radius_or_case=radius_or_case,
        orientation=orientation,
    )


def resolve_device(device: str) -> str:
    if device != "prefer-cuda":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_batch(raw_dir: Path, max_files: int | None = None) -> pd.DataFrame:
    paths = [p for p in sorted(raw_dir.glob("*.xlsx")) if not p.name.startswith("~$")]
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found under {raw_dir}")

    parts: list[pd.DataFrame] = []
    for path in paths:
        meta = parse_file_meta(path)
        df = pd.read_excel(path, engine="openpyxl")
        missing = [col for col in [RE_COL, TARGET_COL, *SPATIAL_COLS] if col not in df.columns]
        if missing:
            raise RuntimeError(f"{path.name} is missing required columns: {missing}")

        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        df["source_file"] = meta.source_file
        df["chip_id"] = meta.chip_id
        df["geometry"] = meta.geometry
        df["radius_or_case"] = meta.radius_or_case
        df["orientation"] = meta.orientation
        df["row_in_source"] = np.arange(len(df), dtype=int)
        parts.append(df)

    batch = pd.concat(parts, ignore_index=True)
    for col in [RE_COL, TARGET_COL, *FEATURE_COLS, *FORCE_COMPONENT_COLS]:
        if col in batch.columns:
            batch[col] = pd.to_numeric(batch[col], errors="coerce")
    batch["experiment_id"] = batch["chip_id"].astype(str) + "__re" + batch[RE_COL].astype("Int64").astype(str)
    return batch


def fit_tabpfn_oof_for_file(
    df_file: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    model_path: Path,
    n_estimators: int,
    folds: int,
    device: str,
    random_state: int,
) -> np.ndarray:
    from tabpfn import TabPFNRegressor

    valid = df_file[target_col].notna().to_numpy()
    pred = np.full(len(df_file), np.nan, dtype=float)
    if valid.sum() < 12:
        return pred

    x = df_file.loc[valid, feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df_file.loc[valid, target_col].to_numpy(dtype=float)
    x_arr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(x)
    n_splits = min(folds, len(y))
    if n_splits < 2:
        return pred

    valid_indices = np.flatnonzero(valid)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x_arr), start=1):
        model = TabPFNRegressor(
            n_estimators=n_estimators,
            model_path=model_path,
            device=device,
            ignore_pretraining_limits=True,
            random_state=random_state + fold_idx,
            show_progress_bar=False,
        )
        model.fit(x_arr[train_idx], y[train_idx])
        pred[valid_indices[test_idx]] = np.asarray(model.predict(x_arr[test_idx]), dtype=float)
    return pred


def add_tabpfn_predictions(
    df: pd.DataFrame,
    *,
    model_path: Path,
    n_estimators: int,
    folds: int,
    device: str,
    random_state: int,
    include_force_components: bool,
) -> pd.DataFrame:
    if not model_path.exists():
        raise RuntimeError(f"TabPFN model checkpoint not found: {model_path}")

    requested_cols = [*FEATURE_COLS, *(FORCE_COMPONENT_COLS if include_force_components else [])]
    feature_cols = [col for col in requested_cols if col in df.columns and col != TARGET_COL]
    if not feature_cols:
        raise RuntimeError("No usable feature columns found for TabPFN.")
    print(f"[tabpfn] feature columns: {', '.join(feature_cols)}", flush=True)

    out = df.copy()
    out["tabpfn_pred_Fzong2"] = np.nan
    for source_file, df_file in out.groupby("source_file", sort=True):
        print(f"[tabpfn] fitting OOF predictions for {source_file} ({len(df_file)} rows)", flush=True)
        pred = fit_tabpfn_oof_for_file(
            df_file,
            feature_cols=feature_cols,
            target_col=TARGET_COL,
            model_path=model_path,
            n_estimators=n_estimators,
            folds=folds,
            device=device,
            random_state=random_state,
        )
        out.loc[df_file.index, "tabpfn_pred_Fzong2"] = pred
    out["tabpfn_residual_Fzong2"] = out[TARGET_COL] - out["tabpfn_pred_Fzong2"]
    return out


def clustering_columns(mode: str, df: pd.DataFrame) -> list[str]:
    if mode == "actual":
        return [TARGET_COL]
    if mode == "actual_pred":
        if df["tabpfn_pred_Fzong2"].notna().any():
            return [TARGET_COL, "tabpfn_pred_Fzong2"]
        return [TARGET_COL]
    if mode == "actual_pred_residual":
        cols = [TARGET_COL, "tabpfn_pred_Fzong2", "tabpfn_residual_Fzong2"]
        return [col for col in cols if col in df.columns and df[col].notna().any()]
    raise ValueError(f"Unsupported clustering mode: {mode}")


def choose_k_and_labels(
    values: pd.DataFrame,
    *,
    min_k: int,
    max_k: int,
    random_state: int,
) -> tuple[np.ndarray, int, float]:
    clean = values.replace([np.inf, -np.inf], np.nan)
    valid_mask = clean.notna().all(axis=1).to_numpy()
    labels = np.full(len(values), -1, dtype=int)
    if valid_mask.sum() < min_k:
        return labels, 0, np.nan

    x_arr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(clean.loc[valid_mask])
    unique_rows = np.unique(np.round(x_arr, decimals=12), axis=0)
    max_possible_k = min(max_k, len(x_arr) - 1, len(unique_rows))
    if max_possible_k < min_k:
        labels[valid_mask] = 0
        return labels, 1, np.nan

    best_score = -np.inf
    best_k = min_k
    best_labels: np.ndarray | None = None
    for k in range(min_k, max_possible_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        candidate = model.fit_predict(x_arr)
        if len(np.unique(candidate)) < 2:
            continue
        score = silhouette_score(x_arr, candidate)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = candidate

    if best_labels is None:
        labels[valid_mask] = 0
        return labels, 1, np.nan
    labels[valid_mask] = best_labels
    return labels, best_k, float(best_score)


def cluster_experiments(
    df: pd.DataFrame,
    *,
    mode: str,
    min_k: int,
    max_k: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["cluster"] = -1
    out["cluster_k"] = 0
    summary_rows: list[dict[str, object]] = []

    for experiment_id, group in out.groupby("experiment_id", sort=True):
        cols = clustering_columns(mode, group)
        labels, best_k, score = choose_k_and_labels(
            group[cols],
            min_k=min_k,
            max_k=max_k,
            random_state=random_state,
        )
        out.loc[group.index, "cluster"] = labels
        out.loc[group.index, "cluster_k"] = best_k

        valid_labels = labels[labels >= 0]
        counts = pd.Series(valid_labels).value_counts().sort_index().to_dict()
        summary_rows.append(
            {
                "experiment_id": experiment_id,
                "source_file": group["source_file"].iloc[0],
                "chip_id": group["chip_id"].iloc[0],
                "geometry": group["geometry"].iloc[0],
                "radius_or_case": group["radius_or_case"].iloc[0],
                "orientation": group["orientation"].iloc[0],
                "re": group[RE_COL].iloc[0],
                "rows": len(group),
                "cluster_space": ",".join(cols),
                "best_k": best_k,
                "silhouette": score,
                "cluster_counts": json.dumps(counts, ensure_ascii=False),
                "Fzong2_min": group[TARGET_COL].min(),
                "Fzong2_median": group[TARGET_COL].median(),
                "Fzong2_max": group[TARGET_COL].max(),
                "tabpfn_pred_median": group["tabpfn_pred_Fzong2"].median(),
                "tabpfn_residual_mae": group["tabpfn_residual_Fzong2"].abs().mean(),
            },
        )

    return out, pd.DataFrame(summary_rows)


def save_file_figures(df: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    for source_file, group in df.groupby("source_file", sort=True):
        re_values = sorted(group[RE_COL].dropna().unique())
        n_cols = 5
        n_rows = int(np.ceil(len(re_values) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.0, 3.0 * n_rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")

        for ax, re_value in zip(axes.ravel(), re_values):
            sub = group[group[RE_COL] == re_value]
            ax.axis("on")
            scatter = ax.scatter(
                sub["x"],
                sub["z"],
                c=sub["cluster"],
                s=14,
                cmap="tab10",
                alpha=0.9,
                linewidths=0,
            )
            ax.set_title(f"Re={int(re_value)}  k={int(sub['cluster_k'].max())}")
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_aspect("equal", adjustable="box")
        fig.suptitle(source_file, y=1.0)
        fig.tight_layout()
        safe_name = Path(source_file).stem.replace(" ", "_")
        fig.savefig(fig_dir / f"{safe_name}_clusters.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster existing Fzong2 values for data/rawdata/2026-07-02.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rawdata/2026-07-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/fzong2_tabpfn_clustering_20260702"))
    parser.add_argument("--model-path", type=Path, default=MODEL_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--device", default="prefer-cuda", help="prefer-cuda, cuda, cpu, or auto")
    parser.add_argument("--cluster-space", choices=["actual", "actual_pred", "actual_pred_residual"], default="actual")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--use-tabpfn",
        action="store_true",
        help="Optional: add out-of-fold TabPFN predictions before clustering. Not needed for basic Fzong2 clustering.",
    )
    parser.add_argument(
        "--include-force-components",
        action="store_true",
        help="Also use fl and FD2 to predict Fzong2. Off by default to avoid near-target leakage.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_batch(args.raw_dir, max_files=args.max_files)
    print(f"[load] rows={len(df)} files={df['source_file'].nunique()} experiments={df['experiment_id'].nunique()}", flush=True)

    device = resolve_device(args.device)
    if not args.use_tabpfn:
        df["tabpfn_pred_Fzong2"] = np.nan
        df["tabpfn_residual_Fzong2"] = np.nan
        cluster_space = "actual"
    else:
        print(f"[tabpfn] device={device} n_estimators={args.n_estimators} folds={args.folds}", flush=True)
        df = add_tabpfn_predictions(
            df,
            model_path=args.model_path,
            n_estimators=args.n_estimators,
            folds=args.folds,
            device=device,
            random_state=args.random_state,
            include_force_components=args.include_force_components,
        )
        cluster_space = args.cluster_space

    clustered, summary = cluster_experiments(
        df,
        mode=cluster_space,
        min_k=args.min_k,
        max_k=args.max_k,
        random_state=args.random_state,
    )

    assignments_cols = [
        "source_file",
        "chip_id",
        "geometry",
        "radius_or_case",
        "orientation",
        "row_in_source",
        "experiment_id",
        RE_COL,
        "x",
        "z",
        TARGET_COL,
        "tabpfn_pred_Fzong2",
        "tabpfn_residual_Fzong2",
        "cluster",
        "cluster_k",
    ]
    assignments_cols = [col for col in assignments_cols if col in clustered.columns]
    clustered[assignments_cols].to_csv(args.out_dir / "cluster_assignments.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.out_dir / "experiment_summary.csv", index=False, encoding="utf-8-sig")
    save_file_figures(clustered, args.out_dir / "figures")

    run_config = {
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "rows": int(len(clustered)),
        "files": int(clustered["source_file"].nunique()),
        "experiments": int(clustered["experiment_id"].nunique()),
        "device": device,
        "n_estimators": args.n_estimators,
        "folds": args.folds,
        "cluster_space": cluster_space,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "use_tabpfn": args.use_tabpfn,
        "include_force_components": args.include_force_components,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
