"""Leakage-resistant FD/FL interpolation and unseen-triangle evaluation.

The script reads the native FD and FL workbooks independently.  It evaluates:

* internal: hold out complete spatial positions in every source workbook;
* external: hold out complete congruent triangle families (including mirrors).

Radius is parsed from the workbook name and included as a model feature.  This
is required because the 2026-06-20 workbooks do not contain a radius column.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "radius", "re", "x", "z", "x_over_h",
    "z_over_h", "prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc",
    "line_dist_to_ab", "line_dist_to_ac",
]
BASE_REQUIRED = [
    "bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c",
    "perimeter", "area", "re", "x", "z", "prep_dist_to_ab",
    "prep_dist_to_ac", "prep_dist_to_bc", "line_dist_to_ab",
    "line_dist_to_ac",
]


def parse_radius(path: Path) -> int:
    match = re.search(r"-(10|20|30)(?=-(?:FL|FD)|$)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse radius 10/20/30 from {path.name}")
    return int(match.group(1))


def geometry_signature(df: pd.DataFrame) -> str:
    """Congruence signature; reflected copies intentionally share a group."""
    sides = sorted(float(pd.to_numeric(df[col], errors="coerce").median()) for col in ("bc", "ab", "ac"))
    return "sides_" + "_".join(f"{value:.5f}" for value in sides)


def stable_seed(text: str, seed: int) -> int:
    return (zlib.crc32(text.encode("utf-8")) + seed) % (2**32)


def load_task(data_root: Path, task: str) -> pd.DataFrame:
    folder = data_root / task
    parts: list[pd.DataFrame] = []
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        target = "u" if task == "FD" else "fl" if "fl" in df.columns else "flx" if "flx" in df.columns else None
        if target is None:
            raise ValueError(f"No FL target column in {path}")
        missing = [column for column in BASE_REQUIRED if column not in df.columns]
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")

        numeric_columns = [*BASE_REQUIRED, target]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=numeric_columns).reset_index(drop=True)
        df["radius"] = parse_radius(path)
        df["x_over_h"] = df["x"] / df["h"]
        df["z_over_h"] = df["z"] / df["h"]
        df["target"] = df[target]
        df["source_file"] = path.name
        df["geometry_group"] = geometry_signature(df)
        df["orientation"] = "reflected" if "反" in path.stem else "original"
        df["position_group"] = (
            path.name + "|" + df["x"].round(8).astype(str) + "|" + df["z"].round(8).astype(str)
        )
        parts.append(df[[*FEATURE_COLUMNS, "target", "source_file", "geometry_group", "orientation", "position_group"]])
    if not parts:
        raise RuntimeError(f"No {task} workbooks found in {folder}")
    return pd.concat(parts, ignore_index=True)


def internal_split(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_groups: set[str] = set()
    for source_file, part in df.groupby("source_file", sort=True):
        groups = part["position_group"].drop_duplicates().to_numpy()
        rng = np.random.default_rng(stable_seed(source_file, seed))
        rng.shuffle(groups)
        n_test = max(1, int(math.ceil(len(groups) * test_fraction)))
        test_groups.update(groups[:n_test].tolist())
    is_test = df["position_group"].isin(test_groups)
    return df.loc[~is_test].copy(), df.loc[is_test].copy()


def balanced_sample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    files = sorted(df["source_file"].unique())
    quota = max(1, max_rows // len(files))
    chunks = []
    used: set[int] = set()
    for source_file, part in df.groupby("source_file", sort=True):
        take = min(quota, len(part))
        sampled = part.sample(n=take, random_state=stable_seed(source_file, seed))
        chunks.append(sampled)
        used.update(sampled.index.tolist())
    sampled = pd.concat(chunks)
    remaining_n = max_rows - len(sampled)
    if remaining_n > 0:
        remaining = df.loc[~df.index.isin(used)]
        sampled = pd.concat([sampled, remaining.sample(n=min(remaining_n, len(remaining)), random_state=seed)])
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def sample_complete_positions(df: pd.DataFrame, max_rows_per_file: int | None, seed: int) -> pd.DataFrame:
    if max_rows_per_file is None:
        return df.copy()
    chunks = []
    for source_file, part in df.groupby("source_file", sort=True):
        if len(part) <= max_rows_per_file:
            chunks.append(part)
            continue
        counts = part.groupby("position_group", sort=False).size()
        # Shuffling a view can corrupt the position-to-count mapping in pandas.
        groups = counts.index.to_numpy(copy=True)
        rng = np.random.default_rng(stable_seed(source_file, seed))
        rng.shuffle(groups)
        selected: list[str] = []
        rows = 0
        for group in groups:
            selected.append(group)
            rows += int(counts[group])
            if rows >= max_rows_per_file:
                break
        chunks.append(part[part["position_group"].isin(selected)])
    return pd.concat(chunks, ignore_index=True)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    target_std = float(np.std(y_true))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 0 and target_std > 0 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson": pearson,
        "nrmse_std": rmse / target_std if target_std > 0 else float("nan"),
    }


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    backend: str,
    model_path: Path,
    device: str,
    n_estimators: int,
    chunk_size: int,
    seed: int,
) -> np.ndarray:
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
    x_train = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    x_test = preprocessor.transform(test[FEATURE_COLUMNS])
    y_train = train["target"].to_numpy(dtype=float)
    if backend == "extra-trees":
        model = ExtraTreesRegressor(n_estimators=max(200, n_estimators), random_state=seed, n_jobs=-1, min_samples_leaf=1)
    else:
        from tabpfn import TabPFNRegressor

        model = TabPFNRegressor(
            model_path=model_path,
            device=device,
            n_estimators=n_estimators,
            ignore_pretraining_limits=True,
            random_state=seed,
            show_progress_bar=True,
        )
    model.fit(x_train, y_train)
    chunks = [x_test[start : start + chunk_size] for start in range(0, len(x_test), chunk_size)]
    return np.concatenate([np.asarray(model.predict(chunk), dtype=float) for chunk in chunks])


def evaluate_split(
    *,
    task: str,
    protocol: str,
    fold: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if protocol == "internal":
        overlap = set(train["position_group"]).intersection(test["position_group"])
        leakage_rule = "no source-file-and-position group appears in both train and test"
    else:
        overlap = set(train["geometry_group"]).intersection(test["geometry_group"])
        leakage_rule = "no congruent triangle geometry appears in both train and test"
    if overlap:
        raise RuntimeError(f"Leakage detected in {task} {protocol} {fold}: {sorted(overlap)[:3]}")

    train_sample = balanced_sample(train, args.max_train_rows, stable_seed(f"{task}|{protocol}|{fold}|train", args.seed))
    test_sample = sample_complete_positions(test, args.max_test_rows_per_file, stable_seed(f"{task}|{protocol}|{fold}|test", args.seed))
    pred = fit_predict(
        train_sample,
        test_sample,
        args.backend,
        args.model_path,
        args.device,
        args.n_estimators,
        args.chunk_size,
        args.seed,
    )
    out = test_sample.copy()
    out["prediction"] = pred
    out["residual"] = out["target"] - out["prediction"]
    out["abs_error"] = out["residual"].abs()
    out["task"] = task
    out["protocol"] = protocol
    out["fold"] = fold
    metrics: dict[str, object] = metric_row(out["target"].to_numpy(), pred)
    metrics.update(
        {
            "task": task,
            "protocol": protocol,
            "fold": fold,
            "train_rows_available": int(len(train)),
            "train_rows_used": int(len(train_sample)),
            "test_rows_available": int(len(test)),
            "test_rows_used": int(len(test_sample)),
            "train_files": int(train["source_file"].nunique()),
            "test_files": int(test["source_file"].nunique()),
            "train_geometry_groups": int(train["geometry_group"].nunique()),
            "test_geometry_groups": int(test["geometry_group"].nunique()),
            "leakage_check_passed": True,
            "leakage_rule": leakage_rule,
        }
    )
    return out, metrics


def add_aggregate_metrics(predictions: pd.DataFrame, backend: str) -> pd.DataFrame:
    rows = []
    for (task, protocol), part in predictions.groupby(["task", "protocol"], sort=True):
        row = metric_row(part["target"].to_numpy(), part["prediction"].to_numpy())
        row.update(
            {
                "task": task,
                "protocol": protocol,
                "backend": backend,
                "rows": int(len(part)),
                "files": int(part["source_file"].nunique()),
                "geometry_groups": int(part["geometry_group"].nunique()),
                "pass_r2_0_80_and_pearson_0_90": bool(row["r2"] >= 0.80 and row["pearson"] >= 0.90),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def save_plots(predictions: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    for (task, protocol), part in predictions.groupby(["task", "protocol"], sort=True):
        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        ax.scatter(part["target"], part["prediction"], s=10, alpha=0.38, c="#1565C0", edgecolors="none")
        lo = float(min(part["target"].min(), part["prediction"].min()))
        hi = float(max(part["target"].max(), part["prediction"].max()))
        ax.plot([lo, hi], [lo, hi], color="#D32F2F", linewidth=1.8, label="理想预测")
        row = summary[(summary["task"] == task) & (summary["protocol"] == protocol)].iloc[0]
        title_protocol = "三角形内部位置预测" if protocol == "internal" else "未知三角形外推"
        ax.set_title(f"{task}：{title_protocol}")
        ax.set_xlabel("真实值")
        ax.set_ylabel("预测值")
        ax.text(0.03, 0.97, f"R² = {row['r2']:.3f}\nPearson = {row['pearson']:.3f}\n样本数 = {int(row['rows'])}", transform=ax.transAxes, va="top")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.2)
        fig.savefig(output_dir / f"{task.lower()}_{protocol}_true_vs_pred.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    fold_metrics = []
    inventory = []
    for task in args.tasks:
        data = load_task(args.data_root, task)
        inventory.append(
            {
                "task": task,
                "rows": int(len(data)),
                "files": int(data["source_file"].nunique()),
                "geometry_groups": int(data["geometry_group"].nunique()),
                "radii": ",".join(map(str, sorted(data["radius"].unique()))),
            }
        )
        if args.protocol in {"internal", "both"}:
            train, test = internal_split(data, args.internal_test_fraction, args.seed)
            pred, metrics = evaluate_split(task=task, protocol="internal", fold="spatial_holdout", train=train, test=test, args=args)
            all_predictions.append(pred)
            fold_metrics.append(metrics)
            print(f"[{task} internal] R2={metrics['r2']:.4f} Pearson={metrics['pearson']:.4f}", flush=True)

        if args.protocol in {"external", "both"}:
            for geometry_group in sorted(data["geometry_group"].unique()):
                is_test = data["geometry_group"] == geometry_group
                train, test = data.loc[~is_test].copy(), data.loc[is_test].copy()
                pred, metrics = evaluate_split(task=task, protocol="external", fold=geometry_group, train=train, test=test, args=args)
                all_predictions.append(pred)
                fold_metrics.append(metrics)
                print(f"[{task} external {geometry_group}] R2={metrics['r2']:.4f} Pearson={metrics['pearson']:.4f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    summary = add_aggregate_metrics(predictions, args.backend)
    file_rows = []
    for (task, protocol, fold, source_file), part in predictions.groupby(["task", "protocol", "fold", "source_file"], sort=True):
        row = metric_row(part["target"].to_numpy(), part["prediction"].to_numpy())
        row.update({"task": task, "protocol": protocol, "fold": fold, "source_file": source_file, "rows": len(part)})
        file_rows.append(row)

    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(args.output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(file_rows).to_csv(args.output_dir / "file_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inventory).to_csv(args.output_dir / "data_inventory.csv", index=False, encoding="utf-8-sig")
    save_plots(predictions, summary, args.output_dir)
    config = {
        "data_root": str(args.data_root),
        "backend": args.backend,
        "protocol": args.protocol,
        "tasks": args.tasks,
        "feature_columns": FEATURE_COLUMNS,
        "model_path": str(args.model_path),
        "device": args.device,
        "n_estimators": args.n_estimators,
        "max_train_rows": args.max_train_rows,
        "max_test_rows_per_file": args.max_test_rows_per_file,
        "internal_test_fraction": args.internal_test_fraction,
        "seed": args.seed,
        "external_group_rule": "sorted median triangle side lengths; mirrors stay together",
        "internal_group_rule": "source workbook + exact rounded (x,z); all Reynolds numbers stay together",
        "acceptance_rule": "aggregate R2 >= 0.80 and Pearson >= 0.90",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FD/FL internal interpolation and unseen-triangle extrapolation.")
    parser.add_argument("--data-root", type=Path, default=Path("data/rawdata/2026-06-20"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_fl_internal_external_20260722"))
    parser.add_argument("--tasks", nargs="+", choices=["FD", "FL"], default=["FD", "FL"])
    parser.add_argument("--protocol", choices=["internal", "external", "both"], default="both")
    parser.add_argument("--backend", choices=["extra-trees", "tabpfn"], default="tabpfn")
    parser.add_argument("--model-path", type=Path, default=Path("models/models/tabpfn-v2.5-regressor-v2.5_default.ckpt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=3000)
    parser.add_argument("--max-test-rows-per-file", type=int, default=300)
    parser.add_argument("--internal-test-fraction", type=float, default=0.20)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
