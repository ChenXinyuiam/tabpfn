"""Experiment-level clustering for the 2026-07-02 Fzong2 batch.

Each experiment is one source_file + re group. The script summarizes every
experiment into distribution and peak-location features, then clusters the 200
experiments with no more than three labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.tabpfn_fzong2_clustering_20260702 import RE_COL, TARGET_COL, load_batch


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den not in {0, 0.0} and np.isfinite(den) else np.nan


def experiment_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for experiment_id, group in df.groupby("experiment_id", sort=True):
        g = group.copy()
        y = pd.to_numeric(g[TARGET_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = y.notna()
        yv = y[valid].to_numpy(dtype=float)
        if len(yv) == 0:
            continue

        x = pd.to_numeric(g.loc[valid, "x"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(g.loc[valid, "z"], errors="coerce").to_numpy(dtype=float)
        h = pd.to_numeric(g.loc[valid, "h"], errors="coerce").to_numpy(dtype=float) if "h" in g.columns else np.nan
        h_med = float(np.nanmedian(h)) if np.ndim(h) else np.nan
        scale = h_med if np.isfinite(h_med) and abs(h_med) > 1e-12 else 1.0

        min_idx = int(np.nanargmin(yv))
        max_idx = int(np.nanargmax(yv))
        q = np.quantile(yv, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        abs_y = np.abs(yv)
        row = {
            "experiment_id": experiment_id,
            "source_file": g["source_file"].iloc[0],
            "chip_id": g["chip_id"].iloc[0],
            "geometry": g["geometry"].iloc[0],
            "radius_or_case": g["radius_or_case"].iloc[0],
            "orientation": g["orientation"].iloc[0],
            "re": int(g[RE_COL].iloc[0]),
            "rows": int(len(yv)),
            "Fzong2_mean": float(np.mean(yv)),
            "Fzong2_std": float(np.std(yv, ddof=0)),
            "Fzong2_min": float(np.min(yv)),
            "Fzong2_q01": float(q[0]),
            "Fzong2_q05": float(q[1]),
            "Fzong2_q10": float(q[2]),
            "Fzong2_q25": float(q[3]),
            "Fzong2_median": float(q[4]),
            "Fzong2_q75": float(q[5]),
            "Fzong2_q90": float(q[6]),
            "Fzong2_q95": float(q[7]),
            "Fzong2_q99": float(q[8]),
            "Fzong2_max": float(np.max(yv)),
            "Fzong2_range": float(np.max(yv) - np.min(yv)),
            "Fzong2_iqr": float(q[5] - q[3]),
            "Fzong2_abs_mean": float(np.mean(abs_y)),
            "Fzong2_abs_q95": float(np.quantile(abs_y, 0.95)),
            "Fzong2_abs_max": float(np.max(abs_y)),
            "Fzong2_negative_ratio": float(np.mean(yv < 0)),
            "Fzong2_positive_ratio": float(np.mean(yv > 0)),
            "Fzong2_left_tail_gap": float(q[4] - q[1]),
            "Fzong2_right_tail_gap": float(q[7] - q[4]),
            "min_x_norm": safe_ratio(x[min_idx], scale),
            "min_z_norm": safe_ratio(z[min_idx], scale),
            "max_x_norm": safe_ratio(x[max_idx], scale),
            "max_z_norm": safe_ratio(z[max_idx], scale),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def choose_experiment_k(x: np.ndarray, max_k: int, random_state: int) -> tuple[np.ndarray, int, float]:
    max_k = min(max_k, len(x) - 1)
    if max_k < 2:
        return np.zeros(len(x), dtype=int), 1, np.nan

    best_labels: np.ndarray | None = None
    best_k = 2
    best_score = -np.inf
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit_predict(x)
        score = silhouette_score(x, labels)
        if score > best_score:
            best_score = float(score)
            best_k = k
            best_labels = labels
    if best_labels is None:
        return np.zeros(len(x), dtype=int), 1, np.nan
    return best_labels, best_k, best_score


def plot_outputs(features: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    x_scaled = make_pipeline(StandardScaler()).fit_transform(features[feature_cols])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x_scaled)
    features["pca1"] = coords[:, 0]
    features["pca2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    scatter = ax.scatter(
        features["pca1"],
        features["pca2"],
        c=features["experiment_cluster"],
        cmap="tab10",
        s=42,
        alpha=0.9,
    )
    ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("Experiment-level Fzong2 clusters")
    fig.colorbar(scatter, ax=ax, label="experiment_cluster")
    fig.tight_layout()
    fig.savefig(fig_dir / "experiment_clusters_pca.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    counts = features.groupby(["re", "experiment_cluster"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_xlabel("Re")
    ax.set_ylabel("Experiment count")
    ax.set_title("Experiment clusters by Reynolds number")
    fig.tight_layout()
    fig.savefig(fig_dir / "experiment_clusters_by_re.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster the 200 source_file + re experiments into <=3 labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rawdata/2026-07-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/fzong2_experiment_level_clustering_20260702"))
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-re", action="store_true", help="Include Re as a clustering feature.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_batch(args.raw_dir)
    features = experiment_features(raw)
    if len(features) == 0:
        raise RuntimeError("No experiment features were generated.")

    feature_cols = [
        col
        for col in features.columns
        if col.startswith("Fzong2_") or col in {"min_x_norm", "min_z_norm", "max_x_norm", "max_z_norm"}
    ]
    if args.include_re:
        feature_cols.append("re")

    x = make_pipeline(StandardScaler()).fit_transform(features[feature_cols])
    labels, chosen_k, silhouette = choose_experiment_k(x, args.max_k, args.random_state)
    features["experiment_cluster"] = labels
    features["experiment_cluster_k"] = chosen_k

    cluster_summary = (
        features.groupby("experiment_cluster")
        .agg(
            experiments=("experiment_id", "size"),
            re_min=("re", "min"),
            re_median=("re", "median"),
            re_max=("re", "max"),
            Fzong2_min_median=("Fzong2_min", "median"),
            Fzong2_median_median=("Fzong2_median", "median"),
            Fzong2_max_median=("Fzong2_max", "median"),
            Fzong2_abs_q95_median=("Fzong2_abs_q95", "median"),
            Fzong2_negative_ratio_median=("Fzong2_negative_ratio", "median"),
        )
        .reset_index()
    )

    label_cols = [
        "experiment_id",
        "source_file",
        "chip_id",
        "geometry",
        "radius_or_case",
        "orientation",
        "re",
        "rows",
        "experiment_cluster",
        "experiment_cluster_k",
        "Fzong2_min",
        "Fzong2_median",
        "Fzong2_max",
        "Fzong2_abs_q95",
        "Fzong2_negative_ratio",
        "min_x_norm",
        "min_z_norm",
        "max_x_norm",
        "max_z_norm",
    ]
    features[label_cols].to_csv(args.out_dir / "experiment_cluster_labels.csv", index=False, encoding="utf-8-sig")
    features.to_csv(args.out_dir / "experiment_features.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(args.out_dir / "experiment_cluster_summary.csv", index=False, encoding="utf-8-sig")
    plot_outputs(features, feature_cols, args.out_dir)

    run_config = {
        "raw_dir": str(args.raw_dir),
        "out_dir": str(args.out_dir),
        "experiments": int(len(features)),
        "max_k": args.max_k,
        "chosen_k": int(chosen_k),
        "silhouette": float(silhouette),
        "feature_cols": feature_cols,
        "include_re": args.include_re,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Experiment-level Fzong2 clustering",
        "",
        f"- Experiments: {len(features)}",
        f"- Chosen cluster_k: {chosen_k}",
        f"- Silhouette: {silhouette:.6f}",
        f"- Max allowed k: {args.max_k}",
        f"- Include Re as feature: {args.include_re}",
        "",
        "## Cluster summary",
        cluster_summary.to_csv(index=False),
        "",
        "## Outputs",
        "- experiment_cluster_labels.csv",
        "- experiment_features.csv",
        "- experiment_cluster_summary.csv",
        "- figures/experiment_clusters_pca.png",
        "- figures/experiment_clusters_by_re.png",
    ]
    (args.out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] experiments={len(features)} chosen_k={chosen_k} silhouette={silhouette:.6f}")
    print(f"[done] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
