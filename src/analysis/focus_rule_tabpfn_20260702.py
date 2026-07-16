"""Physics-rule focus labeling and TabPFN classification for 2026-07-02.

One sample is one source_file + re experiment. Point-level rows are aggregated
into section-level features before modeling. This follows the proposed scheme:
derive DeltaX transition features, geometry/fluid features, and red/blue point
distribution features, then train a supervised classifier from rule labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.analysis.tabpfn_fzong2_clustering_20260702 import RE_COL, TARGET_COL, load_batch


os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


FEATURE_COLUMNS = [
    "re",
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
    "delta_x_mean",
    "delta_x_std",
    "delta_x_min",
    "delta_x_max",
    "delta_x_mean_norm",
    "delta_x_std_norm",
    "delta_x_valid_layers",
    "delta_x_layer_coverage",
    "x_distribution_span",
    "z_distribution_span",
    "red_count",
    "blue_count",
    "red_blue_ratio",
    "red_fraction",
    "blue_fraction",
    "fzong2_abs_mean",
    "fzong2_abs_q95",
    "fzong2_negative_q05",
    "fzong2_positive_q95",
]


def resolve_device(device: str) -> str:
    if device != "prefer-cuda":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def nearest_cross_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    idx = np.searchsorted(b, a)
    distances: list[np.ndarray] = []
    valid_left = idx > 0
    if valid_left.any():
        distances.append(np.abs(a[valid_left] - b[idx[valid_left] - 1]))
    valid_right = idx < len(b)
    if valid_right.any():
        distances.append(np.abs(a[valid_right] - b[idx[valid_right]]))
    if not distances:
        return np.nan
    return float(np.min(np.concatenate(distances)))


def delta_x_features(group: pd.DataFrame, *, n_layers: int) -> dict[str, float]:
    x = pd.to_numeric(group["x"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(group["z"], errors="coerce").to_numpy(dtype=float)
    force = pd.to_numeric(group[TARGET_COL], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(z) & np.isfinite(force)
    x = x[valid]
    z = z[valid]
    force = force[valid]
    if len(x) == 0 or np.nanmax(z) == np.nanmin(z):
        return {
            "delta_x_mean": np.nan,
            "delta_x_std": np.nan,
            "delta_x_min": np.nan,
            "delta_x_max": np.nan,
            "delta_x_valid_layers": 0,
            "delta_x_layer_coverage": 0.0,
        }

    edges = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), n_layers + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    layer_ids = np.digitize(z, edges[1:-1], right=False)
    dx_values: list[float] = []
    for layer in range(n_layers):
        layer_mask = layer_ids == layer
        if not layer_mask.any():
            continue
        red_x = x[layer_mask & (force > 0)]
        blue_x = x[layer_mask & (force < 0)]
        dx = nearest_cross_distance(red_x, blue_x)
        if np.isfinite(dx):
            dx_values.append(dx)

    if not dx_values:
        return {
            "delta_x_mean": np.nan,
            "delta_x_std": np.nan,
            "delta_x_min": np.nan,
            "delta_x_max": np.nan,
            "delta_x_valid_layers": 0,
            "delta_x_layer_coverage": 0.0,
        }
    arr = np.asarray(dx_values, dtype=float)
    return {
        "delta_x_mean": float(np.mean(arr)),
        "delta_x_std": float(np.std(arr, ddof=0)),
        "delta_x_min": float(np.min(arr)),
        "delta_x_max": float(np.max(arr)),
        "delta_x_valid_layers": int(len(arr)),
        "delta_x_layer_coverage": float(len(arr) / n_layers),
    }


def build_section_features(df: pd.DataFrame, *, n_layers: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_global = ["bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c", "perimeter", "area", "R"]
    for experiment_id, group in df.groupby("experiment_id", sort=True):
        f = pd.to_numeric(group[TARGET_COL], errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = pd.to_numeric(group["x"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        z = pd.to_numeric(group["z"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        red_count = int((f > 0).sum())
        blue_count = int((f < 0).sum())
        rows_count = int(f.notna().sum())
        x_span = float(x.max() - x.min()) if x.notna().any() else np.nan
        z_span = float(z.max() - z.min()) if z.notna().any() else np.nan
        dx = delta_x_features(group, n_layers=n_layers)

        row: dict[str, object] = {
            "experiment_id": experiment_id,
            "source_file": group["source_file"].iloc[0],
            "chip_id": group["chip_id"].iloc[0],
            "geometry": group["geometry"].iloc[0],
            "radius_or_case": group["radius_or_case"].iloc[0],
            "orientation": group["orientation"].iloc[0],
            "re": int(group[RE_COL].iloc[0]),
            "rows": rows_count,
            "red_count": red_count,
            "blue_count": blue_count,
            "red_blue_ratio": float(red_count / blue_count) if blue_count else np.nan,
            "red_fraction": float(red_count / rows_count) if rows_count else np.nan,
            "blue_fraction": float(blue_count / rows_count) if rows_count else np.nan,
            "x_distribution_span": x_span,
            "z_distribution_span": z_span,
            "fzong2_abs_mean": float(f.abs().mean()),
            "fzong2_abs_q95": float(f.abs().quantile(0.95)),
            "fzong2_negative_q05": float(f.quantile(0.05)),
            "fzong2_positive_q95": float(f.quantile(0.95)),
            **dx,
        }
        for col in numeric_global:
            row[col] = float(pd.to_numeric(group[col], errors="coerce").median()) if col in group.columns else np.nan
        row["delta_x_mean_norm"] = (
            float(row["delta_x_mean"]) / x_span if np.isfinite(float(row["delta_x_mean"])) and x_span else np.nan
        )
        row["delta_x_std_norm"] = (
            float(row["delta_x_std"]) / x_span if np.isfinite(float(row["delta_x_std"])) and x_span else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_data_audit(raw: pd.DataFrame, features: pd.DataFrame, raw_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    raw_by_file = (
        raw.groupby("source_file", sort=True)
        .agg(
            raw_rows=("source_file", "size"),
            raw_re_count=(RE_COL, "nunique"),
            raw_experiments=("experiment_id", "nunique"),
        )
        .reset_index()
    )
    feature_by_file = (
        features.groupby("source_file", sort=True)
        .agg(
            feature_rows_sum=("rows", "sum"),
            feature_re_count=("re", "nunique"),
            feature_experiments=("experiment_id", "nunique"),
        )
        .reset_index()
    )
    by_file = raw_by_file.merge(feature_by_file, on="source_file", how="outer")
    by_file["row_count_match"] = by_file["raw_rows"].fillna(-1).astype(int) == by_file["feature_rows_sum"].fillna(-1).astype(int)
    by_file["experiment_count_match"] = by_file["raw_experiments"].fillna(-1).astype(int) == by_file[
        "feature_experiments"
    ].fillna(-1).astype(int)
    by_file["all_match"] = by_file["row_count_match"] & by_file["experiment_count_match"]

    audit = {
        "raw_dir": str(raw_dir),
        "raw_dir_resolved": str(raw_dir.resolve()),
        "raw_files": int(raw["source_file"].nunique()),
        "raw_rows": int(len(raw)),
        "raw_experiments": int(raw["experiment_id"].nunique()),
        "feature_experiments": int(len(features)),
        "feature_rows_sum": int(features["rows"].sum()),
        "files_all_match": bool(by_file["all_match"].all()),
        "missing_source_files_in_features": sorted(set(raw["source_file"]) - set(features["source_file"])),
        "extra_source_files_in_features": sorted(set(features["source_file"]) - set(raw["source_file"])),
    }
    return audit, by_file


def apply_rule_labels(
    features: pd.DataFrame,
    *,
    delta_std_norm_threshold: float | None,
    delta_quantile: float,
    ratio_min: float,
    ratio_max: float,
    min_valid_layers: int,
) -> tuple[pd.DataFrame, float]:
    out = features.copy()
    valid_delta = out["delta_x_std_norm"].replace([np.inf, -np.inf], np.nan).dropna()
    if delta_std_norm_threshold is None:
        threshold = float(valid_delta.quantile(delta_quantile))
    else:
        threshold = float(delta_std_norm_threshold)

    ratio = out["red_blue_ratio"].replace([np.inf, -np.inf], np.nan)
    label = (
        out["delta_x_std_norm"].le(threshold)
        & out["delta_x_valid_layers"].ge(min_valid_layers)
        & out["red_count"].gt(0)
        & out["blue_count"].gt(0)
        & ratio.ge(ratio_min)
        & ratio.le(ratio_max)
    )
    out["focus_label"] = label.astype(int)
    out["focus_rule"] = np.where(out["focus_label"] == 1, "focused", "not_focused")
    return out, threshold


def fit_predict_tabpfn_or_fallback(
    data: pd.DataFrame,
    *,
    model_path: Path,
    device: str,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    x_df = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = data["focus_label"].to_numpy(dtype=int)
    groups = data["source_file"].to_numpy()
    preprocessor = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    x = preprocessor.fit_transform(x_df)

    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.full(len(data), -1, dtype=int)
    proba = np.full(len(data), np.nan, dtype=float)
    model_name = "TabPFNClassifier"
    error_message = ""

    try:
        from tabpfn import TabPFNClassifier

        if not model_path.exists():
            raise FileNotFoundError(f"TabPFN classifier checkpoint not found: {model_path}")

        for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
            if len(np.unique(y[train_idx])) < 2:
                raise RuntimeError(f"Fold {fold_idx} training data has one class only.")
            clf = TabPFNClassifier(
                n_estimators=1,
                model_path=model_path,
                device=device,
                ignore_pretraining_limits=True,
                random_state=random_state + fold_idx,
                show_progress_bar=False,
            )
            clf.fit(x[train_idx], y[train_idx])
            pred[test_idx] = clf.predict(x[test_idx])
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(x[test_idx])
                proba[test_idx] = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    except Exception as exc:
        model_name = "RandomForestClassifier_fallback"
        error_message = str(exc)
        pred[:] = -1
        proba[:] = np.nan
        for train_idx, test_idx in splitter.split(x, y, groups):
            clf = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
            )
            clf.fit(x[train_idx], y[train_idx])
            pred[test_idx] = clf.predict(x[test_idx])
            proba[test_idx] = clf.predict_proba(x[test_idx])[:, 1]

    result = data.copy()
    result["pred_focus_label"] = pred
    result["pred_focus_probability"] = proba
    metrics = {
        "model": model_name,
        "fallback_error": error_message,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    return result, metrics


def save_visuals(result: pd.DataFrame, metrics: dict[str, object], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    counts = result["focus_label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(["not_focused", "focused"], [counts.get(0, 0), counts.get(1, 0)], color=["#6b7280", "#f97316"])
    ax.set_ylabel("Experiment count")
    ax.set_title("Rule labels")
    fig.tight_layout()
    fig.savefig(fig_dir / "01_rule_label_counts.png", dpi=240)
    plt.close(fig)

    feature_view = result[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    x = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(feature_view)
    coords = PCA(n_components=2, random_state=42).fit_transform(x)
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for label, color, name in [(0, "#4b5563", "not focused"), (1, "#f97316", "focused")]:
        mask = result["focus_label"] == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=48, color=color, edgecolors="black", linewidths=0.25, label=name)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Section features by rule label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "02_feature_pca_by_rule_label.png", dpi=240)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for label, color, name in [(0, "#4b5563", "not focused"), (1, "#f97316", "focused")]:
        subset = result[result["focus_label"] == label]
        ax.scatter(subset["delta_x_std_norm"], subset["red_blue_ratio"], s=44, color=color, alpha=0.88, label=name)
    ax.set_xlabel("DeltaX_std / X_span")
    ax.set_ylabel("Red_Count / Blue_Count")
    ax.set_title("Rule space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "03_rule_space_delta_vs_ratio.png", dpi=240)
    plt.close(fig)

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    ax.set_title("Grouped CV confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "04_grouped_cv_confusion_matrix.png", dpi=240)
    plt.close(fig)

    label_matrix = result.pivot_table(index="chip_id", columns="re", values="focus_label", aggfunc="first")
    label_matrix = label_matrix.reindex(sorted(label_matrix.index), axis=0).reindex(sorted(label_matrix.columns), axis=1)
    fig, ax = plt.subplots(figsize=(9.2, 7.0))
    im = ax.imshow(label_matrix.to_numpy(dtype=float), cmap="Oranges", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(label_matrix.columns)), [str(int(c)) for c in label_matrix.columns])
    ax.set_yticks(range(len(label_matrix.index)), label_matrix.index, fontsize=7)
    ax.set_xlabel("Re")
    ax.set_ylabel("Chip / file")
    ax.set_title("Rule focus labels by file and Re")
    for i in range(label_matrix.shape[0]):
        for j in range(label_matrix.shape[1]):
            val = label_matrix.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, str(int(val)), ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="focus_label")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_focus_label_matrix_file_by_re.png", dpi=240)
    plt.close(fig)


def write_html(out_dir: Path) -> None:
    figs = [
        ("01_rule_label_counts.png", "Rule label counts"),
        ("02_feature_pca_by_rule_label.png", "PCA view of section features"),
        ("03_rule_space_delta_vs_ratio.png", "Rule space: DeltaX_std_norm vs red-blue ratio"),
        ("04_grouped_cv_confusion_matrix.png", "Grouped cross-validation confusion matrix"),
        ("05_focus_label_matrix_file_by_re.png", "Focus-label matrix by file and Re"),
    ]
    html = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Focus rule TabPFN experiment</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;background:#fafafa;color:#222}section{background:#fff;border:1px solid #ddd;margin:0 0 24px;padding:16px}img{max-width:100%;height:auto;display:block}.meta{line-height:1.7;background:#fff;border:1px solid #ddd;padding:14px 16px;margin-bottom:24px}code{background:#f1f1f1;padding:2px 5px;border-radius:4px}</style>",
        "</head>",
        "<body>",
        "<h1>Focus rule + TabPFN classification experiment</h1>",
        '<div class="meta">',
        "<p>One sample is one <code>source_file + Re</code> experiment.</p>",
        "<p>Point rows are aggregated into section-level features before modeling.</p>",
        "<p>Rule label: low normalized DeltaX_std plus balanced positive/negative Fzong2 points.</p>",
        "</div>",
    ]
    for name, title in figs:
        html.extend(["<section>", f"<h2>{title}</h2>", f'<img src="figures/{name}" alt="{title}">', "</section>"])
    html.extend(["</body>", "</html>"])
    (out_dir / "focus_rule_tabpfn_gallery.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Physics-rule labels plus TabPFN classification for 2026-07-02.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rawdata/2026-07-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/focus_rule_tabpfn_20260702"))
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--delta-quantile", type=float, default=0.35)
    parser.add_argument("--delta-std-norm-threshold", type=float, default=None)
    parser.add_argument("--ratio-min", type=float, default=0.5)
    parser.add_argument("--ratio-max", type=float, default=2.0)
    parser.add_argument("--min-valid-layers", type=int, default=2)
    parser.add_argument("--model-path", type=Path, default=Path("models/models/tabpfn-v2-classifier.ckpt"))
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = load_batch(args.raw_dir)
    features = build_section_features(raw, n_layers=args.n_layers)
    data_audit, data_audit_by_file = build_data_audit(raw, features, args.raw_dir)
    labeled, threshold = apply_rule_labels(
        features,
        delta_std_norm_threshold=args.delta_std_norm_threshold,
        delta_quantile=args.delta_quantile,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        min_valid_layers=args.min_valid_layers,
    )
    device = resolve_device(args.device)
    predictions, metrics = fit_predict_tabpfn_or_fallback(
        labeled,
        model_path=args.model_path,
        device=device,
        random_state=args.random_state,
    )

    features.to_csv(args.out_dir / "section_features.csv", index=False, encoding="utf-8-sig")
    labeled.to_csv(args.out_dir / "section_features_with_rule_labels.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.out_dir / "tabpfn_grouped_cv_predictions.csv", index=False, encoding="utf-8-sig")
    data_audit_by_file.to_csv(args.out_dir / "data_audit_by_file.csv", index=False, encoding="utf-8-sig")
    (args.out_dir / "data_audit.json").write_text(json.dumps(data_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    label_summary = (
        predictions.groupby("focus_label")
        .agg(
            experiments=("experiment_id", "size"),
            delta_x_std_norm_median=("delta_x_std_norm", "median"),
            red_blue_ratio_median=("red_blue_ratio", "median"),
            fzong2_abs_q95_median=("fzong2_abs_q95", "median"),
            re_median=("re", "median"),
        )
        .reset_index()
    )
    label_summary.to_csv(args.out_dir / "rule_label_summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "n_layers": args.n_layers,
        "delta_quantile": args.delta_quantile,
        "delta_std_norm_threshold": threshold,
        "ratio_min": args.ratio_min,
        "ratio_max": args.ratio_max,
        "min_valid_layers": args.min_valid_layers,
        "model_path": str(args.model_path),
        "device": device,
        "data_audit": data_audit,
        "metrics": metrics,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    save_visuals(predictions, metrics, args.out_dir)
    write_html(args.out_dir)

    report = [
        "# Focus rule + TabPFN experiment",
        "",
        f"- Experiments: {len(predictions)}",
        f"- DeltaX layers: {args.n_layers}",
        f"- DeltaX_std_norm threshold: {threshold:.6f}",
        f"- Ratio bounds: {args.ratio_min} to {args.ratio_max}",
        f"- Model: {metrics['model']}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- F1: {metrics['f1']:.6f}",
        "",
        "## Label summary",
        label_summary.to_csv(index=False),
        "",
        "## Outputs",
        "- section_features.csv",
        "- section_features_with_rule_labels.csv",
        "- tabpfn_grouped_cv_predictions.csv",
        "- rule_label_summary.csv",
        "- data_audit.json",
        "- data_audit_by_file.csv",
        "- metrics.json",
        "- focus_rule_tabpfn_gallery.html",
    ]
    (args.out_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[done] wrote {args.out_dir}")
    print(f"[done] model={metrics['model']} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
    print(f"[done] threshold delta_x_std_norm={threshold:.6f}")


if __name__ == "__main__":
    main()
