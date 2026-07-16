"""TabPFN classification from manually transcribed image labels for 2026-07-02.

One sample is one source_file + re experiment. The uploaded label image marks
each experiment as WF, F, or UF. WF and F are merged into the positive class,
and UF is the negative class.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.focus_rule_tabpfn_20260702 import FEATURE_COLUMNS, build_data_audit, build_section_features, resolve_device
from src.analysis.tabpfn_fzong2_clustering_20260702 import RE_COL, load_batch


os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


RE_VALUES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

IMAGE_LABEL_ROWS = [
    ("690-10", "690-340-10-Fzong.xlsx", ["WF", "WF", "WF", "WF", "WF", "WF", "WF", "F", "UF", "UF"]),
    ("720-10", "720-500-10-Fzong.xlsx", ["WF", "WF", "WF", "WF", "WF", "WF", "WF", "F", "UF", "UF"]),
    ("750-400-10", "750-400-10-Fzong.xlsx", ["WF", "WF", "WF", "WF", "WF", "WF", "WF", "UF", "UF", "UF"]),
    ("750-400-20", "750-400-20-Fzong.xlsx", ["UF", "UF", "UF", "UF", "UF", "UF", "UF", "UF", "WF", "WF"]),
    ("750-400-30", "750-400-30-Fzong.xlsx", ["UF", "WF", "UF", "UF", "UF", "UF", "UF", "UF", "UF", "UF"]),
    ("750-400-rev-10", "750-400反-10-Fzong.xlsx", ["F", "WF", "WF", "F", "F", "F", "F", "UF", "UF", "UF"]),
    ("750-400-rev-20", "750-400反-20-Fzong.xlsx", ["WF", "WF", "WF", "WF", "F", "F", "UF", "UF", "UF", "UF"]),
    ("750-400-rev-30", "750-400反-30-Fzong.xlsx", ["UF", "UF", "UF", "UF", "UF", "UF", "WF", "WF", "F", "F"]),
    ("750-320-10", "750-320-10-Fzong.xlsx", ["WF", "WF", "WF", "F", "F", "F", "F", "F", "UF", "UF"]),
    ("750-320-20", "750-320-20-Fzong.xlsx", ["WF", "WF", "F", "F", "UF", "UF", "UF", "UF", "UF", "UF"]),
    ("750-320-30", "750-320-30-Fzong.xlsx", ["UF", "F", "WF", "WF", "WF", "UF", "UF", "UF", "UF", "UF"]),
    ("750-530-10", "750-530-10-Fzong.xlsx", ["UF", "WF", "WF", "WF", "UF", "UF", "UF", "UF", "UF", "UF"]),
    ("750-530-20", "750-530-20-Fzong.xlsx", ["WF", "WF", "F", "F", "F", "F", "UF", "UF", "UF", "UF"]),
    ("750-530-30", "750-530-30-Fzong.xlsx", ["F", "WF", "F", "WF", "WF", "WF", "WF", "WF", "WF", "WF"]),
    ("750-530-rev-10", "750-530-反-10-Fzong.xlsx", ["WF", "WF", "F", "WF", "WF", "UF", "UF", "UF", "UF", "UF"]),
    ("750-530-rev-20", "750-530-反-20-Fzong.xlsx", ["UF", "UF", "UF", "UF", "UF", "UF", "UF", "WF", "WF", "UF"]),
    ("750-530-rev-30", "750-530-反-30-Fzong.xlsx", ["UF", "UF", "UF", "UF", "UF", "UF", "UF", "UF", "UF", "UF"]),
    ("770-10", "770-330-10-Fzong.xlsx", ["WF", "WF", "WF", "WF", "WF", "WF", "WF", "F", "F", "UF"]),
    ("800-10", "800-333-10-Fzong.xlsx", ["WF", "WF", "WF", "F", "F", "F", "F", "UF", "UF", "UF"]),
]


def configure_chinese_font() -> str:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            prop = fm.FontProperties(fname=str(path))
            name = prop.get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return ""


def build_image_labels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chip_label, source_file, labels in IMAGE_LABEL_ROWS:
        if len(labels) != len(RE_VALUES):
            raise ValueError(f"{chip_label} has {len(labels)} labels; expected {len(RE_VALUES)}")
        for re_value, raw_label in zip(RE_VALUES, labels, strict=True):
            rows.append(
                {
                    "chip_label": chip_label,
                    "source_file": source_file,
                    "re": re_value,
                    "image_label": raw_label,
                    "binary_label": 0 if raw_label == "UF" else 1,
                    "binary_label_name": "WF_or_F" if raw_label != "UF" else "UF",
                }
            )
    return pd.DataFrame(rows)


def fit_tabpfn_grouped_cv(
    labeled: pd.DataFrame,
    *,
    model_path: Path,
    device: str,
    random_state: int,
    n_splits: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    from tabpfn import TabPFNClassifier

    x = labeled[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = labeled["binary_label"].to_numpy(dtype=int)
    groups = labeled["source_file"].to_numpy()
    folds = GroupKFold(n_splits=min(n_splits, labeled["source_file"].nunique()))

    pred = np.full(len(labeled), -1, dtype=int)
    proba = np.full(len(labeled), np.nan, dtype=float)
    fold_ids = np.full(len(labeled), -1, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(folds.split(x, y, groups), start=1):
        clf = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            TabPFNClassifier(
                n_estimators=1,
                model_path=model_path,
                device=device,
                random_state=random_state + fold_idx,
            ),
        )
        clf.fit(x[train_idx], y[train_idx])
        pred[test_idx] = clf.predict(x[test_idx])
        probs = clf.predict_proba(x[test_idx])
        class_order = list(clf.named_steps["tabpfnclassifier"].classes_)
        positive_idx = class_order.index(1)
        proba[test_idx] = probs[:, positive_idx]
        fold_ids[test_idx] = fold_idx

    result = labeled.copy()
    result["cv_fold"] = fold_ids
    result["pred_binary_label"] = pred
    result["pred_label_name"] = np.where(pred == 1, "WF_or_F", "UF")
    result["pred_focus_probability"] = proba
    result["correct"] = result["binary_label"].to_numpy(dtype=int) == pred

    metrics = {
        "model": "TabPFNClassifier",
        "validation": "GroupKFold_by_source_file",
        "n_splits": int(min(n_splits, labeled["source_file"].nunique())),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix_labels": ["UF", "WF_or_F"],
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    return result, metrics


def fit_tabpfn_full_predictions(
    labeled: pd.DataFrame,
    all_features: pd.DataFrame,
    *,
    model_path: Path,
    device: str,
    random_state: int,
) -> pd.DataFrame:
    from tabpfn import TabPFNClassifier

    train_x = labeled[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    train_y = labeled["binary_label"].to_numpy(dtype=int)
    all_x = all_features[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        TabPFNClassifier(
            n_estimators=1,
            model_path=model_path,
            device=device,
            random_state=random_state,
        ),
    )
    clf.fit(train_x, train_y)
    pred = clf.predict(all_x)
    probs = clf.predict_proba(all_x)
    class_order = list(clf.named_steps["tabpfnclassifier"].classes_)
    positive_idx = class_order.index(1)

    result = all_features.copy()
    result["full_pred_binary_label"] = pred
    result["full_pred_label_name"] = np.where(pred == 1, "WF_or_F", "UF")
    result["full_pred_focus_probability"] = probs[:, positive_idx]
    return result


def labels_audit(raw: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, object]:
    raw_files = set(raw["source_file"])
    label_files = set(labels["source_file"])
    feature_keys = set(zip(features["source_file"], features["re"], strict=False))
    label_keys = set(zip(labels["source_file"], labels["re"], strict=False))
    return {
        "raw_files": int(raw["source_file"].nunique()),
        "raw_rows": int(len(raw)),
        "raw_experiments": int(raw["experiment_id"].nunique()),
        "feature_experiments": int(len(features)),
        "label_files": int(labels["source_file"].nunique()),
        "label_experiments": int(len(labels)),
        "labeled_feature_experiments": int(len(feature_keys & label_keys)),
        "missing_label_source_files": sorted(raw_files - label_files),
        "extra_label_source_files": sorted(label_files - raw_files),
        "missing_label_experiments": sorted([f"{a}__re{b}" for a, b in feature_keys - label_keys]),
        "extra_label_experiments": sorted([f"{a}__re{b}" for a, b in label_keys - feature_keys]),
    }


def plot_label_heatmaps(cv: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    chips = cv[["chip_label", "source_file"]].drop_duplicates().sort_values("source_file")["chip_label"].tolist()
    true_matrix = cv.pivot(index="chip_label", columns="re", values="binary_label").reindex(index=chips, columns=RE_VALUES)
    pred_matrix = cv.pivot(index="chip_label", columns="re", values="pred_binary_label").reindex(index=chips, columns=RE_VALUES)
    diff_matrix = (true_matrix != pred_matrix).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(14, 8), constrained_layout=True)
    items = [
        (true_matrix, "人工标签", "Blues"),
        (pred_matrix, "TabPFN交叉验证预测", "Blues"),
        (diff_matrix, "错误位置", "Reds"),
    ]
    for ax, (matrix, title, cmap) in zip(axes, items, strict=True):
        ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(RE_VALUES)), RE_VALUES)
        ax.set_yticks(range(len(chips)), chips)
        ax.set_xlabel("雷诺数 Re")
        for i, chip in enumerate(chips):
            for j, re_value in enumerate(RE_VALUES):
                value = matrix.loc[chip, re_value]
                text = "WF/F" if value == 1 else "UF"
                if title == "错误位置":
                    text = "错" if value == 1 else ""
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
    fig.savefig(fig_dir / "label_prediction_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_probability_curves(full: pd.DataFrame, labels: pd.DataFrame, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if {"chip_label", "image_label", "binary_label"}.issubset(full.columns):
        merged = full.copy()
    else:
        merged = full.merge(
            labels[["source_file", "re", "chip_label", "image_label", "binary_label"]],
            on=["source_file", "re"],
            how="left",
        )
    files = sorted(full["source_file"].unique())
    ncols = 4
    nrows = int(np.ceil(len(files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.0 * nrows), sharex=True, sharey=True)
    axes_flat = np.asarray(axes).ravel()
    for ax, source_file in zip(axes_flat, files, strict=False):
        sub = merged[merged["source_file"] == source_file].sort_values("re")
        title = sub["chip_label"].dropna().iloc[0] if sub["chip_label"].notna().any() else source_file.replace("-Fzong.xlsx", "")
        ax.plot(sub["re"], sub["full_pred_focus_probability"], color="#1f77b4", marker="o", linewidth=1.8)
        labeled = sub[sub["binary_label"].notna()]
        colors = np.where(labeled["binary_label"].astype(float).to_numpy() == 1, "#f28e2b", "#4e79a7")
        ax.scatter(labeled["re"], labeled["binary_label"], c=colors, s=38, edgecolor="black", linewidth=0.4, zorder=3)
        ax.axhline(0.5, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.08, 1.08)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(files) :]:
        ax.axis("off")
    fig.suptitle("各芯片随Re变化的WF/F概率曲线", fontsize=15)
    fig.supxlabel("雷诺数 Re")
    fig.supylabel("WF/F概率")
    fig.savefig(fig_dir / "chip_re_probability_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict[str, object], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.8, 4.1), constrained_layout=True)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["预测UF", "预测WF/F"])
    ax.set_yticks([0, 1], ["实际UF", "实际WF/F"])
    ax.set_title("混淆矩阵")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def write_html(out_dir: Path, metrics: dict[str, object], audit: dict[str, object]) -> None:
    html = [
        "<!doctype html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>TabPFN人工标签分类结果</title>",
        "<style>",
        "body{font-family:'Microsoft YaHei','Noto Sans CJK SC',Arial,sans-serif;margin:28px;background:#f7f7f7;color:#222}",
        "section{background:#fff;border:1px solid #ddd;margin:0 0 24px;padding:16px;border-radius:6px}",
        "img{max-width:100%;height:auto;display:block}",
        ".meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}",
        ".item{background:#f1f5fb;border:1px solid #d8e2ef;padding:10px;border-radius:5px}",
        "code{background:#eee;padding:2px 5px;border-radius:4px}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>TabPFN人工标签分类结果</h1>",
        '<section class="meta">',
        f'<div class="item">模型：<code>{metrics["model"]}</code></div>',
        f'<div class="item">验证：按芯片分组 {metrics["n_splits"]} 折</div>',
        f'<div class="item">准确率：{metrics["accuracy"]:.4f}</div>',
        f'<div class="item">F1：{metrics["f1"]:.4f}</div>',
        f'<div class="item">有标签实验：{audit["label_experiments"]}</div>',
        f'<div class="item">缺标签芯片：{", ".join(audit["missing_label_source_files"]) or "无"}</div>',
        "</section>",
        "<section><h2>人工标签与交叉验证预测</h2><img src='figures/label_prediction_heatmaps.png' alt='人工标签与预测热力图'></section>",
        "<section><h2>各芯片随Re变化的WF/F概率</h2><img src='figures/chip_re_probability_curves.png' alt='概率曲线'></section>",
        "<section><h2>混淆矩阵</h2><img src='figures/confusion_matrix.png' alt='混淆矩阵'></section>",
        "</body>",
        "</html>",
    ]
    (out_dir / "tabpfn_image_label_classification_gallery.html").write_text("\n".join(html), encoding="utf-8")


def write_evaluation_tables(cv: pd.DataFrame, full: pd.DataFrame, out_dir: Path) -> None:
    def metric_row(df: pd.DataFrame) -> dict[str, object]:
        y = df["binary_label"].astype(int)
        pred = df["pred_binary_label"].astype(int)
        return {
            "samples": int(len(df)),
            "positive_true": int(y.sum()),
            "negative_true": int((y == 0).sum()),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "mean_pred_focus_probability": float(df["pred_focus_probability"].mean()),
        }

    by_re = [{"re": re_value, **metric_row(group)} for re_value, group in cv.groupby("re", sort=True)]
    by_chip = [
        {"chip_label": chip_label, "source_file": source_file, **metric_row(group)}
        for (chip_label, source_file), group in cv.groupby(["chip_label", "source_file"], sort=True)
    ]
    label_balance = cv.groupby(["re", "image_label"]).size().unstack(fill_value=0).reset_index()
    errors = cv.loc[
        ~cv["correct"],
        ["chip_label", "source_file", "re", "image_label", "binary_label_name", "pred_label_name", "pred_focus_probability", "cv_fold"],
    ].copy()
    unlabeled = full[full["binary_label"].isna()].copy()

    pd.DataFrame(by_re).to_csv(out_dir / "metrics_by_re.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by_chip).to_csv(out_dir / "metrics_by_chip.csv", index=False, encoding="utf-8-sig")
    label_balance.to_csv(out_dir / "label_balance_by_re.csv", index=False, encoding="utf-8-sig")
    errors.to_csv(out_dir / "cv_error_cases_compact.csv", index=False, encoding="utf-8-sig")
    unlabeled.to_csv(out_dir / "unlabeled_830_full_fit_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "worst_re_by_accuracy": pd.DataFrame(by_re).sort_values(["accuracy", "f1"]).head(3).to_dict(orient="records"),
        "best_re_by_accuracy": pd.DataFrame(by_re)
        .sort_values(["accuracy", "f1"], ascending=False)
        .head(3)
        .to_dict(orient="records"),
        "worst_chip_by_accuracy": pd.DataFrame(by_chip).sort_values(["accuracy", "f1"]).head(5).to_dict(orient="records"),
        "unlabeled_predictions": unlabeled[
            ["source_file", "re", "full_pred_label_name", "full_pred_focus_probability"]
        ].to_dict(orient="records"),
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TabPFN classification from uploaded image labels.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rawdata/2026-07-02"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tabpfn_image_label_classification_20260702"))
    parser.add_argument("--model-path", type=Path, default=Path("models/models/tabpfn-v2-classifier.ckpt"))
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    font_name = configure_chinese_font()
    raw = load_batch(args.raw_dir)
    features = build_section_features(raw, n_layers=args.n_layers)
    data_audit, data_audit_by_file = build_data_audit(raw, features, args.raw_dir)
    labels = build_image_labels()
    audit = labels_audit(raw, features, labels)

    labeled = features.merge(labels, on=["source_file", "re"], how="inner", validate="one_to_one")
    if len(labeled) == 0:
        raise RuntimeError("No feature rows matched the image labels.")

    device = resolve_device(args.device)
    cv_predictions, metrics = fit_tabpfn_grouped_cv(
        labeled,
        model_path=args.model_path,
        device=device,
        random_state=args.random_state,
        n_splits=args.n_splits,
    )
    full_predictions = fit_tabpfn_full_predictions(
        labeled,
        features,
        model_path=args.model_path,
        device=device,
        random_state=args.random_state,
    )
    full_predictions = full_predictions.merge(
        labels[["source_file", "re", "chip_label", "image_label", "binary_label", "binary_label_name"]],
        on=["source_file", "re"],
        how="left",
    )

    labels.to_csv(args.out_dir / "image_labels.csv", index=False, encoding="utf-8-sig")
    features.to_csv(args.out_dir / "experiment_features.csv", index=False, encoding="utf-8-sig")
    data_audit_by_file.to_csv(args.out_dir / "data_audit_by_file.csv", index=False, encoding="utf-8-sig")
    cv_predictions.to_csv(args.out_dir / "tabpfn_grouped_cv_predictions.csv", index=False, encoding="utf-8-sig")
    full_predictions.to_csv(args.out_dir / "tabpfn_full_fit_predictions.csv", index=False, encoding="utf-8-sig")

    error_cases = cv_predictions.loc[~cv_predictions["correct"]].copy()
    error_cases.to_csv(args.out_dir / "cv_error_cases.csv", index=False, encoding="utf-8-sig")

    combined_audit = {
        **data_audit,
        **audit,
        "feature_columns": FEATURE_COLUMNS,
        "font_name": font_name,
        "raw_dir": str(args.raw_dir),
        "model_path": str(args.model_path),
        "device": device,
        "note": "The uploaded image visibly contains labels for 19 source files. The unlabeled 830-341-10 file is predicted only by the full-fit model.",
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps({"metrics": metrics, "audit": combined_audit}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_label_heatmaps(cv_predictions, args.out_dir)
    plot_probability_curves(full_predictions, labels, args.out_dir)
    plot_confusion_matrix(metrics, args.out_dir)
    write_html(args.out_dir, metrics, combined_audit)
    write_evaluation_tables(cv_predictions, full_predictions, args.out_dir)

    report = [
        "# TabPFN image-label classification",
        "",
        f"- Raw dir: {args.raw_dir}",
        f"- Raw experiments: {audit['raw_experiments']}",
        f"- Label experiments: {audit['label_experiments']}",
        f"- Labeled feature experiments: {audit['labeled_feature_experiments']}",
        f"- Missing label source files: {', '.join(audit['missing_label_source_files']) or 'none'}",
        f"- Model: {metrics['model']}",
        f"- Device: {device}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        f"- F1: {metrics['f1']:.6f}",
        "",
        "## Outputs",
        "- image_labels.csv",
        "- experiment_features.csv",
        "- tabpfn_grouped_cv_predictions.csv",
        "- tabpfn_full_fit_predictions.csv",
        "- cv_error_cases.csv",
        "- metrics_by_re.csv",
        "- metrics_by_chip.csv",
        "- evaluation_summary.json",
        "- unlabeled_830_full_fit_predictions.csv",
        "- metrics.json",
        "- tabpfn_image_label_classification_gallery.html",
    ]
    (args.out_dir / "analysis_report.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({"metrics": metrics, "audit": combined_audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
