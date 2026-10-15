"""Scientific figures and report for the nested focus benchmark."""

import html

import numpy as np
from sklearn.metrics import confusion_matrix

from .plotting import configure_fonts, plt


def write_report(out_dir, data, metrics, per_group, summary):
    configure_fonts()
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for ax, name, title in zip(
        axes,
        ["baseline", "midpoint", "nested"],
        ["Original baseline", "Fixed midpoint recipe", "Nested selection"],
        strict=True,
    ):
        cm = confusion_matrix(data["focus_label"], data[f"pred_{name}"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=140)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["UF", "F"],
            yticklabels=["UF", "F"],
            xlabel="Predicted",
            ylabel="Actual",
            title=title,
        )
        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    color="white" if cm[row, col] > 70 else "black",
                    fontsize=14,
                )
    fig.savefig(figures / "confusion_matrices.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(per_group))
    ax.bar(x - 0.24, per_group["baseline_accuracy"], width=0.24, color="#387ca6", label="Baseline")
    ax.bar(x, per_group["midpoint_accuracy"], width=0.24, color="#b87932", label="Fixed midpoint")
    ax.bar(x + 0.24, per_group["nested_accuracy"], width=0.24, color="#319268", label="Nested selection")
    ax.set(xticks=x, xticklabels=per_group["geometry_group"], ylim=(0, 1.05), ylabel="Accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.savefig(figures / "geometry_accuracy.png", dpi=170)
    plt.close(fig)

    table_lines = [
        "| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        table_lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['roc_auc']:.4f} |"
        )
    effect = summary["paired_uncertainty"]
    low, high = effect["accuracy_delta_ci95"]
    midpoint_effect = summary["midpoint_paired_uncertainty"]
    mid_low, mid_high = midpoint_effect["accuracy_delta_ci95"]
    paragraphs = [
        f"{len(data)} samples; {data['geometry_group'].nunique()} geometry groups. F=1, UF=0.",
        "The nested row evaluates the full selection procedure. Each outer test geometry is excluded from feature/threshold selection and all fitting.",
        "Candidate and threshold selection uses three inner grouped folds, maximizing accuracy with balanced accuracy as tie-breaker.",
        "Other candidate rows use fixed 0.5 thresholds and are diagnostic comparisons, not a post-test winner selection.",
        f"Nested minus baseline accuracy: {effect['accuracy_delta']:+.2%}. Paired geometry bootstrap 95% interval: [{low:+.2%}, {high:+.2%}].",
        f"Fixed midpoint minus baseline accuracy: {midpoint_effect['accuracy_delta']:+.2%}. Paired geometry bootstrap 95% interval: [{mid_low:+.2%}, {mid_high:+.2%}].",
        "Only eight geometry groups are available. This interval is descriptive; a new independent geometry batch is still needed to establish generalization.",
        "selected_recipe.json is selected on all available grouped OOF data for future prediction; its all-data selection score is not a test result.",
        "fixed_baseline_recipe.json and fixed_midpoint_recipe.json preserve the fixed recipes for prospective comparison. No model is promoted automatically.",
    ]
    (out_dir / "report.md").write_text(
        "# Focus classification optimization\n\n"
        + "\n\n".join(paragraphs[:3])
        + "\n\n"
        + "\n".join(table_lines)
        + "\n\n"
        + "\n\n".join(paragraphs[3:])
        + "\n",
        encoding="utf-8",
    )
    table = metrics[["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1"]].to_html(
        index=False,
        float_format=lambda v: f"{v:.4f}",
        border=0,
    )
    body = "".join(f"<p>{html.escape(text)}</p>" for text in paragraphs)
    (out_dir / "gallery.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Focus classification optimization</title><style>body{margin:24px auto;padding:0 16px;max-width:1100px;"
        "font:15px/1.55 Arial,sans-serif;color:#242424}h1{font-size:25px}table{border-collapse:collapse;width:100%}"
        "td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}"
        "img{display:block;max-width:100%;height:auto;margin:24px 0}.table{overflow:auto}</style>"
        "<h1>Focus classification optimization</h1>" + body + '<div class="table">' + table + "</div>"
        '<img src="figures/confusion_matrices.png" alt="Baseline, fixed midpoint, and nested confusion matrices">'
        '<img src="figures/geometry_accuracy.png" alt="Accuracy by held-out geometry"></html>',
        encoding="utf-8",
    )
