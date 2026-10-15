"""Baseline focus evaluation figures and HTML report."""

import html
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .plotting import plt
from .settings import BOUNDARY_FEATURE, MAIN_MODEL_NAME


def save_figures(data: pd.DataFrame, metrics_table: pd.DataFrame, out_dir: Path) -> None:
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(metrics_table))
    width = 0.24
    for offset, metric, color in [
        (-width, "accuracy", "#4c78a8"),
        (0.0, "balanced_accuracy", "#59a14f"),
        (width, "f1", "#e15759"),
    ]:
        ax.bar(x + offset, metrics_table[metric], width=width, label=metric.replace("_", " "), color=color)
    ax.set_xticks(x, metrics_table["model"], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Grouped out-of-fold model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "01_model_metrics.png", dpi=220)
    plt.close(fig)

    model_names = metrics_table["model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.5), constrained_layout=True)
    for ax, model_name in zip(axes.flat, model_names, strict=True):
        cm = confusion_matrix(data["focus_label"], data[f"pred_{model_name}"], labels=[0, 1])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(1, int(cm.max())))
        ax.set_xticks([0, 1], ["UF", "F"])
        ax.set_yticks([0, 1], ["UF", "F"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(model_name)
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(int(cm[row, column])), ha="center", va="center")
    fig.savefig(figure_dir / "02_confusion_matrices.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for label_value, label_name, color in [(0, "UF", "#4c78a8"), (1, "F", "#e15759")]:
        subset = data[data["focus_label"] == label_value]
        ax.scatter(
            subset[BOUNDARY_FEATURE],
            subset["re"],
            color=color,
            alpha=0.72,
            s=42,
            label=label_name,
        )
    ax.axvline(100, color="#222222", linestyle="--", linewidth=1.2, label="100 micrometers")
    ax.set_xlabel("Smoothed boundary distance (micrometers)")
    ax.set_ylabel("Re")
    ax.set_title("Learning labels and the fixed boundary rule")
    ax.legend()
    ax.grid(True, linewidth=0.35, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "03_boundary_feature_vs_re.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    rng = np.random.default_rng(42)
    for label_value, label_name, color in [(0, "UF", "#4c78a8"), (1, "F", "#e15759")]:
        subset = data[data["focus_label"] == label_value]
        jitter = rng.normal(0, 0.035, size=len(subset))
        ax.scatter(
            np.full(len(subset), label_value) + jitter,
            subset[f"proba_{MAIN_MODEL_NAME}"],
            color=color,
            alpha=0.72,
            s=38,
            label=label_name,
        )
    ax.axhline(0.5, color="#222222", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1], ["UF", "F"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Out-of-fold probability of focus")
    ax.set_title("Main TabPFN out-of-fold probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "04_main_model_probabilities.png", dpi=220)
    plt.close(fig)


def write_gallery(out_dir: Path, metrics_table: pd.DataFrame, audit: dict[str, object]) -> None:
    figures = [
        ("01_model_metrics.png", "Grouped out-of-fold model comparison"),
        ("02_confusion_matrices.png", "Confusion matrices"),
        ("03_boundary_feature_vs_re.png", "Boundary feature and learning labels"),
        ("04_main_model_probabilities.png", "Main model out-of-fold probabilities"),
    ]
    metric_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{float(row['accuracy']):.3f}</td>"
        f"<td>{float(row['balanced_accuracy']):.3f}</td>"
        f"<td>{float(row['precision']):.3f}</td>"
        f"<td>{float(row['recall']):.3f}</td>"
        f"<td>{float(row['f1']):.3f}</td>"
        f"<td>{float(row['mcc']):.3f}</td>"
        "</tr>"
        for row in metrics_table.to_dict("records")
    )
    sections = "".join(
        f"<section><h2>{html.escape(title)}</h2><img src='figures/{name}' alt='{html.escape(title)}'></section>"
        for name, title in figures
    )
    (out_dir / "gallery.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Focus TabPFN classification</title>
  <style>
    body {{ margin: 26px; font-family: Arial, sans-serif; color: #202124; background: #fafafa; }}
    .meta, section {{ background: white; border: 1px solid #dadce0; padding: 16px; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>Focus TabPFN classification</h1>
  <div class="meta">
    <p>Samples: {audit["sample_count"]} | Geometry groups: {audit["geometry_group_count"]} | Labels: F=focused, UF=unfocused</p>
    <p>Evaluation: grouped out-of-fold predictions. All radii, orientations, and Re values of one geometry stay in one fold.</p>
  </div>
  <section>
    <h2>Metrics</h2>
    <table>
      <thead><tr><th>Model</th><th>Accuracy</th><th>Balanced accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>MCC</th></tr></thead>
      <tbody>{metric_rows}</tbody>
    </table>
  </section>
  {sections}
</body>
</html>
""",
        encoding="utf-8",
    )
