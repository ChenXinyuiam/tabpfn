"""Held-out probability quality and geometry-integrity report."""

import html

import numpy as np
import pandas as pd

from .ablation_report import STYLE
from .plotting import configure_fonts, plt


def reliability_bins(data, probability_column, bins=10):
    probability = data[probability_column].to_numpy()
    assignment = np.minimum((probability * bins).astype(int), bins - 1)
    rows = []
    for bin_id in range(bins):
        mask = assignment == bin_id
        if mask.any():
            rows.append(
                {
                    "bin": bin_id,
                    "count": int(mask.sum()),
                    "mean_probability": probability[mask].mean(),
                    "fraction_f": data.loc[mask, "focus_label"].mean(),
                }
            )
    return pd.DataFrame(rows)


def write_report(directory, data, metrics, summary):
    configure_fonts()
    columns = ["accuracy", "balanced_accuracy", "recall", "specificity", "brier_score", "log_loss"]
    mean = metrics.groupby("model", sort=False)[columns].mean().reset_index()
    mean.to_csv(directory / "metrics_mean.csv", index=False, encoding="utf-8-sig")
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#888888")
    histogram, groups = [], []
    for name, color in (("raw", "#387ca6"), ("sigmoid", "#b87932")):
        bins = reliability_bins(data, f"proba_{name}")
        bins["model"] = name
        histogram.append(bins)
        axes[0].plot(bins.mean_probability, bins.fraction_f, marker="o", label=name, color=color)
        axes[1].hist(
            data[f"proba_{name}"], bins=np.linspace(0, 1, 11), histtype="step", linewidth=2, label=name, color=color
        )
        for group_name, group in data.groupby("geometry_group"):
            groups.append(
                {
                    "geometry_group": group_name,
                    "model": name,
                    "accuracy": group[f"pred_{name}"].eq(group.focus_label).mean(),
                    "brier_score": ((group[f"proba_{name}"] - group.focus_label) ** 2).mean(),
                }
            )
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted F probability", ylabel="Observed F fraction")
    axes[1].set(xlim=(0, 1), xlabel="Predicted F probability", ylabel="Predictions across all seeds")
    for ax in axes:
        ax.legend()
    fig.savefig(figures / "reliability.png", dpi=170)
    plt.close(fig)
    pd.concat(histogram).to_csv(directory / "reliability_bins.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(groups).to_csv(directory / "metrics_by_geometry.csv", index=False, encoding="utf-8-sig")
    unique = data.drop_duplicates(["source_file", "R", "re"])
    issues = unique.loc[~unique.geometry_consistent, ["source_file", "R", "re", "geometry_issues"]]
    issues.to_csv(directory / "geometry_issues.csv", index=False, encoding="utf-8-sig")
    error_rows = data.loc[data.pred_raw.ne(data.focus_label) | data.pred_sigmoid.ne(data.focus_label)].copy()
    error_rows.to_csv(directory / "errors.csv", index=False, encoding="utf-8-sig")
    high_confidence = []
    for name in ("raw", "sigmoid"):
        probability = data[f"proba_{name}"]
        wrong = data[f"pred_{name}"].ne(data.focus_label)
        high_confidence.append(
            {
                "model": name,
                "high_confidence_predictions": int(((probability >= 0.9) | (probability <= 0.1)).sum()),
                "high_confidence_errors": int((wrong & ((probability >= 0.9) | (probability <= 0.1))).sum()),
            }
        )
    pd.DataFrame(high_confidence).to_csv(directory / "high_confidence_errors.csv", index=False, encoding="utf-8-sig")
    effect = summary["paired_brier_effect"]
    text = [
        f"{len(unique)} unique samples, {data.geometry_group.nunique()} geometry groups, {data.seed.nunique()} predeclared seeds.",
        "Fixed geometry/Re + width features, four estimators by default, threshold 0.5. Inner grouped OOF predictions fit the sigmoid; outer test labels never enter calibration.",
        f"Sigmoid minus raw Brier score: {effect['brier_delta']:+.5f}; paired geometry bootstrap 95% interval [{effect['ci95'][0]:+.5f}, {effect['ci95'][1]:+.5f}]. Negative is better for this score.",
        "Brier and log loss measure overall probability quality, not calibration alone. Reliability bins are descriptive and sparse; repeated seeds are not independent samples.",
        f"{len(issues)} samples in {issues.source_file.nunique()} source files have inconsistent supplied geometry. Original values and labels are retained as requested; these issues limit scientific interpretation.",
        "This is a development comparison on previously inspected data, not new external validation. Neither calibration nor high probability guarantees a correct future decision. No recipe is promoted automatically.",
        "The saved sigmoid recipe fits its final calibrator on all-data grouped OOF probabilities for future use. Those fitting probabilities are not evaluated as calibrated test predictions.",
    ]
    (directory / "report.md").write_text(
        "# Grouped probability calibration\n\n"
        + "\n\n".join(text)
        + "\n\nReference: https://scikit-learn.org/1.8/modules/calibration.html\n",
        encoding="utf-8",
    )
    (directory / "gallery.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Grouped probability calibration</title><style>{STYLE}</style><h1>Grouped probability calibration</h1>"
        + "".join(f"<p>{html.escape(value)}</p>" for value in text)
        + '<div class="table">'
        + mean.to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}")
        + "</div>"
        + '<h2>Held-out probability reliability</h2><img src="figures/reliability.png" alt="Reliability and score histogram">'
        '<h2>Geometry issues</h2><div class="table">'
        + issues.drop_duplicates(["source_file", "geometry_issues"])[["source_file", "geometry_issues"]].to_html(
            index=False, border=0
        )
        + '</div><h2>Individual seeds</h2><div class="table">'
        + metrics[["seed", "model", *columns]].to_html(index=False, border=0, float_format=lambda v: f"{v:.5f}")
        + '</div><p><a href="https://scikit-learn.org/1.8/modules/calibration.html">Calibration methodology reference</a></p></html>',
        encoding="utf-8",
    )
