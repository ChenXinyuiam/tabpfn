"""Equal-budget benchmark figures and an auditable sample review queue."""

import html

import numpy as np
import pandas as pd

from .build_focus_boundary_features import draw_comparison
from .plotting import configure_fonts, plt
from .quality import quality_reasons

STYLE = """body{max-width:1150px;margin:24px auto;padding:0 16px;font:15px/1.6 Arial,sans-serif;color:#242424}
h1{font-size:25px}h2{font-size:18px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}
td:first-child,th:first-child{text-align:left}.table{overflow:auto}img{display:block;max-width:100%;height:auto}
details{border-bottom:1px solid #ccc;padding:12px 0}summary{cursor:pointer;overflow-wrap:anywhere}input{padding:8px;width:min(90%,480px)}
"""


def write_review(directory, predictions):
    points = pd.read_csv(directory / "features/smoothed_points.csv")
    groups = {key: group for key, group in points.groupby(["source_file", "R", "re"])}
    records, cards = [], []
    for key, group in predictions.groupby(["source_file", "R", "re"], sort=True):
        row = group.iloc[0].to_dict()
        errors = int(group["pred_midpoint"].ne(group["focus_label"]).sum())
        rule_conflict = bool(pd.notna(row["pred_rule_lt_100"]) and row["pred_rule_lt_100"] != row["focus_label"])
        reasons = quality_reasons(row)
        if errors:
            reasons.append("midpoint_oof_error")
        if rule_conflict:
            reasons.append("100um_rule_disagrees_with_label")
        if not reasons:
            continue
        record = {
            "source_file": key[0],
            "R": key[1],
            "re": key[2],
            "label": row["focus_label_name"],
            "error_seeds": errors,
            "total_seeds": len(group),
            "probability_min": group.proba_midpoint.min(),
            "probability_max": group.proba_midpoint.max(),
            "boundary_distance_abs": row["boundary_distance_abs"],
            "boundary_midpoint_x": row["boundary_midpoint_x"],
            "review_reasons": ";".join(reasons),
            "reviewed_label": "",
            "review_note": "",
        }
        records.append(record)
        image = f"review_figures/sample_{len(records):03d}.png"
        draw_comparison(
            groups[key], row, source_file=key[0], radius=key[1], re_value=key[2], out_path=directory / image
        )
        title = f"{key[0]} | R={key[1]:g}, Re={key[2]:g} | label={record['label']} | errors={errors}/{len(group)}"
        cards.append(
            f"<details><summary>{html.escape(title)}</summary><p>{html.escape(record['review_reasons'])}</p>"
            f'<img loading="lazy" src="{image}" alt="Raw and smoothed point distribution"></details>'
        )
    columns = [
        "source_file",
        "R",
        "re",
        "label",
        "error_seeds",
        "total_seeds",
        "probability_min",
        "probability_max",
        "boundary_distance_abs",
        "boundary_midpoint_x",
        "review_reasons",
        "reviewed_label",
        "review_note",
    ]
    review = pd.DataFrame(records, columns=columns)
    review.to_csv(directory / "review_queue.csv", index=False, encoding="utf-8-sig")
    (directory / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Sample review</title><style>{STYLE}</style><h1>Sample review</h1><p><a href="gallery.html">Evaluation</a></p>'
        '<label for="query">Filter samples</label><br><input id="query" type="search">'
        "<p>Flags are review requests, not corrected labels. Original labels remain unchanged.</p>"
        + "".join(cards)
        + '<script>document.getElementById("query").addEventListener("input",e=>{const q=e.target.value.toLowerCase();'
        'document.querySelectorAll("details").forEach(d=>d.hidden=!d.textContent.toLowerCase().includes(q));});</script></html>',
        encoding="utf-8",
    )
    return review


def write_report(directory, predictions, metrics, summary):
    configure_fonts()
    columns = ["accuracy", "balanced_accuracy", "recall", "specificity", "f1", "brier_score"]
    averaged = metrics.groupby("model", sort=False)[columns].mean().reset_index()
    averaged.to_csv(directory / "metrics_mean.csv", index=False, encoding="utf-8-sig")
    per_group = []
    for name in ("context", "width", "midpoint"):
        for geometry, group in predictions.groupby("geometry_group"):
            per_group.append(
                {
                    "model": name,
                    "geometry_group": geometry,
                    "accuracy": group[f"pred_{name}"].eq(group.focus_label).mean(),
                    "unique_samples": len(group) // predictions.seed.nunique(),
                }
            )
    by_group = pd.DataFrame(per_group)
    by_group.to_csv(directory / "metrics_by_geometry.csv", index=False, encoding="utf-8-sig")
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    geometries = sorted(predictions.geometry_group.unique())
    for offset, (name, color) in enumerate(
        zip(("context", "width", "midpoint"), ("#387ca6", "#b87932", "#319268"), strict=True)
    ):
        values = by_group[by_group.model.eq(name)].set_index("geometry_group").loc[geometries, "accuracy"]
        ax.bar(np.arange(len(geometries)) + (offset - 1) * 0.24, values, width=0.24, label=name, color=color)
    ax.set(
        xticks=np.arange(len(geometries)), xticklabels=geometries, ylim=(0, 1.05), ylabel="Mean accuracy across seeds"
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.savefig(figures / "geometry_accuracy.png", dpi=160)
    plt.close(fig)
    calibration = []
    fig, ax = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Reference")
    for name in ("context", "width", "midpoint"):
        p = predictions[f"proba_{name}"]
        bins = np.minimum((p.to_numpy() * 10).astype(int), 9)
        rows = []
        for bin_id in range(10):
            mask = bins == bin_id
            if mask.any():
                row = {
                    "model": name,
                    "bin": bin_id,
                    "prediction_count": int(mask.sum()),
                    "mean_probability": float(p[mask].mean()),
                    "fraction_f": float(predictions.loc[mask, "focus_label"].mean()),
                }
                calibration.append(row)
                rows.append(row)
        ax.plot([r["mean_probability"] for r in rows], [r["fraction_f"] for r in rows], marker="o", label=name)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted F probability", ylabel="Observed F fraction")
    ax.legend()
    fig.savefig(figures / "probability_reliability.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(calibration).to_csv(directory / "probability_reliability.csv", index=False, encoding="utf-8-sig")
    review = write_review(directory, predictions)
    effect = summary["midpoint_minus_width"]
    text = [
        f"{summary['data']['sample_count']} unique samples, {summary['data']['geometry_group_count']} held-out geometry groups, {predictions.seed.nunique()} paired seeds.",
        "All three feature sets use identical folds, seeds, estimator count and a fixed 0.5 threshold. No candidate or seed is selected from these results.",
        f"Midpoint minus width accuracy: {effect['delta']:+.2%}; paired geometry bootstrap 95% interval [{effect['ci95'][0]:+.2%}, {effect['ci95'][1]:+.2%}].",
        "This is a development comparison on previously examined data. It is not an independent external validation; a new geometry batch is still required.",
        "Repeated seeds are not independent samples. Probability reliability is descriptive OOF analysis, not a fitted calibration or a confidence guarantee.",
        f"{len(review)} samples enter the review queue. No labels were changed. Fixed recipes use the first predeclared seed, not the best seed.",
    ]
    (directory / "report.md").write_text(
        "# Controlled focus feature comparison\n\n"
        + "\n\n".join(text)
        + "\n\nSee metrics_mean.csv, metrics_by_seed.csv, review_queue.csv and review.html.\n",
        encoding="utf-8",
    )
    (directory / "gallery.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Controlled focus comparison</title><style>{STYLE}</style><h1>Controlled focus comparison</h1>"
        + "".join(f"<p>{html.escape(line)}</p>" for line in text)
        + '<p><a href="review.html">Sample review queue</a></p>'
        + '<div class="table">'
        + averaged.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
        + "</div>"
        + '<h2>Held-out geometry results</h2><img src="figures/geometry_accuracy.png" alt="Mean accuracy by geometry">'
        '<h2>Probability reliability</h2><img src="figures/probability_reliability.png" alt="OOF probability reliability">'
        '<h2>Individual seeds</h2><div class="table">'
        + metrics[["seed", "model", *columns]].to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}")
        + "</div></html>",
        encoding="utf-8",
    )
