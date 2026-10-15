"""Rebuild a rotation benchmark report from saved outer predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression.evaluation import metric_row


def write_report(out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prediction = pd.read_csv(out / "predictions.csv")
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    names = prediction.groupby("geometry_group").source_file.first().map(lambda name: "-".join(name.split("-")[:2]))
    rows = []
    for method, data in prediction.groupby("method"):
        for angle in (None, 0, 90):
            part = data if angle is None else data[data.rotation_deg == angle]
            geometry_scores = [
                metric_row(g.target.to_numpy(), g.prediction.to_numpy())["r2"]
                for _, g in part.groupby("geometry_group")
            ]
            rows.append(
                {
                    "method": method,
                    "scope": "all_24_angles" if angle is None else f"angle_{angle:03d}",
                    "geometry_mean_r2": float(np.mean(geometry_scores)),
                    "geometry_worst_r2": float(np.min(geometry_scores)),
                    "rows": len(part),
                    **metric_row(part.target.to_numpy(), part.prediction.to_numpy()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "reference_summary.csv", index=False)
    angle_metrics = pd.read_csv(out / "angle_metrics.csv")
    geometry = pd.read_csv(out / "geometry_metrics.csv")
    columns = []
    for scope, data in [
        ("All", geometry),
        ("0 deg", angle_metrics[angle_metrics.rotation_deg == 0]),
        ("90 deg", angle_metrics[angle_metrics.rotation_deg == 90]),
    ]:
        pivot = data.pivot(index="geometry_group", columns="method", values="r2")
        pivot.columns = [scope + " / " + ("baseline" if c == "direct_baseline" else "selected") for c in pivot.columns]
        columns.append(pivot)
    table = pd.concat(columns, axis=1)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.imshow(table.to_numpy(), cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(table)), [names[g] for g in table.index])
    ax.set_title("Unseen-geometry R2 on provider rotation data")
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j, i, f"{table.iloc[i, j]:.3f}", ha="center", va="center", color="black")
    fig.savefig(out / "geometry_comparison.png", dpi=160)
    plt.close(fig)
    table.index = [names[g] for g in table.index]
    table.to_csv(out / "geometry_reference_comparison.csv")
    selections = json.loads((out / "selections.json").read_text(encoding="utf-8"))
    text = (
        "# Rotation-Aware TabPFN FD-u Evaluation\n\n"
        f"Dataset: `{config['snapshot']['root']}`. "
        f"{config['snapshot']['geometry_groups']} independent geometries, "
        f"{config['snapshot']['files']} augmented workbooks.\n\n"
        "The two angular components are defined by the supplied data, not confirmed physical units.\n"
        "All 24 rotation copies of a geometry remain in one outer fold. "
        "Inner selection weights each geometry and the 0/90 degree R2 equally.\n"
        "Direct and component candidates share base points and two scalar labels per point. "
        "The direct model receives two orthogonal projections drawn from the verified training-only identity.\n"
        "Outer truth is read from actual workbooks, not synthesized predictions. "
        "Augmented rows are correlated; they are not independent experimental evidence.\n"
        "Five outer folds and one fixed seed are a research estimate, not a production accuracy guarantee. "
        "Data quality issues and unconfirmed units remain; no production defaults changed.\n\n"
        "## Scope Comparison\n\n```text\n" + summary.to_string(index=False) + "\n```\n\n"
        "## Per Geometry\n\n```text\n" + table.to_string() + "\n```\n\n"
        "## Selected Recipes\n\n```json\n" + json.dumps(selections, indent=2) + "\n```\n"
    )
    (out / "report.md").write_text(text, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(write_report(args.run).to_string(index=False))


if __name__ == "__main__":
    main()
