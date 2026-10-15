"""Compare saved transfer runs on identical native-target test identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression.evaluation import metric_row
from utils.provenance import write_json
from utils.runs import experiment_run


def compare(runs, out):
    summaries, geometry, names = [], [], {}
    reference = None
    checks, snapshot_hash = [], None
    for label, run, method in runs:
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        current_hash = config["snapshot"]["test_sha256"]
        if snapshot_hash is None:
            snapshot_hash = current_hash
        if current_hash != snapshot_hash:
            raise ValueError("Different test snapshots cannot be compared as paired predictions")
        try:
            data = pd.read_csv(run / "predictions.csv")
        except UnicodeDecodeError as error:
            saved = pd.read_csv(run / "reference_summary.csv")
            saved = saved[saved.method == method]
            if len(saved) != 3:
                raise ValueError("Incomplete legacy summary") from error
            scopes = {"all_24_angles": "All 24 angles", "angle_000": "0 degrees", "angle_090": "90 degrees"}
            for row in saved.to_dict("records"):
                row.pop("method")
                row["scope"] = scopes[row["scope"]]
                summaries.append({"experiment": label, **row})
            full = pd.read_csv(run / "geometry_metrics.csv")
            angle = pd.read_csv(run / "angle_metrics.csv")
            for scope, frame in [("All 24 angles", full), ("0 degrees", angle[angle.rotation_deg == 0]),
                                 ("90 degrees", angle[angle.rotation_deg == 90])]:
                part = frame[frame.method == method]
                for row in part.to_dict("records"):
                    row.pop("method")
                    geometry.append({"experiment": label, "scope": scope, **row})
            checks.append({"experiment": label, "source": "saved aggregate metrics",
                           "prediction_identity_verified": False, "reason": str(error),
                           "test_snapshot_sha256": current_hash})
            continue
        data = data[data.method == method].copy()
        if data.empty or data.duplicated(["base_id", "rotation_deg"]).any():
            raise ValueError("Missing method or duplicated predictions")
        if not np.isfinite(data[["target", "prediction"]]).all().all():
            raise ValueError("Nonfinite results")
        identity = data.set_index(["base_id", "rotation_deg"])[["geometry_group", "target"]].sort_index()
        if reference is None:
            reference = identity
        else:
            pd.testing.assert_frame_equal(reference, identity, rtol=1e-12, atol=1e-12)
        names.update(data.groupby("geometry_group").source_file.first().to_dict())
        checks.append({"experiment": label, "source": "row-level predictions",
                       "prediction_identity_verified": True, "test_snapshot_sha256": current_hash})
        for angle in (None, 0, 90):
            part = data if angle is None else data[data.rotation_deg == angle]
            scope = "All 24 angles" if angle is None else f"{angle} degrees"
            scores = []
            for group, chunk in part.groupby("geometry_group"):
                metric = metric_row(chunk.target.to_numpy(), chunk.prediction.to_numpy())
                geometry.append({"experiment": label, "geometry_group": group, "scope": scope, **metric})
                scores.append(metric["r2"])
            summaries.append({"experiment": label, "scope": scope, "geometry_mean_r2": np.mean(scores),
                              "geometry_worst_r2": np.min(scores), "rows": len(part),
                              **metric_row(part.target.to_numpy(), part.prediction.to_numpy())})
    summary = pd.DataFrame(summaries)
    per_geometry = pd.DataFrame(geometry)
    summary.to_csv(out / "comparison.csv", index=False)
    write_json(out / "comparison_checks.json", checks)
    per_geometry.to_csv(out / "geometry_comparison.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, scope in zip(axes, ("All 24 angles", "0 degrees", "90 degrees"), strict=True):
        table = per_geometry[per_geometry.scope == scope].pivot(index="geometry_group", columns="experiment", values="r2")
        for label, _, _ in runs:
            ax.plot(range(len(table)), table[label], marker="o", label=label, linewidth=1.5)
        ax.axhline(.9, color="#a32626", linestyle="--", linewidth=1, label="Target 0.9")
        ax.axhline(0, color="#999999", linewidth=.6)
        ax.set_xticks(range(len(table)), ["-".join(names[g].split("-")[:2]).replace("反", "") for g in table.index],
                      rotation=45, ha="right")
        ax.set_title(scope)
        ax.set_ylabel("Held-out geometry R2")
        ax.grid(axis="y", alpha=.2)
    axes[-1].legend(fontsize=8, loc="best")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=3, metavar=("LABEL", "PATH", "METHOD"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_transfer_comparison/2026-09-29"))
    args = parser.parse_args()
    with experiment_run(args.output_dir, "fd-transfer-comparison", extra_sources=[__file__]) as out:
        runs = [(label, Path(path), method) for label, path, method in args.run]
        write_json(out / "config.json", {"runs": args.run, "target_r2": .9})
        print(compare(runs, out).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
