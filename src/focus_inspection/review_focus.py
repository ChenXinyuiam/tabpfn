"""Sample-level review of completed repeated-seed geometry holdouts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .calibration import checked_probabilities
from .quality import quality_reasons
from .settings import ROOT

KEYS = ["source_file", "R", "re"]


def validate_predictions(data, model):
    required = [*KEYS, "seed", "cv_fold", "geometry_group", "focus_label", f"proba_{model}", f"pred_{model}"]
    missing = sorted(set(required) - set(data))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("Predictions must be nonempty and have complete keys, labels and probabilities")
    numeric_keys = data[["R", "re", "seed", "cv_fold"]].to_numpy(dtype=float)
    if not np.isfinite(numeric_keys).all() or (numeric_keys[:, :2] <= 0).any():
        raise ValueError("Sample keys must be finite; R and Re must be positive")
    if (numeric_keys[:, 2:] % 1 != 0).any() or (numeric_keys[:, 3] < 1).any():
        raise ValueError("Seed and fold identifiers must be integers, with positive folds")
    if data.duplicated([*KEYS, "seed"]).any():
        raise ValueError("Duplicate sample/seed predictions")
    if not data.focus_label.isin([0, 1]).all() or not data[f"pred_{model}"].isin([0, 1]).all():
        raise ValueError("Labels and predictions must be binary F/UF")
    checked_probabilities(data[f"proba_{model}"].to_numpy())
    if data.seed.nunique() < 2:
        raise ValueError("Repeated-seed review requires at least two seeds")
    grouped = data.groupby(KEYS, dropna=False)
    if not grouped.seed.nunique().eq(data.seed.nunique()).all():
        raise ValueError("Every sample must have the same complete seed coverage")
    metadata = [
        "focus_label",
        "geometry_group",
        "cv_fold",
        "source_sha256",
        "boundary_distance_abs",
        "boundary_midpoint_x",
        "geometry_consistent",
        "geometry_issues",
        "quality_flags",
    ]
    for name in metadata:
        if name in data and not grouped[name].nunique(dropna=False).eq(1).all():
            raise ValueError(f"Sample metadata changed between seeds: {name}")
    if not data.groupby(["seed", "geometry_group"]).cv_fold.nunique().eq(1).all():
        raise ValueError("A geometry appears in multiple test folds within one seed")


def summarize_samples(data, model):
    validate_predictions(data, model)
    rows = []
    for key, group in data.groupby(KEYS, sort=True):
        first = group.iloc[0]
        probability = group[f"proba_{model}"]
        wrong = group[f"pred_{model}"].ne(group.focus_label)
        reasons = quality_reasons(first.to_dict())
        errors = int(wrong.sum())
        unstable = group[f"pred_{model}"].nunique() > 1
        persistent = errors == len(group)
        high_errors = int((wrong & ((probability >= 0.9) | (probability <= 0.1))).sum())
        if errors:
            reasons.append("persistent_oof_error" if persistent else "occasional_oof_error")
        if unstable:
            reasons.append("decision_changes_between_seeds")
        if high_errors:
            reasons.append("high_confidence_oof_error")
        width = first.get("boundary_distance_abs", np.nan)
        rule_conflict = bool(pd.notna(width) and int(width < 100) != first.focus_label)
        if rule_conflict:
            reasons.append("100um_rule_disagrees_with_label")
        rows.append(
            {
                **dict(zip(KEYS, key, strict=True)),
                "geometry_group": first.geometry_group,
                "label": "F" if first.focus_label else "UF",
                "seed_count": len(group),
                "error_seeds": errors,
                "persistent_error": persistent,
                "decision_unstable": unstable,
                "high_confidence_error_seeds": high_errors,
                "probability_mean": probability.mean(),
                "probability_min": probability.min(),
                "probability_max": probability.max(),
                "probability_std": probability.std(ddof=0),
                "boundary_distance_abs": width,
                "boundary_midpoint_x": first.get("boundary_midpoint_x", np.nan),
                "geometry_warning": "inconsistent_triangle_geometry" in reasons,
                "geometry_issues": first.get("geometry_issues", ""),
                "rule_conflict": rule_conflict,
                "needs_review": bool(reasons),
                "review_reasons": ";".join(dict.fromkeys(reasons)),
                "reviewed_label": "",
                "review_note": "",
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "persistent_error",
                "error_seeds",
                "high_confidence_error_seeds",
                "decision_unstable",
                "geometry_warning",
                *KEYS,
            ],
            ascending=[False] * 5 + [True] * 3,
            kind="stable",
        )
        .reset_index(drop=True)
    )


def summarize_geometries(data, samples, model):
    rows = []
    for name, group in data.groupby("geometry_group", sort=True):
        unique = samples[samples.geometry_group.eq(name)]
        prediction = group[f"pred_{model}"]
        rows.append(
            {
                "geometry_group": name,
                "unique_samples": len(unique),
                "predictions": len(group),
                "correct_predictions": int(prediction.eq(group.focus_label).sum()),
                "accuracy": prediction.eq(group.focus_label).mean(),
                "f_as_uf_predictions": int((group.focus_label.eq(1) & prediction.eq(0)).sum()),
                "uf_as_f_predictions": int((group.focus_label.eq(0) & prediction.eq(1)).sum()),
                "ever_wrong_samples": int(unique.error_seeds.gt(0).sum()),
                "persistent_error_samples": int(unique.persistent_error.sum()),
                "unstable_samples": int(unique.decision_unstable.sum()),
                "geometry_warning_samples": int(unique.geometry_warning.sum()),
            }
        )
    return pd.DataFrame(rows)


def geometry_accuracy_intervals(groups, *, repeats=5000):
    if len(groups) < 2 or repeats < 100:
        raise ValueError("Intervals require at least two geometries and 100 bootstrap repeats")
    correct = groups.correct_predictions.to_numpy(dtype=float)
    count = groups.predictions.to_numpy(dtype=float)
    draws = np.random.default_rng(42).integers(len(groups), size=(repeats, len(groups)))
    weighted = correct[draws].sum(axis=1) / count[draws].sum(axis=1)
    macro = (correct / count)[draws].mean(axis=1)
    return {
        "sample_weighted_accuracy": float(correct.sum() / count.sum()),
        "sample_weighted_ci95": np.quantile(weighted, [0.025, 0.975]).tolist(),
        "geometry_macro_accuracy": float((correct / count).mean()),
        "geometry_macro_ci95": np.quantile(macro, [0.025, 0.975]).tolist(),
        "unit": "geometry; all samples and seeds stay together",
        "repeats": repeats,
        "random_state": 42,
        "interpretation": "Descriptive development uncertainty, not external validation or a future accuracy guarantee",
    }


def run(evaluation_dir, out_dir, *, model):
    from .review_report import write_report

    evaluation_dir = Path(evaluation_dir)
    status = json.loads((evaluation_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("kind") not in {"focus-ablate", "focus-calibrate"}:
        raise ValueError("Review requires a completed focus-ablate or focus-calibrate run")
    with experiment_run(out_dir, "focus-review") as directory:
        source = evaluation_dir / "oof_predictions.csv"
        data = pd.read_csv(source, encoding="utf-8-sig", float_precision="round_trip")
        samples = summarize_samples(data, model)
        groups = summarize_geometries(data, samples, model)
        summary = {
            "evaluation_dir": str(evaluation_dir.resolve()),
            "model": model,
            "source_sha256": file_sha256(source),
            "unique_samples": len(samples),
            "prediction_rows": len(data),
            "seeds": sorted(data.seed.unique().tolist()),
            "geometry_groups": len(groups),
            "ever_wrong_samples": int(samples.error_seeds.gt(0).sum()),
            "persistent_error_samples": int(samples.persistent_error.sum()),
            "unstable_samples": int(samples.decision_unstable.sum()),
            "geometry_warning_samples": int(samples.geometry_warning.sum()),
            "geometry_audit_available": "geometry_consistent" in data,
            "review_samples": int(samples.needs_review.sum()),
            "high_confidence_error_samples": int(samples.high_confidence_error_seeds.gt(0).sum()),
            "accuracy_intervals": geometry_accuracy_intervals(groups),
            "model_fits": 0,
            "labels_modified": False,
            "review_policy": "Triage only; no automated corrections, exclusions, retraining or model selection",
        }
        samples.to_csv(directory / "sample_summary.csv", index=False, encoding="utf-8-sig")
        samples[samples.needs_review].to_csv(directory / "review_queue.csv", index=False, encoding="utf-8-sig")
        groups.to_csv(directory / "geometry_summary.csv", index=False, encoding="utf-8-sig")
        write_json(directory / "review_summary.json", summary)
        write_report(directory, samples, groups, summary)
        print(f"Reviewed {len(samples)} unique samples; {summary['review_samples']} require review", flush=True)
        return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Saved probability suffix, e.g. raw, width or midpoint")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/focus_inspection/2026-09-04/focus_review")
    args = parser.parse_args()
    run(args.evaluation_dir, args.out_dir, model=args.model)


if __name__ == "__main__":
    main()
