"""Audit zero responses and test training-only positive-floor imputation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.regression.rotation_dataset import ANGLES, load_snapshot
from src.regression.transfer_fd import (
    TransferRecipe,
    guard_training,
    mesh_features,
    predict_components,
    save_metrics,
)
from src.regression.tune_rotated_fd import geometry_folds, observed, queries, sample_context
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run


def positive_floor(context):
    """Infer the floor from training labels only, leaving query truth unchanged."""
    result = context.copy()
    records = []
    for label in ("u0", "u90"):
        values = result[label].to_numpy()
        nonzero = abs(values[values != 0])
        floor = float(nonzero.min() / 10) if len(nonzero) else 0.
        zero = values == 0
        result.loc[zero, label] = floor
        records.append({"component": label, "floor": floor, "zeros": int(zero.sum()),
                        "replaced": int(zero.sum()) if floor > 0 else 0})
    return result, records


def zero_statistics(data):
    boundary = (data[["mesh_a", "mesh_b", "mesh_c"]].abs().min(axis=1) <= 1e-4).to_numpy()
    rows = []
    for label in ("u0", "u90"):
        values = data[label].to_numpy()
        zero = values == 0
        nonzero = abs(values[~zero])
        rows.append({"component": label, "rows": len(data), "zeros": int(zero.sum()),
                     "zero_fraction": float(zero.mean()),
                     "zeros_near_mesh_boundary": int((zero & boundary).sum()),
                     "zeros_interior": int((zero & ~boundary).sum()),
                     "negative_rows": int((values < 0).sum()),
                     "minimum_nonzero_abs": float(nonzero.min()) if len(nonzero) else None,
                     "suggested_positive_floor": float(nonzero.min() / 10) if len(nonzero) else None,
                     "signed_min": float(values.min()), "signed_max": float(values.max())})
    return rows


def audit(base, test, prediction, out):
    overall = zero_statistics(base)
    write_json(out / "zero_summary.json", {"components": overall,
                                          "both_zero_rows": int(((base.u0 == 0) & (base.u90 == 0)).sum()),
                                          "boundary_barycentric_tolerance": 1e-4,
                                          "interpretation": "Boundary coincidence is not proof of an exact physical zero"})
    records = []
    for (source, re), data in base.groupby(["source_file", "re"]):
        records.extend({"source_file": source, "re": re, **row} for row in zero_statistics(data))
    pd.DataFrame(records).to_csv(out / "zero_by_source_re.csv", index=False)
    truth = test[[f"observed_{angle:03d}" for angle in ANGLES]].to_numpy()
    both = ((test.u0 == 0) & (test.u90 == 0)).to_numpy()
    angle_rows = [{"angle": angle, "zero_labels": int((truth[:, i] == 0).sum()),
                   "zero_from_both_reference_components": int(((truth[:, i] == 0) & both).sum()),
                   "zero_with_nonzero_reference_component": int(((truth[:, i] == 0) & ~both).sum()),
                   "near_zero_nonzero_labels": int(((abs(truth[:, i]) <= 1e-10) & (truth[:, i] != 0)).sum())}
                  for i, angle in enumerate(ANGLES)]
    pd.DataFrame(angle_rows).to_csv(out / "zero_by_angle.csv", index=False)
    error_rows = []
    for scope, part in (("all_24_angles", prediction), ("angle_000", prediction[prediction.rotation_deg == 0]),
                        ("angle_090", prediction[prediction.rotation_deg == 90])):
        squared_error = np.square(part.prediction - part.target)
        zero = part.target == 0
        error_rows.append({"scope": scope, "test_rows": len(part), "zero_rows": int(zero.sum()),
                           "zero_error_fraction": float(squared_error[zero].sum() / squared_error.sum()),
                           "zero_rmse": float(np.sqrt(squared_error[zero].mean()))})
    pd.DataFrame(error_rows).to_csv(out / "zero_test_errors.csv", index=False)
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--ensemble-run", type=Path, required=True)
    parser.add_argument("--method", choices=["nested_union_best", "nested_union_blend"], default="nested_union_best")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_zero_sensitivity/2026-09-29"))
    args = parser.parse_args()
    config = json.loads((args.ensemble_run / "config.json").read_text(encoding="utf-8"))
    status = json.loads((args.ensemble_run / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "complete":
        raise ValueError("A completed reference run is required")
    source_config = json.loads((Path(config["input_runs"][0]) / "config.json").read_text(encoding="utf-8"))
    model_args = SimpleNamespace(model_v25=Path(source_config["model_v25"]), model_v3=Path(source_config["model_v3"]),
                                 seed=config["seed"], device=source_config["device"], chunk_size=source_config["chunk_size"])
    for version, expected in source_config["checkpoint_hashes"].items():
        if file_sha256(getattr(model_args, "model_" + version)) != expected:
            raise ValueError("Reference checkpoint changed")
    recipes = {}
    for values in config["recipes"]:
        values["powers"] = tuple(values["powers"])
        recipe = TransferRecipe(**values)
        recipes[recipe.name] = recipe
    selections = json.loads((args.ensemble_run / "selections.json").read_text(encoding="utf-8"))
    with experiment_run(args.output_dir, "fd-zero-sensitivity", extra_sources=[
        __file__, Path(__file__).with_name("transfer_fd.py"), Path(__file__).with_name("transfer_diagnostics.py"),
        Path(__file__).with_name("tune_rotated_fd.py"), Path(__file__).with_name("rotation_dataset.py"),
        Path(__file__).with_name("evaluation.py"),
    ]) as out:
        base, test, info = load_snapshot(args.snapshot)
        if info != config["snapshot"]:
            raise ValueError("Reference dataset changed")
        base, _ = mesh_features(base)
        extra = [c for c in base if c.startswith("mesh_")]
        test = test.merge(base[["base_id", *extra]], on="base_id", validate="one_to_one")
        reference = pd.read_csv(args.ensemble_run / "predictions.csv")
        reference = reference[reference.method == args.method].copy()
        original_truth = reference.set_index(["base_id", "rotation_deg"]).target.sort_index()
        reference["method"] = "original_zero_labels"
        print(pd.DataFrame(audit(base, test, reference, out)).to_string(index=False), flush=True)
        write_json(out / "config.json", {
            "reference_run": str(args.ensemble_run.resolve()), "reference_method": args.method,
            "snapshot": info, "recipes": [asdict(r) for r in recipes.values()],
            "policy": "Training zeros only -> positive min(abs(nonzero training values))/10, per component and fit",
            "test_policy": "All original test truths and test identities preserved, no source file edits",
            "evaluation_status": "Exploratory sensitivity analysis of user-requested imputation, not a new blind test",
        })
        if args.audit_only:
            return
        outputs, thresholds, manifests = [reference], [], []
        for fold, (train_groups, test_groups) in enumerate(geometry_folds(base.geometry_group, 5, model_args.seed), 1):
            selected = selections[fold - 1]["components"]
            if args.method == "nested_union_best":
                selected = [[{**component[0], "weight": 1.}] for component in selected]
            pool = base[base.geometry_group.isin(train_groups)]
            query = test[test.geometry_group.isin(test_groups)]
            required = sorted({row["recipe"] for component in selected for row in component})
            guarded, guard_report = guard_training(pool) if any(recipes[name].guarded for name in required) else (pool, [])
            predictions = {}
            for name in required:
                recipe = recipes[name]
                context = sample_context(guarded if recipe.guarded else pool, recipe.rows, model_args.seed)
                adjusted, rows = positive_floor(context)
                thresholds.extend({"fold": fold, "recipe": name, **row} for row in rows)
                manifests.append({"fold": fold, "recipe": name, "train_groups": sorted(train_groups),
                                  "test_groups": sorted(test_groups), "base_ids": context.base_id.tolist(),
                                  "quarantine": guard_report if recipe.guarded else []})
                predictions[name] = predict_components(adjusted, query, recipe, model_args)
                print(f"zero-floor fold={fold} {name}", flush=True)
            components = np.column_stack([sum(row["weight"] * predictions[row["recipe"]][:, c] for row in selected[c])
                                          for c in (0, 1)])
            theta = np.deg2rad(ANGLES)
            values = (components[:, 0, None] * np.cos(theta) + components[:, 1, None] * np.sin(theta)).ravel()
            result = queries(query, ANGLES)[[
                "base_id", "source_file", "source_row", "geometry_group", "radius", "re", "x", "z", "rotation_deg"
            ]].copy()
            result["target"], result["prediction"] = observed(query, ANGLES), values
            result["method"], result["fold"] = "training_positive_floor", fold
            outputs.append(result)
            pd.DataFrame(thresholds).to_csv(out / "training_floors.csv", index=False)
            write_json(out / "splits.json", manifests)
        prediction = pd.concat(outputs, ignore_index=True)
        adjusted_truth = prediction[prediction.method == "training_positive_floor"].set_index(
            ["base_id", "rotation_deg"]
        ).target.sort_index()
        pd.testing.assert_series_equal(original_truth, adjusted_truth, rtol=1e-12, atol=1e-12)
        prediction.to_csv(out / "predictions.csv", index=False)
        print(save_metrics(prediction, out).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
