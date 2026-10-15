"""Audit -> grouped evaluation -> model context -> new-triangle prediction."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.data.audit import audit_sources, load_snapshot
from src.evaluation.system import evaluate_system
from src.geometry.triangle import query_grid
from src.physics.force import FORCE_TO_N
from src.regression.models import ModelSpec
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

from .artifacts import (
    checkpoint_identity,
    environment_identity,
    implementation_identity,
    load_bundle,
    source_files,
    train_bundle,
)
from .engine import diagnostics, predict_system
from .reporting import write_report


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    unexpected = set(config) - {"regression", "classifier", "seed", "inner_splits", "window_size", "physics"}
    if unexpected:
        raise ValueError("Unknown configuration fields: " + str(sorted(unexpected)))
    for task in ("regression", "classifier"):
        settings = config.setdefault(task, {})
        if settings.get("checkpoint"):
            settings["checkpoint"] = str((Path(path).resolve().parent / settings["checkpoint"]).resolve())
        config[task] = ModelSpec(**settings).to_dict()
    config.setdefault("seed", 42)
    config.setdefault("inner_splits", 3)
    config.setdefault("window_size", 3)
    if type(config["window_size"]) is not int or config["window_size"] < 3 or config["window_size"] % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    return config


def audit_run(manifest, out_dir):
    with experiment_run(out_dir, "system-audit", extra_sources=source_files()) as directory:
        snapshot = audit_sources(manifest, directory)
        write_report(directory, "Data audit", json.loads((directory / "audit.json").read_text(encoding="utf-8")))
    return directory, snapshot["status"]


def evaluate_run(snapshot, config, out_dir, *, task="fields", folds=5, protocol="geometry"):
    points, samples, meta = load_snapshot(snapshot)
    input_hash = file_sha256(Path(snapshot) / "snapshot.json")
    checkpoints = checkpoint_identity(config)
    with experiment_run(out_dir, "system-evaluate", extra_sources=source_files()) as directory:
        results = evaluate_system(points, samples, meta, config, task=task, n_splits=folds, protocol=protocol)
        results["metrics"]["target_contracts"] = meta["targets"]
        results["metrics"]["physics"] = config.get("physics")
        results["metrics"]["snapshot"] = str(Path(snapshot).resolve())
        for name in ("fold_manifest", "point_oof", "focus_oof", "by_geometry_re"):
            results[name].to_csv(directory / (name + ".csv"), index=False)
        write_json(directory / "metrics.json", results["metrics"])
        load_snapshot(snapshot)
        if input_hash != file_sha256(Path(snapshot) / "snapshot.json") or checkpoints != checkpoint_identity(config):
            raise ValueError("Input or checkpoint changed during evaluation")
        record = {
            "snapshot_hash": file_sha256(Path(snapshot) / "snapshot.json"),
            "config": config,
            "task": task,
            "protocol": protocol,
            "folds": folds,
            "implementation": implementation_identity(),
            "checkpoints": checkpoints,
            "environment": environment_identity(config),
            "metrics_sha256": file_sha256(directory / "metrics.json"),
        }
        write_json(directory / "evaluation.json", record)
        write_report(directory, "Grouped evaluation", results["metrics"])
    return directory


def train_run(snapshot, config, out_dir, *, mode="fields", evaluation=None):
    record = None
    if evaluation:
        evaluation = Path(evaluation)
        record = json.loads((evaluation / "evaluation.json").read_text(encoding="utf-8"))
        expected = {
            "snapshot_hash": file_sha256(Path(snapshot) / "snapshot.json"),
            "config": config,
            "task": mode,
            "implementation": implementation_identity(),
            "checkpoints": checkpoint_identity(config),
            "environment": environment_identity(config),
        }
        if any(record[key] != value for key, value in expected.items()):
            raise ValueError("Evaluation does not match this dataset, configuration, task or implementation")
        if record["metrics_sha256"] != file_sha256(evaluation / "metrics.json"):
            raise ValueError("Evaluation metrics changed")
        status = json.loads((evaluation / "run_status.json").read_text(encoding="utf-8"))
        if status["status"] != "complete":
            raise ValueError("Evaluation run is incomplete")
        record = {"path": str(evaluation.resolve()), **record}
    with experiment_run(out_dir, "system-train", extra_sources=source_files()) as directory:
        bundle = train_bundle(snapshot, directory, config, mode, record)
        write_report(directory, "Model context bundle", bundle)
    return directory


def predict_run(bundle_path, request, out_dir):
    fitted, training, samples, meta, bundle = load_bundle(bundle_path)
    query = query_grid(request)
    if "provided_fzong" in request:
        provided = request["provided_fzong"]
        target = meta["targets"].get("Fzong", {})
        if target.get("quantity") != "force" or provided.get("component") != target.get("component"):
            raise ValueError("Provided Fzong must match the observed-field training component")
        values = np.asarray(provided["values"], dtype=float)
        if values.shape != (len(query),) or not np.isfinite(values).all() or provided.get("unit") not in FORCE_TO_N:
            raise ValueError("provided_fzong must be finite, unit-declared and aligned with every query row")
        query["provided_force"] = values * FORCE_TO_N[provided["unit"]] / FORCE_TO_N[target["unit"]]
    with experiment_run(out_dir, "system-predict", extra_sources=source_files()) as directory:
        write_json(directory / "request.json", request)
        query.to_csv(directory / "query_points.csv", index=False)
        points, focus, status = predict_system(fitted, query, meta, bundle["config"])
        risk = diagnostics(training, query, fitted["fields"])
        if not risk.empty:
            risk["unvalidated_model"] = bundle["release_status"] != "evaluated"
            risk["needs_review"] |= risk.unvalidated_model
        if not focus.empty:
            context = fitted["focus_features"]
            known = focus.sample_id.isin(context.sample_id)
            reasons = []
            for row in focus.itertuples():
                flags = [row.review_reason] if row.review_reason else []
                if row.geometry_family_id not in set(context.geometry_family_id):
                    flags.append("new_geometry")
                if bundle["release_status"] != "evaluated":
                    flags.append("unvalidated_model")
                if not risk.empty and risk.loc[risk.sample_id.eq(row.sample_id), "needs_review"].any():
                    flags.append("upstream_field_review")
                # The current boundary feature uses physical horizontal rows.
                widths = context.unique_coordinate_points
                if row.unique_coordinate_points not in set(widths) or row.z_row_count not in set(context.z_row_count):
                    flags.append("sampling_differs_from_training")
                for column in ("re", "radius_um", "boundary_distance_norm", "boundary_midpoint_norm"):
                    value = getattr(row, column)
                    if np.isfinite(value) and not context[column].min() <= value <= context[column].max():
                        flags.append("outside_focus_feature_range")
                reasons.append(";".join(sorted(set(flags))))
            focus["known_sample_identity"] = known
            focus["review_reason"] = reasons
            focus["review_status"] = np.where(focus.review_reason.ne(""), "REVIEW", "OK")
        points.to_csv(directory / "point_predictions.csv", index=False)
        focus.to_csv(directory / "focus_predictions.csv", index=False)
        risk.to_csv(directory / "diagnostics.csv", index=False)
        status.update(
            {
                "evaluation": False,
                "query_rows": len(query),
                "focus_samples": len(focus),
                "bundle": str(Path(bundle_path).resolve()),
                "bundle_hash": file_sha256(Path(bundle_path) / "bundle.json"),
                "release_status": bundle["release_status"],
                "targets": meta["targets"],
                "physics": bundle["config"].get("physics"),
                "review_point_records": int(risk.needs_review.sum()) if len(risk) else 0,
            }
        )
        write_json(directory / "prediction.json", status)
        write_report(directory, "New triangle prediction", status)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a synthetic five-fold cascade integration example")
    demo.add_argument("--out-dir", type=Path, default=Path("results/system/demo"))
    audit = commands.add_parser("audit", help="Audit explicit source manifest into a versioned snapshot")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--out-dir", type=Path, default=Path("results/system/audit"))
    for name in ("evaluate", "train"):
        item = commands.add_parser(name)
        item.add_argument("--snapshot", type=Path, required=True)
        item.add_argument("--config", type=Path)
        item.add_argument("--out-dir", type=Path, default=Path("results/system") / name)
        if name == "evaluate":
            item.add_argument("--task", choices=["fields", "FD", "FL", "focus", "cascade"], default="fields")
            item.add_argument("--folds", type=int, default=5)
            item.add_argument("--protocol", choices=["geometry", "spatial"], default="geometry")
        else:
            item.add_argument("--mode", choices=["fields", "focus", "cascade"], default="fields")
            item.add_argument("--evaluation", type=Path)
    predict = commands.add_parser("predict")
    predict.add_argument("--bundle", type=Path, required=True)
    predict.add_argument("--request", type=Path, required=True)
    predict.add_argument("--out-dir", type=Path, default=Path("results/system/predict"))
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo

        print(run_demo(args.out_dir))
    elif args.command == "audit":
        directory, status = audit_run(args.manifest, args.out_dir)
        print(f"Audit status: {status}; {directory}")
        if status == "REJECT":
            raise SystemExit(2)
    elif args.command == "evaluate":
        print(
            evaluate_run(
                args.snapshot,
                read_config(args.config),
                args.out_dir,
                task=args.task,
                folds=args.folds,
                protocol=args.protocol,
            )
        )
    elif args.command == "train":
        print(
            train_run(args.snapshot, read_config(args.config), args.out_dir, mode=args.mode, evaluation=args.evaluation)
        )
    elif args.command == "predict":
        print(predict_run(args.bundle, json.loads(args.request.read_text(encoding="utf-8")), args.out_dir))
    else:
        for name in ("prediction.json", "metrics.json", "audit.json", "bundle.json"):
            if (args.run / name).is_file():
                write_report(
                    args.run, "Rebuilt system report", json.loads((args.run / name).read_text(encoding="utf-8"))
                )
                break
        else:
            raise ValueError("No completed numerical report inputs found")


if __name__ == "__main__":
    main()
