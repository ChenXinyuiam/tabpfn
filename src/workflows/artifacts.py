"""Reconstructible model context bundles with data, implementation and checkpoint hashes."""

import json
import shutil
from importlib.metadata import version
from pathlib import Path

from src.data.audit import load_snapshot
from src.geometry.triangle import FEATURE_VERSION
from utils.provenance import file_sha256, write_json

from .engine import fit_system

ROOT = Path(__file__).resolve().parents[2]


def source_files():
    paths = []
    for folder in ("geometry", "physics", "evaluation", "workflows"):
        paths.extend((ROOT / "src" / folder).glob("*.py"))
    paths.extend(
        ROOT / path
        for path in (
            "src/data/contracts.py",
            "src/data/audit.py",
            "src/regression/models.py",
            "src/focus_inspection/system_features.py",
            "src/focus_inspection/focus_boundary_feature.py",
            "src/focus_inspection/modeling.py",
            "src/regression/predict_fields.py",
            "src/regression/fd_to_fl.py",
            "src/cli.py",
        )
    )
    return sorted(set(paths))


def implementation_identity():
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in source_files()}


def environment_identity(config):
    names = ["numpy", "pandas", "scikit-learn", "scipy"]
    if any(config.get(kind, {}).get("backend") == "tabpfn" for kind in ("regression", "classifier")):
        names.extend(["tabpfn", "torch"])
    return {name: version(name) for name in names}


def checkpoint_identity(config):
    result = {}
    for task in ("regression", "classifier"):
        settings = config.get(task, {})
        if settings.get("backend") == "tabpfn":
            path = Path(settings["checkpoint"]).resolve()
            result[str(path)] = file_sha256(path)
    return result


def save_bundle(snapshot_path, directory, config, mode, fitted, evaluation=None):
    directory, snapshot_path = Path(directory), Path(snapshot_path)
    _, _, meta = load_snapshot(snapshot_path)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "snapshot"
    target.mkdir(exist_ok=False)
    for name in ("snapshot.json", "points.csv", "samples.csv", "audit.json", "issues.csv"):
        shutil.copy2(snapshot_path / name, target / name)
    if fitted["focus_features"] is not None:
        fitted["focus_features"].to_csv(directory / "training_focus_features.csv", index=False)
    if fitted["crossfit"] is not None:
        fitted["crossfit"].to_csv(directory / "training_oof_fields.csv", index=False)
        write_json(directory / "crossfit_audit.json", fitted["crossfit_audit"])
    data = {
        "bundle_version": 1,
        "feature_version": FEATURE_VERSION,
        "mode": mode,
        "config": config,
        "targets": meta["targets"],
        "classes": meta["classes"],
        "snapshot_hash": file_sha256(target / "snapshot.json"),
        "implementation": implementation_identity(),
        "environment": environment_identity(config),
        "checkpoints": checkpoint_identity(config),
        "evaluation": evaluation,
        "release_status": "evaluated" if evaluation else "unvalidated",
        "serialization": "training_context; estimators are deterministically reconstructed, not pickled",
        "derived_files": {p.name: file_sha256(p) for p in directory.glob("training_*.csv")},
    }
    write_json(directory / "bundle.json", data)
    return data


def load_bundle(directory):
    directory = Path(directory)
    bundle = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
    if bundle.get("bundle_version") != 1 or bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Unsupported model bundle version")
    if bundle["implementation"] != implementation_identity():
        raise ValueError("Implementation changed; re-evaluate and rebuild the bundle")
    if bundle["environment"] != environment_identity(bundle["config"]):
        raise ValueError("Model dependency versions changed")
    if bundle["checkpoints"] != checkpoint_identity(bundle["config"]):
        raise ValueError("Checkpoint hash mismatch")
    if bundle["snapshot_hash"] != file_sha256(directory / "snapshot/snapshot.json"):
        raise ValueError("Bundle snapshot metadata changed")
    for name, digest in bundle.get("derived_files", {}).items():
        if Path(name).name != name or file_sha256(directory / name) != digest:
            raise ValueError("Derived training context changed")
    points, samples, meta = load_snapshot(directory / "snapshot")
    if meta["targets"] != bundle["targets"] or meta["classes"] != bundle["classes"]:
        raise ValueError("Bundle and snapshot contracts differ")
    # Reuse the saved out-of-fold context; no new cross-fitting is needed at inference.
    import pandas as pd

    from src.workflows.engine import fit_classifier, fit_fields

    fitted = {"mode": bundle["mode"], "fields": {}, "classifier": None, "focus_features": None}
    if bundle["mode"] in {"fields", "cascade"}:
        fitted["fields"] = fit_fields(points, bundle["config"])
    if bundle["mode"] in {"focus", "cascade"}:
        features = pd.read_csv(directory / "training_focus_features.csv")
        fitted["classifier"] = fit_classifier(features, meta, bundle["config"])
        fitted["focus_features"] = features
    return fitted, points, samples, meta, bundle


def train_bundle(snapshot_path, directory, config, mode, evaluation=None):
    points, samples, meta = load_snapshot(snapshot_path)
    before = file_sha256(Path(snapshot_path) / "snapshot.json")
    checkpoints = checkpoint_identity(config)
    fitted = fit_system(points, samples, meta, config, mode)
    load_snapshot(snapshot_path)
    if before != file_sha256(Path(snapshot_path) / "snapshot.json") or checkpoints != checkpoint_identity(config):
        raise ValueError("Training inputs or checkpoints changed during fitting")
    return save_bundle(snapshot_path, directory, config, mode, fitted, evaluation)
