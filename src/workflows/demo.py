"""Reproducible synthetic integration example; not a scientific benchmark."""

from pathlib import Path

from src.geometry.triangle import query_grid
from utils.provenance import write_json
from utils.runs import experiment_run


def example_request(width=100, height=70, experiment="example", re_values=None):
    return {
        "coordinate_frame": "global_xz",
        "geometry": {"unit": "um", "vertices": [[width / 2, height], [0, 0], [width, 0]]},
        "grid": {"nx": 7, "nz": 7, "include_boundary": False},
        "re_values": re_values or [6, 12, 18],
        "radius_um": 1200,
        "experiment_id": experiment,
    }


def synthetic_dataset(root, *, geometries=6, classes=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources, labels = [], []
    classes = classes or ["UF", "F", "WF"]
    for i in range(geometries):
        req = example_request(100 + i * 10, 70 + i * 4, f"geometry-{i}")
        points = query_grid(req)
        for task in ("FD", "FL", "Fzong"):
            frame = points[["x", "z", "re"]].copy()
            centered = points.x - (100 + i * 10) / 2
            fd, fl = centered * 0.02 + points.re * 0.01, -centered * 0.08 + points.re * 0.02
            frame["value"] = fd if task == "FD" else fl if task == "FL" else fd + fl
            filename = f"{i}-{task}.csv"
            frame.to_csv(root / filename, index=False)
            sources.append(
                {
                    "path": filename,
                    "task": task,
                    "experiment_id": req["experiment_id"],
                    "geometry": req["geometry"],
                    "length_unit": "um",
                    "radius_unit": "um",
                    "radius": 1200,
                    "coordinate_frame": "global_xz",
                    "value_origin": "simulated",
                    "target": {"column": "value", "name": task, "quantity": "force", "unit": "nN", "component": "x"},
                }
            )
        for j, re in enumerate([6, 12, 18]):
            labels.append(
                {"experiment_id": req["experiment_id"], "re": re, "radius_um": 1200, "label": classes[j % len(classes)]}
            )
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "sources": sources,
            "classes": classes,
            "labels": labels,
            "label_provenance": "Synthetic integration labels; not experimental or physical validation",
        },
    )
    return root / "manifest.json"


def example_config():
    from .system import read_config

    result = read_config(None)
    result["regression"]["n_estimators"] = 8
    result["classifier"]["n_estimators"] = 8
    result["inner_splits"] = 2
    result["physics"] = {
        "confirmed": True,
        "formula": "signed_component_sum",
        "version": "synthetic-test-v1",
        "reference": "Synthetic test definition, not a claim about repository FD-u",
        "component": "x",
        "output_unit": "nN",
    }
    return result


def run_demo(out_dir):
    from .artifacts import source_files
    from .system import audit_run, evaluate_run, predict_run, train_run

    with experiment_run(out_dir, "system-synthetic-demo", extra_sources=source_files()) as directory:
        manifest = synthetic_dataset(directory / "inputs")
        config = example_config()
        write_json(directory / "model_config.json", config)
        snapshot, status = audit_run(manifest, directory / "audit")
        if status != "ACCEPT":
            raise RuntimeError("Synthetic fixture did not pass audit")
        evaluation = evaluate_run(snapshot, config, directory / "evaluation", task="cascade", folds=5)
        bundle = train_run(snapshot, config, directory / "models", mode="cascade", evaluation=evaluation)
        query = example_request(135, 81, "new-synthetic-triangle")
        write_json(directory / "query.json", query)
        prediction = predict_run(bundle, query, directory / "prediction")
        write_json(
            directory / "demo.json",
            {
                "synthetic": True,
                "scientific_validation": False,
                "snapshot": str(snapshot.resolve()),
                "evaluation": str(evaluation.resolve()),
                "bundle": str(bundle.resolve()),
                "prediction": str(prediction.resolve()),
            },
        )
    return directory
