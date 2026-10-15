"""Baseline focus evaluation CLI; public imports remain backward compatible."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.provenance import write_json
from utils.runs import experiment_run, latest_artifact

from .dataset import (
    build_labeled_dataset,
)
from .dataset import (
    geometry_group_from_source_file as geometry_group_from_source_file,
)
from .dataset import (
    load_focus_labels as load_focus_labels,
)
from .dataset import (
    load_geometry_features as load_geometry_features,
)
from .dataset import (
    summary_id_from_source_file as summary_id_from_source_file,
)
from .evaluation import evaluate_predictions, make_grouped_folds, metrics_by_geometry
from .modeling import grouped_tabpfn_oof, resolve_device
from .plotting import configure_fonts
from .reporting import save_figures, write_gallery
from .settings import (
    BOUNDARY_FEATURE,
    BOUNDARY_MIDPOINT_FEATURE,
    CONTEXT_FEATURES,
    DEFAULT_BOUNDARY_FEATURES,
    DEFAULT_LABEL_FILE,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_RAW_DIR,
    MAIN_MODEL_NAME,
    MODEL_FEATURE_SETS,
)
from .settings import (
    GEOMETRY_FEATURES as GEOMETRY_FEATURES,
)


def _run(
    *,
    raw_dir: Path,
    label_file: Path,
    boundary_features_file: Path,
    out_dir: Path,
    model_path: Path,
    device: str,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    configure_fonts()
    data, data_audit = build_labeled_dataset(raw_dir, label_file, boundary_features_file)
    folds, fold_assignments, leakage_audit = make_grouped_folds(
        data,
        n_splits=n_splits,
        random_state=random_state,
    )
    data["cv_fold"] = fold_assignments["cv_fold"].to_numpy(dtype=int)

    y = data["focus_label"].to_numpy(dtype=int)
    data["pred_rule_lt_100"] = (data[BOUNDARY_FEATURE] < 100).astype(int)
    metric_records = [{"model": "rule_lt_100", **evaluate_predictions(y, data["pred_rule_lt_100"].to_numpy(), None)}]

    for model_offset, (model_name, feature_columns) in enumerate(MODEL_FEATURE_SETS.items(), start=1):
        prediction, probability = grouped_tabpfn_oof(
            data,
            folds,
            feature_columns=feature_columns,
            model_path=model_path,
            device=device,
            random_state=random_state + model_offset * 100,
        )
        data[f"pred_{model_name}"] = prediction
        data[f"proba_{model_name}"] = probability
        metric_records.append({"model": model_name, **evaluate_predictions(y, prediction, probability)})

    metrics_table = pd.DataFrame(metric_records)
    main_errors = data[data[f"pred_{MAIN_MODEL_NAME}"] != data["focus_label"]].copy()
    geometry_metrics = metrics_by_geometry(data, f"pred_{MAIN_MODEL_NAME}")
    fold_metrics = []
    for fold_id, fold_data in data.groupby("cv_fold", sort=True):
        fold_result = evaluate_predictions(
            fold_data["focus_label"].to_numpy(dtype=int),
            fold_data[f"pred_{MAIN_MODEL_NAME}"].to_numpy(dtype=int),
            fold_data[f"proba_{MAIN_MODEL_NAME}"].to_numpy(dtype=float),
        )
        fold_metrics.append({"fold": int(fold_id), "samples": int(len(fold_data)), **fold_result})

    output_columns = [
        "source_file",
        "summary_id",
        "geometry_group",
        "R",
        "re",
        "focus_label_name",
        "focus_label",
        "cv_fold",
        *CONTEXT_FEATURES[2:],
        BOUNDARY_FEATURE,
        BOUNDARY_MIDPOINT_FEATURE,
        "boundary_gap_signed",
        "boundary_distance_norm",
        "pred_rule_lt_100",
        *[column for model_name in MODEL_FEATURE_SETS for column in (f"pred_{model_name}", f"proba_{model_name}")],
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    data[output_columns].to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    data[
        [
            "source_file",
            "summary_id",
            "geometry_group",
            "R",
            "re",
            *CONTEXT_FEATURES[2:],
            BOUNDARY_FEATURE,
            BOUNDARY_MIDPOINT_FEATURE,
            "focus_label_name",
            "focus_label",
        ]
    ].to_csv(out_dir / "labeled_features.csv", index=False, encoding="utf-8-sig")
    metrics_table.drop(columns="confusion_matrix").to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    fold_assignments.to_csv(out_dir / "fold_assignments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metrics).drop(columns="confusion_matrix").to_csv(
        out_dir / "main_model_metrics_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    geometry_metrics.to_csv(out_dir / "main_model_metrics_by_geometry.csv", index=False, encoding="utf-8-sig")
    main_errors[output_columns].to_csv(out_dir / "main_model_errors.csv", index=False, encoding="utf-8-sig")

    audit = {
        **data_audit,
        "label_file": str(label_file.resolve()),
        "boundary_features_file": str(boundary_features_file.resolve()),
        "model_path": str(model_path.resolve()),
        "model": "TabPFNClassifier",
        "device": device,
        "main_model": MAIN_MODEL_NAME,
        "feature_sets": MODEL_FEATURE_SETS,
        "rule_baseline": f"{BOUNDARY_FEATURE} < 100 micrometers",
        "cross_validation": "StratifiedGroupKFold_by_geometry_size",
        "n_splits": n_splits,
        "random_state": random_state,
        "leakage_checks": leakage_audit,
        "preprocessing": "SimpleImputer and StandardScaler fitted separately inside each training fold",
    }
    metrics_json = {
        row["model"]: {key: value for key, value in row.items() if key != "model"} for row in metric_records
    }
    write_json(out_dir / "evaluation.json", {"metrics": metrics_json, "audit": audit})

    save_figures(data, metrics_table, out_dir)
    write_gallery(out_dir, metrics_table, audit)
    report_metric_lines = [
        "| Model | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | ROC AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_table.to_dict("records"):
        report_metric_lines.append(
            f"| {row['model']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
            f"{row['mcc']:.4f} | {row['roc_auc']:.4f} |"
        )

    report_lines = [
        "# Focus TabPFN classification",
        "",
        f"- Samples: {len(data)}",
        f"- Labels: {data_audit['label_counts']}",
        f"- Geometry groups: {data_audit['geometry_group_count']}",
        f"- Cross-validation: {audit['cross_validation']} ({n_splits} folds)",
        f"- Main model: {MAIN_MODEL_NAME}",
        f"- Device: {device}",
        "",
        "## Metrics",
        "",
        *report_metric_lines,
        "",
        "## Leakage controls",
        "",
        "- One geometry size and all of its radii, orientations, and Re values stay in one fold.",
        "- Imputation and scaling are fitted only on each training fold.",
        "- The fixed 100 micrometer baseline is not tuned on evaluation labels.",
        "- All reported TabPFN metrics use out-of-fold predictions.",
    ]
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return data, metrics_table, audit


def run(**kwargs):
    with experiment_run(kwargs.pop("out_dir"), "focus-baseline") as run_dir:
        result, metrics, audit = _run(out_dir=run_dir, **kwargs)
        audit["output_dir"] = str(run_dir)
        return result, metrics, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-resistant TabPFN focus classification.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument(
        "--boundary-features", type=Path, help="Defaults to the latest completed feature run, then the legacy flat file"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="prefer-cuda")
    parser.add_argument(
        "--n-splits",
        type=int,
        default=8,
        help="Grouped folds; 8 leaves out one geometry size per fold for this batch.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    device = resolve_device(args.device)
    _, metrics, audit = run(
        raw_dir=args.raw_dir,
        label_file=args.label_file,
        boundary_features_file=args.boundary_features
        or latest_artifact(DEFAULT_BOUNDARY_FEATURES.parent, "features.csv"),
        out_dir=args.out_dir,
        model_path=args.model_path,
        device=device,
        n_splits=args.n_splits,
        random_state=args.random_state,
    )
    print(json.dumps({"metrics": metrics.to_dict("records"), "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
