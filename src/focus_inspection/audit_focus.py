"""Audit focus inputs and labels without loading or fitting TabPFN."""

import argparse
from pathlib import Path

from utils.provenance import write_json
from utils.runs import experiment_run

from .build_focus_boundary_features import run as build_features
from .dataset import build_feature_dataset, build_labeled_dataset
from .geometry_integrity import GEOMETRY_COLUMNS, write_geometry_audit
from .quality import FeatureConfig, quality_reasons
from .settings import DEFAULT_LABEL_FILE, DEFAULT_RAW_DIR, ROOT


def run(raw_dir, out_dir, *, label_file=None, config=None, strict=False):
    config = config or FeatureConfig()
    with experiment_run(out_dir, "focus-audit") as directory:
        build_features(raw_dir, directory / "features", make_plots=False, config=config)
        if label_file is not None:
            data, audit = build_labeled_dataset(raw_dir, label_file, directory / "features/features.csv")
        else:
            data = build_feature_dataset(raw_dir, directory / "features/features.csv")
            audit = {"sample_count": len(data), "labels_checked": False}
        data["all_quality_flags"] = [";".join(quality_reasons(row)) for row in data.to_dict("records")]
        data.to_csv(directory / "audited_samples.csv", index=False, encoding="utf-8-sig")
        write_geometry_audit(data, directory / "geometry_audit.csv")
        warnings = data.loc[data["all_quality_flags"].ne("")]
        audit.update(
            {
                "quality_warning_samples": len(warnings),
                "geometry_warning_samples": int((~data.geometry_consistent).sum()),
                "raw_data_modified": False,
            }
        )
        write_json(directory / "audit.json", audit)
        lines = [
            "# Focus input audit",
            "",
            "Original values and labels are unchanged.",
            "",
            f"Samples: {len(data)}; review flags: {len(warnings)}; inconsistent geometry: {audit['geometry_warning_samples']}.",
        ]
        for source, group in warnings.groupby("source_file", sort=True):
            lines += [
                "",
                f"## {source}",
                "",
                f"Flagged samples: {len(group)}.",
                "",
                "Flags: " + "; ".join(sorted(set(group.all_quality_flags.str.split(";").explode()))),
                "",
            ]
            for _, sample in group.groupby(["R", "geometry_issues"], dropna=False):
                row = sample.iloc[0]
                lines.append("Geometry checks: " + (row["geometry_issues"] or "none"))
                lines.append(f"R={row['R']:g}; Re=" + ",".join(f"{v:g}" for v in sorted(sample.re.unique())))
                lines.append(
                    "Supplied geometry: " + ", ".join(f"{column}={row[column]:g}" for column in GEOMETRY_COLUMNS)
                )
                lines.append("")
        (directory / "issues.md").write_text("\n".join(lines), encoding="utf-8")
        if strict and len(warnings):
            raise ValueError(f"Input audit contains review flags; see {directory / 'issues.md'}")
        print(f"Saved audit for {len(data)} samples to {directory}")
        return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--label-file", type=Path, default=DEFAULT_LABEL_FILE)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/focus_inspection/2026-09-04/focus_data_audit")
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument("--length-unit", choices=("um", "mm", "m"), default="um")
    parser.add_argument("--invalid-rows", choices=("error", "drop"), default="error")
    parser.add_argument("--strict", action="store_true", help="Return failure if any review flags exist")
    args = parser.parse_args()
    run(
        args.raw_dir,
        args.out_dir,
        label_file=None if args.skip_labels else args.label_file,
        config=FeatureConfig(args.window_size, args.length_unit, args.invalid_rows),
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
