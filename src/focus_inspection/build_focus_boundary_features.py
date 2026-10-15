"""Build smoothed Fzong boundary features for an Excel batch."""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.focus_inspection.focus_boundary_feature import (
    extract_focus_boundary_feature,
    resolve_source_columns,
)
from utils.provenance import file_sha256, write_json

from .plotting import configure_fonts, plt
from .quality import FEATURE_SCHEMA_VERSION, FeatureConfig, checked_source_rows, feature_code_hashes, quality_reasons

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "rawdata" / "2026-09-04"
DEFAULT_OUT_DIR = ROOT / "results" / "focus_inspection" / "2026-09-04" / "focus_boundary_features"

RED = "#d62728"
BLUE = "#1f77b4"
GRAY = "#777777"


def fmt_number(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def slugify(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", stem, flags=re.UNICODE)
    return stem.strip("_") or "sample"


def _scatter_signs(ax: plt.Axes, points: pd.DataFrame, sign_column: str, *, title: str) -> None:
    for sign, color, label in [(-1, BLUE, "Fzong < 0"), (1, RED, "Fzong > 0"), (0, GRAY, "Fzong = 0")]:
        subset = points[points[sign_column] == sign]
        if not subset.empty:
            ax.scatter(subset["x"], subset["z"], c=color, s=25, alpha=0.86, edgecolors="none", label=label)

    changed = points[points["sign_changed"]]
    if not changed.empty:
        ax.scatter(
            changed["x"],
            changed["z"],
            facecolors="none",
            edgecolors="#111111",
            s=58,
            linewidths=0.9,
            label="sign changed",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.25)


def draw_comparison(
    points: pd.DataFrame,
    feature: dict[str, object],
    *,
    source_file: str,
    radius: float,
    re_value: float,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True, sharex=True, sharey=True)
    _scatter_signs(axes[0], points, "sign_raw", title="Before smoothing")
    _scatter_signs(axes[1], points, "sign_smoothed", title="After centered median smoothing")

    raw_red = float(feature["rightmost_red_x_raw"])
    raw_blue = float(feature["leftmost_blue_x_raw"])
    raw_midpoint = float(feature["boundary_midpoint_x_raw"])
    smooth_red = float(feature["rightmost_red_x"])
    smooth_blue = float(feature["leftmost_blue_x"])
    smooth_midpoint = float(feature["boundary_midpoint_x"])
    if np.isfinite(raw_red):
        axes[0].axvline(raw_red, color=RED, linestyle="--", linewidth=1.0, alpha=0.8)
    if np.isfinite(raw_blue):
        axes[0].axvline(raw_blue, color=BLUE, linestyle="--", linewidth=1.0, alpha=0.8)
    if np.isfinite(raw_midpoint):
        axes[0].axvline(raw_midpoint, color="#111111", linestyle=":", linewidth=1.2, alpha=0.9)
    if np.isfinite(smooth_red):
        axes[1].axvline(smooth_red, color=RED, linestyle="--", linewidth=1.0, alpha=0.8)
    if np.isfinite(smooth_blue):
        axes[1].axvline(smooth_blue, color=BLUE, linestyle="--", linewidth=1.0, alpha=0.8)
    if np.isfinite(smooth_midpoint):
        axes[1].axvline(
            smooth_midpoint,
            color="#111111",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label="interval midpoint",
        )

    raw_feature = float(feature["boundary_distance_abs_raw"])
    smooth_feature = float(feature["boundary_distance_abs"])
    fig.suptitle(
        f"{source_file} | R={fmt_number(radius)} | Re={fmt_number(re_value)}\n"
        f"boundary distance: {fmt_number(raw_feature)} -> {fmt_number(smooth_feature)} | "
        f"midpoint x: {fmt_number(raw_midpoint)} -> {fmt_number(smooth_midpoint)} | "
        f"changed points: {feature['smoothed_sign_changes']}",
        fontsize=11,
    )

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside right upper", fontsize=8, frameon=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def build_gallery(features: pd.DataFrame, gallery_path: Path) -> None:
    cards: list[str] = []
    for row in features.sort_values(["source_file", "R", "re"]).to_dict(orient="records"):
        image_rel = Path(str(row["comparison_image"])).resolve().relative_to(gallery_path.parent.resolve()).as_posix()
        title = f"{row['source_file']} | R={fmt_number(float(row['R']))} | Re={fmt_number(float(row['re']))}"
        stats = (
            f"distance {fmt_number(float(row['boundary_distance_abs_raw']))} -> "
            f"{fmt_number(float(row['boundary_distance_abs']))} | "
            f"midpoint x {fmt_number(float(row['boundary_midpoint_x_raw']))} -> "
            f"{fmt_number(float(row['boundary_midpoint_x']))} | "
            f"changed points {row['smoothed_sign_changes']} | "
            f"R-B-R {row['immediate_rbr_before']} -> {row['immediate_rbr_after']}"
        )
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(stats)}</p>"
            f"<img src='{html.escape(image_rel)}' alt='{html.escape(title)}'>"
            "</section>"
        )

    gallery_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fzong boundary feature comparisons</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #202124; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ margin: 0 0 22px; color: #5f6368; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 560px), 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #dadce0; border-radius: 6px; padding: 12px; background: #fff; }}
    .card h2 {{ margin: 0 0 6px; font-size: 15px; }}
    .card p {{ margin: 0 0 10px; font-size: 13px; color: #5f6368; }}
    img {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>Fzong boundary feature comparisons</h1>
  <p class="meta">Each image compares raw signs with centered median smoothing.</p>
  <main class="grid">
{"".join(cards)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def run(
    raw_dir: Path,
    out_dir: Path,
    *,
    window_size: int = 3,
    make_plots: bool = True,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or FeatureConfig(window_size=window_size)
    configure_fonts()
    paths = [path for path in sorted(raw_dir.glob("*Fzong*.xlsx")) if not path.name.startswith("~$")]
    if not paths:
        raise FileNotFoundError(f"No *Fzong*.xlsx point-data files found in {raw_dir}")

    feature_records: list[dict[str, object]] = []
    point_records: list[pd.DataFrame] = []
    file_audit = []
    for path in paths:
        source_hash = file_sha256(path)
        source = pd.read_excel(path, engine="openpyxl")
        source = source.drop(
            columns=[column for column in source.columns if str(column).startswith("Unnamed")], errors="ignore"
        )
        columns = resolve_source_columns(source)
        try:
            clean, quality = checked_source_rows(source, columns, config)
        except ValueError as error:
            raise ValueError(f"{path.name}: {error}") from error
        file_audit.append({"source_file": path.name, "source_sha256": source_hash, **quality})
        coordinate_count = len(clean[["x", "z"]].drop_duplicates())

        for (radius, re_value), sample in clean.groupby(["R", "re"], sort=True):
            feature, points = extract_focus_boundary_feature(sample, window_size=config.window_size)
            feature["relative_coordinate_coverage"] = len(points) / coordinate_count
            feature["dropped_input_rows"] = quality["dropped_rows"]
            feature["quality_flags"] = ";".join(quality_reasons(feature))
            image_path = (
                out_dir
                / "figures"
                / slugify(path.name)
                / f"R{fmt_number(float(radius))}"
                / f"{slugify(path.name)}__R{fmt_number(float(radius))}__Re{fmt_number(float(re_value))}.png"
            )
            if make_plots:
                draw_comparison(
                    points,
                    feature,
                    source_file=path.name,
                    radius=float(radius),
                    re_value=float(re_value),
                    out_path=image_path,
                )
            feature_records.append(
                {
                    "source_file": path.name,
                    "source_sha256": source_hash,
                    "R": float(radius),
                    "re": float(re_value),
                    "target_col": columns.target,
                    "source_length_unit": config.length_unit,
                    "coordinate_unit": "um",
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    **feature,
                    "comparison_image": str(image_path) if make_plots else "",
                }
            )
            detail = points.copy()
            detail.insert(0, "source_file", path.name)
            detail.insert(1, "R", float(radius))
            detail.insert(2, "re", float(re_value))
            point_records.append(detail)
        if file_sha256(path) != source_hash:
            raise ValueError(f"{path.name} changed while it was being read; rerun with stable inputs")

    features = pd.DataFrame(feature_records).sort_values(["source_file", "R", "re"]).reset_index(drop=True)
    point_audit = pd.concat(point_records, ignore_index=True).sort_values(["source_file", "R", "re", "z", "x"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "features.csv", index=False, encoding="utf-8-sig")
    point_audit.to_csv(out_dir / "smoothed_points.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(file_audit).to_csv(out_dir / "input_quality.csv", index=False, encoding="utf-8-sig")
    features.loc[features["quality_flags"].ne(""), ["source_file", "R", "re", "quality_flags"]].to_csv(
        out_dir / "quality_warnings.csv", index=False, encoding="utf-8-sig"
    )
    if make_plots:
        build_gallery(features, out_dir / "gallery.html")

    summary = {
        "raw_dir": str(raw_dir.resolve()),
        "sample_key": ["source_file", "R", "re"],
        "sample_count": int(len(features)),
        "figures_generated": make_plots,
        "source_file_count": int(features["source_file"].nunique()),
        "smoothing_method": f"centered_median_{config.window_size}",
        "feature_config": config.to_dict(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_code_hashes": feature_code_hashes(),
        "quality_warning_samples": int(features["quality_flags"].ne("").sum()),
        "dropped_input_rows": sum(row["dropped_rows"] for row in file_audit),
        "samples_with_changed_signs": int((features["smoothed_sign_changes"] > 0).sum()),
        "changed_coordinate_signs": int(features["smoothed_sign_changes"].sum()),
        "samples_with_changed_feature": int(
            (
                ~np.isclose(features["boundary_distance_abs_raw"], features["boundary_distance_abs"], equal_nan=True)
            ).sum()
        ),
        "samples_without_both_signs": int((~features["has_both_signs"]).sum()),
        "immediate_rbr_before": int(features["immediate_rbr_before"].sum()),
        "immediate_rbr_after": int(features["immediate_rbr_after"].sum()),
        "broad_rbr_rows_before": int(features["broad_rbr_rows_before"].sum()),
        "broad_rbr_rows_after": int(features["broad_rbr_rows_after"].sum()),
    }
    write_json(out_dir / "run_summary.json", summary)
    return features, point_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build smoothed Fzong boundary features by Excel, R, and Re.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing raw .xlsx files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--window-size", type=int, default=3, help="Odd centered median window size; default: 3.")
    parser.add_argument("--no-plots", action="store_true", help="Extract training features without rendering figures.")
    parser.add_argument(
        "--length-unit",
        choices=("um", "mm", "m"),
        default="um",
        help="Source x/z and geometry length unit; R retains its source convention",
    )
    parser.add_argument("--invalid-rows", choices=("error", "drop"), default="error")
    args = parser.parse_args()

    from utils.runs import experiment_run

    with experiment_run(args.out_dir, "focus-features") as out_dir:
        args.out_dir = out_dir
        features, point_audit = run(
            args.raw_dir,
            out_dir,
            make_plots=not args.no_plots,
            config=FeatureConfig(args.window_size, args.length_unit, args.invalid_rows),
        )
    print(f"Saved {len(features)} sample features to {args.out_dir / 'features.csv'}")
    print(f"Saved {len(point_audit)} coordinate audit rows to {args.out_dir / 'smoothed_points.csv'}")
    if not args.no_plots:
        print(f"Saved comparison gallery to {args.out_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
