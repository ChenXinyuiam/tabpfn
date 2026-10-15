"""Plot Fzong sign distributions by source Excel, curvature radius, and Re.

Sample unit:
    one source Excel file + one curvature radius R + one Reynolds number re

Each output image is one triangle point cloud. Points with positive Fzong are
red, negative Fzong are blue, and exact zeros are gray. Raw rows are preserved
by default; exact duplicate x/z coordinates are slightly spread in display
coordinates so overplotted rows remain visible.
"""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .focus_boundary_feature import (
    SourceColumns as Columns,  # noqa: F401 - compatibility export
)
from .focus_boundary_feature import (
    _find_column as find_column,  # noqa: F401 - compatibility export
)
from .focus_boundary_feature import (
    collapse_duplicate_coordinates,
)
from .focus_boundary_feature import (
    numeric_source_rows as numeric_rows,
)
from .focus_boundary_feature import (
    resolve_source_columns as resolve_columns,
)
from .plotting import configure_fonts, plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = ROOT / "data" / "rawdata" / "2026-09-04"
DEFAULT_OUT_DIR = ROOT / "results" / "focus_inspection" / "2026-09-04" / "fzong_sign_triangles_by_re"

RED = "#d62728"
BLUE = "#1f77b4"
GRAY = "#777777"
HULL = "#222222"


def force_sign(values: pd.Series) -> np.ndarray:
    return np.sign(values.to_numpy(dtype=float)).astype(int)


def aggregate_duplicate_points(sample: pd.DataFrame) -> pd.DataFrame:
    grouped = collapse_duplicate_coordinates(sample).rename(
        columns={
            "force_raw": "force",
            "sign_raw": "sign",
            "positive_source_rows": "positive_rows",
            "negative_source_rows": "negative_rows",
            "zero_source_rows": "zero_rows",
            "duplicate_sign_conflict": "mixed_duplicate_sign",
        }
    )
    grouped["duplicate_group_size"] = grouped["source_rows"]
    grouped["display_spread"] = False
    grouped["x_plot"] = grouped["x"]
    grouped["z_plot"] = grouped["z"]
    return grouped


def points_for_plot(sample: pd.DataFrame, duplicate_mode: str) -> pd.DataFrame:
    if duplicate_mode == "mean":
        return aggregate_duplicate_points(sample)
    if duplicate_mode == "raw":
        out = sample.copy()
        out["sign"] = force_sign(out["force"])
        out["source_rows"] = 1
        out["positive_rows"] = (out["force"] > 0).astype(int)
        out["negative_rows"] = (out["force"] < 0).astype(int)
        out["zero_rows"] = (out["force"] == 0).astype(int)
        out["mixed_duplicate_sign"] = False
        out["duplicate_group_size"] = out.groupby(["x", "z"])["force"].transform("size").astype(int)
        out = add_display_coordinates(out)
        return out
    raise ValueError(f"Unsupported duplicate mode: {duplicate_mode}")


def add_display_coordinates(points: pd.DataFrame) -> pd.DataFrame:
    out = points.copy()
    out["x_plot"] = out["x"].astype(float)
    out["z_plot"] = out["z"].astype(float)
    out["display_spread"] = False

    duplicate_mask = out.duplicated(["x", "z"], keep=False)
    if not duplicate_mask.any():
        return out

    spacing_values: list[float] = []
    for col in ["x", "z"]:
        unique_values = np.sort(out[col].dropna().unique().astype(float))
        diffs = np.diff(unique_values)
        spacing_values.extend([float(v) for v in diffs if v > 0])
    spread_radius = 0.1 * min(spacing_values) if spacing_values else 0.5

    for _, idx in out[duplicate_mask].groupby(["x", "z"], sort=False).groups.items():
        indices = list(idx)
        n = len(indices)
        if n <= 1:
            continue
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for row_idx, angle in zip(indices, angles, strict=True):
            out.at[row_idx, "x_plot"] = float(out.at[row_idx, "x"]) + spread_radius * math.cos(float(angle))
            out.at[row_idx, "z_plot"] = float(out.at[row_idx, "z"]) + spread_radius * math.sin(float(angle))
            out.at[row_idx, "display_spread"] = True
    return out


def slugify(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", stem, flags=re.UNICODE)
    return stem.strip("_") or "sample"


def monotonic_chain(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=float)

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def fmt_number(value: float) -> str:
    if math.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def draw_sample(
    plot_df: pd.DataFrame,
    *,
    source_file: str,
    radius: float,
    re_value: float,
    target_col: str,
    raw_rows: int,
    out_path: Path,
) -> dict[str, object]:
    pos = plot_df[plot_df["sign"] > 0]
    neg = plot_df[plot_df["sign"] < 0]
    zero = plot_df[plot_df["sign"] == 0]
    mixed = plot_df[plot_df["mixed_duplicate_sign"]]

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    if len(neg):
        ax.scatter(neg["x_plot"], neg["z_plot"], c=BLUE, s=28, alpha=0.86, edgecolors="none", label="Fzong < 0")
    if len(pos):
        ax.scatter(pos["x_plot"], pos["z_plot"], c=RED, s=28, alpha=0.86, edgecolors="none", label="Fzong > 0")
    if len(zero):
        ax.scatter(zero["x_plot"], zero["z_plot"], c=GRAY, s=28, alpha=0.86, edgecolors="none", label="Fzong = 0")
    if len(mixed):
        ax.scatter(
            mixed["x_plot"],
            mixed["z_plot"],
            facecolors="none",
            edgecolors="#111111",
            s=58,
            linewidths=0.75,
            label="mixed duplicate sign",
        )

    spread_rows = int(plot_df["display_spread"].sum()) if "display_spread" in plot_df.columns else 0
    duplicate_groups = (
        int((plot_df["duplicate_group_size"] > 1).sum()) if "duplicate_group_size" in plot_df.columns else 0
    )

    points = plot_df[["x", "z"]].drop_duplicates().dropna().to_numpy(dtype=float)
    if len(points) >= 3:
        hull = monotonic_chain(points)
        if len(hull) >= 3:
            hull_closed = np.vstack([hull, hull[0]])
            ax.plot(hull_closed[:, 0], hull_closed[:, 1], color=HULL, linewidth=1.0, alpha=0.55)

    subtitle = (
        f"rows={raw_rows}, points={len(plot_df)}, "
        f"red={len(pos)}, blue={len(neg)}, zero={len(zero)}, "
        f"spread_rows={spread_rows}, duplicate_points={duplicate_groups}"
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(
        f"{source_file} | R={fmt_number(radius)} | Re={fmt_number(re_value)} | {target_col} sign\n{subtitle}",
        fontsize=11,
    )
    ax.grid(True, linewidth=0.35, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "source_file": source_file,
        "R": radius,
        "re": re_value,
        "target_col": target_col,
        "raw_rows": raw_rows,
        "plotted_points": int(len(plot_df)),
        "positive_points": int(len(pos)),
        "negative_points": int(len(neg)),
        "zero_points": int(len(zero)),
        "mixed_duplicate_points": int(len(mixed)),
        "display_spread_rows": spread_rows,
        "duplicate_coordinate_rows": int((plot_df["duplicate_group_size"] > 1).sum())
        if "duplicate_group_size" in plot_df.columns
        else 0,
        "duplicate_coordinate_groups": int(
            plot_df.loc[plot_df["duplicate_group_size"] > 1, ["x", "z"]].drop_duplicates().shape[0]
        )
        if "duplicate_group_size" in plot_df.columns
        else 0,
        "image_path": str(out_path),
    }


def build_gallery(summary: pd.DataFrame, gallery_path: Path, figure_root: Path) -> None:
    cards: list[str] = []
    for row in summary.sort_values(["source_file", "R", "re"]).to_dict(orient="records"):
        image_rel = Path(row["image_path"]).resolve().relative_to(gallery_path.parent.resolve()).as_posix()
        title = f"{row['source_file']} | R={fmt_number(float(row['R']))} | Re={fmt_number(float(row['re']))}"
        stats = (
            f"points {row['plotted_points']} | red {row['positive_points']} | "
            f"blue {row['negative_points']} | spread rows {row['display_spread_rows']}"
        )
        cards.append(
            "<section class='card'>"
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(stats)}</p>"
            f"<img src='{html.escape(image_rel)}' alt='{html.escape(title)}'>"
            "</section>"
        )

    body = "\n".join(cards)
    gallery_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fzong sign triangle gallery by Re</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #202124; }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ margin: 0 0 22px; color: #5f6368; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid #dadce0; border-radius: 6px; padding: 12px; background: #fff; }}
    .card h2 {{ margin: 0 0 6px; font-size: 15px; }}
    .card p {{ margin: 0 0 10px; font-size: 13px; color: #5f6368; }}
    img {{ width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
  <h1>Fzong sign triangle gallery by Re</h1>
  <p class="meta">Figure folder: {html.escape(str(figure_root))}</p>
  <main class="grid">
{body}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def run(raw_dir: Path, out_dir: Path, duplicate_mode: str) -> pd.DataFrame:
    configure_fonts()
    paths = [p for p in sorted(raw_dir.glob("*.xlsx")) if not p.name.startswith("~$")]
    if not paths:
        raise FileNotFoundError(f"No .xlsx files found in {raw_dir}")

    figure_dir = out_dir / "figures"
    records: list[dict[str, object]] = []
    for path in paths:
        df = pd.read_excel(path, engine="openpyxl")
        df = df.drop(columns=[col for col in df.columns if str(col).startswith("Unnamed")], errors="ignore")
        columns = resolve_columns(df)
        clean = numeric_rows(df, columns)
        if clean.empty:
            raise ValueError(f"{path.name} has no valid numeric x/z/R/re/{columns.target} rows")

        for (radius, re_value), sample in clean.groupby(["R", "re"], sort=True):
            plot_df = points_for_plot(sample, duplicate_mode=duplicate_mode)
            image_path = (
                figure_dir
                / slugify(path.name)
                / f"R{fmt_number(float(radius))}"
                / f"{slugify(path.name)}__R{fmt_number(float(radius))}__Re{fmt_number(float(re_value))}.png"
            )
            records.append(
                draw_sample(
                    plot_df,
                    source_file=path.name,
                    radius=float(radius),
                    re_value=float(re_value),
                    target_col=columns.target,
                    raw_rows=len(sample),
                    out_path=image_path,
                )
            )

    summary = pd.DataFrame(records).sort_values(["source_file", "R", "re"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    build_gallery(summary, out_dir / "gallery.html", figure_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Fzong sign triangle images by Excel, R, and Re.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing raw .xlsx files.")
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for figures and summary."
    )
    parser.add_argument(
        "--duplicate-mode",
        choices=["mean", "raw"],
        default="raw",
        help="mean: merge duplicate x/z rows within the same Excel/R/Re; raw: plot every row.",
    )
    args = parser.parse_args()

    summary = run(args.raw_dir, args.out_dir, args.duplicate_mode)
    print(f"Saved {len(summary)} sample figures to {args.out_dir / 'figures'}")
    print(f"Saved summary to {args.out_dir / 'summary.csv'}")
    print(f"Saved gallery to {args.out_dir / 'gallery.html'}")


if __name__ == "__main__":
    main()
