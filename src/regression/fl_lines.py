"""Interpolate the two reference reverse-FL lines at matching FD locations."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, QhullError
from sklearn.neighbors import NearestNeighbors

DEFAULT_OUTPUT_DIR = Path("results/fl_reverse_fd_point_predictions")


@dataclass(frozen=True)
class LineJob:
    label: str
    fl_path: Path
    fd_path: Path


DEFAULT_JOBS = [
    LineJob(
        label="750-400反",
        fl_path=Path("data/rawdata/2026-05-01/750-400-改-反-FL.xlsx"),
        fd_path=Path("data/rawdata/2026-05-25/FD数据/10/750-400反.xlsx"),
    ),
    LineJob(
        label="750-530反",
        fl_path=Path("data/rawdata/2026-05-01/750-530-改-反-FL.xlsx"),
        fd_path=Path("data/rawdata/2026-05-25/FD数据/10/750-530反.xlsx"),
    ),
]


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def reduce_duplicate_points(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
    if len(unique_points) == len(points):
        return points, values

    reduced = np.zeros(len(unique_points), dtype=float)
    counts = np.zeros(len(unique_points), dtype=int)
    for i, group in enumerate(inverse):
        if np.isfinite(values[i]):
            reduced[group] += float(values[i])
            counts[group] += 1
    valid = counts > 0
    return unique_points[valid], reduced[valid] / counts[valid]


def interpolate_with_method(
    source_points: np.ndarray,
    source_values: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source_points, source_values = reduce_duplicate_points(source_points, source_values)
    preds = np.full(len(target_points), np.nan, dtype=float)
    methods = np.full(len(target_points), "missing", dtype=object)

    for method in ["cubic", "linear", "nearest"]:
        missing = np.isnan(preds)
        if not missing.any():
            break
        if method in {"cubic", "linear"} and len(source_points) < 4:
            continue
        try:
            current = griddata(source_points, source_values, target_points[missing], method=method)
        except QhullError:
            continue
        except ValueError:
            continue

        current = np.asarray(current, dtype=float)
        fillable = np.isfinite(current)
        if not fillable.any():
            continue
        missing_indices = np.flatnonzero(missing)
        fill_indices = missing_indices[fillable]
        preds[fill_indices] = current[fillable]
        methods[fill_indices] = method

    return preds, methods


def inside_convex_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    try:
        hull = ConvexHull(hull_points)
    except Exception:
        return np.ones(len(points), dtype=bool)
    a, b = hull.equations[:, :-1], hull.equations[:, -1]
    return np.all((points @ a.T + b) <= 1e-12, axis=1)


def midpoint_axis(values: pd.Series) -> np.ndarray:
    unique = np.sort(pd.to_numeric(values, errors="coerce").dropna().unique().astype(float))
    if len(unique) < 2:
        return unique
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.sort(np.unique(np.concatenate([unique, midpoints])))


def read_fl(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    required = ["X", "Z", "雷诺数", "FL"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required FL columns: {missing}")
    df = numeric_frame(df, required)
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def read_fd(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    required = ["Re", "X [ m ]", "Z [ m ]"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"{path} is missing required FD columns: {missing}")
    df = numeric_frame(df, required)
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def build_dense_fl(fl_df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    original_points = fl_df[["X", "Z"]].drop_duplicates().to_numpy(dtype=float)
    x_full = midpoint_axis(fl_df["X"])
    z_full = midpoint_axis(fl_df["Z"])
    grid_x, grid_z = np.meshgrid(x_full, z_full)
    candidate_points = np.column_stack([grid_x.ravel(), grid_z.ravel()])
    target_points = candidate_points[inside_convex_hull(candidate_points, original_points)]

    original_key = {
        (round(float(row.X), 15), round(float(row.Z), 15), float(row.雷诺数)): float(row.FL)
        for row in fl_df[["X", "Z", "雷诺数", "FL"]].itertuples(index=False)
    }

    rows: list[pd.DataFrame] = []
    for re_value in sorted(fl_df["雷诺数"].dropna().unique()):
        subset = fl_df[fl_df["雷诺数"] == re_value]
        source_points = subset[["X", "Z"]].to_numpy(dtype=float)
        source_values = subset["FL"].to_numpy(dtype=float)
        preds, methods = interpolate_with_method(source_points, source_values, target_points)

        generated = np.ones(len(target_points), dtype=bool)
        for i, point in enumerate(target_points):
            key = (round(float(point[0]), 15), round(float(point[1]), 15), float(re_value))
            if key in original_key:
                preds[i] = original_key[key]
                methods[i] = "original"
                generated[i] = False

        rows.append(
            pd.DataFrame(
                {
                    "source_file": source_name,
                    "雷诺数": re_value,
                    "X": target_points[:, 0],
                    "Z": target_points[:, 1],
                    "FL": preds,
                    "generated": generated,
                    "FL_interp_method": methods,
                }
            )
        )

    return pd.concat(rows, ignore_index=True)


def predict_fd_points(fd_df: pd.DataFrame, dense_fl: pd.DataFrame, fl_source_name: str) -> pd.DataFrame:
    out = fd_df.copy()
    out["FL_pred"] = np.nan
    out["FL_pred_method"] = "missing_re"
    out["FL_nearest_distance_m"] = np.nan
    out["FL_nearest_distance_um"] = np.nan
    out["FL_source_file"] = fl_source_name

    for re_value in sorted(out["Re"].dropna().unique()):
        fd_mask = out["Re"] == re_value
        fl_subset = dense_fl[dense_fl["雷诺数"] == re_value].dropna(subset=["X", "Z", "FL"])
        if fl_subset.empty:
            continue

        source_points = fl_subset[["X", "Z"]].to_numpy(dtype=float)
        source_values = fl_subset["FL"].to_numpy(dtype=float)
        target_points = out.loc[fd_mask, ["X [ m ]", "Z [ m ]"]].to_numpy(dtype=float)
        preds, methods = interpolate_with_method(source_points, source_values, target_points)

        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(source_points)
        distances, _ = nn.kneighbors(target_points)

        out.loc[fd_mask, "FL_pred"] = preds
        out.loc[fd_mask, "FL_pred_method"] = methods
        out.loc[fd_mask, "FL_nearest_distance_m"] = distances[:, 0]
        out.loc[fd_mask, "FL_nearest_distance_um"] = distances[:, 0] * 1e6

    return out


def run_job(job: LineJob, output_dir: Path) -> dict[str, object]:
    fl_df = read_fl(job.fl_path)
    fd_df = read_fd(job.fd_path)

    dense_fl = build_dense_fl(fl_df, job.fl_path.name)
    predicted_fd = predict_fd_points(fd_df, dense_fl, job.fl_path.name)

    dense_path = output_dir / f"{job.label}_FL_interpolated.xlsx"
    pred_path = output_dir / f"{job.label}_FD_with_FL_pred.xlsx"
    dense_fl.to_excel(dense_path, index=False, engine="openpyxl")
    predicted_fd.to_excel(pred_path, index=False, engine="openpyxl")

    method_counts = predicted_fd["FL_pred_method"].value_counts(dropna=False).to_dict()
    return {
        "label": job.label,
        "fl_file": job.fl_path.as_posix(),
        "fd_file": job.fd_path.as_posix(),
        "interpolated_file": dense_path.as_posix(),
        "prediction_file": pred_path.as_posix(),
        "fl_rows_in": int(len(fl_df)),
        "fl_rows_interpolated": int(len(dense_fl)),
        "fd_rows": int(len(predicted_fd)),
        "fd_rows_predicted": int(predicted_fd["FL_pred"].notna().sum()),
        "fd_prediction_methods": method_counts,
        "nearest_distance_um_mean": float(predicted_fd["FL_nearest_distance_um"].mean()),
        "nearest_distance_um_p95": float(predicted_fd["FL_nearest_distance_um"].quantile(0.95)),
        "nearest_distance_um_max": float(predicted_fd["FL_nearest_distance_um"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FL at FD locations from two reverse-FL line files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [run_job(job, args.output_dir) for job in DEFAULT_JOBS]

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
