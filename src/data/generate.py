"""
Generate a complete dense (X, Z) grid by 2-D interpolation within each cleaned file.

Root cause of the previous blank-checkerboard heatmaps
-------------------------------------------------------
  The old code only generated midpoint (X, Z) pairs — the original grid positions
  were absent from the output file.  When pivoted into a heatmap the alternating
  original/midpoint columns and rows produced a classic checkerboard of blanks.

Fixed strategy
--------------
  1. Build the FULL dense grid: X_full = orig_X ∪ X_mid, Z_full = orig_Z ∪ Z_mid.
  2. Create a complete meshgrid over X_full × Z_full and filter to points inside
     the convex hull of the original data.
  3. Interpolate FL (per col[3] level) and all geometric columns at every point
     in the dense grid.
  4. For positions that already existed in the original cleaned data, substitute
     the exact original values back — no interpolation error at known points.
  5. Mark each row: generated=False (original) or generated=True (new midpoint).

Output
------
  data/processed/2026-04-22/{label}_generated.xlsx
  Contains the full dense grid: originals + all new interpolated midpoints.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, QhullError

from src.config import PUBLISHED_PROCESSED_ROOT, BASE_COL, RE_COL
from utils.paths import select_batch_dir

# ── helpers ──────────────────────────────────────────────────────────────────

def _detect_spacing(vals: np.ndarray) -> float:
    """Return the smallest gap between sorted unique values."""
    u = np.sort(np.unique(vals))
    if len(u) < 2:
        return float("nan")
    return float(np.min(np.diff(u)))


def _midpoint_grid(vals: np.ndarray, spacing: float) -> np.ndarray:
    """Return midpoints between consecutive unique values."""
    u = np.sort(np.unique(vals))
    return u[:-1] + spacing / 2.0


def _in_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """Boolean mask: which rows of `points` lie inside the convex hull."""
    try:
        hull = ConvexHull(hull_points)
    except Exception:
        return np.ones(len(points), dtype=bool)
    # A point p is inside if A @ p + b <= 0 for all half-spaces
    A, b = hull.equations[:, :-1], hull.equations[:, -1]
    inside = np.all((points @ A.T + b) <= 1e-10, axis=1)
    return inside


def interpolate_column(
    src_XZ: np.ndarray,
    src_vals: np.ndarray,
    tgt_XZ: np.ndarray,
    method: str = "cubic",
) -> np.ndarray:
    """Interpolate src_vals at tgt_XZ; fall back to linear then nearest."""
    if len(src_XZ) == 0:
        return np.full(len(tgt_XZ), np.nan)

    unique_points, inverse = np.unique(src_XZ, axis=0, return_inverse=True)
    if len(unique_points) != len(src_XZ):
        reduced_vals = np.zeros(len(unique_points), dtype=float)
        counts = np.zeros(len(unique_points), dtype=int)
        for idx, inv_idx in enumerate(inverse):
            reduced_vals[inv_idx] += float(src_vals[idx])
            counts[inv_idx] += 1
        src_XZ = unique_points
        src_vals = reduced_vals / np.clip(counts, 1, None)

    result = np.full(len(tgt_XZ), np.nan, dtype=float)
    methods = [method, "linear", "nearest"]
    for current_method in methods:
        try:
            missing = np.isnan(result)
            if not missing.any():
                break
            target = tgt_XZ if current_method == method else tgt_XZ[missing]
            current = griddata(src_XZ, src_vals, target, method=current_method)
            if current_method == method:
                result = current.astype(float, copy=False)
            else:
                result[missing] = current
        except QhullError:
            continue
    return result


# ── per-file generation ──────────────────────────────────────────────────────

def input_label(path: Path) -> str:
    return path.stem.removesuffix("_cleaned")


def is_source_file(path: Path) -> bool:
    return (
        path.suffix.lower() == ".xlsx"
        and not path.name.startswith("~$")
        and not path.name.startswith("_")
        and "_generated" not in path.stem
    )


def generate_for_file(path: Path) -> pd.DataFrame | None:
    label = input_label(path)
    df = pd.read_excel(path)

    cols = df.columns.tolist()
    col_id = BASE_COL if BASE_COL in df.columns else cols[0]
    col_sp = RE_COL if RE_COL in df.columns else cols[3]

    # ── detect grid spacing ───────────────────────────────────────────────
    dx = _detect_spacing(df["X"].values)
    dz = _detect_spacing(df["Z"].values)
    print(f"  [{label}]  X-step={dx:.1f}  Z-step={dz:.1f}")

    X_mid = _midpoint_grid(df["X"].values, dx)
    Z_mid = _midpoint_grid(df["Z"].values, dz)

    if len(X_mid) == 0 or len(Z_mid) == 0:
        print(f"  [{label}] WARNING: too few grid points, skipping.")
        return None

    # ── build FULL dense grid: orig ∪ midpoints in both axes ─────────────
    X_orig = np.sort(np.unique(df["X"].values)).astype(float)
    Z_orig = np.sort(np.unique(df["Z"].values)).astype(float)
    X_full = np.sort(np.unique(np.concatenate([X_orig, X_mid])))
    Z_full = np.sort(np.unique(np.concatenate([Z_orig, Z_mid])))

    grid_X, grid_Z = np.meshgrid(X_full, Z_full)
    cand_XZ = np.column_stack([grid_X.ravel(), grid_Z.ravel()])

    # ── filter: keep only points inside the original convex hull ─────────
    orig_XZ = df[["X", "Z"]].drop_duplicates().values.astype(float)
    in_hull_mask = _in_hull(cand_XZ, orig_XZ)
    new_XZ = cand_XZ[in_hull_mask]

    if len(new_XZ) == 0:
        print(f"  [{label}] No new points inside convex hull.")
        return None

    # build a set of original (X,Z) tuples for fast lookup
    orig_XZ_set = {(float(r[0]), float(r[1])) for r in orig_XZ}
    is_new = np.array([
        (float(p[0]), float(p[1])) not in orig_XZ_set for p in new_XZ
    ])
    print(f"  [{label}]  total dense-grid pts inside hull: {len(new_XZ)} "
          f"({is_new.sum()} new midpoints, {(~is_new).sum()} original)")

    # ── identify columns by role ──────────────────────────────────────────
    lambda_cols = [c for c in cols if str(c).startswith("lambda")]
    # Float columns that vary only with (X,Z) — identify by checking
    # variance across col_sp groups at the same (X,Z)
    geom_float_cols = []
    for c in cols:
        if c in ["X", "Z", "FL", col_id, col_sp, "geometry_valid"] + lambda_cols:
            continue
        if df[c].dtype not in [np.float64, float]:
            continue
        # check if variance across col_sp levels at same (X,Z) is negligible
        var = df.groupby(["X", "Z"])[c].std().max()
        if var < 1e-9 or np.isnan(var):
            geom_float_cols.append(c)

    # Constant integer columns (same value throughout the file)
    const_int_cols = {}
    for c in cols:
        if c in ["X", "Z", "FL", col_id, col_sp, "geometry_valid"] + lambda_cols + geom_float_cols:
            continue
        if df[c].nunique() == 1:
            const_int_cols[c] = df[c].iloc[0]

    print(f"  [{label}]  geom_float_cols : {geom_float_cols}")
    print(f"  [{label}]  const_int_cols  : {list(const_int_cols.keys())}")
    print(f"  [{label}]  lambda_cols     : {lambda_cols}")

    # ── interpolate geometry columns (once, independent of col_sp) ────────
    # Use the first col_sp level for lambda/geom (identical across levels)
    ref_rows = df.groupby(["X", "Z"]).first().reset_index()
    ref_XZ   = ref_rows[["X", "Z"]].values.astype(float)

    # Interpolated values at all dense-grid points
    interp_geom: dict[str, np.ndarray] = {}
    for c in geom_float_cols + lambda_cols:
        interp_geom[c] = interpolate_column(ref_XZ, ref_rows[c].values.astype(float), new_XZ)

    # Build a lookup: (X,Z) → exact original geom values
    orig_geom_lookup: dict[tuple, dict] = {}
    for _, row in ref_rows.iterrows():
        key = (float(row["X"]), float(row["Z"]))
        orig_geom_lookup[key] = {c: row[c] for c in geom_float_cols + lambda_cols}

    # Restore exact original geometry values where available
    for i, pt in enumerate(new_XZ):
        key = (float(pt[0]), float(pt[1]))
        if key in orig_geom_lookup:
            for c in geom_float_cols + lambda_cols:
                interp_geom[c][i] = orig_geom_lookup[key][c]

    # ── interpolate FL per col_sp level; restore originals exactly ────────
    sp_levels = sorted(df[col_sp].unique())
    rows_list = []

    for sp in sp_levels:
        sub = df[df[col_sp] == sp]
        sub_XZ = sub[["X", "Z"]].values.astype(float)
        fl_interp = interpolate_column(sub_XZ, sub["FL"].values.astype(float), new_XZ)

        # Build exact FL lookup for this sp level
        orig_fl_lookup = {
            (float(r["X"]), float(r["Z"])): float(r["FL"])
            for _, r in sub.iterrows()
        }
        for i, pt in enumerate(new_XZ):
            key = (float(pt[0]), float(pt[1]))
            if key in orig_fl_lookup:
                fl_interp[i] = orig_fl_lookup[key]

        chunk = pd.DataFrame({
            col_id: df[col_id].iloc[0],
            "X":    new_XZ[:, 0].astype(int),
            "Z":    new_XZ[:, 1],
            col_sp: sp,
        })
        for c, v in const_int_cols.items():
            chunk[c] = v
        for c in geom_float_cols:
            chunk[c] = interp_geom[c]
        chunk["FL"] = fl_interp
        for c in lambda_cols:
            chunk[c] = interp_geom[c]
        chunk["geometry_valid"] = True
        chunk["generated"] = ~is_new   # False = original, True = new midpoint
        # flip: generated=True means it IS a new point
        chunk["generated"] = is_new

        rows_list.append(chunk)

    gen_df = pd.concat(rows_list, ignore_index=True)
    keep_cols = [c for c in cols if c in gen_df.columns] + ["generated"]
    gen_df = gen_df[keep_cols]

    n_orig_rows = int((~is_new).sum()) * len(sp_levels)
    n_new_rows  = int(is_new.sum())    * len(sp_levels)
    print(f"  [{label}]  total rows: {len(gen_df)} "
          f"({n_orig_rows} original + {n_new_rows} new midpoints, "
          f"× {len(sp_levels)} Re levels)\n")
    return gen_df

def main():
    parser = argparse.ArgumentParser(description="Generate dense interpolated data for one processed batch.")
    parser.add_argument("--batch", default=None, help="Batch under data/processed. Defaults to the latest batch.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Explicit processed data directory.")
    parser.add_argument("--pattern", default="*.xlsx")
    args = parser.parse_args()

    data_dir = args.data_dir or select_batch_dir(PUBLISHED_PROCESSED_ROOT, args.batch)
    input_files = sorted(path for path in data_dir.glob(args.pattern) if is_source_file(path))
    if not input_files:
        print(f"No source .xlsx files found in {data_dir}.")
        return

    print(f"Generating dense data for batch {data_dir.name}: {len(input_files)} file(s).\n")

    for path in input_files:
        label = input_label(path)
        gen_df = generate_for_file(path)
        if gen_df is None:
            continue

        out_path = data_dir / f"{label}_generated.xlsx"
        gen_df.to_excel(out_path, index=False)
        print(f"  [{label}] saved -> {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
