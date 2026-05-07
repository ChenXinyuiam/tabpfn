"""
Per-Re generation: each Re level is treated as an independent spatial dataset.

Key difference from generate_data.py
-------------------------------------
  Previous approach : all Re levels share one generation pass; FL is interpolated
                      per Re using the shared (X,Z) grid of the whole file.
  This approach     : for EACH Re level the dataset is isolated first (only rows
                      with that Re).  The dense (X,Z) grid, the convex hull, and
                      the interpolant are all built exclusively from that Re
                      level's data.  Data from other Re values is strictly banned.

  Consequence: slightly different convex-hull boundaries per Re (confirmed above —
  some Re levels are genuinely missing certain (X,Z) points).

Output
------
  data/processed/2026-04-22/{label}_generated_per_re.xlsx
  Columns: same as cleaned file + 'generated' (True = new midpoint, False = original)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull

from src.config import PUBLISHED_PROCESSED_ROOT, BASE_COL, RE_COL
from utils.paths import select_batch_dir

# ── helpers (identical to generate_data.py) ──────────────────────────────────

def _detect_spacing(vals: np.ndarray) -> float:
    u = np.sort(np.unique(vals))
    return float(np.min(np.diff(u))) if len(u) >= 2 else float("nan")

def _midpoint_grid(vals: np.ndarray, spacing: float) -> np.ndarray:
    u = np.sort(np.unique(vals))
    return u[:-1] + spacing / 2.0

def _in_hull(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    try:
        hull = ConvexHull(hull_points)
    except Exception:
        return np.ones(len(points), dtype=bool)
    A, b = hull.equations[:, :-1], hull.equations[:, -1]
    return np.all((points @ A.T + b) <= 1e-10, axis=1)

def _interpolate(src_XZ, src_vals, tgt_XZ):
    """Cubic → linear → nearest fallback interpolation."""
    result = griddata(src_XZ, src_vals, tgt_XZ, method="cubic")
    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = griddata(src_XZ, src_vals, tgt_XZ[nan_mask], method="linear")
    nan_mask2 = np.isnan(result)
    if nan_mask2.any():
        result[nan_mask2] = griddata(src_XZ, src_vals, tgt_XZ[nan_mask2], method="nearest")
    return result

# ── per-Re generation for one file ───────────────────────────────────────────

def input_label(path_or_name: str | Path) -> str:
    return Path(path_or_name).stem.removesuffix("_cleaned")

def is_source_file(path: Path) -> bool:
    return (
        path.suffix.lower() == ".xlsx"
        and not path.name.startswith("~$")
        and not path.name.startswith("_")
        and "_generated" not in path.stem
    )

def generate_per_re(src: Path) -> pd.DataFrame | None:
    label = input_label(src)
    df    = pd.read_excel(src)
    cols  = df.columns.tolist()

    col_id = BASE_COL if BASE_COL in df.columns else cols[0]
    col_sp = RE_COL if RE_COL in df.columns else cols[3]

    sp_levels = sorted(df[col_sp].unique())

    # ── column roles (shared across Re levels) ────────────────────────────
    lambda_cols = [c for c in cols if str(c).startswith("lambda")]

    geom_float_cols = []
    for c in cols:
        if c in ["X", "Z", "FL", col_id, col_sp, "geometry_valid"] + lambda_cols:
            continue
        if df[c].dtype not in [np.float64, float]:
            continue
        var = df.groupby(["X", "Z"])[c].std().max()
        if var < 1e-9 or np.isnan(var):
            geom_float_cols.append(c)

    const_int_cols = {
        c: df[c].iloc[0]
        for c in cols
        if c not in ["X", "Z", "FL", col_id, col_sp, "geometry_valid"]
           + lambda_cols + geom_float_cols
        and df[c].nunique() == 1
    }

    print(f"  [{label}]  Re levels: {sp_levels}")

    all_chunks = []

    for sp in sp_levels:
        sub = df[df[col_sp] == sp].copy()   # ← ONLY this Re level

        sub_XZ  = sub[["X", "Z"]].drop_duplicates().values.astype(float)
        sub_map  = sub.set_index(["X", "Z"])  # for exact-value lookup

        # ── dense grid for THIS Re level only ─────────────────────────────
        dx = _detect_spacing(sub["X"].values)
        dz = _detect_spacing(sub["Z"].values)
        X_orig = np.sort(np.unique(sub["X"].values)).astype(float)
        Z_orig = np.sort(np.unique(sub["Z"].values)).astype(float)
        X_mid  = _midpoint_grid(X_orig, dx)
        Z_mid  = _midpoint_grid(Z_orig, dz)
        X_full = np.sort(np.unique(np.concatenate([X_orig, X_mid])))
        Z_full = np.sort(np.unique(np.concatenate([Z_orig, Z_mid])))

        grid_X, grid_Z = np.meshgrid(X_full, Z_full)
        cand_XZ = np.column_stack([grid_X.ravel(), grid_Z.ravel()])

        # filter to convex hull of THIS Re level's (X,Z)
        inside = _in_hull(cand_XZ, sub_XZ)
        dense_XZ = cand_XZ[inside]

        orig_set = {(float(r[0]), float(r[1])) for r in sub_XZ}
        is_new   = np.array([(float(p[0]), float(p[1])) not in orig_set for p in dense_XZ])

        # ── interpolate FL from THIS Re level only ────────────────────────
        fl_src  = sub[["X", "Z", "FL"]].dropna()
        fl_interp = _interpolate(
            fl_src[["X", "Z"]].values.astype(float),
            fl_src["FL"].values.astype(float),
            dense_XZ,
        )
        # restore exact original FL at original positions
        orig_fl = {(float(r.X), float(r.Z)): float(r.FL) for r in fl_src.itertuples()}
        for i, pt in enumerate(dense_XZ):
            key = (float(pt[0]), float(pt[1]))
            if key in orig_fl:
                fl_interp[i] = orig_fl[key]

        # ── interpolate geom columns from THIS Re level only ──────────────
        ref_rows = sub.groupby(["X", "Z"]).first().reset_index()
        ref_XZ   = ref_rows[["X", "Z"]].values.astype(float)

        interp_geom: dict[str, np.ndarray] = {}
        for c in geom_float_cols + lambda_cols:
            interp_geom[c] = _interpolate(ref_XZ, ref_rows[c].values.astype(float), dense_XZ)

        # restore exact original geometry values
        orig_geom = {
            (float(r.X), float(r.Z)): {c: getattr(r, c) for c in geom_float_cols + lambda_cols}
            for r in ref_rows.itertuples()
        }
        for i, pt in enumerate(dense_XZ):
            key = (float(pt[0]), float(pt[1]))
            if key in orig_geom:
                for c in geom_float_cols + lambda_cols:
                    interp_geom[c][i] = orig_geom[key][c]

        # ── assemble chunk for this Re level ─────────────────────────────
        chunk = pd.DataFrame({
            col_id: df[col_id].iloc[0],
            "X":    dense_XZ[:, 0].astype(int),
            "Z":    dense_XZ[:, 1],
            col_sp: sp,
        })
        for c, v in const_int_cols.items():
            chunk[c] = v
        for c in geom_float_cols:
            chunk[c] = interp_geom[c]
        chunk["FL"]            = fl_interp
        for c in lambda_cols:
            chunk[c]           = interp_geom[c]
        chunk["geometry_valid"] = True
        chunk["generated"]      = is_new

        all_chunks.append(chunk)
        print(f"    Re={sp:>2}  (X,Z) pts: {len(sub_XZ)} orig → {len(dense_XZ)} dense "
              f"({is_new.sum()} new)")

    gen_df = pd.concat(all_chunks, ignore_index=True)
    keep   = [c for c in cols if c in gen_df.columns] + ["generated"]
    gen_df = gen_df[keep]

    total_new  = gen_df["generated"].sum()
    total_orig = (~gen_df["generated"]).sum()
    print(f"  [{label}]  total {len(gen_df)} rows  "
          f"({int(total_orig)} original + {int(total_new)} new)\n")
    return gen_df

def main():
    parser = argparse.ArgumentParser(description="Generate per-Re dense interpolated data for one processed batch.")
    parser.add_argument("--batch", default=None, help="Batch under data/processed. Defaults to the latest batch.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Explicit processed data directory.")
    parser.add_argument("--pattern", default="*.xlsx")
    args = parser.parse_args()

    data_dir = args.data_dir or select_batch_dir(PUBLISHED_PROCESSED_ROOT, args.batch)
    input_files = sorted(path for path in data_dir.glob(args.pattern) if is_source_file(path))
    if not input_files:
        print(f"No source .xlsx files found in {data_dir}.")
        return

    print(f"Generating per-Re dense data for batch {data_dir.name}: {len(input_files)} file(s).\n")
    for path in input_files:
        label = input_label(path)
        gen_df = generate_per_re(path)
        if gen_df is None:
            continue
        out = data_dir / f"{label}_generated_per_re.xlsx"
        gen_df.to_excel(out, index=False)
        print(f"  [{label}] saved -> {out}\n")

    print("Done.")

if __name__ == "__main__":
    main()
