"""Audit a provider's rotation exports without treating copies as new geometries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression.evaluation import geometry_signature, parse_radius, sample_complete_positions, stable_seed
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run

ANGLES = tuple(range(0, 360, 15))
CORE = ["bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c", "re", "x", "z", "bary_a", "bary_b", "bary_c"]


def read_workbook(path, engine):
    frame = pd.read_excel(path, engine=engine, sheet_name=0)
    required = [*CORE, "rotation_deg", "u"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    for column in frame:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & numeric.isna()).any():
            raise ValueError(f"{path}: nonnumeric {column}")
        frame[column] = numeric
    # Incomplete predictors are quarantined consistently across all rotations.
    # Missing labels or angle identities cannot support a rotation audit.
    point_required = ["rotation_deg", "u"]
    if frame.empty or not np.isfinite(frame[point_required].to_numpy()).all():
        raise ValueError(f"{path}: missing/non-finite required data")
    if np.isinf(frame.to_numpy()).any():
        raise ValueError(f"{path}: infinite optional data")
    return frame


def check_alignment(base, rotated, angle):
    if len(base) != len(rotated) or not np.allclose(base[CORE], rotated[CORE], rtol=1e-10, atol=1e-7, equal_nan=True):
        raise ValueError(f"Rotation {angle}: canonical coordinates/row identities are not aligned")
    if not np.allclose(rotated.rotation_deg, angle, rtol=0, atol=1e-8):
        raise ValueError(f"Wrong rotation_deg in angle_{angle:03d}")


def prepare(root: Path, out: Path, *, engine="calamine", test_rows=100, seed=42):
    base_paths = sorted((root / "angle_000").glob("*.xlsx"))
    if not base_paths:
        raise ValueError("Expected angle_000/*.xlsx")
    expected = {p.name for p in base_paths}
    for angle in ANGLES:
        found = {p.name for p in (root / f"angle_{angle:03d}").glob("*.xlsx")}
        if found != expected:
            raise ValueError(f"Inconsistent source files at angle {angle}")
    base_parts, test_parts, inventory, checks, rejected = [], [], [], [], []
    for path in base_paths:
        base = read_workbook(path, engine)
        quarter_path = root / "angle_090" / path.name
        quarter = read_workbook(quarter_path, engine)
        check_alignment(base, base, 0)
        check_alignment(base, quarter, 90)
        valid_base = np.isfinite(base[CORE].to_numpy()).all(axis=1)
        for index in np.flatnonzero(~valid_base):
            rejected.append(
                {
                    "source_file": path.name,
                    "source_row": int(index + 2),
                    "missing_columns": ",".join(base[CORE].columns[base[CORE].iloc[index].isna()]),
                    "excluded_rotation_copies": len(ANGLES),
                }
            )
        if not valid_base.any():
            raise ValueError(f"No complete geometry rows: {path}")
        for col in ("bc", "ab", "ac", "h"):
            good = base.loc[valid_base, col]
            if (good <= 0).any() or not np.allclose(good, good.iloc[0]):
                raise ValueError(f"Mixed/invalid geometry: {path.name} {col}")
        if (base.re <= 0).any():
            raise ValueError("Nonpositive Re")
        base["u0"] = base.u
        base["u90"] = quarter.u.to_numpy()
        base["radius"] = parse_radius(path)
        base["source_file"] = path.name
        base["source_row"] = np.arange(len(base)) + 2
        base["geometry_group"] = geometry_signature(base)
        base["base_id"] = path.name + "|" + base.source_row.astype(str)
        base["position_group"] = path.name + "|" + base.x.round(8).astype(str) + "|" + base.z.round(8).astype(str)
        sample = sample_complete_positions(base.loc[valid_base], test_rows, stable_seed(path.name, seed))
        source_indices = sample.source_row.to_numpy() - 2
        for angle in ANGLES:
            rotated_path = root / f"angle_{angle:03d}" / path.name
            rotated = base if angle == 0 else quarter if angle == 90 else read_workbook(rotated_path, engine)
            check_alignment(base, rotated, angle)
            theta = np.deg2rad(angle)
            expected_u = base.u0.to_numpy() * np.cos(theta) + base.u90.to_numpy() * np.sin(theta)
            residual = rotated.u.to_numpy() - expected_u
            valid = np.isclose(rotated.u.to_numpy(), expected_u, rtol=1e-8, atol=1e-7)
            checks.append(
                {
                    "source_file": str(rotated_path.relative_to(root)),
                    "rows": len(rotated),
                    "sha256": file_sha256(rotated_path),
                    "rotation_deg": angle,
                    "formula_mismatch_rows": int((~valid).sum()),
                    "max_abs_rotation_residual": float(abs(residual).max()),
                }
            )
            if not valid.all():
                pd.DataFrame(checks).to_csv(out / "rotation_checks.csv", index=False)
                raise ValueError(f"Unverified rotation identity: {rotated_path}")
            # Test labels come from each actual workbook, never from predictions.
            sample[f"observed_{angle:03d}"] = rotated.u.to_numpy()[source_indices]
        inventory.append(
            {
                "source_file": path.name,
                "geometry_group": base.geometry_group.iloc[0],
                "base_rows": len(base),
                "accepted_base_rows": int(valid_base.sum()),
                "rejected_base_rows": int((~valid_base).sum()),
                "test_base_rows": len(sample),
                "radius_label": int(base.radius.iloc[0]),
                "u0_min": float(base.u0.min()),
                "u0_max": float(base.u0.max()),
                "u90_min": float(base.u90.min()),
                "u90_max": float(base.u90.max()),
                "missing_area": int(base.area.isna().sum()),
                "missing_prep_dist": int(
                    base[["prep_dist_to_ab", "prep_dist_to_ac", "prep_dist_to_bc"]].isna().any(axis=1).sum()
                ),
                "area_mismatch": int(
                    (base.area.notna() & ~np.isclose(base.area, base.bc * base.h / 2, rtol=0.02)).sum()
                ),
                "outside_barycentric": int(
                    ((base[["bary_a", "bary_b", "bary_c"]] < -1e-4) | (base[["bary_a", "bary_b", "bary_c"]] > 1.0001))
                    .any(axis=1)
                    .sum()
                ),
                "duplicate_re_x_z": int(base.duplicated(["re", "x", "z"]).sum()),
            }
        )
        base_parts.append(base.loc[valid_base].copy())
        test_parts.append(sample)
        print(f"Audited all 24 angles: {path.name}, {len(base)} base rows", flush=True)
    base_data, test_data = pd.concat(base_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)
    # Pickle is only used for our local, hashed snapshot, never untrusted downloads.
    base_data.to_pickle(out / "base.pkl")
    test_data.to_pickle(out / "test.pkl")
    pd.DataFrame(inventory).to_csv(out / "data_inventory.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "rotation_checks.csv", index=False)
    pd.DataFrame(rejected, columns=["source_file", "source_row", "missing_columns", "excluded_rotation_copies"]).to_csv(
        out / "rejected_rows.csv", index=False
    )
    write_json(
        out / "snapshot.json",
        {
            "root": str(root.resolve()),
            "engine": engine,
            "seed": seed,
            "test_rows_per_file": test_rows,
            "base_rows": len(base_data),
            "geometry_groups": int(base_data.geometry_group.nunique()),
            "files": len(checks),
            "rotation_rows_checked": sum(row["rows"] for row in checks),
            "base_sha256": file_sha256(out / "base.pkl"),
            "test_sha256": file_sha256(out / "test.pkl"),
            "meaning": "native u; empirical two-component rotation identity verified, physical units unconfirmed",
        },
    )
    return base_data, test_data


def load_snapshot(path: Path):
    import json

    info = json.loads((path / "snapshot.json").read_text(encoding="utf-8"))
    for name in ("base", "test"):
        if file_sha256(path / f"{name}.pkl") != info[f"{name}_sha256"]:
            raise ValueError("Snapshot hash mismatch")
    return pd.read_pickle(path / "base.pkl"), pd.read_pickle(path / "test.pkl"), info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/rawdata/2026-09-29"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_rotation_audit/2026-09-29"))
    parser.add_argument("--engine", choices=["calamine", "openpyxl"], default="calamine")
    parser.add_argument("--test-rows", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.test_rows <= 0:
        parser.error("test-rows must be positive")
    with experiment_run(
        args.output_dir, "fd-rotation-audit", extra_sources=[__file__, Path(__file__).with_name("evaluation.py")]
    ) as out:
        prepare(args.data_root, out, engine=args.engine, test_rows=args.test_rows, seed=args.seed)


if __name__ == "__main__":
    main()
