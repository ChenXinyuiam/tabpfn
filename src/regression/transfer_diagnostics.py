"""Non-destructive source consistency evidence for FD transfer experiments."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.regression.rotation_dataset import load_snapshot
from utils.provenance import write_json
from utils.runs import experiment_run


def repeated_conditions(base):
    """Report fields copied between Re values; never infer corrected labels."""
    records = []
    for source, data in base.groupby("source_file"):
        for first, second in combinations(sorted(data.re.unique()), 2):
            a = data[data.re == first][["x", "z", "u0", "u90"]].drop_duplicates()
            b = data[data.re == second][["x", "z", "u0", "u90"]].drop_duplicates()
            merged = a.merge(b, on=["x", "z"], suffixes=("_a", "_b"))
            if len(merged) < 10:
                continue
            same = np.isclose(merged.u0_a, merged.u0_b, rtol=1e-12, atol=1e-12) & np.isclose(
                merged.u90_a, merged.u90_b, rtol=1e-12, atol=1e-12
            )
            nonzero = (merged.u0_a.abs() + merged.u90_a.abs()) > 1e-8
            if nonzero.sum() and float(same[nonzero].mean()) > 0.95:
                records.append({
                    "source_file": source, "re_a": float(first), "re_b": float(second),
                    "rows_a": len(a), "rows_b": len(b), "matched_positions": len(merged),
                    "nonzero_matched": int(nonzero.sum()), "identical_nonzero_fraction": float(same[nonzero].mean()),
                    "u0_max_abs_difference": float((merged.u0_a - merged.u0_b).abs().max()),
                    "u90_max_abs_difference": float((merged.u90_a - merged.u90_b).abs().max()),
                })
    return pd.DataFrame(records)


def identical_components(base):
    records = []
    for (source, reynolds), data in base.groupby(["source_file", "re"]):
        nonzero = (data.u0.abs() + data.u90.abs()) > 1e-8
        if nonzero.sum() < 10:
            continue
        for sign in (-1, 1):
            equal = np.isclose(data.u0, sign * data.u90, rtol=1e-12, atol=1e-12)
            if float(equal[nonzero].mean()) > .95:
                records.append({"source_file": source, "re": float(reynolds), "rows": len(data),
                                "nonzero_rows": int(nonzero.sum()), "u0_equals_sign_times_u90": sign,
                                "equal_nonzero_fraction": float(equal[nonzero].mean()),
                                "max_abs_difference": float((data.u0 - sign * data.u90).abs().max())})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fd_transfer_diagnostics/2026-09-29"))
    args = parser.parse_args()
    with experiment_run(args.output_dir, "fd-transfer-diagnostics", extra_sources=[__file__]) as out:
        base, _, info = load_snapshot(args.snapshot)
        write_json(out / "config.json", {"snapshot": info, "policy": "Evidence only, no repair or exclusion"})
        repeated = repeated_conditions(base)
        repeated.to_csv(out / "repeated_re_fields.csv", index=False)
        components = identical_components(base)
        components.to_csv(out / "identical_component_fields.csv", index=False)
        flagged = set()
        for row in repeated.to_dict("records"):
            flagged.update([(row["source_file"], row["re_a"]), (row["source_file"], row["re_b"])])
        for row in components.to_dict("records"):
            flagged.add((row["source_file"], row["re"]))
        mask = [(source, float(reynolds)) in flagged for source, reynolds in zip(base.source_file, base.re)]
        base.loc[mask, ["source_file", "source_row", "re", "x", "z", "u0", "u90"]].to_csv(
            out / "suspect_source_rows.csv", index=False, encoding="utf-8-sig"
        )
        statistics = base.groupby(["source_file", "re"])[["u0", "u90"]].agg(["size", "min", "max", "mean", "std"])
        statistics.to_csv(out / "source_re_statistics.csv")
        print(repeated.to_string(index=False), flush=True)
        print(components.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
