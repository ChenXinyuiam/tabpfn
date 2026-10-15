"""Redundant triangle-geometry checks and filename-independent identity."""

import numpy as np
import pandas as pd

GEOMETRY_COLUMNS = ["bc", "ab", "ac", "h", "angle_a", "angle_b", "angle_c", "area", "perimeter"]


def physical_geometry_id(row) -> str:
    sides = np.sort([float(row[name]) for name in ("bc", "ab", "ac")])
    if not np.isfinite(sides).all() or (sides <= 0).any():
        raise ValueError("Physical geometry identity requires three finite positive sides")
    return ":".join(f"{value:.3f}" for value in sides)


def geometry_issues(row) -> list[str]:
    values = np.asarray([row[name] for name in GEOMETRY_COLUMNS], dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        return ["invalid_geometry_value"]
    bc, ab, ac, height, a, b, c, area, perimeter = values
    issues = []
    sides = np.sort([bc, ab, ac])
    if sides[0] + sides[1] <= sides[2]:
        issues.append("triangle_inequality")
    if height > min(ab, ac) + 1.0:
        issues.append("height_exceeds_side")
    if max(a, b, c) >= 180 or abs(a + b + c - 180) > 0.5:
        issues.append("angle_sum")
    if not np.isclose(area, bc * height / 2, rtol=0.02, atol=1):
        issues.append("area_disagrees_with_base_height")
    if not np.isclose(perimeter, bc + ab + ac, rtol=0.02, atol=1):
        issues.append("perimeter_disagrees_with_sides")
    if 0 < min(a, b, c) and max(a, b, c) < 180:
        scale = bc / np.sin(np.deg2rad(a))
        if not np.isclose(ab, scale * np.sin(np.deg2rad(c)), rtol=0.02, atol=1):
            issues.append("ab_disagrees_with_angles")
        if not np.isclose(ac, scale * np.sin(np.deg2rad(b)), rtol=0.02, atol=1):
            issues.append("ac_disagrees_with_angles")
    if not np.isclose(height, ab * np.sin(np.deg2rad(b)), rtol=0.02, atol=1) or not np.isclose(
        height, ac * np.sin(np.deg2rad(c)), rtol=0.02, atol=1
    ):
        issues.append("height_disagrees_with_angles")
    return issues


def attach_geometry_audit(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["physical_geometry_id"] = [physical_geometry_id(row) for row in result.to_dict("records")]
    result["geometry_issues"] = [";".join(geometry_issues(row)) for row in result.to_dict("records")]
    result["geometry_consistent"] = result["geometry_issues"].eq("")
    result["named_geometry_group"] = result["geometry_group"]
    # A shared name OR matching physical sides connects samples. Merging is
    # conservative: a filename alias cannot create a new independent holdout.
    names = sorted(result["named_geometry_group"].unique())
    parent = {name: name for name in names}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for _, group in result.groupby("physical_geometry_id"):
        aliases = sorted(group["named_geometry_group"].unique())
        for name in aliases[1:]:
            parent[root(name)] = root(aliases[0])
    members = {}
    for name in names:
        members.setdefault(root(name), []).append(name)
    labels = {name: "+".join(members[root(name)]) for name in names}
    result["geometry_group"] = result["named_geometry_group"].map(labels)
    return result


def write_geometry_audit(data, path):
    columns = [
        "source_file",
        "R",
        "re",
        "named_geometry_group",
        "geometry_group",
        "physical_geometry_id",
        "geometry_consistent",
        "geometry_issues",
        *GEOMETRY_COLUMNS,
    ]
    data[columns].to_csv(path, index=False, encoding="utf-8-sig")
