"""Triangle geometry in an explicitly declared global x/z frame (lengths in um)."""

import hashlib
import json

import numpy as np
import pandas as pd

LENGTH_TO_UM = {"m": 1e6, "mm": 1e3, "um": 1.0}
VERTEX_COLUMNS = ["ax", "az", "bx", "bz", "cx", "cz"]
FEATURES = [
    *VERTEX_COLUMNS,
    "ab",
    "ac",
    "bc",
    "h",
    "area",
    "perimeter",
    "angle_a",
    "angle_b",
    "angle_c",
    "re",
    "radius_um",
    "x",
    "z",
    "x_over_h",
    "z_over_h",
    "distance_ab",
    "distance_ac",
    "distance_bc",
]
FEATURE_VERSION = "triangle-xz-v1"


def identity(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


def vertices(spec):
    if spec.get("unit") not in LENGTH_TO_UM:
        raise ValueError("Geometry unit must be explicitly m, mm or um")
    values = np.asarray(spec["vertices"], dtype=float) * LENGTH_TO_UM[spec["unit"]]
    if values.shape != (3, 2) or not np.isfinite(values).all():
        raise ValueError("vertices must be finite [[Ax,Az],[Bx,Bz],[Cx,Cz]]")
    a, b, c = values
    scale = max(np.linalg.norm(a - b), np.linalg.norm(a - c), np.linalg.norm(b - c))
    cross = abs(np.linalg.det(np.stack([b - a, c - a])))
    if scale <= 0 or cross <= 1e-10 * scale**2:
        raise ValueError("Degenerate triangle")
    return values


def geometry_family(values):
    a, b, c = values
    return "g_" + identity(sorted(round(float(np.linalg.norm(p - q)), 6) for p, q in [(a, b), (a, c), (b, c)]))


def inside(points, triangle, *, boundary=True):
    a, b, c = triangle
    uv = np.linalg.solve(np.column_stack([b - a, c - a]), (np.asarray(points) - a).T).T
    weights = np.column_stack([1 - uv.sum(axis=1), uv])
    return (weights >= -1e-9).all(axis=1) if boundary else (weights > 1e-9).all(axis=1)


def feature_frame(points, triangle, re_value, radius_um, experiment_id):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected nonempty finite x/z coordinates")
    if not inside(points, triangle).all():
        raise ValueError("Coordinates outside the declared triangle")
    if not np.isfinite([re_value, radius_um]).all() or min(re_value, radius_um) <= 0:
        raise ValueError("Re and radius_um must be positive finite values")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be nonempty")
    a, b, c = triangle
    ab, ac, bc = (float(np.linalg.norm(p - q)) for p, q in [(a, b), (a, c), (b, c)])
    area = abs(float(np.linalg.det(np.stack([b - a, c - a])))) / 2
    h = 2 * area / bc
    out = pd.DataFrame(points, columns=["x", "z"])
    for column, value in zip(VERTEX_COLUMNS, triangle.ravel(), strict=True):
        out[column] = value
    for column, value in {
        "ab": ab,
        "ac": ac,
        "bc": bc,
        "h": h,
        "area": area,
        "perimeter": ab + ac + bc,
        "re": re_value,
        "radius_um": radius_um,
    }.items():
        out[column] = value
    for name, opposite, left, right in [("a", bc, ab, ac), ("b", ac, ab, bc), ("c", ab, ac, bc)]:
        out["angle_" + name] = np.degrees(
            np.arccos(np.clip((left**2 + right**2 - opposite**2) / (2 * left * right), -1, 1))
        )
    for name, p, q in [("ab", a, b), ("ac", a, c), ("bc", b, c)]:
        delta = q - p
        out["distance_" + name] = abs(
            delta[0] * (points[:, 1] - p[1]) - delta[1] * (points[:, 0] - p[0])
        ) / np.linalg.norm(delta)
    out["x_over_h"], out["z_over_h"] = (points[:, 0] - b[0]) / h, (points[:, 1] - b[1]) / h
    out["geometry_family_id"] = geometry_family(triangle)
    out["geometry_id"] = identity(np.round(triangle, 6).tolist())
    out["experiment_id"] = experiment_id
    out["sample_id"] = "s_" + identity([experiment_id, out.geometry_id.iloc[0], float(re_value), float(radius_um)])
    out["point_id"] = ["p_" + identity([round(float(x), 8), round(float(z), 8)]) for x, z in points]
    return out


def query_grid(request):
    if request.get("coordinate_frame") != "global_xz":
        raise ValueError("coordinate_frame must be global_xz; transform other frames explicitly")
    triangle = vertices(request["geometry"])
    if "positions" in request:
        unit = request.get("position_unit")
        if unit not in LENGTH_TO_UM:
            raise ValueError("position_unit must be explicit")
        points = np.asarray(request["positions"], dtype=float) * LENGTH_TO_UM[unit]
    else:
        grid = request["grid"]
        nx, nz = grid["nx"], grid["nz"]
        if any(type(n) is not int or n < 3 for n in (nx, nz)) or nx * nz > 1_000_000:
            raise ValueError("Grid needs integer nx,nz >= 3 and at most 1000000 candidate points")
        x = np.linspace(triangle[:, 0].min(), triangle[:, 0].max(), nx)
        z = np.linspace(triangle[:, 1].min(), triangle[:, 1].max(), nz)
        xx, zz = np.meshgrid(x, z)
        points = np.column_stack([xx.ravel(), zz.ravel()])
        points = points[inside(points, triangle, boundary=grid.get("include_boundary", False))]
    if len(points) > 1_000_000:
        raise ValueError("Too many query points")
    reynolds = request["re_values"]
    if not reynolds or len(set(reynolds)) != len(reynolds):
        raise ValueError("re_values must be nonempty and unique")
    frames = [
        feature_frame(points, triangle, float(re_value), float(request["radius_um"]), request["experiment_id"])
        for re_value in reynolds
    ]
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["sample_id", "point_id"]).any():
        raise ValueError("Duplicate query coordinates")
    return result
