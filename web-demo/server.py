"""TriFocus ForceLab local demo server.

This is a dependency-free demonstration backend. It serves the static website
and exposes the API contract expected by the frontend:

  GET  /api/health
  POST /api/predict-triangle

The prediction engine is deterministic and intentionally lightweight. It is
shaped like a real inference service so it can be replaced by TabPFN-backed
model code later without changing the frontend contract.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


RE_VALUES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
POINT_COUNT = 200
RADIUS_GROUPS = [10, 20, 30]
MODEL_VERSION = "demo-backend-v2-radius"
FEATURE_VERSION = "triangle-force-features-v2-radius"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b


def angle_from_sides(a: float, b: float, c: float) -> float:
    cos_value = clamp((b * b + c * c - a * a) / (2 * b * c), -1.0, 1.0)
    return math.degrees(math.acos(cos_value))


def validate_sides(ab: float, ac: float, bc: float) -> None:
    if not all(math.isfinite(v) and v > 0 for v in (ab, ac, bc)):
        raise ValueError("side lengths must be positive numbers")
    if ab + ac <= bc or ab + bc <= ac or ac + bc <= ab:
        raise ValueError("side lengths cannot form a triangle")


def validate_radius_group(value: Any) -> int:
    try:
        radius_group = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("radius_group must be one of 10, 20, or 30") from exc
    if radius_group not in RADIUS_GROUPS:
        raise ValueError("radius_group must be one of 10, 20, or 30")
    return radius_group


def build_geometry(ab: float, ac: float, bc: float) -> dict[str, Any]:
    x_a = (ab * ab + bc * bc - ac * ac) / (2 * bc)
    y_a = math.sqrt(max(0.0, ab * ab - x_a * x_a))
    vertices = {
        "A": {"x": x_a, "y": y_a},
        "B": {"x": 0.0, "y": 0.0},
        "C": {"x": bc, "y": 0.0},
    }
    return {
        "sides": {"ab": ab, "ac": ac, "bc": bc},
        "vertices": vertices,
        "area": (bc * y_a) / 2,
        "perimeter": ab + ac + bc,
        "height": y_a,
        "centroid": {"x": (x_a + bc) / 3, "y": y_a / 3},
        "angles": {
            "A": angle_from_sides(bc, ab, ac),
            "B": angle_from_sides(ac, ab, bc),
            "C": angle_from_sides(ab, ac, bc),
        },
    }


def halton(index: int, base: int) -> float:
    factor = 1.0
    result = 0.0
    i = index
    while i > 0:
        factor /= base
        result += factor * (i % base)
        i //= base
    return result


def point_line_distance(point: dict[str, float], a: dict[str, float], b: dict[str, float]) -> float:
    dx = b["x"] - a["x"]
    dy = b["y"] - a["y"]
    numerator = abs(dy * point["x"] - dx * point["y"] + b["x"] * a["y"] - b["y"] * a["x"])
    return numerator / math.hypot(dx, dy)


def generate_points(geometry: dict[str, Any]) -> list[dict[str, Any]]:
    vertices = geometry["vertices"]
    a = vertices["A"]
    b = vertices["B"]
    c = vertices["C"]
    points: list[dict[str, Any]] = []

    for i in range(1, POINT_COUNT + 1):
        u = halton(i, 2)
        v = halton(i, 3)
        if u + v > 1:
            u = 1 - u
            v = 1 - v

        x = b["x"] + u * (c["x"] - b["x"]) + v * (a["x"] - b["x"])
        y = b["y"] + u * (c["y"] - b["y"]) + v * (a["y"] - b["y"])
        point = {"point_id": f"p{i:04d}", "x": x, "y": y}
        point["distBase"] = point_line_distance(point, b, c)
        point["distLeft"] = point_line_distance(point, a, b)
        point["distRight"] = point_line_distance(point, a, c)
        point["distCentroid"] = math.hypot(x - geometry["centroid"]["x"], y - geometry["centroid"]["y"])
        point["xNorm"] = safe_div(x, geometry["sides"]["bc"])
        point["yNorm"] = safe_div(y, geometry["height"])
        points.append(point)

    return points


def focus_model(geometry: dict[str, Any], re_value: int, radius_group: int) -> dict[str, float]:
    sides = geometry["sides"]
    ab = sides["ab"]
    ac = sides["ac"]
    bc = sides["bc"]
    aspect = safe_div(bc, geometry["height"])
    symmetry = 1 - min(abs(ab - ac) / max(ab, ac), 0.95)
    aspect_score = math.exp(-((aspect - 1.45) / 0.68) ** 2)
    geometry_score = clamp(0.22 + 0.52 * symmetry + 0.32 * aspect_score, 0.16, 0.98)
    curvature_factor = 20 / radius_group
    opt_re = clamp(31 + (aspect - 1.25) * 12 + (1 - symmetry) * 10 + (radius_group - 20) * 0.42, 18, 56)
    re_score = math.exp(-((re_value - opt_re) / 16) ** 2)
    radius_score = clamp(0.88 + 0.12 * curvature_factor, 0.78, 1.12)
    score = clamp(geometry_score * radius_score * (0.18 + 0.88 * re_score), 0.03, 0.98)
    skew = (ab - ac) / (ab + ac)
    fx = clamp(
        geometry["centroid"]["x"] + (re_value - 36) * 0.0035 * bc - skew * 0.09 * bc
        + (20 - radius_group) * 0.0015 * bc,
        min(geometry["vertices"]["A"]["x"], 0, bc) + bc * 0.12,
        max(geometry["vertices"]["A"]["x"], 0, bc) - bc * 0.10,
    )
    fy = clamp(
        geometry["centroid"]["y"]
        + math.sin(re_value / 9) * 0.045 * geometry["height"]
        + (1 - aspect_score) * 0.04 * geometry["height"],
        geometry["height"] * 0.16,
        geometry["height"] * 0.78,
    )
    return {"score": score, "x": fx, "y": fy, "optRe": opt_re, "geometryScore": geometry_score}


def predict_for_re(
    geometry: dict[str, Any], base_points: list[dict[str, Any]], re_value: int, radius_group: int
) -> dict[str, Any]:
    focus = focus_model(geometry, re_value, radius_group)
    scale = max(geometry["sides"]["bc"], geometry["height"])
    re_factor = re_value / 60
    skew = (geometry["sides"]["ab"] - geometry["sides"]["ac"]) / (geometry["sides"]["ab"] + geometry["sides"]["ac"])
    points: list[dict[str, Any]] = []

    for point in base_points:
        dx = focus["x"] - point["x"]
        dy = focus["y"] - point["y"]
        nx = dx / scale
        ny = dy / scale
        dist = math.hypot(nx, ny)
        wall_bias = safe_div(point["distRight"] - point["distLeft"], geometry["sides"]["bc"])
        vertical_bias = point["yNorm"] - 0.5
        convergence = 0.038 * focus["score"] * dist
        swirl = 0.016 * (1 - focus["score"]) * (0.45 + re_factor)
        total_base_x = convergence * nx + swirl * -ny + 0.0025 * wall_bias
        total_base_y = convergence * ny + swirl * nx - 0.0018 * vertical_bias
        fl_mix = 0.57 + 0.07 * math.sin(re_value / 12)
        fl_x = total_base_x * fl_mix + 0.0036 * wall_bias + 0.001 * skew
        fl_y = total_base_y * fl_mix - 0.0028 * vertical_bias + 0.0012 * math.sin(math.pi * point["xNorm"])
        dean_radius_factor = 20 / radius_group
        fd_x = (total_base_x - fl_x) * dean_radius_factor
        fd_y = (total_base_y - fl_y) * dean_radius_factor
        total_x = fl_x + fd_x
        total_y = fl_y + fd_y
        focus_distance = math.hypot(point["x"] - focus["x"], point["y"] - focus["y"])

        points.append(
            {
                **point,
                "re": re_value,
                "fl_x_pred": fl_x,
                "fl_y_pred": fl_y,
                "fl_magnitude_pred": math.hypot(fl_x, fl_y),
                "fd_x_pred": fd_x,
                "fd_y_pred": fd_y,
                "fd_magnitude_pred": math.hypot(fd_x, fd_y),
                "total_force_x": total_x,
                "total_force_y": total_y,
                "total_force_magnitude": math.hypot(total_x, total_y),
                "focus_score": focus["score"],
                "focus_distance_um": focus_distance,
                "focus_region_flag": 1 if focus_distance < scale * 0.08 else 0,
            }
        )

    return {"re": re_value, "focus": focus, "points": points}


def predict_triangle(payload: dict[str, Any]) -> dict[str, Any]:
    started_at = time.perf_counter()
    ab = float(payload.get("side_ab_um", payload.get("sideAB", 0)))
    ac = float(payload.get("side_ac_um", payload.get("sideAC", 0)))
    bc = float(payload.get("side_bc_um", payload.get("sideBC", 0)))
    radius_group = validate_radius_group(payload.get("radius_group", payload.get("radiusGroup")))
    validate_sides(ab, ac, bc)

    geometry = build_geometry(ab, ac, bc)
    base_points = generate_points(geometry)
    sweep = [predict_for_re(geometry, base_points, re_value, radius_group) for re_value in RE_VALUES]
    best = max(sweep, key=lambda item: item["focus"]["score"])
    focused_re = [item["re"] for item in sweep if item["focus"]["score"] >= 0.52]
    score = best["focus"]["score"]
    status = "focused" if score >= 0.72 else "weak_focus" if score >= 0.48 else "not_focused"

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    generated_rows = len(sweep) * len(base_points)

    return {
        "status": "ok",
        "input": {"side_ab_um": ab, "side_ac_um": ac, "side_bc_um": bc, "radius_group": radius_group},
        "geometry": geometry,
        "basePoints": base_points,
        "sweep": sweep,
        "bestRe": best["re"],
        "focus": {
            "status": status,
            "confidence": score,
            "x": best["focus"]["x"],
            "y": best["focus"]["y"],
            "reRange": [focused_re[0], focused_re[-1]] if focused_re else None,
            "method": "force_rule_demo",
        },
        "prediction_meta": {
            "engine": "python-demo-backend",
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "radius_group": radius_group,
            "fd_model_route": f"demo_fd_radius_{radius_group}",
            "point_count_per_re": POINT_COUNT,
            "generated_rows": generated_rows,
            "re_values": RE_VALUES,
            "inference_latency_ms": latency_ms,
            "data_source": "generated_from_geometry_and_radius_group",
            "fl_output_mode": "directional_projection_xy",
            "fd_output_mode": "force_vector",
            "uses_tabpfn": False,
            "notes": [
                "Deterministic demo backend.",
                "Replace this module with TabPFN inference for production predictions.",
            ],
        },
    }


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, directory: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        if self.path == "/api/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "engine": "python-demo-backend",
                    "model_loaded": True,
                    "model_version": MODEL_VERSION,
                    "uses_tabpfn": False,
                    "point_count_per_re": POINT_COUNT,
                    "re_values": RE_VALUES,
                    "supported_radius_groups": RADIUS_GROUPS,
                },
            )
            return
        if self.path == "/api/schema":
            self.send_json(
                HTTPStatus.OK,
                {
                    "service": "TriFocus ForceLab demo API",
                    "version": MODEL_VERSION,
                    "endpoints": {
                        "GET /api/health": {
                            "purpose": "Check whether the demo backend is running.",
                            "returns": ["status", "engine", "model_version", "point_count_per_re", "re_values"],
                        },
                        "POST /api/predict-triangle": {
                            "purpose": "Generate force-field predictions from triangle geometry and curvature radius group.",
                            "input": {
                                "side_ab_um": "number, triangle side AB in micrometers",
                                "side_ac_um": "number, triangle side AC in micrometers",
                                "side_bc_um": "number, triangle side BC in micrometers",
                                "radius_group": "required integer enum: 10, 20, or 30",
                            },
                            "output": [
                                "geometry",
                                "basePoints",
                                "sweep",
                                "bestRe",
                                "focus",
                                "prediction_meta",
                            ],
                        },
                    },
                    "prediction_contract": {
                        "point_count_per_re": POINT_COUNT,
                        "re_values": RE_VALUES,
                        "supported_radius_groups": RADIUS_GROUPS,
                        "fl_output_mode": "directional_projection_xy",
                        "fd_output_mode": "force_vector",
                        "generated_rows_per_request": POINT_COUNT * len(RE_VALUES),
                        "uses_tabpfn": False,
                        "production_note": "Replace the deterministic predictor with TabPFN FD/FL inference behind the same endpoint.",
                    },
                },
            )
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        if self.path != "/api/predict-triangle":
            self.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            self.send_json(HTTPStatus.OK, predict_triangle(payload))
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": str(exc)})
        except Exception as exc:  # pragma: no cover - user-facing demo safety
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TriFocus ForceLab demo server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    class Handler(DemoHandler):
        def __init__(self, *handler_args: Any, **handler_kwargs: Any) -> None:
            super().__init__(*handler_args, directory=str(root), **handler_kwargs)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TriFocus ForceLab demo: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
