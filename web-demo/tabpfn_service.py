"""TriFocus TabPFN-ready CPU inference API.

This service exposes the same API contract as the Node demo backend:

  GET  /api/health
  GET  /api/schema
  POST /api/predict-triangle

It can run in three modes:

1. training_csv: fit TabPFN regressors at service startup from a CSV file.
2. joblib: load four pre-fitted sklearn/joblib models.
3. fallback: keep the API alive with the deterministic force-rule predictor.

The fallback mode is intentionally explicit: health and metadata report
uses_tabpfn=false. It is useful while the server model files are being prepared.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

RE_VALUES = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
POINT_COUNT = 400
MODEL_VERSION = "tabpfn-cpu-service-v1"
FEATURE_VERSION = "triangle-tabpfn-features-v1"
REFERENCE_CURVATURE_RADIUS_UM = 1200.0
MIN_CURVATURE_RADIUS_UM = 300.0
MAX_CURVATURE_RADIUS_UM = 5000.0

FEATURE_COLUMNS = [
    "re",
    "x_m",
    "z_m",
    "bc_m",
    "ab_m",
    "ac_m",
    "h_m",
    "angle_a",
    "angle_b",
    "angle_c",
    "area_m2",
    "perimeter_m",
    "x_over_h",
    "z_over_h",
    "dist_ab_m",
    "dist_ac_m",
    "dist_bc_m",
    "dist_centroid_m",
    "centroid_x_m",
    "centroid_y_m",
]

TARGET_COLUMNS = {
    "fl_x": "fl_x_pred",
    "fl_y": "fl_y_pred",
    "fd_x": "fd_x_pred",
    "fd_y": "fd_y_pred",
}


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


def validate_curvature_radius(value: float) -> None:
    if not math.isfinite(value) or value < MIN_CURVATURE_RADIUS_UM or value > MAX_CURVATURE_RADIUS_UM:
        raise ValueError("curvature radius must be between 300 and 5000 um")


def dean_curvature_factor(curvature_radius_um: float) -> float:
    return clamp(REFERENCE_CURVATURE_RADIUS_UM / curvature_radius_um, 0.35, 2.35)


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


def build_point(
    geometry: dict[str, Any],
    *,
    point_id: str,
    x: float,
    y: float,
) -> dict[str, Any]:
    vertices = geometry["vertices"]
    a = vertices["A"]
    b = vertices["B"]
    c = vertices["C"]
    point = {
        "point_id": point_id,
        "x": x,
        "y": y,
        "x_um": x,
        "z_um": y,
    }
    point["distBase"] = point_line_distance(point, b, c)
    point["distLeft"] = point_line_distance(point, a, b)
    point["distRight"] = point_line_distance(point, a, c)
    point["distCentroid"] = math.hypot(x - geometry["centroid"]["x"], y - geometry["centroid"]["y"])
    point["xNorm"] = safe_div(x, geometry["sides"]["bc"])
    point["yNorm"] = safe_div(y, geometry["height"])
    return point


def point_is_inside_triangle(x: float, y: float, geometry: dict[str, Any]) -> bool:
    a = geometry["vertices"]["A"]
    b = geometry["vertices"]["B"]
    c = geometry["vertices"]["C"]
    denominator = (b["y"] - c["y"]) * (a["x"] - c["x"]) + (c["x"] - b["x"]) * (a["y"] - c["y"])
    weight_a = ((b["y"] - c["y"]) * (x - c["x"]) + (c["x"] - b["x"]) * (y - c["y"])) / denominator
    weight_b = ((c["y"] - a["y"]) * (x - c["x"]) + (a["x"] - c["x"]) * (y - c["y"])) / denominator
    weight_c = 1.0 - weight_a - weight_b
    tolerance = 1e-9
    return weight_a >= -tolerance and weight_b >= -tolerance and weight_c >= -tolerance


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
        points.append(build_point(geometry, point_id=f"p{i:04d}", x=x, y=y))

    return points


def resolve_points(payload: dict[str, Any], geometry: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    raw_positions = payload.get("positions")
    if raw_positions is None:
        return generate_points(geometry), "halton_sampled_positions"
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError("positions must be a non-empty list when provided")
    if len(raw_positions) > 10000:
        raise ValueError("positions cannot contain more than 10000 points")

    points: list[dict[str, Any]] = []
    point_ids = set()
    for index, raw in enumerate(raw_positions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"positions[{index - 1}] must be an object")
        x_value = raw.get("x_um", raw.get("x"))
        z_value = raw.get("z_um", raw.get("z", raw.get("y_um", raw.get("y"))))
        if x_value is None or z_value is None:
            raise ValueError(f"positions[{index - 1}] must contain x_um and z_um")
        x = float(x_value)
        y = float(z_value)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"positions[{index - 1}] contains a non-finite coordinate")
        if not point_is_inside_triangle(x, y, geometry):
            raise ValueError(f"positions[{index - 1}] is outside the triangle")
        point_id = str(raw.get("point_id", f"p{index:04d}"))
        if not point_id.strip() or point_id in point_ids:
            raise ValueError("point_id values must be nonempty and unique")
        point_ids.add(point_id)
        points.append(build_point(geometry, point_id=point_id, x=x, y=y))
    return points, "provided_positions"


def focus_model(geometry: dict[str, Any], re_value: int, curvature_radius_um: float = REFERENCE_CURVATURE_RADIUS_UM) -> dict[str, float]:
    sides = geometry["sides"]
    ab = sides["ab"]
    ac = sides["ac"]
    bc = sides["bc"]
    aspect = safe_div(bc, geometry["height"])
    symmetry = 1 - min(abs(ab - ac) / max(ab, ac), 0.95)
    aspect_score = math.exp(-(((aspect - 1.45) / 0.68) ** 2))
    geometry_score = clamp(0.22 + 0.52 * symmetry + 0.32 * aspect_score, 0.16, 0.98)
    dean_gain = dean_curvature_factor(curvature_radius_um)
    opt_re = clamp(31 + (aspect - 1.25) * 12 + (1 - symmetry) * 10 + (dean_gain - 1) * 7.5, 18, 56)
    re_score = math.exp(-(((re_value - opt_re) / 16) ** 2))
    radius_score = clamp(0.9 + 0.1 * dean_gain, 0.78, 1.14)
    score = clamp(geometry_score * radius_score * (0.18 + 0.88 * re_score), 0.03, 0.98)
    skew = (ab - ac) / (ab + ac)
    fx = clamp(
        geometry["centroid"]["x"] + (re_value - 36) * 0.0035 * bc - skew * 0.09 * bc + (dean_gain - 1) * 0.035 * bc,
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


def deterministic_force(point: dict[str, Any], geometry: dict[str, Any], re_value: int, curvature_radius_um: float = REFERENCE_CURVATURE_RADIUS_UM) -> dict[str, float]:
    focus = focus_model(geometry, re_value, curvature_radius_um)
    scale = max(geometry["sides"]["bc"], geometry["height"])
    re_factor = re_value / 60
    skew = (geometry["sides"]["ab"] - geometry["sides"]["ac"]) / (geometry["sides"]["ab"] + geometry["sides"]["ac"])
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
    dean_factor = dean_curvature_factor(curvature_radius_um)
    fd_x = (total_base_x - fl_x) * dean_factor
    fd_y = (total_base_y - fl_y) * dean_factor
    return {"fl_x": fl_x, "fl_y": fl_y, "fd_x": fd_x, "fd_y": fd_y}


def build_feature_rows(geometry: dict[str, Any], points: list[dict[str, Any]]) -> list[dict[str, float]]:
    sides = geometry["sides"]
    angles = geometry["angles"]
    rows: list[dict[str, float]] = []
    for re_value in RE_VALUES:
        for point in points:
            rows.append(
                {
                    "re": float(re_value),
                    "x_m": point["x"] * 1e-6,
                    "z_m": point["y"] * 1e-6,
                    "bc_m": sides["bc"] * 1e-6,
                    "ab_m": sides["ab"] * 1e-6,
                    "ac_m": sides["ac"] * 1e-6,
                    "h_m": geometry["height"] * 1e-6,
                    "angle_a": angles["A"],
                    "angle_b": angles["B"],
                    "angle_c": angles["C"],
                    "area_m2": geometry["area"] * 1e-12,
                    "perimeter_m": geometry["perimeter"] * 1e-6,
                    "x_over_h": safe_div(point["x"], geometry["height"]),
                    "z_over_h": safe_div(point["y"], geometry["height"]),
                    "dist_ab_m": point["distLeft"] * 1e-6,
                    "dist_ac_m": point["distRight"] * 1e-6,
                    "dist_bc_m": point["distBase"] * 1e-6,
                    "dist_centroid_m": point["distCentroid"] * 1e-6,
                    "centroid_x_m": geometry["centroid"]["x"] * 1e-6,
                    "centroid_y_m": geometry["centroid"]["y"] * 1e-6,
                }
            )
    return rows


class ModelManager:
    def __init__(self) -> None:
        self.ready = False
        self.mode = "fallback"
        self.error: str | None = None
        self.models: dict[str, Any] = {}
        self.preprocessor: Any = None
        self.revision = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.uses_tabpfn = False

    def load(self) -> None:
        with self.lock:
            self._load()

    def _load(self) -> None:
        self.ready = False
        self.error = None
        self.models = {}
        self.preprocessor = None
        self.uses_tabpfn = False
        self.revision = uuid.uuid4().hex
        requested = os.environ.get("TRIFOCUS_MODEL_MODE", "auto").strip().lower()
        try:
            if requested in {"auto", "joblib"} and self._joblib_paths_available():
                self._load_joblib_models()
                return
            if requested in {"auto", "training_csv", "tabpfn"} and os.environ.get("TRIFOCUS_TRAINING_CSV"):
                self._fit_tabpfn_from_csv()
                return
            self.mode = "fallback"
            self.error = "No TabPFN model configuration found. Set TRIFOCUS_TRAINING_CSV or four TRIFOCUS_*_MODEL paths."
        except Exception as exc:  # pragma: no cover - deployment diagnostics
            self.models = {}
            self.preprocessor = None
            self.mode = "fallback"
            self.error = f"{type(exc).__name__}: {exc}"

    def _joblib_paths_available(self) -> bool:
        return all(os.environ.get(name) for name in ["TRIFOCUS_FLX_MODEL", "TRIFOCUS_FLY_MODEL", "TRIFOCUS_FDX_MODEL", "TRIFOCUS_FDY_MODEL"])

    def _load_joblib_models(self) -> None:
        import pathlib

        import joblib

        paths = {
            "fl_x": Path(os.environ["TRIFOCUS_FLX_MODEL"]),
            "fl_y": Path(os.environ["TRIFOCUS_FLY_MODEL"]),
            "fd_x": Path(os.environ["TRIFOCUS_FDX_MODEL"]),
            "fd_y": Path(os.environ["TRIFOCUS_FDY_MODEL"]),
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing joblib model files: " + ", ".join(missing))
        if os.name != "nt":
            # Models trained on Windows can contain pathlib.WindowsPath objects.
            # Linux cannot instantiate those during pickle/joblib loading.
            pathlib.WindowsPath = pathlib.PosixPath  # type: ignore[attr-defined, assignment]
        self.models = {key: joblib.load(path) for key, path in paths.items()}
        self.uses_tabpfn = any(
            isinstance(model, dict) and model.get("kind") == "tabpfn_regressor_bundle"
            or "tabpfn" in type(model).__module__.lower()
            or any("tabpfn" in type(step).__module__.lower() for step in getattr(model, "named_steps", {}).values())
            for model in self.models.values()
        )
        self.mode = "joblib"
        self.ready = True

    def _fit_tabpfn_from_csv(self) -> None:
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MinMaxScaler
        from tabpfn import TabPFNRegressor

        training_path = Path(os.environ["TRIFOCUS_TRAINING_CSV"])
        if not training_path.exists():
            raise FileNotFoundError(f"Training CSV not found: {training_path}")
        df = pd.read_csv(training_path)
        missing = [column for column in FEATURE_COLUMNS + list(TARGET_COLUMNS.values()) if column not in df.columns]
        if missing:
            raise ValueError(f"Training CSV missing columns: {missing}")

        numeric = df[FEATURE_COLUMNS + list(TARGET_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna(subset=list(TARGET_COLUMNS.values()))
        max_rows = int(os.environ.get("TRIFOCUS_MAX_TRAIN_ROWS", "2000"))
        if len(numeric) > max_rows:
            numeric = numeric.sample(n=max_rows, random_state=42)
        if len(numeric) < 20:
            raise ValueError(f"Training CSV has too few usable rows: {len(numeric)}")

        self.preprocessor = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler())
        x_train = self.preprocessor.fit_transform(numeric[FEATURE_COLUMNS])
        n_estimators = int(os.environ.get("TRIFOCUS_TABPFN_ESTIMATORS", "1"))
        self.models = {}
        for key, target_column in TARGET_COLUMNS.items():
            model = TabPFNRegressor(
                device="cpu",
                n_estimators=n_estimators,
                ignore_pretraining_limits=True,
                random_state=42,
            )
            model.fit(x_train, numeric[target_column].to_numpy(dtype=float))
            self.models[key] = model
        self.mode = "training_csv"
        self.uses_tabpfn = True
        self.ready = True

    def predict(self, rows: list[dict[str, float]]) -> dict[str, list[float]]:
        with self.lock:
            return self._predict_rows(rows)

    def _predict_rows(self, rows: list[dict[str, float]]) -> dict[str, list[float]]:
        if not self.ready:
            raise RuntimeError(self.error or "TabPFN model is not ready")

        import numpy as np
        import pandas as pd

        frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        if frame.empty or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("Prediction feature rows must be nonempty and finite")
        predictions: dict[str, list[float]] = {}
        if self.mode == "training_csv":
            x_pred = self.preprocessor.transform(frame)
            for key, model in self.models.items():
                predictions[key] = np.asarray(model.predict(x_pred), dtype=float).tolist()
            return checked_model_predictions(predictions, len(frame))

        if self.mode == "joblib":
            for key, model in self.models.items():
                if isinstance(model, dict) and model.get("kind") == "constant":
                    pred = np.full(len(frame), float(model["value"]))
                elif isinstance(model, dict) and model.get("kind") == "tabpfn_regressor_bundle":
                    pred = self._predict_bundle(model, frame)
                else:
                    try:
                        pred = model.predict(frame)
                    except Exception:
                        pred = model.predict(frame.to_numpy(dtype=float))
                predictions[key] = np.asarray(pred, dtype=float).tolist()
            return checked_model_predictions(predictions, len(frame))

        raise RuntimeError(f"Unsupported model mode: {self.mode}")

    def _predict_bundle(self, bundle: dict[str, Any], frame: Any) -> Any:
        import numpy as np

        columns = bundle.get("feature_columns", FEATURE_COLUMNS)
        values = frame[columns].to_numpy(dtype=float)
        preprocessor = bundle["preprocessor"]
        medians = np.asarray(preprocessor["medians"], dtype=float)
        min_values = np.asarray(preprocessor["min_values"], dtype=float)
        span = np.asarray(preprocessor["span"], dtype=float)
        filled = np.where(np.isfinite(values), values, medians)
        transformed = (filled - min_values) / span
        if not np.isfinite(transformed).all():
            raise ValueError("Invalid stored preprocessing values")
        chunk_size = int(os.environ.get("TRIFOCUS_PREDICT_CHUNK_SIZE", "256"))
        if chunk_size < 1:
            raise ValueError("TRIFOCUS_PREDICT_CHUNK_SIZE must be positive")
        result = np.empty(len(transformed), dtype=float)
        for start in range(0, len(transformed), chunk_size):
            chunk = transformed[start:start + chunk_size]
            values = np.asarray(bundle["model"].predict(chunk), dtype=float)
            if values.shape != (len(chunk),) or not np.isfinite(values).all():
                raise ValueError("Model returned invalid prediction rows")
            result[start:start + len(chunk)] = values
        return result

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "mode": self.mode,
            "uses_tabpfn": self.ready and self.uses_tabpfn,
            "model_revision": self.revision,
            "error": self.error,
        }


MODEL_MANAGER = ModelManager()
PREDICTION_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
CACHE_LOCK = threading.RLock()


def checked_model_predictions(predictions, row_count):
    import numpy as np

    result = {}
    for key in TARGET_COLUMNS:
        if key not in predictions:
            raise ValueError(f"Model output missing {key}")
        values = np.asarray(predictions[key], dtype=float)
        if values.shape != (row_count,) or not np.isfinite(values).all():
            raise ValueError(f"Model output {key} must have {row_count} finite values")
        result[key] = values.tolist()
    return result


def predict_triangle(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Prediction request must be a JSON object")
    started_at = time.perf_counter()
    cache_capacity = int(os.environ.get("TRIFOCUS_CACHE_MAX_ENTRIES", "8"))
    if cache_capacity < 0:
        raise ValueError("TRIFOCUS_CACHE_MAX_ENTRIES must be nonnegative")
    cache_enabled = os.environ.get("TRIFOCUS_CACHE", "1") == "1" and cache_capacity > 0
    require_model = payload.get("require_model", False)
    if not isinstance(require_model, bool):
        raise ValueError("require_model must be boolean")
    if require_model and not MODEL_MANAGER.ready:
        raise RuntimeError(MODEL_MANAGER.error or "Model required but unavailable")
    ab = float(payload.get("side_ab_um", payload.get("sideAB", 0)))
    ac = float(payload.get("side_ac_um", payload.get("sideAC", 0)))
    bc = float(payload.get("side_bc_um", payload.get("sideBC", 0)))
    curvature_radius_um = float(
        payload.get(
            "curvature_radius_um",
            payload.get("curvatureRadius", payload.get("radius_um", REFERENCE_CURVATURE_RADIUS_UM)),
        )
    )
    validate_sides(ab, ac, bc)
    validate_curvature_radius(curvature_radius_um)

    geometry = build_geometry(ab, ac, bc)
    base_points, position_source = resolve_points(payload, geometry)
    point_signature = json.dumps(
        [(point["point_id"], point["x"], point["y"]) for point in base_points],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    points_hash = hashlib.sha256(point_signature.encode("utf-8")).hexdigest()
    cache_key = json.dumps([ab, ac, bc, curvature_radius_um, points_hash, position_source, MODEL_MANAGER.mode, MODEL_MANAGER.ready, MODEL_MANAGER.revision, require_model])
    with CACHE_LOCK:
        while len(PREDICTION_CACHE) > cache_capacity:
            PREDICTION_CACHE.popitem(last=False)
        cached = copy.deepcopy(PREDICTION_CACHE.get(cache_key)) if cache_enabled else None
        if cached is not None:
            PREDICTION_CACHE.move_to_end(cache_key)
    if cached is not None:
        cached["prediction_meta"]["cache_hit"] = True
        cached["prediction_meta"]["inference_latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
        return cached

    feature_rows = build_feature_rows(geometry, base_points)
    model_predictions: dict[str, list[float]] | None = None
    uses_tabpfn = False
    model_error = None
    if MODEL_MANAGER.ready:
        try:
            model_predictions = checked_model_predictions(MODEL_MANAGER.predict(feature_rows), len(feature_rows))
            uses_tabpfn = MODEL_MANAGER.uses_tabpfn
        except Exception as exc:  # pragma: no cover - defensive service mode
            if require_model:
                raise RuntimeError(f"Model inference failed: {exc}") from exc
            model_error = f"{type(exc).__name__}: {exc}"

    sweep = []
    row_index = 0
    dean_factor = dean_curvature_factor(curvature_radius_um)
    for re_value in RE_VALUES:
        focus = focus_model(geometry, re_value, curvature_radius_um)
        points = []
        for point in base_points:
            if model_predictions is not None:
                fl_x = model_predictions["fl_x"][row_index]
                fl_y = model_predictions["fl_y"][row_index]
                fd_x = model_predictions["fd_x"][row_index] * dean_factor
                fd_y = model_predictions["fd_y"][row_index] * dean_factor
            else:
                force = deterministic_force(point, geometry, re_value, curvature_radius_um)
                fl_x = force["fl_x"]
                fl_y = force["fl_y"]
                fd_x = force["fd_x"]
                fd_y = force["fd_y"]

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
                    "fl_pred": fl_x,
                    "fd_pred": fd_x,
                    "fnet_pred": total_x,
                    "fnet_x": total_x,
                    "fnet_z": total_y,
                    "fnet_magnitude": math.hypot(total_x, total_y),
                    "total_force_x": total_x,
                    "total_force_y": total_y,
                    "total_force_magnitude": math.hypot(total_x, total_y),
                    "focus_score": focus["score"],
                    "focus_distance_um": focus_distance,
                    "focus_region_flag": 1 if focus_distance < max(geometry["sides"]["bc"], geometry["height"]) * 0.08 else 0,
                }
            )
            row_index += 1
        sweep.append({"re": re_value, "focus": focus, "points": points})

    best = max(sweep, key=lambda item: item["focus"]["score"])
    focused_re = [item["re"] for item in sweep if item["focus"]["score"] >= 0.52]
    score = best["focus"]["score"]
    status = "focused" if score >= 0.72 else "weak_focus" if score >= 0.48 else "not_focused"
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    result = {
        "status": "ok",
        "input": {
            "side_ab_um": ab,
            "side_ac_um": ac,
            "side_bc_um": bc,
            "curvature_radius_um": curvature_radius_um,
            "positions_supplied": position_source == "provided_positions",
            "point_count": len(base_points),
        },
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
            "method": "heuristic_focus_rule",
            "confidence_is_probability": False,
        },
        "prediction_meta": {
            "engine": "python-tabpfn-cpu-service" if uses_tabpfn else "python-regression-service" if model_predictions is not None else "python-tabpfn-service-fallback",
            "model_version": MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "model_mode": MODEL_MANAGER.mode,
            "point_count_per_re": len(base_points),
            "default_point_count_per_re": POINT_COUNT,
            "generated_rows": len(feature_rows),
            "re_values": RE_VALUES,
            "feature_columns": FEATURE_COLUMNS,
            "inference_latency_ms": latency_ms,
            "data_source": position_source,
            "curvature_radius_um": curvature_radius_um,
            "reference_curvature_radius_um": REFERENCE_CURVATURE_RADIUS_UM,
            "dean_curvature_factor": dean_factor,
            "fd_model_route": "native_target_with_heuristic_radius_correction",
            "fl_output_mode": "native_projection_with_unverified_transverse_component",
            "fd_output_mode": "native_u_with_unverified_curvature_correction",
            "fnet_definition": "fnet_pred = fl_pred + fd_pred; vector magnitude = hypot(fnet_x, fnet_z)",
            "uses_tabpfn": uses_tabpfn,
            "uses_model": model_predictions is not None,
            "model_revision": MODEL_MANAGER.revision,
            "physical_units_verified": False,
            "fnet_physically_validated": False,
            "radius_is_model_feature": False,
            "curvature_correction_method": "heuristic_clipped_reference_radius_ratio",
            "focus_is_learned_prediction": False,
            "warnings": [
                "FD training target u has unverified physical units; FD+FL is a legacy arithmetic output, not a validated force sum.",
                "Zero transverse components may be placeholders, not learned vector components.",
                "Radius correction and focus status are heuristic, not learned or calibrated model predictions.",
            ],
            "model_error": model_error or MODEL_MANAGER.error,
            "cache_hit": False,
        },
    }
    if cache_enabled and model_error is None:
        with CACHE_LOCK:
            PREDICTION_CACHE[cache_key] = copy.deepcopy(result)
            PREDICTION_CACHE.move_to_end(cache_key)
            while len(PREDICTION_CACHE) > cache_capacity:
                PREDICTION_CACHE.popitem(last=False)
    return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("TRIFOCUS_ACCESS_LOG", "0") == "1":
            super().log_message(fmt, *args)

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "engine": "python-tabpfn-cpu-service",
                    "model_loaded": MODEL_MANAGER.ready,
                    "model_version": MODEL_VERSION,
                    "uses_tabpfn": MODEL_MANAGER.ready,
                    "model": MODEL_MANAGER.health(),
                    "point_count_per_re": POINT_COUNT,
                    "re_values": RE_VALUES,
                    "curvature_radius_um": {
                        "min": MIN_CURVATURE_RADIUS_UM,
                        "max": MAX_CURVATURE_RADIUS_UM,
                        "reference": REFERENCE_CURVATURE_RADIUS_UM,
                    },
                },
            )
            return
        if self.path == "/api/schema":
            self.send_json(
                HTTPStatus.OK,
                {
                    "service": "TriFocus TabPFN CPU API",
                    "version": MODEL_VERSION,
                    "feature_columns": FEATURE_COLUMNS,
                    "target_columns": TARGET_COLUMNS,
                    "prediction_contract": {
                        "point_count_per_re": POINT_COUNT,
                        "supports_provided_positions": True,
                        "position_columns": ["x_um", "z_um"],
                        "re_values": RE_VALUES,
                        "generated_rows_per_request": POINT_COUNT * len(RE_VALUES),
                        "curvature_radius_um": {
                            "min": MIN_CURVATURE_RADIUS_UM,
                            "max": MAX_CURVATURE_RADIUS_UM,
                            "reference": REFERENCE_CURVATURE_RADIUS_UM,
                            "effect": "Dean force is corrected by reference_radius / curvature_radius.",
                        },
                        "uses_tabpfn": MODEL_MANAGER.ready and MODEL_MANAGER.uses_tabpfn,
                        "model_mode": MODEL_MANAGER.mode,
                    },
                },
            )
            return
        self.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict-triangle":
            self.send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            self.send_json(HTTPStatus.OK, predict_triangle(payload))
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": str(exc)})
        except Exception as exc:  # pragma: no cover
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "error": str(exc)})


def write_feature_template(path: Path) -> None:
    rows = [
        {
            **{column: "" for column in FEATURE_COLUMNS},
            **{column: "" for column in TARGET_COLUMNS.values()},
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TriFocus TabPFN CPU inference service.")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8100")))
    parser.add_argument("--write-training-template", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing fl_x/fl_y/fd_x/fd_y.joblib")
    parser.add_argument("--predict-json", type=Path, default=None, help="Run one prediction from a JSON request and exit")
    parser.add_argument("--output-json", type=Path, default=Path("results/triangle_force_prediction.json"))
    args = parser.parse_args()

    if args.write_training_template:
        write_feature_template(args.write_training_template)
        print(f"Wrote template: {args.write_training_template}")
        return

    if args.model_dir is not None:
        model_env = {
            "TRIFOCUS_FLX_MODEL": "fl_x.joblib",
            "TRIFOCUS_FLY_MODEL": "fl_y.joblib",
            "TRIFOCUS_FDX_MODEL": "fd_x.joblib",
            "TRIFOCUS_FDY_MODEL": "fd_y.joblib",
        }
        for env_name, file_name in model_env.items():
            os.environ[env_name] = str(args.model_dir / file_name)
        os.environ["TRIFOCUS_MODEL_MODE"] = "joblib"

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("POSTHOG_DISABLED", "1")
    MODEL_MANAGER.load()
    if args.predict_json is not None:
        payload = json.loads(args.predict_json.read_text(encoding="utf-8"))
        result = predict_triangle(payload)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "output_json": args.output_json.as_posix(),
                    "uses_tabpfn": result["prediction_meta"]["uses_tabpfn"],
                    "point_count_per_re": result["prediction_meta"]["point_count_per_re"],
                    "generated_rows": result["prediction_meta"]["generated_rows"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print(f"TriFocus TabPFN CPU service: http://{args.host}:{args.port}")
    print(json.dumps(MODEL_MANAGER.health(), ensure_ascii=False))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping service.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
