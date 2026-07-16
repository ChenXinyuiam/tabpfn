# TriFocus ForceLab API Contract

This document describes the demo backend used by the website.

## Base URL

```text
http://127.0.0.1:8000
```

## GET /api/health

Checks whether the backend is running.

Example response:

```json
{
  "status": "ok",
  "engine": "python-demo-backend",
  "model_loaded": true,
  "model_version": "demo-backend-v2-radius",
  "uses_tabpfn": false,
  "point_count_per_re": 200,
  "re_values": [6, 12, 18, 24, 30, 36, 42, 48, 54, 60],
  "supported_radius_groups": [10, 20, 30]
}
```

## GET /api/schema

Returns the API shape, data contract, Re sweep, sample count, and current model boundary.

Use this endpoint during judging if someone asks what data the demo produces.

## POST /api/predict-triangle

Generates a complete force-field prediction from three triangle side lengths and a required curvature-radius group.

Request:

```json
{
  "side_ab_um": 690,
  "side_ac_um": 520,
  "side_bc_um": 750,
  "radius_group": 20
}
```

Response fields:

- `geometry`: triangle vertices, area, perimeter, aspect ratio, inradius, and centroid.
- `basePoints`: 200 generated internal sampling points.
- `sweep`: prediction rows grouped by Reynolds number.
- `bestRe`: the Reynolds number with the best predicted focusing score.
- `focus`: predicted focus center, radius, score, status, and Re window.
- `prediction_meta`: engine, model version, `radius_group`, FD model route, generated row count, latency, output modes, and TabPFN status.

## Current Model Boundary

The current backend is a deterministic demonstration surrogate. It is designed to make the full website pipeline demonstrable:

```text
three side lengths + radius_group (10 / 20 / 30)
  -> triangle validation and geometry
  -> 200 internal sample points
  -> Re sweep
  -> radius-specific FD route
  -> FL-x / FL-y / Dean-x / Dean-y
  -> total force field
  -> focus window and focus location
```

It does not call TabPFN yet. The production upgrade should keep the same endpoint and replace only the predictor implementation:

```text
POST /api/predict-triangle
  -> generate feature table
  -> route the FD model by radius_group
  -> call TabPFN FD/FL models
  -> aggregate force field
  -> return the same response contract
```
