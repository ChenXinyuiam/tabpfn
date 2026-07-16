# TriFocus ForceLab Demo

This is the presentation-ready demo website for triangle-section force-field prediction.

## Recommended full demo

Run from the repository root:

```powershell
python web-demo\server.py --host 127.0.0.1 --port 8000
```

Keep that terminal window open, then open:

```text
http://127.0.0.1:8000
```

This mode serves:

- static website files;
- `GET /api/health`;
- `GET /api/schema`;
- `POST /api/predict-triangle`.

## BT Panel / Node deployment

For BT Panel Node project deployment, use:

```bash
npm start
```

This runs `server.js`, which serves the website and the same API endpoints without extra Node dependencies.

Read `BAOTA_DEPLOYMENT.md` before uploading files to the server.

The frontend calls the Python API first. The backend receives three side lengths and `radius_group`, then generates the triangle, 200 internal sample points, the Re sweep, force-field values, and the focus window.

## Offline fallback

If the local server is not convenient, open:

```text
file:///D:/2026/tab/FDFLPredictions/web-demo/index.html
```

The page will automatically switch to browser-only surrogate mode.

## Current demo scope

- Input `AB`, `AC`, and `BC` side lengths plus required curvature-radius group `R = 10/20/30`.
- Route the demo FD calculation by `radius_group`; changing R changes Dean-force and focus results.
- Generate a valid triangle and 200 internal sample points.
- Sweep `Re = 6, 12, ..., 60`.
- Show `FL-x`, `FL-y`, `Dean-x`, `Dean-y`, total force, focus region, and focus Re window.
- Support Chinese/English UI switching.
- Provide three demo presets: balanced, tall, and asymmetric triangles.
- Export CSV, PNG, and a Markdown report.
- Display generated row count, inference latency, and engine status.

## Useful files

- `API_CONTRACT.md`: backend request/response contract and model boundary.
- `DEMO_SCRIPT.md`: 3-minute Chinese demo script and judge Q&A.
- `BAOTA_DEPLOYMENT.md`: BT Panel Node project upload and update guide.
- `run_demo.ps1`: one-command Windows launcher.

## Important model note

The current backend is still a deterministic demo surrogate. It is a real API-shaped backend, but it does **not** use TabPFN yet.

Production replacement path:

```text
POST /api/predict-triangle
  -> generate 200 points x Re values
  -> build model feature table
  -> route the FD model by radius_group
  -> call TabPFN FD/FL inference
  -> return force field and focus result
```
