# TriFocus ForceLab Demo

This folder contains a static demo website for the triangle-section force-field prediction requirements.

Open:

```text
web-demo/index.html
```

Local static server preview:

```powershell
python -m http.server 8000 -d web-demo
```

Then open:

```text
http://localhost:8000
```

Server deployment can use any static host such as Nginx, Apache, or a CDN-backed object bucket. Upload the whole `web-demo/` folder as the site root.

Current scope:

- Input only `AB`, `AC`, and `BC` side lengths.
- Generate a valid triangle and 200 internal sample points.
- Sweep `Re = 6, 12, ..., 60`.
- Show `FL-x`, `FL-y`, `Dean-x`, `Dean-y`, total force, focus region, and focus Re window.
- Support Chinese/English UI switching.
- Export CSV and PNG.

The current prediction engine is a deterministic front-end surrogate for demonstration. The UI and data fields are aligned with `docs/WEBSITE_REQUIREMENTS.md`; the surrogate can later be replaced by a real backend call to `POST /api/predict-triangle`.
