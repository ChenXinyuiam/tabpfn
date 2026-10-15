"""Offline, escaped reports built only from saved numerical artifacts."""

import html
import json
from pathlib import Path

import pandas as pd


def write_report(directory, title, summary):
    directory = Path(directory)
    sections = []
    for name in ("issues.csv", "by_geometry_re.csv", "focus_oof.csv", "focus_predictions.csv", "diagnostics.csv"):
        path = directory / name
        if not path.exists() or path.stat().st_size < 3:
            continue
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        sections.append(
            f"<h2>{html.escape(name)}</h2><p>{len(frame)} rows; preview limited to 30.</p>"
            f'<div class="table">{frame.head(30).to_html(index=False, escape=True)}</div>'
        )
    pictures = []
    points_path = directory / "point_predictions.csv"
    if points_path.is_file():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        points = pd.read_csv(points_path)
        figures = directory / "figures"
        figures.mkdir(exist_ok=True)
        for index, (sample_id, part) in enumerate(points.groupby("sample_id", sort=True)):
            if index >= 6:
                break
            for field in ("FD_pred", "FL_pred", "Fzong_pred"):
                if field not in part:
                    continue
                fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
                scatter = ax.scatter(part.x, part.z, c=part[field], s=12, cmap="coolwarm")
                ax.set(xlabel="x (um)", ylabel="z (um)", title=f"{field} | Re={part.re.iloc[0]:g}")
                ax.set_aspect("equal")
                fig.colorbar(scatter, ax=ax)
                name = f"{index}-{field}.png"
                fig.savefig(figures / name, dpi=130)
                plt.close(fig)
                pictures.append(
                    f'<figure><img src="figures/{name}" alt="{field} spatial prediction"><figcaption>{html.escape(sample_id)}</figcaption></figure>'
                )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:15px/1.7 system-ui,sans-serif;color:#20352b;margin:32px auto;padding:0 20px;max-width:1150px}}h1{{font-size:28px}}h2{{font-size:20px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f3;padding:20px}}.table{{overflow:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border:1px solid #d6dfda}}img{{max-width:100%;height:auto}}figure{{display:inline-block;margin:10px 10px 10px 0;max-width:480px}}@media print{{body{{margin:0}}pre{{font-size:9px}}}}</style>
<h1>{html.escape(title)}</h1><p>Versioned run. Prediction diagnostics are not held-out accuracy measurements.</p>
{"".join(sections)}{"".join(pictures)}<h2>Run summary</h2><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, default=str))}</pre></html>"""
    (directory / "report.html").write_text(body, encoding="utf-8")
