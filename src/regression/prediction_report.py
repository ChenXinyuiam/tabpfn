"""Native-target field statistics without mixing FD-u and FL units."""

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors

from src.focus_inspection.plotting import configure_fonts, plt
from utils.provenance import file_sha256, write_json
from utils.runs import experiment_run


def statistics(data, keys):
    records = []
    for key, group in data.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        values = group.loc[np.isfinite(group.prediction), "prediction"]
        n = len(values)
        distances = group.nearest_training_distance_um.dropna()
        records.append({**dict(zip(keys, key, strict=True)), "query_rows": len(group), "predicted_rows": n,
                        "missing_predictions": len(group) - n, "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)), "min": float(values.min()), "q25": float(values.quantile(.25)),
                        "median": float(values.median()), "q75": float(values.quantile(.75)), "max": float(values.max()),
                        "positive_fraction": float(values.gt(0).mean()) if n else np.nan,
                        "zero_fraction": float(values.eq(0).mean()) if n else np.nan,
                        "review_rows": int(group.needs_review.sum()),
                        "training_position_overlap_rows": int(group.training_position_overlap.sum()),
                        "outside_hull_rows": int(group.spatial_relation.eq("outside_training_hull").sum()),
                        "missing_training_re_rows": int(group.spatial_relation.eq("missing_training_re").sum()),
                        "nearest_training_distance_median_um": float(distances.median()),
                        "nearest_training_distance_p95_um": float(distances.quantile(.95))})
    return pd.DataFrame(records)


def write_report(directory, data, audits, inventory):
    configure_fonts()
    by_field = statistics(data, ["field"])
    by_pair = statistics(data, ["pair_key", "radius_label", "field"])
    by_re = statistics(data, ["pair_key", "radius_label", "field", "re"])
    for name, frame in [("by_field", by_field), ("by_pair", by_pair), ("by_re", by_re)]:
        frame.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")
    reasons = data.assign(reason=data.review_reasons.str.split(";")).explode("reason")
    reasons = reasons[reasons.reason.ne("")]
    reason_counts = reasons.groupby(["field", "reason"]).size().rename("rows").reset_index()
    reason_counts.to_csv(directory / "review_reasons.csv", index=False, encoding="utf-8-sig")
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    panels = []
    plotted = 0
    for pair, group in data.groupby("pair_key", sort=True):
        limits = {}
        for field, part in group.groupby("field"):
            finite = part.prediction[np.isfinite(part.prediction)]
            limits[field] = max(float(finite.abs().max()) if len(finite) else 0, 1e-30)
        for re_value in sorted(group.re.dropna().unique()):
            frame = group[group.re.eq(re_value)]
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
            counts = []
            for ax, field in zip(axes, ["FD_u", "FL"], strict=True):
                current = frame[frame.field.eq(field)]
                valid = np.isfinite(current[["x", "z", "prediction"]]).all(axis=1)
                points = current.loc[valid]
                plotted += len(points)
                # A shared color scale within each field/pair makes Re slices comparable.
                norm = colors.Normalize(-limits[field], limits[field])
                outside = points.spatial_relation.ne("inside_training_hull")
                plot = ax.scatter(points.loc[~outside, "x"], points.loc[~outside, "z"], c=points.loc[~outside, "prediction"], s=9, cmap="RdBu_r",
                    norm=norm, rasterized=True, label="Inside training hull")
                if outside.any():
                    ax.scatter(points.loc[outside, "x"], points.loc[outside, "z"], c=points.loc[outside, "prediction"], s=12, marker="x", linewidths=.5, cmap="RdBu_r", norm=norm, rasterized=True, label="Outside/unsupported")
                ax.set(xlabel="x (um)", ylabel="z (um)", title=f"{field}: {len(points)}/{len(current)} rows; unsupported={int(outside.sum())}")
                ax.legend(fontsize=7, loc="best")
                ax.set_aspect("equal", adjustable="box")
                fig.colorbar(plot, ax=ax, label=f"{field} (native target units)")
                counts.append(f"{field}={len(points)}")
            name = f"field_{len(panels) + 1:03d}.png"
            fig.savefig(figures / name, dpi=145)
            plt.close(fig)
            panels.append((f"{pair} | Re={re_value:g} | {', '.join(counts)}", name))
    summary = {"selected_pairs": int(data.pair_key.nunique()), "query_rows": len(data),
               "predicted_rows": int(data.prediction.notna().sum()), "plotted_rows": plotted,
               "review_rows": int(data.needs_review.sum()), "unpaired_workbooks": sum(row["status"] != "paired" for row in inventory),
               "training_input_rows": sum(row["training_input_rows"] for row in audits),
               "training_used_rows": sum(row["training_used_rows"] for row in audits),
               "evaluation": False, "error_metrics": None, "force_sum_computed": False,
               "target_units_verified": False, "radius_label_unit_verified": False,
               "source_rows_preserved": True, "synthetic_training_rows": 0}
    write_json(directory / "statistics.json", summary)
    messages = [
        f"已处理 {summary['selected_pairs']} 对工作簿，合计 {summary['query_rows']} 个目标位置记录；成功预测 {summary['predicted_rows']} 个，空间图涵盖 {plotted} 个。两种目标分开统计，不混合求均值。",
        "FD_u 使用 FD 文件的原始 u 作为目标，在配对 FL 的位置预测；FL 使用原始 fl，在配对 FD 的位置预测。FL 不使用插值生成的伪标签训练。",
        "u 的物理含义和目标单位尚未确认，保留原始数值，不把 FD_u 与 FL 相加。文件名中的 10/20/30 只作为半径标签记录，不擅自映射到服务中的微米曲率半径。",
        f"本次使用 {summary['training_used_rows']} 个训练记录（原始输入 {summary['training_input_rows']} 个），训练上限、各 Re 数量和单位转换见审计文件。输入数据不被覆盖。",
        f"目录中另有 {summary['unpaired_workbooks']} 份未配对工作簿，详情见配对清单。未选中的工作簿不包含在本次预测统计中。",
        "距离与凸包按相同 Re、实际用于训练的点计算；越界、缺少训练 Re、退化凸包和训练坐标重合均提示复核。距离小不等于预测可靠。",
        "这不是留出测试，不报告 MAE、RMSE 或 R²，也不把训练坐标上的拟合结果当作独立准确率。未知几何性能仍需完整几何留出验证。",
        "每张图保留原目标位置，不做插值补图。相同坐标可能重合；标题列出所含记录数。每种目标在同一文件的全部 Re 中使用一致色标，但 FD_u 与 FL 的色标和单位不能互比。",
    ]
    visible = by_pair[["pair_key", "field", "query_rows", "predicted_rows", "mean", "median", "min", "max", "outside_hull_rows", "review_rows"]].rename(
        columns={"pair_key": "样本对", "field": "目标", "query_rows": "目标记录", "predicted_rows": "已预测", "mean": "均值", "median": "中位数", "min": "最小值", "max": "最大值", "outside_hull_rows": "凸包外位置", "review_rows": "复核记录"})
    css = "body{max-width:1150px;margin:24px auto;padding:0 16px;font:15px/1.6 Arial,sans-serif;color:#242424}h1{font-size:25px}h2{font-size:18px}table{border-collapse:collapse;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:right}.table{overflow:auto}img{display:block;max-width:100%;height:auto}nav{display:flex;gap:14px;flex-wrap:wrap}details{padding:12px 0;border-bottom:1px solid #ddd}summary{cursor:pointer;overflow-wrap:anywhere}input{padding:8px;width:min(90%,480px)}"
    (directory / "report.md").write_text("# FD-u / FL 预测统计\n\n" + "\n\n".join(messages) + "\n", encoding="utf-8")
    (directory / "gallery.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>FD-u / FL 预测统计</title><style>{css}</style><h1>FD-u / FL 预测统计</h1>'
        + "".join(f"<p>{html.escape(text)}</p>" for text in messages)
        + '<nav><a href="predictions.csv">逐点预测</a><a href="by_re.csv">逐 Re 统计</a><a href="pair_inventory.csv">配对清单</a><a href="prediction_audit.json">训练审计</a></nav>'
        '<h2>分目标统计</h2><div class="table">' + visible.to_html(index=False, border=0, float_format=lambda v: f"{v:.5g}")
        + '</div><h2>空间分布</h2><label>筛选文件或 Re <input id="query" type="search"></label>'
        + "".join(f'<details {"open" if index == 0 else ""}><summary>{html.escape(title)}</summary><img src="figures/{name}" alt="FD-u 与 FL 在原始坐标上的预测分布"></details>' for index, (title, name) in enumerate(panels))
        + '<script>document.getElementById("query").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("details").forEach(d=>d.hidden=!d.querySelector("summary").textContent.toLowerCase().includes(q));});</script></html>', encoding="utf-8")


def regenerate_report(source, out_dir):
    source = Path(source)
    status = json.loads((source / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("kind") != "regression-predict":
        raise ValueError("Report requires a completed regression-predict run")
    data = pd.read_csv(source / "predictions.csv", encoding="utf-8-sig", float_precision="round_trip", keep_default_na=False,
                       na_values={"prediction": [""], "re": [""], "x": [""], "z": [""], "nearest_training_distance_um": [""]})
    if data.empty or data.duplicated(["pair_key", "field", "query_file", "source_row"]).any():
        raise ValueError("Prediction rows must be nonempty and unique")
    audits = json.loads((source / "prediction_audit.json").read_text(encoding="utf-8"))["directions"]
    inventory = pd.read_csv(source / "pair_inventory.csv", keep_default_na=False).to_dict("records")
    files = ["predictions.csv", "prediction_audit.json", "pair_inventory.csv", "protocol.json"]
    with experiment_run(out_dir, "regression-report", extra_sources=[Path(__file__)]) as directory:
        hashes = {name: file_sha256(source / name) for name in files}
        for name in files:
            shutil.copy2(source / name, directory / name)
        write_report(directory, data, audits, inventory)
        if any(file_sha256(source / name) != sha for name, sha in hashes.items()):
            raise ValueError("Source prediction run changed while creating report")
        write_json(directory / "report_provenance.json", {"prediction_run": str(source.resolve()), "source_hashes": hashes, "model_fits": 0})
        return directory


def main():
    parser = argparse.ArgumentParser(description="Regenerate FD-u/FL statistics without rerunning models")
    parser.add_argument("--prediction-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/regression_prediction/reports"))
    args = parser.parse_args()
    regenerate_report(args.prediction_run, args.out_dir)


if __name__ == "__main__":
    main()
