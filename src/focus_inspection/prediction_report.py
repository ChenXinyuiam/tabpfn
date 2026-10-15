"""Prediction-only figures and a searchable Chinese batch report."""

import html

import numpy as np

from utils.provenance import write_json

from .ablation_report import STYLE
from .plotting import configure_fonts, plt
from .prediction_statistics import summarize_predictions

REASONS = {
    "inconsistent_triangle_geometry": "几何字段矛盾",
    "invalid_rows_dropped": "无效点已按配置排除",
    "conflicting_duplicate_points": "重复坐标力符号冲突",
    "multiple_sign_transitions": "平滑后仍有多次符号变化",
    "undefined_boundary": "边界无法定义",
    "coordinate_grid_differs_within_file": "同文件坐标覆盖不一致",
    "missing_model_features": "模型输入存在缺失",
    "outside_training_range": "超出训练特征范围",
    "unseen_geometry": "训练中未见的几何",
    "training_sample_not_independent": "与训练数据内容重合",
    "same_key_changed_or_unverified_content": "同名样本内容改变或未核实",
    "near_decision_threshold": "接近分类阈值",
    "legacy_recipe_config_unverified": "旧模型配置未完全核实",
    "duplicate_input_content": "本批次存在重复内容",
}


def draw_figures(directory, data, tables):
    configure_fonts()
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, color in [("UF", "#387ca6"), ("F", "#b94a48")]:
        rows = data[data.focus_prediction.eq(label)]
        axes[0].hist(
            rows.focus_probability,
            bins=np.linspace(0, 1, 11),
            histtype="step",
            linewidth=2,
            color=color,
            label=f"Predicted {label} (n={len(rows)})",
        )
        counts = tables["boundary_plot_counts"]
        points = counts[counts.focus_prediction.eq(label)]
        axes[1].scatter(
            points.boundary_distance_abs,
            points.boundary_midpoint_x,
            s=30 * points.samples,
            alpha=0.7,
            facecolors="none" if label == "UF" else color,
            edgecolors=color,
            marker="o" if label == "UF" else "^",
            label=f"Predicted {label}",
        )
    axes[0].axvline(data.threshold.iloc[0], color="#444444", linestyle="--", label="Fixed threshold")
    axes[0].set(xlim=(0, 1), xlabel="Predicted F probability", ylabel="Samples")
    axes[1].axvline(100, color="#444444", linestyle="--", label="100 um reference")
    axes[1].set(
        xlabel="Boundary width (um)", ylabel="Boundary midpoint x (um)", title="Marker area = coincident sample count"
    )
    axes[1].margins(x=0.08, y=0.12)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.savefig(figures / "probability_and_boundary.png", dpi=160)
    plt.close(fig)

    rows = data.assign(file_radius=data.source_file + " | R=" + data.R.map(lambda v: f"{v:g}"))
    grid = rows.pivot(index="file_radius", columns="re", values="focus_probability").sort_index(axis=1)
    heatmap_files = []
    for start in range(0, len(grid), 30):
        part = grid.iloc[start : start + 30]
        fig, ax = plt.subplots(
            figsize=(max(8, min(16, 0.55 * len(part.columns) + 5)), max(3, 0.34 * len(part) + 1.5)),
            constrained_layout=True,
        )
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("#dedede")
        im = ax.imshow(np.ma.masked_invalid(part.to_numpy()), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(part)), part.index, fontsize=8)
        ax.set_xticks(
            range(len(part.columns)),
            [f"{v:g}" for v in part.columns],
            fontsize=8,
            rotation=45 if len(part.columns) > 15 else 0,
        )
        ax.set(xlabel="Re", ylabel="Excel x R")
        fig.colorbar(im, ax=ax, label="Predicted F probability")
        name = f"probability_map_{start // 30 + 1:02d}.png"
        fig.savefig(figures / name, dpi=150)
        plt.close(fig)
        heatmap_files.append(name)
    by_re = tables["by_reynolds"]
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    positions = np.arange(len(by_re))
    ax.bar(positions, by_re.predicted_uf, label="Predicted UF", color="#387ca6")
    ax.bar(positions, by_re.predicted_f, bottom=by_re.predicted_uf, label="Predicted F", color="#b94a48")
    ax.set_xticks(positions, [f"{v:g}" for v in by_re.re])
    ax.set(xlabel="Re", ylabel="Samples (not accuracy)")
    ax.legend()
    fig.savefig(figures / "counts_by_re.png", dpi=150)
    plt.close(fig)
    return heatmap_files


def write_prediction_report(directory, data):
    summary, tables = summarize_predictions(data)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False, encoding="utf-8-sig")
    write_json(directory / "prediction_statistics.json", summary)
    maps = draw_figures(directory, data, tables)
    messages = [
        f"本批 {summary['samples']} 个 Excel × R × Re 样本，来自 {summary['files']} 个文件、{summary['radii']} 个 R、{summary['reynolds_numbers']} 个 Re。",
        f"模型预测 F：{summary['predicted_f']}，UF：{summary['predicted_uf']}；预测 F 占比 {summary['predicted_f_rate']:.1%}。这是预测构成，不是准确率，也不是真实聚焦比例。",
        f"待复核 {summary['review_samples']} 个，其中数据质量提示 {summary['data_quality_warning_samples']} 个；与训练内容重合 {summary['known_training_samples']} 个。提示可重叠，不能相加。未触发提示也不保证预测正确。",
        f"分类阈值固定为 {summary['threshold']:g}。预测不读取本批标签，不拟合新阈值、不重新选择特征。报告不计算准确率。",
        f"宽度可用 {summary['width_available_samples']}/{summary['samples']}，中点可用 {summary['midpoint_available_samples']}/{summary['samples']}。缺失值保留为空，不以零代替；散点图只显示两项都可用的样本。",
        f"边界图涵盖 {summary['boundary_plot_samples']} 个样本，同一预测类别、相同坐标的样本以标记面积表示数量；UF 用空心圆，F 用三角形。每个样本仍保留在热图和明细表中。",
        "Re 统计为当前采样组成，未配平文件、R 或几何组，不应据此认定 Re 的因果作用。热图每格对应一个原始样本，灰色表示缺少该组合，不进行插值或平均。",
    ]
    if "duplicate_content_extra_samples" in summary:
        messages.append(
            f"按文件内容校验值 × R × Re 去重后为 {summary['verified_content_sample_units']} 个已核实内容单元，额外重复 {summary['duplicate_content_extra_samples']} 个；原记录均保留，主表统计不静默去重。"
        )
    reasons = tables["review_reasons"].copy()
    reasons["reason"] = reasons.reason.map(lambda v: REASONS.get(v, v))
    reasons = reasons.rename(
        columns={"reason": "复核原因", "samples": "样本数", "fraction_of_all_samples": "占全批比例"}
    )
    columns = {
        "source_file": "文件",
        "R": "R",
        "re": "Re",
        "focus_prediction": "模型预测",
        "decision": "处理状态",
        "focus_probability": "F 概率",
        "boundary_distance_abs": "区间宽度 (um)",
        "boundary_midpoint_x": "中点 x (um)",
    }
    body = []
    for _, row in data.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(
                "缺失"
                if isinstance(value, float) and not np.isfinite(value)
                else (f"{value:.5g}" if isinstance(value, float) else str(value))
            )
        translated = "；".join(REASONS.get(v, v) for v in str(row.review_reasons).split(";") if v)
        values.append(translated or "未触发复核")
        attrs = f'data-review="{int(row.needs_review)}" data-training="{int(row.known_training_sample)}" data-quality="{int(row.data_quality_warning)}" data-new="{int(not row.known_training_sample)}"'
        body.append(f"<tr {attrs}>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    (directory / "report.md").write_text("# 聚焦预测统计\n\n" + "\n\n".join(messages) + "\n", encoding="utf-8")
    (directory / "gallery.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>聚焦预测统计</title><style>{STYLE}td,th{{white-space:nowrap}}nav{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}select{{padding:8px}}input{{box-sizing:border-box}}.figures{{display:grid;gap:20px}}</style>"
        "<h1>聚焦预测统计</h1>"
        + "".join(f"<p>{html.escape(message)}</p>" for message in messages)
        + '<nav><a href="predictions.csv">预测明细</a><a href="by_file.csv">文件统计</a><a href="by_radius.csv">R 统计</a><a href="by_reynolds.csv">Re 统计</a><a href="feature_statistics.csv">特征分布</a></nav>'
        '<h2>概率与边界特征</h2><img src="figures/probability_and_boundary.png" alt="预测概率分布与边界宽度、中点坐标散点图">'
        '<h2>各样本的 F 概率</h2><div class="figures">'
        + "".join(f'<img src="figures/{name}" alt="按文件、R 与 Re 排列的 F 概率热图">' for name in maps)
        + '</div><h2>Re 样本构成</h2><img src="figures/counts_by_re.png" alt="各 Re 的预测 F 与 UF 数量">'
        '<h2>复核原因</h2><div class="table">'
        + reasons.to_html(index=False, border=0, float_format=lambda v: f"{v:.1%}")
        + '</div><h2>预测明细</h2><nav><label>搜索 <input id="query" type="search"></label><label>范围 <select id="scope"><option value="all">全部样本</option><option value="review">待复核</option><option value="quality">数据质量提示</option><option value="training">训练内容重合</option><option value="new">非训练原内容</option></select></label></nav>'
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{html.escape(v)}</th>" for v in [*columns.values(), "复核原因"])
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        '<script>const query=document.getElementById("query"),scope=document.getElementById("scope");function filter(){const q=query.value.toLowerCase();document.querySelectorAll("tbody tr[data-review]").forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q)||(scope.value!=="all"&&r.dataset[scope.value]!=="1");});}query.addEventListener("input",filter);scope.addEventListener("change",filter);</script></html>',
        encoding="utf-8",
    )
    return summary
