"""Human-readable sample review without changing saved evaluation artifacts."""

import html

from .ablation_report import STYLE
from .plotting import configure_fonts, plt


def write_report(directory, samples, groups, summary):
    configure_fonts()
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(groups) * 0.5)), constrained_layout=True)
    ordered = groups.sort_values(["accuracy", "geometry_group"])
    bars = ax.barh(ordered.geometry_group, ordered.accuracy, color="#387ca6")
    ax.bar_label(
        bars,
        labels=[f"{a:.1%}  (n={n})" for a, n in zip(ordered.accuracy, ordered.unique_samples, strict=True)],
        padding=4,
    )
    ax.set(xlim=(0, 1.22), xlabel="Mean held-out accuracy across seeds", ylabel="Geometry")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "geometry_accuracy.png", dpi=160)
    plt.close(fig)
    intervals = summary["accuracy_intervals"]
    text = [
        f"模型：{summary['model']}。{summary['unique_samples']} 个去重样本，{summary['geometry_groups']} 个几何组，{len(summary['seeds'])} 次固定种子实验，共 {summary['prediction_rows']} 条预测记录。同组样本不视为统计独立。",
        f"平均准确率 {intervals['sample_weighted_accuracy']:.2%}；按几何组重采样的描述性 95% 区间为 {intervals['sample_weighted_ci95'][0]:.2%} 至 {intervals['sample_weighted_ci95'][1]:.2%}。各几何组等权平均准确率为 {intervals['geometry_macro_accuracy']:.2%}。",
        f"曾误判 {summary['ever_wrong_samples']} 个样本，其中每次都误判 {summary['persistent_error_samples']} 个、跨次判断不一致 {summary['unstable_samples']} 个；曾出现高置信误判 {summary['high_confidence_error_samples']} 个。高置信按 F 概率 >=0.9 或 <=0.1 统计。",
        f"几何矛盾涉及 {summary['geometry_warning_samples']} 个样本，综合复核队列共 {summary['review_samples']} 个样本。误判与数据疑点可能重叠，不能直接相加。",
        "复核优先级不是标签正确性的判定。原始数值、F/UF 标签和已保存的模型评估均未修改，也未自动删除样本或重新训练。",
        "同一几何组的全部样本和重复种子一起重采样。只有少量几何组且数据已参与开发，这些区间不代表新数据上的准确率保证；概率范围反映重复运行差异，不是预测置信区间。",
    ]
    if not summary["geometry_audit_available"]:
        text.append("该历史实验未记录几何一致性审计，几何警告计数为未评估，不能解释为没有问题。")
    columns = {
        "source_file": "文件",
        "R": "R",
        "re": "Re",
        "label": "标签",
        "error_seeds": "误判次数",
        "seed_count": "实验次数",
        "probability_min": "最低 F 概率",
        "probability_max": "最高 F 概率",
        "boundary_distance_abs": "区间宽度 (um)",
        "boundary_midpoint_x": "中点 x (um)",
    }
    rows = []
    for _, row in samples.iterrows():
        tags = []
        for name, label in [
            ("persistent_error", "持续误判"),
            ("decision_unstable", "判断不稳定"),
            ("geometry_warning", "几何矛盾"),
            ("rule_conflict", "100um 规则与标签不符"),
        ]:
            if row[name]:
                tags.append(label)
        if row.error_seeds and not row.persistent_error:
            tags.append("部分实验误判")
        if row.needs_review and not tags:
            tags.append("数据质量提示")
        if not tags:
            tags.append("未触发复核")
        values = []
        for name in columns:
            value = row[name]
            values.append(html.escape(f"{value:.5g}" if isinstance(value, float) else str(value)))
        status = "、".join(tags)
        values.append(html.escape(status))
        rows.append(
            f'<tr data-review="{int(row.needs_review)}" data-error="{int(row.error_seeds > 0)}" data-geometry="{int(row.geometry_warning)}">'
            + "".join(f"<td>{v}</td>" for v in values)
            + "</tr>"
        )
    (directory / "report.md").write_text("# 聚焦分类复核\n\n" + "\n\n".join(text) + "\n", encoding="utf-8")
    (directory / "gallery.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>聚焦分类复核</title><style>{STYLE}select{{padding:8px}}td,th{{white-space:nowrap}}nav{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}input{{box-sizing:border-box}}</style>"
        "<h1>聚焦分类复核</h1>"
        + "".join(f"<p>{html.escape(t)}</p>" for t in text)
        + '<nav><a href="review_queue.csv">复核清单</a><a href="sample_summary.csv">全部样本</a><a href="geometry_summary.csv">分组统计</a></nav>'
        '<h2>各几何组表现</h2><img src="figures/geometry_accuracy.png" alt="各几何组的平均留出准确率与去重样本数">'
        '<h2>样本明细</h2><nav><label>搜索 <input id="query" type="search"></label>'
        '<label>范围 <select id="scope"><option value="all">全部样本</option><option value="review">待复核</option><option value="error">曾误判</option><option value="geometry">几何矛盾</option></select></label></nav>'
        '<div class="table"><table><thead><tr>'
        + "".join(f"<th>{html.escape(v)}</th>" for v in [*columns.values(), "状态"])
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        '<script>const query=document.getElementById("query"),scope=document.getElementById("scope");function filter(){const q=query.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q)||(scope.value!=="all"&&r.dataset[scope.value]!=="1");});}query.addEventListener("input",filter);scope.addEventListener("change",filter);</script></html>',
        encoding="utf-8",
    )
