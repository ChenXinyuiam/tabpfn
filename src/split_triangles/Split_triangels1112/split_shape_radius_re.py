# -*- coding: utf-8 -*-
"""
这个文件用于将数据集按照  聚焦状态-形状-曲率半径-雷诺数  进行分层并落盘为层级目录。
目录结构: <output_dir>/<focus>/<shape>/<R>/<Re>/data.csv
"""

import os
import math
import pandas as pd

# ========================= 配置区 =========================
# 五个基本形状
ALLOWED_SHAPES = {"23_125_32", "125_23_32", "40_40_100", "25_90_65", "90_25_65"}
# 3 个曲率半径
ALLOWED_RADII = {10, 20, 30}
# 10 个雷诺数 6, 12, ..., 60
ALLOWED_RE = set(range(6, 61, 6))

# 候选列名（支持中英文/多版本命名）
ANGLE_COLS = ["底角θ1", "底角θ2", "顶角θ"]
RE_COL_CANDIDATES = ["Re(雷诺数)", "Re", "雷诺数"]
R_COL_CANDIDATES = ["R(曲率半径）", "R(曲率半径)", "曲率半径", "R"]
FOCUS_COL_CANDIDATES = ["聚焦判定", "聚焦", "focus", "label", "is_focus", "是否聚焦"]

# ========================= 工具函数 =========================
def find_existing_column(df, candidates):
    """
    查找 candidates 中第一个存在于 df 的列名；若不存在则做空格归一化再匹配；仍失败则抛错。
    """
    for c in candidates:
        if c in df.columns:
            return c
    normalized = {"".join(k.split()): k for k in df.columns}
    for c in candidates:
        key = "".join(c.split())
        if key in normalized:
            return normalized[key]
    raise KeyError(f"无法找到列: {candidates}, 实际列: {list(df.columns)}")

def to_int_like_str(v):
    """
    将 v 转为“整数-like”的字符串：125.000000 -> '125'；NaN/inf 按原样字符串化。
    """
    try:
        f = float(v)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(v)
    except Exception:
        return str(v)

def normalize_shape(df):
    """
    将三角角度列拼接为形状字符串：θ1_θ2_θ。（如 '32_23_125'）
    """
    for col in ANGLE_COLS:
        if col not in df.columns:
            raise KeyError(f"缺失角度列: {col}")
    parts = [df[c].map(to_int_like_str) for c in ANGLE_COLS]
    return parts[0].str.cat(parts[1], sep="_").str.cat(parts[2], sep="_")

def round_to_allowed_re(x):
    """
    将 Re 四舍五入到最邻近的 6 的倍数，范围裁剪到 [6, 60]；否则返回 NA。
    """
    if pd.isna(x):
        return pd.NA
    try:
        xf = float(x)
    except Exception:
        return pd.NA
    nearest = int(round(xf / 6.0) * 6)
    if nearest < 6 or nearest > 60:
        return pd.NA
    return nearest

# ========================= 主函数 =========================
def main():
    # === 路径按需修改 ===
    input_csv = r"D:\pmc\tabpfn_project\data_tabpfn\数据更新20251004.csv"
    output_dir = r"D:\pmc\tabpfn_project\data\triangle_groups_fixed_v2_focus"
    drop_invalid = True  # 是否丢弃不在允许集合内的样本

    print(f"开始分组: {input_csv}")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    # 规范化列
    focus_col = find_existing_column(df, FOCUS_COL_CANDIDATES)
    r_col     = find_existing_column(df, R_COL_CANDIDATES)
    re_col    = find_existing_column(df, RE_COL_CANDIDATES)

    # 形状规范化
    df["shape"] = normalize_shape(df)

    # 数值规范化
    df["_R"] = pd.to_numeric(df[r_col], errors="coerce").astype("Int64")
    df["_Re_bucket"] = df[re_col].apply(round_to_allowed_re).astype("Int64")

    # 聚焦状态 -> 字符串标签（防止目录名出现 nan/空白）
    focus_series = df[focus_col].astype(str).fillna("NA").replace({"nan": "NA", "None": "NA"})

    # 可选：过滤，仅保留允许的形状/R/Re
    if drop_invalid:
        mask = (
            df["shape"].isin(ALLOWED_SHAPES)
            & df["_R"].isin(ALLOWED_RADII)
            & df["_Re_bucket"].isin(ALLOWED_RE)
        )
        before, after = len(df), int(mask.sum())
        df = df[mask].copy()
        print(f"过滤非法样本: {before - after} 条；保留 {after} 条。")

    # 分组：聚焦状态 -> 形状 -> R -> Re
    grouped = df.groupby([focus_series, "shape", "_R", "_Re_bucket"], dropna=True)
    count = 0
    for (focus_tag, shape, radius, re_val), g in grouped:
        # 二次防御：仅落盘到允许集合
        if shape not in ALLOWED_SHAPES or int(radius) not in ALLOWED_RADII or int(re_val) not in ALLOWED_RE:
            continue

        # 目录层级：<output>/<focus>/<shape>/<R>/<Re>/
        folder = os.path.join(
            output_dir,
            str(focus_tag),
            str(shape),
            str(int(radius)),
            str(int(re_val)),
        )
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, "data.csv")

        # 清理中间列后落盘
        g_out = g.drop(columns=["shape", "_R", "_Re_bucket"], errors="ignore")
        g_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        count += 1

    print(f"✅ 完成分组，共生成 {count} 个子表。输出目录: {output_dir}")

if __name__ == "__main__":
    main()
