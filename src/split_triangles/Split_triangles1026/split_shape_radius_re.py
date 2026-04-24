'''
这个文件用于将数据集按照 形状-曲率半径-雷诺数进行分层.
'''

import os
import math
import pandas as pd

# ========================= 配置区 =========================
ALLOWED_SHAPES = {"23_125_32", "125_23_32", "40_40_100", "25_90_65", "90_25_65"} # 五个基本形状
ALLOWED_RADII = {10, 20, 30} # 3个曲率半径
ALLOWED_RE = set(range(6, 61, 6))  #10个雷诺数 6,12,...,60

ANGLE_COLS = ["底角θ1", "底角θ2", "顶角θ"]
RE_COL_CANDIDATES = ["Re(雷诺数)"]
R_COL_CANDIDATES = ["R(曲率半径）"]

# ========================= 工具函数 =========================
def find_existing_column(df, candidates):
    '''
    查找candiates列是否在df中. 如果存在, 返回列名. 如果不存在, 抛出错误.
    :param df:
    :param candidates:
    :return:
    '''
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
    '''
    将v转换为整数like字符串. 如果v是NaN, 返回NaN. 如果v是inf, 返回inf. 如果v是-inf, 返回-inf.
    如果v是字符串, 返回v. 如果v是数字, 返回str(int(round(v))).
    例如125.000000转换为'125', 125.000001转换为'125', 125.999999转换为'126', 126.000000转换为'126'.
    :param v: 要转换的值
    :return: 转换后的字符串
    '''
    try:
        f = float(v)
        if math.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(v)
    except Exception:
        return str(v)

def normalize_shape(df):
    '''
        将角度拼接起来, 如'角度θ1':32,'角度θ2':23,'角度θ':125拼接为'32_23_125'.
    :param df: 要转换的DataFrame
    :return: 转换后的字符串
    '''
    for col in ANGLE_COLS:
        if col not in df.columns:
            raise KeyError(f"缺失角度列: {col}")
    parts = [df[c].map(to_int_like_str) for c in ANGLE_COLS]
    return parts[0].str.cat(parts[1], sep="_").str.cat(parts[2], sep="_")

def round_to_allowed_re(x):
    '''
        基本无用的函数，目的是为了让非6倍数的雷诺数转换为6倍数.
    :param x: 要转换的值
    :return: 转换后的值
    '''
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
    input_csv = r"D:\\pmc\\tabpfn_project\\data_tabpfn\\数据更新20251004.csv"
    output_dir = r"D:\\pmc\\tabpfn_project\\data_tabpfn\\triangle_groups_fixed"
    drop_invalid = True

    print(f"开始分组: {input_csv}")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    df["shape"] = normalize_shape(df)
    r_col = find_existing_column(df, R_COL_CANDIDATES)
    re_col = find_existing_column(df, RE_COL_CANDIDATES)

    df["_R"] = pd.to_numeric(df[r_col], errors="coerce").astype("Int64")
    df["_Re_bucket"] = df[re_col].apply(round_to_allowed_re).astype("Int64")

    if drop_invalid:
        df = df[df["shape"].isin(ALLOWED_SHAPES) & df["_R"].isin(ALLOWED_RADII) & df["_Re_bucket"].isin(ALLOWED_RE)]

    grouped = df.groupby(["shape", "_R", "_Re_bucket"], dropna=True)
    count = 0
    for (shape, radius, re_val), g in grouped:
        if shape not in ALLOWED_SHAPES or int(radius) not in ALLOWED_RADII or int(re_val) not in ALLOWED_RE:
            continue
        folder = os.path.join(output_dir, f"{shape}", f"{int(radius)}", f"{int(re_val)}")
        os.makedirs(folder, exist_ok=True)
        out_path = os.path.join(folder, "data.csv")
        g.drop(columns=["shape", "_R", "_Re_bucket"], errors="ignore").to_csv(out_path, index=False, encoding="utf-8-sig")
        count += 1

    print(f"✅ 完成分组，共生成 {count} 个子表。输出目录: {output_dir}")

if __name__ == "__main__":
    main()
