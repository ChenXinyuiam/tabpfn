# split_triangles_fixed_cv5.py
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# ========= 配置项 =========
INPUT_CSV  = r"D:\pmc\tabpfn_project\data_tabpfn\测试集数据20251008.csv"
OUTPUT_DIR = r"D:\pmc\tabpfn_project\data\triangle_groups_fixed"  # 将落成 fold_*/R_*/Re_*/data.csv

# 列名（含全角括号容错）
ANGLE_LEFT_COL   = "底角θ1"
ANGLE_RIGHT_COL  = "底角θ2"
ANGLE_TOP_COL    = "顶角θ"
RE_COL           = "Re(雷诺数)"
R_COL_CANDIDATES = ["R(曲率半径）", "R(曲率半径)"]  # 兼容全角/半角

# 固定白名单
ALLOWED_SHAPES = {"25_90_65", "40_40_100", "23_125_32", "90_25_65", "125_23_32"}
ALLOWED_Rs     = [10, 20, 30]
ALLOWED_REs    = list(range(6, 61, 6))  # 6,12,...,60

# CV 配置
FOLDS = 5
SEED  = 42  # 桶内洗牌固定种子，保证可复现

DATA_FILENAME = "data.csv"

# ========= 工具函数 =========
def pick_radius_column(df: pd.DataFrame) -> str:
    for c in R_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"未发现曲率半径列，尝试的列名：{R_COL_CANDIDATES}")

def shape_label(row: pd.Series) -> str:
    # 形状标签格式：底角θ1_顶角θ_底角θ2（例如 23_125_32）
    a1 = int(round(float(row[ANGLE_LEFT_COL])))
    at = int(round(float(row[ANGLE_TOP_COL])))
    a2 = int(round(float(row[ANGLE_RIGHT_COL])))
    return f"{a1}_{at}_{a2}"

def snap_or_none(value, allowed: List[int]):
    if pd.isna(value):
        return None
    try:
        iv = int(round(float(value)))
    except Exception:
        return None
    return iv if iv in allowed else None

def split_into_5(idx: np.ndarray, seed: int = SEED) -> List[np.ndarray]:
    """
    将一组索引按规则切 5 份：每份≈1/5，余数全部加到第 1 份。
    返回长度为 5 的索引数组列表。
    """
    rng = np.random.default_rng(seed)
    shuffled = idx.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    base = n // FOLDS
    rem  = n % FOLDS
    sizes = [base + rem] + [base] * (FOLDS - 1) if n > 0 else [0] * FOLDS

    offsets = np.cumsum([0] + sizes[:-1])
    return [shuffled[offsets[i]: offsets[i] + sizes[i]] for i in range(FOLDS)]

def write_csv_with_header(path: Path, df: pd.DataFrame):
    """
    若 df 为空，也写出空 CSV 但保留表头，确保目录结构与文件可被下游稳定读取。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

# ========= 主流程 =========
def main():
    root = Path(OUTPUT_DIR)
    root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # 字段校验
    required = {ANGLE_LEFT_COL, ANGLE_RIGHT_COL, ANGLE_TOP_COL, RE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列：{missing}")
    r_col = pick_radius_column(df)

    # 生成形状标签 & 白名单贴合
    df = df.copy()
    df["__shape__"] = df.apply(shape_label, axis=1)
    df["__R__"]  = df[r_col].apply(lambda x: snap_or_none(x, ALLOWED_Rs))
    df["__Re__"] = df[RE_COL].apply(lambda x: snap_or_none(x, ALLOWED_REs))

    df["__shape_ok__"] = df["__shape__"].isin(ALLOWED_SHAPES)
    df["__R_ok__"]     = df["__R__"].notna()
    df["__Re_ok__"]    = df["__Re__"].notna()
    df["__ok__"]       = df["__shape_ok__"] & df["__R_ok__"] & df["__Re_ok__"]

    ok_df      = df[df["__ok__"]].copy()
    dropped_df = df[~df["__ok__"]].copy()

    # 审计元信息：各 shape/R/Re 的原始样本量
    meta_rows = []
    for shp in sorted(ALLOWED_SHAPES):
        for Rv in ALLOWED_Rs:
            for Rev in ALLOWED_REs:
                n = int(((ok_df["__shape__"] == shp) & (ok_df["__R__"] == Rv) & (ok_df["__Re__"] == Rev)).sum())
                meta_rows.append({"shape": shp, "R": Rv, "Re": Rev, "n": n})
    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(root / "_meta_counts.csv", index=False, encoding="utf-8-sig")

    # 准备输出空模板（保持 header 一致）
    header_df = ok_df.drop(columns=[c for c in ok_df.columns if c.startswith("__")], errors="ignore").head(0)

    # === 关键动作：按 “每个 R × Re 桶” 在全量形状聚合后切 5 份 ===
    allocation_rows = []  # 记录每个桶在 5 折的样本数
    for Rv in ALLOWED_Rs:
        for Rev in ALLOWED_REs:
            mask = (ok_df["__R__"] == Rv) & (ok_df["__Re__"] == Rev)
            bucket_df = ok_df[mask].copy()

            # 切 5 份（余数→第 1 折）
            idx_parts = split_into_5(bucket_df.index.to_numpy(), seed=SEED)

            # 写出 5 折目录与 data.csv（为空也写 header）
            for i in range(FOLDS):
                fold_dir = root / f"fold_{i+1}" / f"R_{Rv}" / f"Re_{Rev}"
                part_idx = idx_parts[i]
                if part_idx.size > 0:
                    out_df = ok_df.loc[part_idx].drop(columns=[c for c in ok_df.columns if c.startswith("__")], errors="ignore")
                else:
                    out_df = header_df.copy()
                write_csv_with_header(fold_dir / DATA_FILENAME, out_df)

            # 记录分配详情
            allocation_rows.append({
                "R": Rv, "Re": Rev, "bucket_size": int(bucket_df.shape[0]),
                **{f"fold_{j+1}": int(idx_parts[j].size) for j in range(FOLDS)}
            })

    alloc_df = pd.DataFrame(allocation_rows)
    alloc_df.to_csv(root / "_cv5_allocation.csv", index=False, encoding="utf-8-sig")

    # 输出不合规样本，供回溯治理
    if not dropped_df.empty:
        dropped_df.to_csv(root / "_dropped_rows.csv", index=False, encoding="utf-8-sig")

    # 控制台反馈
    print("=== CV5 目录结构已生成（按 R×Re 桶 1/5 抽取，余数入第 1 折）===")
    print(f"输出根目录：{root}")
    print(f"折目录：{[f'fold_{i}' for i in range(1, FOLDS+1)]}")
    print("曲率半径桶：", ALLOWED_Rs)
    print("雷诺数桶：", ALLOWED_REs)
    if not alloc_df.empty:
        tot = {f"fold_{i}": int(alloc_df[f"fold_{i}"].sum()) for i in range(1, FOLDS+1)}
        print("折间样本量汇总：", tot)
        print(f"分配详情审计：{root / '_cv5_allocation.csv'}")
    if not dropped_df.empty:
        print(f"⚠️ 不在白名单的样本数：{len(dropped_df)} （详见 _dropped_rows.csv）")

if __name__ == "__main__":
    main()
