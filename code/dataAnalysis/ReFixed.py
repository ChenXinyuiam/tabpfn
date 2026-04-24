# -*- coding: utf-8 -*-
"""
读取单个截面的 xlsx 文件，
按空间位置 (X, Z) 分组，
筛选出 Re = 6,12,...,60 都齐全的位置，
并输出多组图：每组图 10 个子图，每个子图对应一个位置，
子图内容为该位置的 Re-FL 散点图。
"""

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1) 配置
# =========================
INPUT_FILE = Path(r"D:\2026\others\tabpfn\data\processed\2026-04-22\830-342-FL.xlsx")   # 改成你的文件路径
OUTPUT_DIR = Path(r"D:\2026\others\tabpfn\dataAnalysis\ReFixed\ReFixed830-342-FL")

# 列名配置
COL_X = "X"
COL_Z = "Z"
COL_RE = "雷诺数"
COL_FL = "FL"

# 目标 Re 序列
TARGET_RE_LIST = list(range(6, 66, 6))   # [6, 12, ..., 60]

# 每张图放几个位置子图
N_SUBPLOTS_PER_FIG = 10

# 如果 X/Z 是浮点数，做一下四舍五入，避免“看起来同一位置但浮点不完全相等”
ROUND_DECIMALS = 6


# =========================
# 2) 工具函数
# =========================
def load_data(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"文件不存在: {fp}")

    df = pd.read_excel(fp)

    required_cols = [COL_X, COL_Z, COL_RE, COL_FL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 强制转数值
    df[COL_X] = pd.to_numeric(df[COL_X], errors="coerce")
    df[COL_Z] = pd.to_numeric(df[COL_Z], errors="coerce")
    df[COL_RE] = pd.to_numeric(df[COL_RE], errors="coerce")
    df[COL_FL] = pd.to_numeric(df[COL_FL], errors="coerce")

    # 去掉关键列缺失
    df = df.dropna(subset=[COL_X, COL_Z, COL_RE, COL_FL]).reset_index(drop=True)

    # 浮点坐标做 rounding，便于分组
    df["X_round"] = df[COL_X].round(ROUND_DECIMALS)
    df["Z_round"] = df[COL_Z].round(ROUND_DECIMALS)

    return df


def find_valid_positions(df: pd.DataFrame):
    """
    找出 Re 完整覆盖 TARGET_RE_LIST 的位置。
    返回一个列表，每个元素是某位置对应的数据子表。
    """
    valid_groups = []

    grouped = df.groupby(["X_round", "Z_round"], sort=True)

    for (xv, zv), g in grouped:
        # 同一位置同一 Re 可能有多行，这里先按 Re 聚合取均值
        g2 = g.groupby(COL_RE, as_index=False)[COL_FL].mean()

        re_values = sorted(g2[COL_RE].astype(int).tolist())
        if re_values == TARGET_RE_LIST:
            g2["X_round"] = xv
            g2["Z_round"] = zv
            valid_groups.append(g2)

    return valid_groups


def plot_groups(valid_groups, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if not valid_groups:
        print("没有找到任何一个位置同时具备 10 个 Re 点。")
        return

    n_positions = len(valid_groups)
    n_figs = math.ceil(n_positions / N_SUBPLOTS_PER_FIG)

    print(f"找到满足条件的位置数: {n_positions}")
    print(f"将输出图像数量: {n_figs}")

    for fig_idx in range(n_figs):
        start = fig_idx * N_SUBPLOTS_PER_FIG
        end = min((fig_idx + 1) * N_SUBPLOTS_PER_FIG, n_positions)
        chunk = valid_groups[start:end]

        n_subplots = len(chunk)
        nrows, ncols = 5, 2   # 固定 10 个子图布局

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 18))
        axes = axes.flatten()

        for ax_idx, ax in enumerate(axes):
            if ax_idx < n_subplots:
                g = chunk[ax_idx].sort_values(by=COL_RE)

                x_val = g["X_round"].iloc[0]
                z_val = g["Z_round"].iloc[0]

                ax.scatter(g[COL_RE], g[COL_FL])

                # 可选：连线更容易看趋势
                ax.plot(g[COL_RE], g[COL_FL])

                ax.set_title(f"X={x_val}, Z={z_val}")
                ax.set_xlabel("Re")
                ax.set_ylabel("FL")
                ax.set_xticks(TARGET_RE_LIST)
            else:
                ax.axis("off")

        fig.suptitle(
            f"Re-FL scatter by position | Figure {fig_idx + 1}/{n_figs}",
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        save_path = output_dir / f"re_fl_scatter_group_{fig_idx + 1:02d}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"已保存: {save_path}")


# =========================
# 3) 主程序
# =========================
def main():
    df = load_data(INPUT_FILE)
    df = preprocess(df)

    valid_groups = find_valid_positions(df)
    plot_groups(valid_groups, OUTPUT_DIR)

    # 额外导出位置清单
    if valid_groups:
        pos_rows = []
        for g in valid_groups:
            pos_rows.append({
                "X_round": g["X_round"].iloc[0],
                "Z_round": g["Z_round"].iloc[0],
            })

        pos_df = pd.DataFrame(pos_rows)
        pos_df.to_csv(OUTPUT_DIR / "valid_positions.csv", index=False, encoding="utf-8-sig")
        print(f"已保存位置清单: {OUTPUT_DIR / 'valid_positions.csv'}")


if __name__ == "__main__":
    main()