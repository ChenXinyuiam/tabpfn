import sys
import pathlib
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
INPUT_DIR = pathlib.Path("data/scaled")
OUTPUT_DIR = pathlib.Path("data/processedv2")

# 顶点定义（基于你这组数据的正确映射）
# A = 顶角（与底边相对） = 角度1
# B = 左下角             = 角度2
# C = 右下角             = 角度3
ANGLE_A_COL = "角度1"   # 顶角
ANGLE_B_COL = "角度2"   # 左下角
ANGLE_C_COL = "角度3"   # 右下角

BASE_COL = "三角形底边"   # a = BC
HEIGHT_COL = "高"
AREA_COL = "面积"

# 点到三边距离
D_BASE_COL = "点到底边距离"   # d_a，对应边 a=BC
D_LEFT_COL = "点到左边距离"   # 到左边 AB 的距离，即 d_c
D_RIGHT_COL = "点到右边距离"  # 到右边 AC 的距离，即 d_b

# 由正弦定理恢复的边长
# c = AB = 左边长度
# b = AC = 右边长度
LEFT_SIDE_COL = "左边长度"
RIGHT_SIDE_COL = "右边长度"

# 新增输出列
AREA_CALC_COL = "面积_calc"
ANGLE_SUM_COL = "角和"
VALID_GEOMETRY_COL = "geometry_valid"

# barycentric coordinates
# λ_A 对应顶点 A（顶角），其对边是底边 a
# λ_B 对应顶点 B（左下角），其对边是右边 b
# λ_C 对应顶点 C（右下角），其对边是左边 c
LAMBDA_A_COL = "lambda_A"
LAMBDA_B_COL = "lambda_B"
LAMBDA_C_COL = "lambda_C"
LAMBDA_SUM_COL = "lambda_sum"

LAMBDA_A_NORM_COL = "lambda_A_norm"
LAMBDA_B_NORM_COL = "lambda_B_norm"
LAMBDA_C_NORM_COL = "lambda_C_norm"
# ------------------------------------------------------------------


def safe_sin_deg(series: pd.Series) -> pd.Series:
    return np.sin(np.deg2rad(series.astype(float)))


def compute_geometry_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        BASE_COL, HEIGHT_COL,
        ANGLE_A_COL, ANGLE_B_COL, ANGLE_C_COL,
        D_BASE_COL, D_LEFT_COL, D_RIGHT_COL
    }

    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] geometry columns missing, skip barycentric features: {missing}")
        return df

    df = df.copy()

    # 转为数值，非法值转 NaN
    numeric_cols = [
        BASE_COL, HEIGHT_COL, AREA_COL,
        ANGLE_A_COL, ANGLE_B_COL, ANGLE_C_COL,
        D_BASE_COL, D_LEFT_COL, D_RIGHT_COL
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) 保留原面积，新增重算面积
    df[AREA_CALC_COL] = df[BASE_COL] * df[HEIGHT_COL] / 2.0

    # 2) 基本几何校验
    df[ANGLE_SUM_COL] = df[ANGLE_A_COL] + df[ANGLE_B_COL] + df[ANGLE_C_COL]

    # 顶角 A 的正弦不能太小
    sin_A = safe_sin_deg(df[ANGLE_A_COL])
    valid_sin_A = np.abs(sin_A) > 1e-15

    # 基础条件
    valid_base = df[BASE_COL] > 0
    valid_height = df[HEIGHT_COL] > 0
    valid_area = df[AREA_CALC_COL] > 0
    valid_angles = np.abs(df[ANGLE_SUM_COL] - 180.0) < 1.0  # 允许少量测量误差
    valid_dist = (
        (df[D_BASE_COL] >= 0) &
        (df[D_LEFT_COL] >= 0) &
        (df[D_RIGHT_COL] >= 0)
    )

    df[VALID_GEOMETRY_COL] = (
        valid_base & valid_height & valid_area &
        valid_angles & valid_sin_A & valid_dist
    )

    # 3) 用正弦定理恢复边长
    # a = BC = 底边
    # b = AC = 右边长度
    # c = AB = 左边长度
    #
    # a / sin(A) = b / sin(B) = c / sin(C)
    #
    # b = a * sin(B) / sin(A)
    # c = a * sin(C) / sin(A)

    df[LEFT_SIDE_COL] = np.nan   # c = AB
    df[RIGHT_SIDE_COL] = np.nan  # b = AC

    valid = df[VALID_GEOMETRY_COL]

    df.loc[valid, RIGHT_SIDE_COL] = (
        df.loc[valid, BASE_COL] *
        safe_sin_deg(df.loc[valid, ANGLE_B_COL]) /
        sin_A.loc[valid]
    )

    df.loc[valid, LEFT_SIDE_COL] = (
        df.loc[valid, BASE_COL] *
        safe_sin_deg(df.loc[valid, ANGLE_C_COL]) /
        sin_A.loc[valid]
    )

    # 4) 计算 barycentric coordinates
    # λ_A = a * d_a / (2S)
    # λ_B = b * d_b / (2S)  <- B 的对边是右边 AC
    # λ_C = c * d_c / (2S)  <- C 的对边是左边 AB
    #
    # 注意：
    # d_a = 点到底边距离
    # d_b = 点到右边距离
    # d_c = 点到左边距离

    denom = 2.0 * df[AREA_CALC_COL]

    df[LAMBDA_A_COL] = np.nan
    df[LAMBDA_B_COL] = np.nan
    df[LAMBDA_C_COL] = np.nan

    df.loc[valid, LAMBDA_A_COL] = (
        df.loc[valid, BASE_COL] * df.loc[valid, D_BASE_COL] / denom.loc[valid]
    )

    df.loc[valid, LAMBDA_B_COL] = (
        df.loc[valid, RIGHT_SIDE_COL] * df.loc[valid, D_RIGHT_COL] / denom.loc[valid]
    )

    df.loc[valid, LAMBDA_C_COL] = (
        df.loc[valid, LEFT_SIDE_COL] * df.loc[valid, D_LEFT_COL] / denom.loc[valid]
    )

    df[LAMBDA_SUM_COL] = (
        df[LAMBDA_A_COL] + df[LAMBDA_B_COL] + df[LAMBDA_C_COL]
    )

    # 5) 归一化 barycentric coordinates，减小测量误差影响
    df[LAMBDA_A_NORM_COL] = np.nan
    df[LAMBDA_B_NORM_COL] = np.nan
    df[LAMBDA_C_NORM_COL] = np.nan

    valid_lambda_sum = valid & df[LAMBDA_SUM_COL].notna() & (np.abs(df[LAMBDA_SUM_COL]) > 1e-15)

    df.loc[valid_lambda_sum, LAMBDA_A_NORM_COL] = (
        df.loc[valid_lambda_sum, LAMBDA_A_COL] / df.loc[valid_lambda_sum, LAMBDA_SUM_COL]
    )
    df.loc[valid_lambda_sum, LAMBDA_B_NORM_COL] = (
        df.loc[valid_lambda_sum, LAMBDA_B_COL] / df.loc[valid_lambda_sum, LAMBDA_SUM_COL]
    )
    df.loc[valid_lambda_sum, LAMBDA_C_NORM_COL] = (
        df.loc[valid_lambda_sum, LAMBDA_C_COL] / df.loc[valid_lambda_sum, LAMBDA_SUM_COL]
    )

    return df


def process_file(src: pathlib.Path, dst: pathlib.Path) -> None:
    df = pd.read_excel(src, engine="openpyxl")
    df = compute_geometry_features(df)
    df.to_excel(dst, index=False, engine="openpyxl")


def main() -> None:
    if not INPUT_DIR.exists():
        sys.exit(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(
        f for f in INPUT_DIR.glob("*.xlsx")
        if not f.name.startswith("~$")
    )

    if not xlsx_files:
        sys.exit(f"No .xlsx files found in {INPUT_DIR}")

    print(f"Processing {len(xlsx_files)} file(s) -> {OUTPUT_DIR}/")
    for src in xlsx_files:
        dst = OUTPUT_DIR / src.name
        print(f"  {src.name}", end=" ... ", flush=True)
        process_file(src, dst)
        print("done")

    print("All files processed.")


if __name__ == "__main__":
    main()