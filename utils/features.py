"""Feature engineering helpers."""

import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "三角形底边",
    "X",
    "Z",
    "雷诺数",
    "周长",
    "面积",
    "高",
    "角度1",
    "角度2",
    "角度3",
    "重心坐标X",
    "重心坐标Y",
    "点到底边距离",
    "点到左边距离",
    "点到右边距离",
    "点到重心距离",
    "面积_calc",
    "角和",
    "左边长度",
    "右边长度",
    "lambda_A",
    "lambda_B",
    "lambda_C",
    "lambda_sum",
    "lambda_A_norm",
    "lambda_B_norm",
    "lambda_C_norm",
]

ENGINEERED_FEATURE_COLUMNS = [
    "angle_sum",
    "angle_diff_12",
    "angle_diff_23",
    "angle_diff_13",
    "angle_ratio_12",
    "angle_ratio_23",
    "aspect_ratio",
    "area_perimeter_ratio",
    "x_base_ratio",
    "z_height_ratio",
    "x_centroid_offset",
    "z_centroid_offset",
    "d_base_height_ratio",
    "d_left_side_ratio",
    "d_right_side_ratio",
    "distance_balance_lr",
    "lambda_balance_bc",
    "re_x",
    "re_z",
    "re_area",
    "re_height",
]


def available_feature_columns(df: pd.DataFrame, include_engineered: bool = True) -> list[str]:
    """Return configured feature columns that are present in a DataFrame."""
    columns = RAW_FEATURE_COLUMNS + (ENGINEERED_FEATURE_COLUMNS if include_engineered else [])
    return [column for column in columns if column in df.columns]


def _safe_div(numerator, denominator):
    return numerator / np.clip(denominator, 1e-12, None)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction and ratio features based on processed Excel columns."""
    df = df.copy()

    angle_1 = df["角度1"]
    angle_2 = df["角度2"]
    angle_3 = df["角度3"]
    base = df["三角形底边"]
    height = df["高"]
    perimeter = df["周长"]
    area = df["面积_calc"] if "面积_calc" in df.columns else df["面积"]
    re_values = df["雷诺数"]

    df["angle_sum"] = angle_1 + angle_2 + angle_3
    df["angle_diff_12"] = np.abs(angle_1 - angle_2)
    df["angle_diff_23"] = np.abs(angle_2 - angle_3)
    df["angle_diff_13"] = np.abs(angle_1 - angle_3)
    df["angle_ratio_12"] = _safe_div(angle_1, angle_2)
    df["angle_ratio_23"] = _safe_div(angle_2, angle_3)

    df["aspect_ratio"] = _safe_div(base, height)
    df["area_perimeter_ratio"] = _safe_div(area, perimeter)
    df["x_base_ratio"] = _safe_div(df["X"], base)
    df["z_height_ratio"] = _safe_div(df["Z"], height)
    df["x_centroid_offset"] = df["X"] - df["重心坐标X"]
    df["z_centroid_offset"] = df["Z"] - df["重心坐标Y"]

    df["d_base_height_ratio"] = _safe_div(df["点到底边距离"], height)
    df["d_left_side_ratio"] = _safe_div(df["点到左边距离"], df["左边长度"])
    df["d_right_side_ratio"] = _safe_div(df["点到右边距离"], df["右边长度"])
    df["distance_balance_lr"] = _safe_div(
        df["点到左边距离"] - df["点到右边距离"],
        df["点到左边距离"] + df["点到右边距离"],
    )
    df["lambda_balance_bc"] = df["lambda_B_norm"] - df["lambda_C_norm"]

    df["re_x"] = re_values * df["X"]
    df["re_z"] = re_values * df["Z"]
    df["re_area"] = re_values * area
    df["re_height"] = re_values * height

    return df
