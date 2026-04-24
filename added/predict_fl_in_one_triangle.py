# -*- coding: utf-8 -*-
"""
单个截面内 FL 预测：
1. 随机划分点
2. 按空间位置划分点

输入：一个 .xlsx 文件
输出：两种划分下的预测结果、指标、散点图
"""

from pathlib import Path
import os
import shutil
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1) 离线 TabPFN 配置
# =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"D:\tabpfn_project"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

logging.getLogger("posthog").setLevel(logging.ERROR)

LOCAL_TABPFN_DIR = Path(r"D:\tabpfn_project\models\offline_tabpfn_2_5")

LOCAL_MODEL_FILES = {
    "tabpfn-v2.5-regressor-v2.5_default.ckpt": LOCAL_TABPFN_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    "config.json": LOCAL_TABPFN_DIR / "config.json",
}

for filename, path in LOCAL_MODEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"缺少离线 TabPFN 文件: {path}")


def _patched_hf_hub_download(*args, **kwargs):
    repo_id = kwargs.get("repo_id", None)
    filename = kwargs.get("filename", None)

    if repo_id is None and len(args) >= 1:
        repo_id = args[0]
    if filename is None and len(args) >= 2:
        filename = args[1]

    if repo_id != "Prior-Labs/tabpfn_2_5":
        raise FileNotFoundError(
            f"离线 patch 不支持该 repo_id: repo_id={repo_id}, filename={filename}"
        )

    if filename not in LOCAL_MODEL_FILES:
        raise FileNotFoundError(
            f"离线 patch 缺少该文件的本地映射: filename={filename}"
        )

    src = LOCAL_MODEL_FILES[filename]

    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA 未设置。")

    target_dir = Path(appdata) / "tabpfn"
    target_dir.mkdir(parents=True, exist_ok=True)

    temp_local_path = target_dir / f"__incoming__{filename}"
    shutil.copy2(src, temp_local_path)

    print(f"[offline patch] copied {filename} -> {temp_local_path}")
    return str(temp_local_path)


import huggingface_hub
import huggingface_hub.file_download
import tabpfn.model_loading as _tabpfn_model_loading

huggingface_hub.hf_hub_download = _patched_hf_hub_download
huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download
_tabpfn_model_loading.hf_hub_download = _patched_hf_hub_download

from tabpfn import TabPFNRegressor

# =========================================================
# 2) 配置
# =========================================================
DATA_FILE = Path(r"D:\tabpfn_project\data\single_section.xlsx")   # 改成你的文件
RESULTS_ROOT = Path(r"D:\tabpfn_project\results\single_section_split_compare")

TARGET = "FL"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TRANSFORM_MODE = "identity"

# 空间划分方式:
# "x_right"  -> 右侧 20% 作为验证
# "x_left"   -> 左侧 20% 作为验证
# "z_top"    -> 上方 20% 作为验证
# "z_bottom" -> 下方 20% 作为验证
SPATIAL_SPLIT_MODE = "x_right"

MIN_TRAIN_ROWS = 8

# 若想手动指定输入特征，可在这里填 list；否则自动用 engineered features
MANUAL_FEATURES = None

RAW_REQUIRED_COLS = [
    "三角形底边", "X", "Z", "雷诺数", "周长", "面积", "高",
    "角度1", "角度2", "角度3",
    "重心坐标X", "重心坐标Y",
    "点到底边距离", "点到左边距离", "点到右边距离", "点到重心距离",
    "FL",
    "面积_calc", "角和", "geometry_valid",
    "左边长度", "右边长度",
    "lambda_A", "lambda_B", "lambda_C", "lambda_sum",
    "lambda_A_norm", "lambda_B_norm", "lambda_C_norm",
]


# =========================================================
# 3) 工具函数
# =========================================================
def transform_y(y: np.ndarray, mode: str) -> np.ndarray:
    y = y.astype(float)
    if mode == "log1p":
        return np.log1p(np.clip(y, 1e-15, None))
    if mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    return y


def inverse_y(y: np.ndarray, mode: str) -> np.ndarray:
    y = y.astype(float)
    if mode == "log1p":
        return np.expm1(y)
    if mode == "sqrt":
        return y ** 2
    return y


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data(fp: Path) -> pd.DataFrame:
    if not fp.exists():
        raise FileNotFoundError(f"文件不存在: {fp}")

    df = pd.read_excel(fp)
    df["__row_id__"] = np.arange(len(df))

    missing = [c for c in ["FL", "X", "Z", "三角形底边", "高", "雷诺数"] if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 尽量转成数值
    numeric_candidate_cols = [c for c in df.columns if not c.startswith("__")]
    for c in numeric_candidate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-12

    # 基础归一化
    df["x_norm"] = df["X"] / np.clip(df["三角形底边"], eps, None)
    df["z_norm"] = df["Z"] / np.clip(df["高"], eps, None)

    df["d_base_norm"] = df["点到底边距离"] / np.clip(df["高"], eps, None)
    df["d_left_norm"] = df["点到左边距离"] / np.clip(df["左边长度"], eps, None)
    df["d_right_norm"] = df["点到右边距离"] / np.clip(df["右边长度"], eps, None)
    df["d_centroid_norm"] = df["点到重心距离"] / np.clip(df["高"], eps, None)

    # Re 非线性
    df["Re_norm"] = (df["雷诺数"] - 6.0) / 54.0
    df["Re2"] = df["Re_norm"] ** 2
    df["Re3"] = df["Re_norm"] ** 3

    # 空间 × Re 交互
    df["x_Re"] = df["x_norm"] * df["Re_norm"]
    df["z_Re"] = df["z_norm"] * df["Re_norm"]
    df["x_Re2"] = df["x_norm"] * df["Re2"]
    df["z_Re2"] = df["z_norm"] * df["Re2"]

    df["d_base_Re"] = df["d_base_norm"] * df["Re_norm"]
    df["d_left_Re"] = df["d_left_norm"] * df["Re_norm"]
    df["d_right_Re"] = df["d_right_norm"] * df["Re_norm"]
    df["d_centroid_Re"] = df["d_centroid_norm"] * df["Re_norm"]

    # lambda 重心系交互
    if {"lambda_A_norm", "lambda_B_norm", "lambda_C_norm"}.issubset(df.columns):
        df["lambdaA_Re"] = df["lambda_A_norm"] * df["Re_norm"]
        df["lambdaB_Re"] = df["lambda_B_norm"] * df["Re_norm"]
        df["lambdaC_Re"] = df["lambda_C_norm"] * df["Re_norm"]

    # 几何比例
    df["aspect_ratio_base_h"] = df["三角形底边"] / np.clip(df["高"], eps, None)
    df["perimeter_area_ratio"] = df["周长"] / np.clip(df["面积"], eps, None)
    df["left_right_ratio"] = df["左边长度"] / np.clip(df["右边长度"], eps, None)
    df["area_calc_diff"] = df["面积_calc"] - df["面积"]

    # Re 事件特征
    re_values = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
    for r in re_values:
        df[f"Re_{r}"] = (df["雷诺数"] == r).astype(float)

    df["is_re_30_42"] = df["雷诺数"].between(30, 42).astype(float)
    df["is_re_36"] = (df["雷诺数"] == 36).astype(float)
    df["is_re_42"] = (df["雷诺数"] == 42).astype(float)

    df["is_re_42_x_norm"] = df["is_re_42"] * df["x_norm"]
    df["is_re_42_z_norm"] = df["is_re_42"] * df["z_norm"]
    df["is_re_36_x_norm"] = df["is_re_36"] * df["x_norm"]
    df["is_re_36_z_norm"] = df["is_re_36"] * df["z_norm"]

    if "geometry_valid" in df.columns:
        if df["geometry_valid"].dtype == bool:
            df["geometry_valid_num"] = df["geometry_valid"].astype(float)
        else:
            df["geometry_valid_num"] = df["geometry_valid"]

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    if MANUAL_FEATURES is not None:
        missing = [c for c in MANUAL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"手动指定特征不存在: {missing}")
        return MANUAL_FEATURES

    drop_cols = [
        "X", "Z", "FL",
        "重心坐标X", "重心坐标Y",
        "lambda_A", "lambda_B", "lambda_C", "lambda_sum",
        "__row_id__",
    ]
    feature_cols = [
        c for c in df.columns
        if not c.startswith("__") and c not in drop_cols
    ]
    return feature_cols


def drop_all_nan_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    return [c for c in feature_cols if not df[c].isna().all()]


def drop_constant_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    kept = []
    for c in feature_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        if s.nunique() <= 1:
            continue
        kept.append(c)
    return kept


def choose_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = get_feature_columns(df)
    feature_cols = drop_all_nan_feature_cols(df, feature_cols)
    feature_cols = drop_constant_feature_cols(df, feature_cols)
    if len(feature_cols) == 0:
        raise RuntimeError("没有可用特征列。")
    return feature_cols


def split_random(df: pd.DataFrame, test_size: float, random_state: int):
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    test_df = df.loc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df


def split_spatial(df: pd.DataFrame, mode: str, test_size: float):
    df = df.copy()

    if mode == "x_right":
        threshold = df["x_norm"].quantile(1 - test_size)
        test_df = df[df["x_norm"] >= threshold].copy()
        train_df = df[df["x_norm"] < threshold].copy()

    elif mode == "x_left":
        threshold = df["x_norm"].quantile(test_size)
        test_df = df[df["x_norm"] <= threshold].copy()
        train_df = df[df["x_norm"] > threshold].copy()

    elif mode == "z_top":
        threshold = df["z_norm"].quantile(1 - test_size)
        test_df = df[df["z_norm"] >= threshold].copy()
        train_df = df[df["z_norm"] < threshold].copy()

    elif mode == "z_bottom":
        threshold = df["z_norm"].quantile(test_size)
        test_df = df[df["z_norm"] <= threshold].copy()
        train_df = df[df["z_norm"] > threshold].copy()

    else:
        raise ValueError("SPATIAL_SPLIT_MODE 必须是 x_right/x_left/z_top/z_bottom 之一")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"空间划分失败，train/test 为空。mode={mode}")

    return train_df, test_df


def prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    test_df[target_col] = pd.to_numeric(test_df[target_col], errors="coerce")

    train_before = len(train_df)
    train_df = train_df[train_df[target_col].notna()].copy()
    train_after = len(train_df)

    if train_after < MIN_TRAIN_ROWS:
        raise RuntimeError(f"训练样本过少：{train_after}")

    train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    test_df[feature_cols] = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    test_df[feature_cols] = test_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    medians = train_df[feature_cols].median(axis=0, numeric_only=True).fillna(0.0)
    x_train_df = train_df[feature_cols].fillna(medians)
    x_test_df = test_df[feature_cols].fillna(medians)

    x_train = x_train_df.to_numpy(dtype=float)
    x_test = x_test_df.to_numpy(dtype=float)

    y_train = train_df[target_col].to_numpy(dtype=float)
    y_test = test_df[target_col].to_numpy(dtype=float)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    info = {
        "train_rows_before": int(train_before),
        "train_rows_after": int(train_after),
        "train_target_nan_dropped": int(train_before - train_after),
        "test_rows": int(len(test_df)),
        "test_target_nan": int(pd.isna(y_test).sum()),
    }

    return train_df, test_df, x_train_scaled, y_train, x_test_scaled, y_test, info


def fit_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
):
    (
        train_df_clean,
        test_df_clean,
        x_train,
        y_train,
        x_test,
        y_test,
        info
    ) = prepare_xy(train_df, test_df, feature_cols, target_col)

    y_train_t = transform_y(y_train, TRANSFORM_MODE)

    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(x_train, y_train_t)

    pred_t = model.predict(x_test)
    pred = inverse_y(pred_t, TRANSFORM_MODE)
    pred = np.asarray(pred, dtype=float)

    pred_df = test_df_clean.copy()
    pred_df["FL_true"] = y_test
    pred_df["FL_pred"] = pred
    pred_df["abs_error"] = np.where(
        pd.notna(pred_df["FL_true"]),
        np.abs(pred_df["FL_true"] - pred_df["FL_pred"]),
        np.nan,
    )

    valid_mask = np.isfinite(y_test) & np.isfinite(pred)
    y_eval = y_test[valid_mask]
    pred_eval = pred[valid_mask]

    if len(y_eval) == 0:
        metrics = {
            "rows": int(len(y_test)),
            "rows_eval": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }
    else:
        metrics = {
            "rows": int(len(y_test)),
            "rows_eval": int(len(y_eval)),
            "mae": float(mean_absolute_error(y_eval, pred_eval)),
            "rmse": rmse(y_eval, pred_eval),
            "r2": float(r2_score(y_eval, pred_eval)) if len(y_eval) >= 2 else np.nan,
        }

    return metrics, pred_df, info


def save_scatter_plot(pred_df: pd.DataFrame, save_path: Path, title: str):
    df_plot = pred_df[pd.notna(pred_df["FL_true"]) & pd.notna(pred_df["FL_pred"])].copy()
    if len(df_plot) == 0:
        return

    plt.figure(figsize=(6, 6))
    plt.scatter(df_plot["FL_true"], df_plot["FL_pred"], alpha=0.7)

    min_v = min(df_plot["FL_true"].min(), df_plot["FL_pred"].min())
    max_v = max(df_plot["FL_true"].max(), df_plot["FL_pred"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("FL_true")
    plt.ylabel("FL_pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# 4) 主程序
# =========================================================
def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    df = load_data(DATA_FILE)
    df = add_engineered_features(df)

    feature_cols = choose_feature_columns(df)

    print(f"数据文件: {DATA_FILE.name}")
    print(f"总样本数: {len(df)}")
    print("本次使用特征列：")
    for c in feature_cols:
        print(f"  - {c}")

    # -------------------------
    # A. 随机划分
    # -------------------------
    print("\n" + "=" * 80)
    print("版本 A：随机划分点")

    train_a, test_a = split_random(df, TEST_SIZE, RANDOM_STATE)
    metrics_a, pred_a, info_a = fit_and_predict(train_a, test_a, feature_cols, TARGET)
    pred_a["split_type"] = "random"

    pred_a.to_csv(RESULTS_ROOT / "pred_random.csv", index=False, encoding="utf-8-sig")
    save_scatter_plot(pred_a, RESULTS_ROOT / "scatter_random.png", "Random Split")

    print(
        f"[Random] train={info_a['train_rows_after']}, test={metrics_a['rows_eval']}, "
        f"MAE={metrics_a['mae']:.6f}, RMSE={metrics_a['rmse']:.6f}, R2={metrics_a['r2']:.6f}"
    )

    # -------------------------
    # B. 空间划分
    # -------------------------
    print("\n" + "=" * 80)
    print(f"版本 B：按空间划分点（mode={SPATIAL_SPLIT_MODE}）")

    train_b, test_b = split_spatial(df, SPATIAL_SPLIT_MODE, TEST_SIZE)
    metrics_b, pred_b, info_b = fit_and_predict(train_b, test_b, feature_cols, TARGET)
    pred_b["split_type"] = f"spatial_{SPATIAL_SPLIT_MODE}"

    pred_b.to_csv(RESULTS_ROOT / "pred_spatial.csv", index=False, encoding="utf-8-sig")
    save_scatter_plot(pred_b, RESULTS_ROOT / "scatter_spatial.png", f"Spatial Split - {SPATIAL_SPLIT_MODE}")

    print(
        f"[Spatial] train={info_b['train_rows_after']}, test={metrics_b['rows_eval']}, "
        f"MAE={metrics_b['mae']:.6f}, RMSE={metrics_b['rmse']:.6f}, R2={metrics_b['r2']:.6f}"
    )

    # -------------------------
    # 汇总
    # -------------------------
    summary_df = pd.DataFrame([
        {
            "split_type": "random",
            **metrics_a,
            **info_a,
        },
        {
            "split_type": f"spatial_{SPATIAL_SPLIT_MODE}",
            **metrics_b,
            **info_b,
        }
    ])
    summary_df.to_csv(RESULTS_ROOT / "summary_compare.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 完成：单截面内随机划分 vs 空间划分")
    print(f"结果目录：{RESULTS_ROOT}")


if __name__ == "__main__":
    main()