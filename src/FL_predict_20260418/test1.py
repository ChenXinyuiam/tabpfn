# -*- coding: utf-8 -*-
"""
单个三角形截面内 FL 预测诊断实验

版本 A：同一个截面内，随机划分点
版本 B：同一个截面内，按空间区域划分

用途：
1. 验证模型是否能在同一截面内学到 FL
2. 对比随机划分 vs 空间外推的性能差异
3. 为“为什么跨截面 R² < 0”提供诊断依据
"""

from pathlib import Path
import os
import shutil
import logging

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

LOCAL_TABPFN_DIR = Path(r"D:\tabpfn_project\offline_tabpfn_2_5")

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
DATA_FILE = Path(r"D:\tabpfn_project\data\data20260418\720-500-FL.xlsx")  # 改成你的单个截面文件
RESULTS_ROOT = Path(r"D:\tabpfn_project\results\single_section_diagnosis")

TARGET = "FL"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TRANSFORM_MODE = "identity"

# 若你想手动固定特征，就改这里；否则设为 None 自动使用除 FL 外所有列
MANUAL_FEATURES = None

# 版本 B 的空间划分方式：可选 "X" 或 "Z"
SPATIAL_SPLIT_AXIS = "X"

# 版本 B 的验证区域比例
SPATIAL_TEST_RATIO = 0.2


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
    df = pd.read_excel(fp)
    df["__row_id__"] = np.arange(len(df))
    return df


def validate_data(df: pd.DataFrame):
    if TARGET not in df.columns:
        raise ValueError(f"缺少目标列: {TARGET}")

    required_spatial_cols = {"X", "Z"}
    missing = [c for c in required_spatial_cols if c not in df.columns]
    if missing:
        raise ValueError(f"版本 B 需要空间列 X/Z，当前缺少: {missing}")


def get_feature_columns(df: pd.DataFrame):
    if MANUAL_FEATURES is not None:
        missing = [c for c in MANUAL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"手动指定的特征不存在: {missing}")
        return MANUAL_FEATURES

    feature_cols = [c for c in df.columns if c not in [TARGET, "__row_id__"]]
    return feature_cols


def check_numeric(df: pd.DataFrame, cols: list[str], df_name: str):
    bad_cols = []
    for c in cols:
        try:
            pd.to_numeric(df[c], errors="raise")
        except Exception:
            bad_cols.append(c)
    if bad_cols:
        raise ValueError(f"{df_name} 存在非数值列或非法值: {bad_cols}")


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    x_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.astype(float)

    x_test = test_df[feature_cols].values
    y_test = test_df[target_col].values.astype(float)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test


def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    x_train, y_train, x_test, y_test = prepare_xy(train_df, test_df, feature_cols, target_col)

    y_train_t = transform_y(y_train, TRANSFORM_MODE)

    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(x_train, y_train_t)

    pred_t = model.predict(x_test)
    pred = inverse_y(pred_t, TRANSFORM_MODE)
    pred = np.asarray(pred, dtype=float)

    metrics = {
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": rmse(y_test, pred),
        "r2": float(r2_score(y_test, pred)),
    }

    pred_df = test_df.copy()
    pred_df["FL_true"] = y_test
    pred_df["FL_pred"] = pred
    pred_df["abs_error"] = np.abs(pred_df["FL_true"] - pred_df["FL_pred"])

    return metrics, pred_df


def split_random_points(df: pd.DataFrame, test_size: float, random_state: int):
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_df = df.loc[train_idx].copy().reset_index(drop=True)
    test_df = df.loc[test_idx].copy().reset_index(drop=True)
    return train_df, test_df


def split_spatial_region(df: pd.DataFrame, axis: str = "X", test_ratio: float = 0.2):
    if axis not in ["X", "Z"]:
        raise ValueError("axis 只能是 'X' 或 'Z'")

    sorted_df = df.sort_values(by=axis).reset_index(drop=True)
    n_test = max(1, int(len(sorted_df) * test_ratio))

    # 取右侧/上侧连续区域作为验证集
    test_df = sorted_df.iloc[-n_test:].copy().reset_index(drop=True)
    train_df = sorted_df.iloc[:-n_test].copy().reset_index(drop=True)

    return train_df, test_df


def save_scatter_plot(pred_df: pd.DataFrame, save_path: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(pred_df["FL_true"], pred_df["FL_pred"], alpha=0.7)

    min_v = min(pred_df["FL_true"].min(), pred_df["FL_pred"].min())
    max_v = max(pred_df["FL_true"].max(), pred_df["FL_pred"].max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("FL_true")
    plt.ylabel("FL_pred")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# =========================================================
# 4) 主流程
# =========================================================
def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到数据文件: {DATA_FILE}")

    df = load_data(DATA_FILE)
    validate_data(df)

    feature_cols = get_feature_columns(df)

    # 最小数值检查
    check_numeric(df, feature_cols + [TARGET], "原始数据")

    print(f"数据文件: {DATA_FILE.name}")
    print(f"总行数: {len(df)}")
    print("使用特征列:")
    for c in feature_cols:
        print(f"  - {c}")

    summary_rows = []

    # =====================================================
    # 版本 A：同一个截面内，随机划分点
    # =====================================================
    print("\n" + "=" * 80)
    print("版本 A：同一个截面内，随机划分点")

    train_a, test_a = split_random_points(
        df=df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metrics_a, pred_a = fit_and_predict(
        train_df=train_a,
        test_df=test_a,
        feature_cols=feature_cols,
        target_col=TARGET,
    )

    pred_a["split_type"] = "random_points"
    pred_a.to_csv(RESULTS_ROOT / "predictions_version_A_random_points.csv", index=False, encoding="utf-8-sig")

    save_scatter_plot(
        pred_df=pred_a,
        save_path=RESULTS_ROOT / "scatter_version_A_random_points.png",
        title="Version A - Random Point Split",
    )

    summary_rows.append({
        "version": "A_random_points",
        **metrics_a,
    })

    print(
        f"A结果: train={metrics_a['rows_train']}, test={metrics_a['rows_test']}, "
        f"MAE={metrics_a['mae']:.6f}, RMSE={metrics_a['rmse']:.6f}, R2={metrics_a['r2']:.6f}"
    )

    # =====================================================
    # 版本 B：同一个截面内，按空间区域划分
    # =====================================================
    print("\n" + "=" * 80)
    print(f"版本 B：同一个截面内，按空间区域划分（axis={SPATIAL_SPLIT_AXIS}）")

    train_b, test_b = split_spatial_region(
        df=df,
        axis=SPATIAL_SPLIT_AXIS,
        test_ratio=SPATIAL_TEST_RATIO,
    )

    metrics_b, pred_b = fit_and_predict(
        train_df=train_b,
        test_df=test_b,
        feature_cols=feature_cols,
        target_col=TARGET,
    )

    pred_b["split_type"] = f"spatial_region_{SPATIAL_SPLIT_AXIS}"
    pred_b.to_csv(RESULTS_ROOT / "predictions_version_B_spatial_region.csv", index=False, encoding="utf-8-sig")

    save_scatter_plot(
        pred_df=pred_b,
        save_path=RESULTS_ROOT / "scatter_version_B_spatial_region.png",
        title=f"Version B - Spatial Region Split ({SPATIAL_SPLIT_AXIS})",
    )

    summary_rows.append({
        "version": f"B_spatial_region_{SPATIAL_SPLIT_AXIS}",
        **metrics_b,
    })

    print(
        f"B结果: train={metrics_b['rows_train']}, test={metrics_b['rows_test']}, "
        f"MAE={metrics_b['mae']:.6f}, RMSE={metrics_b['rmse']:.6f}, R2={metrics_b['r2']:.6f}"
    )

    # =====================================================
    # 输出汇总
    # =====================================================
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_ROOT / "summary_single_section_diagnosis.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("诊断实验完成")
    print(f"结果目录: {RESULTS_ROOT}")
    print("\n解释建议：")
    print("1. 如果 A 明显好而 B 明显差，说明模型更擅长局部插值，不擅长空间外推。")
    print("2. 如果 A 和 B 都好，而跨截面仍差，说明主要问题在跨截面泛化。")
    print("3. 如果 A 和 B 都差，说明当前特征或标签本身就不足以支撑预测。")


if __name__ == "__main__":
    main()