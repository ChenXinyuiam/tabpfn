# -*- coding: utf-8 -*-
"""
FAIL!!

2个xlsx文件做 Leave-One-File-Out:
每次 1 个文件训练，1 个文件验证，预测 FL

特征:
    使用除 ["X", "Z", "FL"] 外的所有列

要求:
    - 每个 .xlsx 文件都有相同字段
    - 总共应有 7 个 .xlsx 文件
    - 目标列为 FL
    - 所有数据已经清洗完毕，不再做缺失值处理和数值转换
"""

from pathlib import Path
import os
import shutil
import logging

import numpy as np
import pandas as pd

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
DATA_ROOT = Path(r"D:\tabpfn_project\data\temp")
RESULTS_ROOT = Path(r"D:\tabpfn_project\results\results_20260418temp")
TARGET = "FL"

# 不作为输入特征的列
EXCLUDE_COLS = ["X", "Z", "FL"]

TRANSFORM_MODE = "identity"


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


def load_one_excel(fp: Path) -> pd.DataFrame:
    df = pd.read_excel(fp)
    df["__source_file__"] = fp.stem
    return df


def validate_columns(df: pd.DataFrame):
    required = {TARGET, "X", "Z"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"文件缺少必要列: {missing}")


def get_feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in EXCLUDE_COLS and not c.startswith("__")]


def prepare_xy(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str], target_col: str):
    x_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.astype(float)

    x_val = val_df[feature_cols].values
    y_val = val_df[target_col].values.astype(float)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    return train_df, val_df, x_train_scaled, y_train, x_val_scaled, y_val


def fit_and_predict(train_df, val_df, feature_cols, target_col, transform_mode="identity"):
    (
        train_df_clean,
        val_df_clean,
        x_train,
        y_train,
        x_val,
        y_val
    ) = prepare_xy(train_df, val_df, feature_cols, target_col)

    y_train_t = transform_y(y_train, transform_mode)

    model = TabPFNRegressor(
        ignore_pretraining_limits=True,
    )
    model.fit(x_train, y_train_t)

    pred_t = model.predict(x_val)
    pred = inverse_y(pred_t, transform_mode)
    pred = np.asarray(pred, dtype=float)

    metrics = {
        "rows": int(len(y_val)),
        "mae": float(mean_absolute_error(y_val, pred)),
        "rmse": rmse(y_val, pred),
        "r2": float(r2_score(y_val, pred)),
    }

    pred_df = val_df_clean.copy()
    pred_df["FL_true"] = y_val
    pred_df["FL_pred"] = pred
    pred_df["abs_error"] = np.abs(pred_df["FL_true"] - pred_df["FL_pred"])

    return metrics, pred_df


# =========================================================
# 4) 主流程
# =========================================================
def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    xlsx_files = sorted(DATA_ROOT.glob("*.xlsx"))
    if len(xlsx_files) != 2:
        raise RuntimeError(
            f"当前目录下找到 {len(xlsx_files)} 个 .xlsx 文件；该脚本按“1个训练 + 1个验证”固定写法要求总数必须正好为 2 个。"
        )

    print("发现以下 2 个文件：")
    for i, fp in enumerate(xlsx_files, 1):
        print(f"  {i}. {fp.name}")

    all_data = {}
    for fp in xlsx_files:
        df = load_one_excel(fp)
        validate_columns(df)
        all_data[fp.name] = df

    sample_df = next(iter(all_data.values()))
    feature_cols = get_feature_columns(sample_df)

    print("\n本次使用的输入特征列：")
    for c in feature_cols:
        print(f"  - {c}")

    if len(feature_cols) == 0:
        raise RuntimeError("没有可用特征列，请检查列名。")

    all_fold_metrics = []
    all_pred_rows = []

    for val_fp in xlsx_files:
        val_name = val_fp.name
        train_names = [fp.name for fp in xlsx_files if fp.name != val_name]

        print("\n" + "=" * 80)
        print(f"验证文件: {val_name}")
        print("训练文件:")
        for name in train_names:
            print(f"  - {name}")

        train_df = pd.concat([all_data[name] for name in train_names], axis=0, ignore_index=True)
        val_df = all_data[val_name].copy()

        metrics, pred_df = fit_and_predict(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col=TARGET,
            transform_mode=TRANSFORM_MODE,
        )

        metrics_row = {
            "val_file": val_name,
            "train_files": " | ".join(train_names),
            **metrics,
        }
        all_fold_metrics.append(metrics_row)

        pred_df["val_file"] = val_name
        pred_df["train_files"] = " | ".join(train_names)
        all_pred_rows.append(pred_df)

        print(
            f"[{val_name}] rows={metrics['rows']}, "
            f"MAE={metrics['mae']:.6f}, "
            f"RMSE={metrics['rmse']:.6f}, "
            f"R2={metrics['r2']:.6f}"
        )

    metrics_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.concat(all_pred_rows, axis=0, ignore_index=True)

    metrics_df.to_csv(RESULTS_ROOT / "fl_leave_one_file_out_metrics.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(RESULTS_ROOT / "fl_leave_one_file_out_predictions.csv", index=False, encoding="utf-8-sig")

    total_rows = metrics_df["rows"].sum()
    weights = metrics_df["rows"] / total_rows

    summary_df = pd.DataFrame([{
        "files_total": len(xlsx_files),
        "rows_total": int(total_rows),
        "mae_weighted": float(np.average(metrics_df["mae"], weights=weights)),
        "rmse_weighted": float(np.average(metrics_df["rmse"], weights=weights)),
        "r2_weighted": float(np.average(metrics_df["r2"], weights=weights)),
    }])

    summary_df.to_csv(RESULTS_ROOT / "fl_leave_one_file_out_summary.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 完成：1个文件训练，1个文件验证（共7折）")
    print(f"结果目录：{RESULTS_ROOT}")


if __name__ == "__main__":
    main()