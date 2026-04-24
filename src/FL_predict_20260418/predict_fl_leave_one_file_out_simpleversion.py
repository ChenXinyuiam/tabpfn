# -*- coding: utf-8 -*-
"""
基于多个 xlsx 文件做按文件分组的 Group K-Fold 回归预测 FL。

当前版本针对真实数据做了鲁棒化处理：
1. 使用 GroupKFold(n_splits=5)
2. 同一文件不会同时出现在训练集和验证集
3. 训练集内 y=FL 为 NaN 的样本会被删除
4. X 中的 NaN / inf 用训练集各列中位数填补
5. 验证集 y 为 NaN 的样本仍可预测，但不参与指标计算
6. 自动检查特征列是否一致、是否全空、是否常数等基础问题
"""

from pathlib import Path
import os
import shutil
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
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
DATA_ROOT = Path(r"D:\tabpfn_project\data\processed\2026-04-21")
RESULTS_ROOT = Path(r"D:\tabpfn_project\results\results_2026_04_21_groupkfold")
TARGET = "FL"
N_SPLITS = 5

# 不作为输入特征的列
EXCLUDE_COLS = ["X", "Z", "FL", "重心坐标X", "重心坐标Y", "点到重心距离"]

TRANSFORM_MODE = "identity"

# 若训练集删除 NaN 目标后样本太少，则该折直接报错
MIN_TRAIN_ROWS = 8


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
    df["__source_file__"] = fp.name
    return df



def validate_columns(df: pd.DataFrame, fp: Path):
    required = {TARGET, "X", "Z"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"文件 {fp.name} 缺少必要列: {missing}")



def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS and not c.startswith("__")]



def coerce_numeric_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def drop_all_nan_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    kept = [c for c in feature_cols if not df[c].isna().all()]
    dropped = sorted(set(feature_cols) - set(kept))
    if dropped:
        print("\n[INFO] 以下特征列全为 NaN，已自动丢弃：")
        for c in dropped:
            print(f"  - {c}")
    return kept



def drop_constant_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    kept = []
    dropped = []
    for c in feature_cols:
        s = df[c].dropna()
        if s.empty:
            dropped.append(c)
        elif s.nunique() <= 1:
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        print("\n[INFO] 以下特征列为常数列/空列，已自动丢弃：")
        for c in dropped:
            print(f"  - {c}")
    return kept



def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_df = train_df.copy()
    val_df = val_df.copy()

    # 目标列转数值
    train_df[target_col] = pd.to_numeric(train_df[target_col], errors="coerce")
    val_df[target_col] = pd.to_numeric(val_df[target_col], errors="coerce")

    # 训练集：目标缺失不能用于拟合
    train_rows_before = len(train_df)
    train_df = train_df[train_df[target_col].notna()].copy()
    train_rows_after = len(train_df)
    dropped_train_target_nan = train_rows_before - train_rows_after

    if train_rows_after < MIN_TRAIN_ROWS:
        raise RuntimeError(
            f"训练集中可用于拟合的样本仅 {train_rows_after} 行，少于 MIN_TRAIN_ROWS={MIN_TRAIN_ROWS}。"
        )

    # 特征转数值，并处理 inf
    train_df[feature_cols] = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    val_df[feature_cols] = val_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    val_df[feature_cols] = val_df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # 用训练集各列中位数填补 X 缺失
    medians = train_df[feature_cols].median(axis=0, numeric_only=True)
    # 若某列中位数仍为 NaN，说明该列在训练集中全空；这里补 0，且这种列理论上已在前面被剔除了
    medians = medians.fillna(0.0)

    x_train_df = train_df[feature_cols].fillna(medians)
    x_val_df = val_df[feature_cols].fillna(medians)

    x_train = x_train_df.to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)

    x_val = x_val_df.to_numpy(dtype=float)
    y_val = val_df[target_col].to_numpy(dtype=float)

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    prep_info = {
        "train_rows_before": int(train_rows_before),
        "train_rows_after": int(train_rows_after),
        "dropped_train_target_nan": int(dropped_train_target_nan),
        "val_rows": int(len(val_df)),
        "val_target_nan": int(pd.isna(y_val).sum()),
        "train_feature_nan_after_impute": int(np.isnan(x_train_scaled).sum()),
        "val_feature_nan_after_impute": int(np.isnan(x_val_scaled).sum()),
    }

    return train_df, val_df, x_train_scaled, y_train, x_val_scaled, y_val, prep_info



def fit_and_predict(train_df, val_df, feature_cols, target_col, transform_mode="identity"):
    (
        train_df_clean,
        val_df_clean,
        x_train,
        y_train,
        x_val,
        y_val,
        prep_info,
    ) = prepare_xy(train_df, val_df, feature_cols, target_col)

    if not np.isfinite(y_train).all():
        raise RuntimeError("训练目标 y_train 在清洗后仍包含 NaN 或 inf。")
    if not np.isfinite(x_train).all():
        raise RuntimeError("训练特征 x_train 在填补后仍包含 NaN 或 inf。")
    if not np.isfinite(x_val).all():
        raise RuntimeError("验证特征 x_val 在填补后仍包含 NaN 或 inf。")

    y_train_t = transform_y(y_train, transform_mode)
    if not np.isfinite(y_train_t).all():
        raise RuntimeError("目标变换后的 y_train_t 包含 NaN 或 inf。")

    model = TabPFNRegressor(ignore_pretraining_limits=True)
    model.fit(x_train, y_train_t)

    pred_t = model.predict(x_val)
    pred = inverse_y(pred_t, transform_mode)
    pred = np.asarray(pred, dtype=float)

    pred_df = val_df_clean.copy()
    pred_df["FL_true"] = y_val
    pred_df["FL_pred"] = pred
    pred_df["abs_error"] = np.where(
        pd.notna(pred_df["FL_true"]),
        np.abs(pred_df["FL_true"] - pred_df["FL_pred"]),
        np.nan,
    )

    valid_metric_mask = np.isfinite(y_val) & np.isfinite(pred)
    y_val_metric = y_val[valid_metric_mask]
    pred_metric = pred[valid_metric_mask]

    if len(y_val_metric) == 0:
        metrics = {
            "rows": int(len(y_val)),
            "rows_eval": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }
    else:
        metrics = {
            "rows": int(len(y_val)),
            "rows_eval": int(len(y_val_metric)),
            "mae": float(mean_absolute_error(y_val_metric, pred_metric)),
            "rmse": rmse(y_val_metric, pred_metric),
            "r2": float(r2_score(y_val_metric, pred_metric)) if len(y_val_metric) >= 2 else np.nan,
        }

    return metrics, pred_df, prep_info



def build_master_dataframe(data_root: Path):
    xlsx_files = sorted(
        [f for f in data_root.glob("*.xlsx") if not f.name.startswith("~$")],
        key=lambda p: p.name,
    )

    if len(xlsx_files) < N_SPLITS:
        raise RuntimeError(
            f"文件数 {len(xlsx_files)} 小于 n_splits={N_SPLITS}，无法执行 GroupKFold。"
        )

    frames = []
    for fp in xlsx_files:
        df = load_one_excel(fp)
        validate_columns(df, fp)
        frames.append(df)

    full_df = pd.concat(frames, axis=0, ignore_index=True)
    return full_df, xlsx_files



def choose_feature_columns(full_df: pd.DataFrame) -> List[str]:
    feature_cols = get_feature_columns(full_df)
    feature_cols = drop_all_nan_feature_cols(full_df, feature_cols)
    feature_cols = drop_constant_feature_cols(full_df, feature_cols)
    if len(feature_cols) == 0:
        raise RuntimeError("没有可用特征列，请检查 EXCLUDE_COLS 或输入表字段。")
    return feature_cols


# =========================================================
# 4) 主流程
# =========================================================
def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    full_df, xlsx_files = build_master_dataframe(DATA_ROOT)

    # 尽量统一数值型列
    candidate_numeric_cols = [c for c in full_df.columns if not c.startswith("__")]
    full_df = coerce_numeric_columns(full_df, candidate_numeric_cols)

    feature_cols = choose_feature_columns(full_df)

    unique_files = sorted(full_df["__source_file__"].dropna().unique().tolist())
    print(f"发现 {len(unique_files)} 个文件，采用 GroupKFold(n_splits={N_SPLITS})")
    for i, name in enumerate(unique_files, 1):
        print(f"  {i}. {name}")

    print("\n本次使用的输入特征列：")
    for c in feature_cols:
        print(f"  - {c}")

    groups = full_df["__source_file__"].values
    gkf = GroupKFold(n_splits=N_SPLITS)

    all_fold_metrics = []
    all_pred_rows = []
    fold_assignment_rows = []
    prep_rows = []

    X_dummy = np.zeros(len(full_df))
    y_dummy = np.zeros(len(full_df))

    for fold_id, (train_idx, val_idx) in enumerate(gkf.split(X_dummy, y_dummy, groups=groups), start=1):
        train_df = full_df.iloc[train_idx].copy()
        val_df = full_df.iloc[val_idx].copy()

        train_files = sorted(train_df["__source_file__"].dropna().unique().tolist())
        val_files = sorted(val_df["__source_file__"].dropna().unique().tolist())

        print("\n" + "=" * 80)
        print(f"Fold {fold_id}/{N_SPLITS}")
        print("验证文件:")
        for name in val_files:
            print(f"  - {name}")
        print("训练文件:")
        for name in train_files:
            print(f"  - {name}")

        metrics, pred_df, prep_info = fit_and_predict(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            target_col=TARGET,
            transform_mode=TRANSFORM_MODE,
        )

        prep_rows.append({
            "fold": fold_id,
            "train_files": " | ".join(train_files),
            "val_files": " | ".join(val_files),
            **prep_info,
        })

        metrics_row = {
            "fold": fold_id,
            "val_files": " | ".join(val_files),
            "train_files": " | ".join(train_files),
            **metrics,
        }
        all_fold_metrics.append(metrics_row)

        pred_df["fold"] = fold_id
        pred_df["val_files"] = " | ".join(val_files)
        pred_df["train_files"] = " | ".join(train_files)
        all_pred_rows.append(pred_df)

        for f in val_files:
            fold_assignment_rows.append({"fold": fold_id, "role": "val", "file": f})
        for f in train_files:
            fold_assignment_rows.append({"fold": fold_id, "role": "train", "file": f})

        print(
            f"[Fold {fold_id}] rows={metrics['rows']}, rows_eval={metrics['rows_eval']}, "
            f"drop_train_y_nan={prep_info['dropped_train_target_nan']}, "
            f"val_y_nan={prep_info['val_target_nan']}, "
            f"MAE={metrics['mae'] if pd.notna(metrics['mae']) else np.nan}, "
            f"RMSE={metrics['rmse'] if pd.notna(metrics['rmse']) else np.nan}, "
            f"R2={metrics['r2'] if pd.notna(metrics['r2']) else np.nan}"
        )

    metrics_df = pd.DataFrame(all_fold_metrics)
    preds_df = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    assignment_df = pd.DataFrame(fold_assignment_rows)
    prep_df = pd.DataFrame(prep_rows)

    metrics_df.to_csv(RESULTS_ROOT / "fl_groupkfold_metrics.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(RESULTS_ROOT / "fl_groupkfold_predictions.csv", index=False, encoding="utf-8-sig")
    assignment_df.to_csv(RESULTS_ROOT / "fl_groupkfold_fold_assignments.csv", index=False, encoding="utf-8-sig")
    prep_df.to_csv(RESULTS_ROOT / "fl_groupkfold_preprocessing_log.csv", index=False, encoding="utf-8-sig")

    valid_metric_rows = metrics_df[metrics_df["rows_eval"] > 0].copy()
    if len(valid_metric_rows) > 0:
        total_rows_eval = valid_metric_rows["rows_eval"].sum()
        weights = valid_metric_rows["rows_eval"] / total_rows_eval
        summary_df = pd.DataFrame([{
            "files_total": len(unique_files),
            "rows_total": int(metrics_df["rows"].sum()),
            "rows_eval_total": int(total_rows_eval),
            "n_splits": N_SPLITS,
            "mae_weighted": float(np.average(valid_metric_rows["mae"], weights=weights)),
            "rmse_weighted": float(np.average(valid_metric_rows["rmse"], weights=weights)),
            "r2_weighted": float(np.average(valid_metric_rows["r2"].fillna(0.0), weights=weights)),
        }])
    else:
        summary_df = pd.DataFrame([{
            "files_total": len(unique_files),
            "rows_total": int(metrics_df["rows"].sum()),
            "rows_eval_total": 0,
            "n_splits": N_SPLITS,
            "mae_weighted": np.nan,
            "rmse_weighted": np.nan,
            "r2_weighted": np.nan,
        }])

    summary_df.to_csv(RESULTS_ROOT / "fl_groupkfold_summary.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 完成：按文件分组的 5 折交叉验证")
    print(f"结果目录：{RESULTS_ROOT}")


if __name__ == "__main__":
    main()
