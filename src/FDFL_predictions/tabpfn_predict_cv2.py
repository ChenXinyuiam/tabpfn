# -*- coding: utf-8 -*-
"""
从 triangle_groups_fixed_v2_focus/combined 读取各叶子的 fold1..fold5.csv
执行全局 5 折交叉验证：
  - 第 k 折测试集 = 所有叶子的 foldk.csv
  - 训练集 = 其余四折的合并
输出：
  - fold_global_metrics.csv（每折全局）
  - fold_bucket_metrics.csv（每折分叶子工况）
  - global_weighted_summary.csv（加权汇总）
"""

import os
import shutil
from pathlib import Path

# =========================================================
# 1) 离线 TabPFN 配置：必须放在任何 huggingface / tabpfn 导入之前
# =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = r"D:\tabpfn_project"

# 这里放你离线准备好的模型文件目录
LOCAL_TABPFN_DIR = Path(r"D:\tabpfn_project\offline_tabpfn_2_5")

# 至少要准备这些文件
LOCAL_MODEL_FILES = {
    "tabpfn-v2.5-regressor-v2.5_default.ckpt": LOCAL_TABPFN_DIR / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
    "config.json": LOCAL_TABPFN_DIR / "config.json",
    # 如果后续报还缺别的文件，就继续往这里加
    # "criterion_config.json": LOCAL_TABPFN_DIR / "criterion_config.json",
}

for filename, path in LOCAL_MODEL_FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"缺少离线 TabPFN 文件: {path}")


def _patched_hf_hub_download(*args, **kwargs):
    """
    拦截 HuggingFace 下载请求，改为从本地离线目录复制文件到 C 盘缓存临时位置，
    避免 Windows 跨盘 rename 失败。
    """
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


# 必须在导入 tabpfn 前 patch
import huggingface_hub
import huggingface_hub.file_download

huggingface_hub.hf_hub_download = _patched_hf_hub_download
huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download

import tabpfn.model_loading as _tabpfn_model_loading
_tabpfn_model_loading.hf_hub_download = _patched_hf_hub_download

# =========================================================
# 2) 常规依赖导入
# =========================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# =========================================================
# 3) 配置
# =========================================================
ROOT = r"D:\tabpfn_project\data\triangle_groups_fixed_v2_focus\combined"
RESULTS_ROOT = r"D:\pmc\tabpfn_project\results"
N_SPLITS = 5

TARGETS = ["FD", "FL"]
SELECTED_FEATURES = [
    "底角θ1",
    "底角θ2",
    "Re(雷诺数)",
    "R(曲率半径）",
    "x/H",
    "流速(m/s)",
    "r(粒径)",
    "z/H",
]

TABPFN_MAX_TIME = 30
TABPFN_IGNORE_PRETRAINING_LIMITS = True

# =========================================================
# 4) 指标函数
# =========================================================
def mape(y_true, y_pred):
    return float(
        np.mean(
            np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))
        ) * 100.0
    )


def smape(y_true, y_pred):
    return float(
        np.mean(
            2.0 * np.abs(y_pred - y_true)
            / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-12, None)
        ) * 100.0
    )


# =========================================================
# 5) I/O 与预处理
# =========================================================
def discover_leaves(combined_root: Path):
    """
    扫描 combined/<shape>/<R>/<Re>/ 下的 fold1..fold5.csv
    返回：
        list[
            {
                "shape": s,
                "R": r,
                "Re": re,
                "fold_paths": {1: path, ..., 5: path}
            }
        ]
    只要某叶子至少存在一个 fold*.csv 就纳入。
    """
    leaves = []
    for shape_dir in sorted([p for p in combined_root.iterdir() if p.is_dir()]):
        for r_dir in sorted([p for p in shape_dir.iterdir() if p.is_dir()]):
            for re_dir in sorted([p for p in r_dir.iterdir() if p.is_dir()]):
                fold_paths = {}
                for k in range(1, N_SPLITS + 1):
                    fp = re_dir / f"fold{k}.csv"
                    if fp.is_file():
                        fold_paths[k] = fp

                if fold_paths:
                    leaves.append(
                        {
                            "shape": shape_dir.name,
                            "R": r_dir.name,
                            "Re": re_dir.name,
                            "fold_paths": fold_paths,
                        }
                    )
    return leaves


def load_fold_df(fp: Path, bucket_tag: str):
    df = pd.read_csv(fp)
    df["__bucket__"] = bucket_tag
    return df


def check_columns(df: pd.DataFrame, features, target):
    missing = [c for c in list(features) + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, features, target):
    check_columns(train_df, features, target)
    check_columns(test_df, features, target)

    x_train = train_df[features].values
    x_test = test_df[features].values

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train = train_df[target].values.reshape(-1, 1)
    y_test = test_df[target].values.reshape(-1, 1)

    y_train_clip = np.clip(y_train, 1e-15, None)
    y_train_log = np.log1p(y_train_clip).ravel()

    return x_train_scaled, x_test_scaled, y_train_log, y_test.ravel(), scaler


# =========================================================
# 6) 主流程
# =========================================================
def main():
    combined_root = Path(ROOT)
    results_root = Path(RESULTS_ROOT)
    results_root.mkdir(parents=True, exist_ok=True)

    leaves = discover_leaves(combined_root)
    if not leaves:
        raise RuntimeError("combined 目录下未发现任何叶子折文件。")

    print(f"发现叶子工况 {len(leaves)} 个；将执行 {N_SPLITS} 折全局评估。")
    print(f"HF_HUB_OFFLINE = {os.environ.get('HF_HUB_OFFLINE')}")
    print(f"HF_HOME = {os.environ.get('HF_HOME')}")
    print(f"LOCAL_TABPFN_DIR = {LOCAL_TABPFN_DIR}")

    for target in TARGETS:
        fold_metrics_all = []
        per_bucket_metrics = []

        for k in range(1, N_SPLITS + 1):
            test_parts = []
            train_parts = []

            for leaf in leaves:
                bucket = f"{leaf['shape']}/{leaf['R']}/{leaf['Re']}"

                # 第 k 折作为测试
                if k in leaf["fold_paths"]:
                    test_parts.append(load_fold_df(leaf["fold_paths"][k], bucket))

                # 其余折作为训练
                for j in range(1, N_SPLITS + 1):
                    if j == k:
                        continue
                    if j in leaf["fold_paths"]:
                        train_parts.append(load_fold_df(leaf["fold_paths"][j], bucket))

            if not test_parts or not train_parts:
                print(f"[警告] 第 {k} 折存在空训练/测试集，跳过。")
                continue

            test_df = pd.concat(test_parts, axis=0, ignore_index=True)
            train_df = pd.concat(train_parts, axis=0, ignore_index=True)

            x_train, x_test, y_train_log, y_test, _ = preprocess(
                train_df, test_df, SELECTED_FEATURES, target
            )

            reg = AutoTabPFNRegressor(
                max_time=TABPFN_MAX_TIME,
                ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS,
            )

            reg.fit(x_train, y_train_log)

            y_pred_log = reg.predict(x_test)
            y_pred = np.expm1(y_pred_log)

            fold_row = {
                "target": target,
                "fold": f"fold{k}",
                "rows": int(len(y_test)),
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "mape": mape(y_test, y_pred),
                "smape": smape(y_test, y_pred),
            }
            fold_metrics_all.append(fold_row)

            gb = test_df.copy()
            gb[f"{target}_pred"] = y_pred

            for bucket_id, sub in gb.groupby("__bucket__"):
                y_true_bucket = sub[target].values
                y_pred_bucket = sub[f"{target}_pred"].values

                per_bucket_metrics.append(
                    {
                        "target": target,
                        "fold": f"fold{k}",
                        "bucket": bucket_id,
                        "rows": int(len(y_true_bucket)),
                        "mse": mean_squared_error(y_true_bucket, y_pred_bucket),
                        "mae": mean_absolute_error(y_true_bucket, y_pred_bucket),
                        "r2": r2_score(y_true_bucket, y_pred_bucket)
                        if len(y_true_bucket) > 1
                        else np.nan,
                        "mape": mape(y_true_bucket, y_pred_bucket),
                        "smape": smape(y_true_bucket, y_pred_bucket),
                    }
                )

        out_dir = results_root / f"from_combined_{target}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_folds = pd.DataFrame(fold_metrics_all)
        df_folds.to_csv(
            out_dir / "fold_global_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        df_buckets = pd.DataFrame(per_bucket_metrics)
        df_buckets.to_csv(
            out_dir / "fold_bucket_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if not df_folds.empty:
            total_rows = df_folds["rows"].sum()
            weights = df_folds["rows"] / (total_rows if total_rows > 0 else 1)

            global_summary = pd.DataFrame(
                [
                    {
                        "target": target,
                        "rows_total": int(total_rows),
                        "mse_w": float(np.average(df_folds["mse"], weights=weights)),
                        "mae_w": float(np.average(df_folds["mae"], weights=weights)),
                        "r2_w": float(np.average(df_folds["r2"], weights=weights)),
                        "mape_w": float(np.average(df_folds["mape"], weights=weights)),
                        "smape_w": float(np.average(df_folds["smape"], weights=weights)),
                    }
                ]
            )

            global_summary.to_csv(
                out_dir / "global_weighted_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

        print(f"[{target}] ✅ 评估完成：输出目录 {out_dir}")


if __name__ == "__main__":
    main()