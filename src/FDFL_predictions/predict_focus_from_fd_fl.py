# -*- coding: utf-8 -*-
"""
两阶段聚焦判定：
第一阶段：用原始特征预测 FD / FL
第二阶段：用 [原始特征 + FD_pred + FL_pred] 预测 聚焦判定（二分类）

标签列：
    聚焦判定 ∈ {"聚焦", "不聚焦"}
"""

import os
import shutil
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier

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

from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# =========================================================
# 2) 配置
# =========================================================
ROOT = r"D:\tabpfn_project\data\triangle_groups_fixed_v2_focus\combined"
RESULTS_ROOT = r"D:\tabpfn_project\results_focus_stage2"
N_SPLITS = 5

BASE_FEATURES = [
    "底角θ1",
    "底角θ2",
    "Re(雷诺数)",
    "R(曲率半径）",
    "z/H",
    "流速(m/s)",
    "r(粒径)",
    "X",
]

# 如果你已经在 FL 里验证过这些交互特征很有效，可以直接启用
USE_ENGINEERED_FEATURES = True

FD_TRANSFORM = "log1p"
FL_TRANSFORM = "identity"

FD_MAX_TIME = 30
FL_MAX_TIME = 60

# 第二阶段分类器
# 这里先用 RandomForestClassifier，稳定、简单、对二分类很适合
CLASSIFIER_RANDOM_STATE = 42

# =========================================================
# 3) 工具函数
# =========================================================
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["angle_sum"] = df["底角θ1"] + df["底角θ2"]
    df["angle_diff"] = np.abs(df["底角θ1"] - df["底角θ2"])
    df["angle_ratio"] = df["底角θ1"] / np.clip(df["底角θ2"], 1e-12, None)

    df["Re_R"] = df["Re(雷诺数)"] * df["R(曲率半径）"]
    df["Re_r"] = df["Re(雷诺数)"] * df["r(粒径)"]
    df["Re_X"] = df["Re(雷诺数)"] * df["X"]
    df["v_r"] = df["流速(m/s)"] * df["r(粒径)"]
    df["v_R"] = df["流速(m/s)"] * df["R(曲率半径）"]
    df["X_zH"] = df["X"] * df["z/H"]

    df["R_zH"] = df["R(曲率半径）"] / np.clip(df["z/H"], 1e-12, None)
    df["r_zH"] = df["r(粒径)"] / np.clip(df["z/H"], 1e-12, None)
    df["X_zH_ratio"] = df["X"] / np.clip(df["z/H"], 1e-12, None)
    df["R_r_ratio"] = df["R(曲率半径）"] / np.clip(df["r(粒径)"], 1e-12, None)

    return df


def get_stage1_features():
    if not USE_ENGINEERED_FEATURES:
        return BASE_FEATURES.copy()

    return BASE_FEATURES + [
        "angle_sum",
        "angle_diff",
        "angle_ratio",
        "Re_R",
        "Re_r",
        "Re_X",
        "v_r",
        "v_R",
        "X_zH",
        "R_zH",
        "r_zH",
        "X_zH_ratio",
        "R_r_ratio",
    ]


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


def discover_leaves(combined_root: Path):
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


def load_fold_df(fp: Path, bucket_tag: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["__bucket__"] = bucket_tag
    df = add_engineered_features(df)
    return df


def check_required_columns(df: pd.DataFrame, columns: list[str]):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")


def fit_regressor(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], target: str, transform_mode: str, max_time: int):
    check_required_columns(train_df, features + [target])
    check_required_columns(test_df, features + [target])

    x_train = train_df[features].values
    x_test = test_df[features].values

    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_train = train_df[target].values.astype(float)
    y_train_t = transform_y(y_train, transform_mode)

    model = AutoTabPFNRegressor(
        max_time=max_time,
        ignore_pretraining_limits=True,
    )
    model.fit(x_train_scaled, y_train_t)

    train_pred_t = model.predict(x_train_scaled)
    test_pred_t = model.predict(x_test_scaled)

    train_pred = inverse_y(train_pred_t, transform_mode)
    test_pred = inverse_y(test_pred_t, transform_mode)

    return train_pred, test_pred


def build_stage2_features(df: pd.DataFrame) -> list[str]:
    # 第二阶段分类器使用：原始特征 + 第一阶段预测得到的 FD/FL
    return BASE_FEATURES + ["FD_pred", "FL_pred"]


def encode_focus_label(y: pd.Series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str))
    return y_enc, le


# =========================================================
# 4) 主流程
# =========================================================
def main():
    combined_root = Path(ROOT)
    results_root = Path(RESULTS_ROOT)
    results_root.mkdir(parents=True, exist_ok=True)

    leaves = discover_leaves(combined_root)
    if not leaves:
        raise RuntimeError("combined 目录下未发现任何叶子折文件。")

    print(f"发现叶子工况 {len(leaves)} 个；开始两阶段聚焦判定。")

    stage1_features = get_stage1_features()
    fold_metrics = []
    prediction_rows = []

    for k in range(1, N_SPLITS + 1):
        print(f"\n========== fold{k} ==========")

        train_parts, test_parts = [], []

        for leaf in leaves:
            bucket = f"{leaf['shape']}/{leaf['R']}/{leaf['Re']}"

            if k in leaf["fold_paths"]:
                test_parts.append(load_fold_df(leaf["fold_paths"][k], bucket))

            for j in range(1, N_SPLITS + 1):
                if j == k:
                    continue
                if j in leaf["fold_paths"]:
                    train_parts.append(load_fold_df(leaf["fold_paths"][j], bucket))

        if not train_parts or not test_parts:
            print(f"[警告] fold{k} 训练集或测试集为空，跳过。")
            continue

        train_df = pd.concat(train_parts, axis=0, ignore_index=True)
        test_df = pd.concat(test_parts, axis=0, ignore_index=True)

        # 必要列检查
        check_required_columns(train_df, stage1_features + ["FD", "FL", "聚焦判定"])
        check_required_columns(test_df, stage1_features + ["FD", "FL", "聚焦判定"])

        # --------------------------
        # 第一阶段：预测 FD / FL
        # --------------------------
        print(f"[fold{k}] 第一阶段：预测 FD")
        fd_train_pred, fd_test_pred = fit_regressor(
            train_df=train_df,
            test_df=test_df,
            features=stage1_features,
            target="FD",
            transform_mode=FD_TRANSFORM,
            max_time=FD_MAX_TIME,
        )

        print(f"[fold{k}] 第一阶段：预测 FL")
        fl_train_pred, fl_test_pred = fit_regressor(
            train_df=train_df,
            test_df=test_df,
            features=stage1_features,
            target="FL",
            transform_mode=FL_TRANSFORM,
            max_time=FL_MAX_TIME,
        )

        # 把第一阶段预测写回数据表
        train_stage2 = train_df.copy()
        test_stage2 = test_df.copy()

        train_stage2["FD_pred"] = fd_train_pred
        train_stage2["FL_pred"] = fl_train_pred
        test_stage2["FD_pred"] = fd_test_pred
        test_stage2["FL_pred"] = fl_test_pred

        # --------------------------
        # 第二阶段：预测 聚焦判定
        # --------------------------
        stage2_features = build_stage2_features(train_stage2)
        check_required_columns(train_stage2, stage2_features + ["聚焦判定"])
        check_required_columns(test_stage2, stage2_features + ["聚焦判定"])

        x_train_cls = train_stage2[stage2_features].values
        x_test_cls = test_stage2[stage2_features].values

        scaler_cls = MinMaxScaler()
        x_train_cls = scaler_cls.fit_transform(x_train_cls)
        x_test_cls = scaler_cls.transform(x_test_cls)

        y_train_cls, label_encoder = encode_focus_label(train_stage2["聚焦判定"])
        y_test_cls = label_encoder.transform(test_stage2["聚焦判定"].astype(str))

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=CLASSIFIER_RANDOM_STATE,
            n_jobs=-1,
        )
        clf.fit(x_train_cls, y_train_cls)

        y_pred_cls = clf.predict(x_test_cls)
        y_prob_cls = clf.predict_proba(x_test_cls)

        # 正类索引：优先找“聚焦”
        class_names = list(label_encoder.classes_)
        if "聚焦" in class_names:
            pos_idx = class_names.index("聚焦")
            y_prob_pos = y_prob_cls[:, pos_idx]
        else:
            pos_idx = 1 if y_prob_cls.shape[1] > 1 else 0
            y_prob_pos = y_prob_cls[:, pos_idx]

        # 指标
        acc = accuracy_score(y_test_cls, y_pred_cls)
        prec = precision_score(y_test_cls, y_pred_cls, average="binary", pos_label=pos_idx, zero_division=0)
        rec = recall_score(y_test_cls, y_pred_cls, average="binary", pos_label=pos_idx, zero_division=0)
        f1 = f1_score(y_test_cls, y_pred_cls, average="binary", pos_label=pos_idx, zero_division=0)

        try:
            auc = roc_auc_score((y_test_cls == pos_idx).astype(int), y_prob_pos)
        except Exception:
            auc = np.nan

        cm = confusion_matrix(y_test_cls, y_pred_cls, labels=list(range(len(class_names))))

        fold_metrics.append(
            {
                "fold": f"fold{k}",
                "rows": len(y_test_cls),
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "roc_auc": auc,
            }
        )

        # 保存预测明细
        pred_df = test_stage2.copy()
        pred_df["聚焦判定_true"] = test_stage2["聚焦判定"].astype(str)
        pred_df["聚焦判定_pred"] = label_encoder.inverse_transform(y_pred_cls)
        pred_df["聚焦_prob_聚焦"] = y_prob_pos
        pred_df["fold"] = f"fold{k}"
        prediction_rows.append(pred_df)

        print(
            f"[fold{k}] accuracy={acc:.4f}, precision={prec:.4f}, "
            f"recall={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}"
        )
        print(f"[fold{k}] confusion_matrix=\n{cm}")

    # -------------------------------------------------
    # 输出结果
    # -------------------------------------------------
    metrics_df = pd.DataFrame(fold_metrics)
    preds_df = pd.concat(prediction_rows, axis=0, ignore_index=True) if prediction_rows else pd.DataFrame()

    metrics_df.to_csv(results_root / "focus_fold_metrics.csv", index=False, encoding="utf-8-sig")
    preds_df.to_csv(results_root / "focus_fold_predictions.csv", index=False, encoding="utf-8-sig")

    if not metrics_df.empty:
        total_rows = metrics_df["rows"].sum()
        weights = metrics_df["rows"] / total_rows

        summary_df = pd.DataFrame(
            [
                {
                    "rows_total": int(total_rows),
                    "accuracy_w": float(np.average(metrics_df["accuracy"], weights=weights)),
                    "precision_w": float(np.average(metrics_df["precision"], weights=weights)),
                    "recall_w": float(np.average(metrics_df["recall"], weights=weights)),
                    "f1_w": float(np.average(metrics_df["f1"], weights=weights)),
                    "roc_auc_w": float(np.average(metrics_df["roc_auc"].fillna(0), weights=weights)),
                }
            ]
        )
        summary_df.to_csv(results_root / "focus_weighted_summary.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 聚焦判定两阶段建模完成。")
    print(f"结果输出目录：{results_root}")


if __name__ == "__main__":
    main()