# fd_fl_common.py
import os
import shutil
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================== 离线 TabPFN ==================
def setup_offline_tabpfn(local_model_dir):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(local_model_dir.parent)

    logging.getLogger("posthog").setLevel(logging.ERROR)

    LOCAL_MODEL_FILES = {
        "tabpfn-v2.5-regressor-v2.5_default.ckpt": local_model_dir / "tabpfn-v2.5-regressor-v2.5_default.ckpt",
        "config.json": local_model_dir / "config.json",
    }

    for name, path in LOCAL_MODEL_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing local file: {path}")

    def _patched_hf_hub_download(*args, **kwargs):
        repo_id = kwargs.get("repo_id", args[0] if len(args) > 0 else None)
        filename = kwargs.get("filename", args[1] if len(args) > 1 else None)

        if repo_id != "Prior-Labs/tabpfn_2_5":
            raise FileNotFoundError(f"Unsupported repo: {repo_id}")

        if filename not in LOCAL_MODEL_FILES:
            raise FileNotFoundError(f"Missing local mapping: {filename}")

        src = LOCAL_MODEL_FILES[filename]

        target_dir = Path(os.environ["APPDATA"]) / "tabpfn"
        target_dir.mkdir(parents=True, exist_ok=True)

        temp = target_dir / f"__incoming__{filename}"
        shutil.copy2(src, temp)

        print(f"[offline] {filename} -> {temp}")
        return str(temp)

    import huggingface_hub
    import huggingface_hub.file_download
    import tabpfn.model_loading as ml

    huggingface_hub.hf_hub_download = _patched_hf_hub_download
    huggingface_hub.file_download.hf_hub_download = _patched_hf_hub_download
    ml.hf_hub_download = _patched_hf_hub_download


# ================== 指标 ==================
def mape(y, yhat):
    return np.mean(np.abs((y - yhat) / np.clip(np.abs(y), 1e-12, None))) * 100


def smape(y, yhat):
    return np.mean(2 * np.abs(yhat - y) / np.clip(np.abs(y) + np.abs(yhat), 1e-12, None)) * 100


# ================== 数据 ==================
def discover_leaves(root, n_splits):
    leaves = []
    for shape in Path(root).iterdir():
        for R in shape.iterdir():
            for Re in R.iterdir():
                fold_paths = {}
                for k in range(1, n_splits + 1):
                    fp = Re / f"fold{k}.csv"
                    if fp.exists():
                        fold_paths[k] = fp

                if fold_paths:
                    leaves.append({
                        "shape": shape.name,
                        "R": R.name,
                        "Re": Re.name,
                        "fold_paths": fold_paths
                    })
    return leaves


def load_df(fp, bucket):
    df = pd.read_csv(fp)
    df["__bucket__"] = bucket
    # ===== 几何组合特征 =====
    df["angle_sum"] = df["底角θ1"] + df["底角θ2"]
    df["angle_diff"] = np.abs(df["底角θ1"] - df["底角θ2"])
    df["angle_ratio"] = df["底角θ1"] / np.clip(df["底角θ2"], 1e-12, None)

    # ===== 流动-几何交互特征 =====
    df["Re_R"] = df["Re(雷诺数)"] * df["R(曲率半径）"]
    df["Re_r"] = df["Re(雷诺数)"] * df["r(粒径)"]
    df["Re_X"] = df["Re(雷诺数)"] * df["X"]
    df["v_r"] = df["流速(m/s)"] * df["r(粒径)"]
    df["v_R"] = df["流速(m/s)"] * df["R(曲率半径）"]
    df["X_zH"] = df["X"] * df["z/H"]

    # ===== 比值 / 无量纲风格特征 =====
    df["R_zH"] = df["R(曲率半径）"] / np.clip(df["z/H"], 1e-12, None)
    df["r_zH"] = df["r(粒径)"] / np.clip(df["z/H"], 1e-12, None)
    df["X_zH_ratio"] = df["X"] / np.clip(df["z/H"], 1e-12, None)
    df["R_r_ratio"] = df["R(曲率半径）"] / np.clip(df["r(粒径)"], 1e-12, None)

    return df


# ================== 目标变换 ==================
def transform_y(y, mode):
    if mode == "log1p":
        return np.log1p(np.clip(y, 1e-15, None))
    elif mode == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    return y


def inverse_y(y, mode):
    if mode == "log1p":
        return np.expm1(y)
    elif mode == "sqrt":
        return y ** 2
    return y


# ================== 核心训练 ==================
def run_experiment(config):
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

    leaves = discover_leaves(config["data_root"], config["n_splits"])
    print(f"Total leaves: {len(leaves)}")

    all_metrics = []

    for k in range(1, config["n_splits"] + 1):
        train_parts, test_parts = [], []

        for leaf in leaves:
            bucket = f"{leaf['shape']}/{leaf['R']}/{leaf['Re']}"

            if k in leaf["fold_paths"]:
                test_parts.append(load_df(leaf["fold_paths"][k], bucket))

            for j in range(1, config["n_splits"] + 1):
                if j != k and j in leaf["fold_paths"]:
                    train_parts.append(load_df(leaf["fold_paths"][j], bucket))

        train_df = pd.concat(train_parts)
        test_df = pd.concat(test_parts)

        Xtr = train_df[config["features"]].values
        Xte = test_df[config["features"]].values

        scaler = MinMaxScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        ytr = train_df[config["target"]].values
        yte = test_df[config["target"]].values

        ytr_t = transform_y(ytr, config["transform"])

        model = AutoTabPFNRegressor(max_time=config["max_time"],ignore_pretraining_limits=True,)
        model.fit(Xtr, ytr_t)

        pred = inverse_y(model.predict(Xte), config["transform"])

        r2 = r2_score(yte, pred)
        print(f"[{config['target']}] fold{k} R2={r2:.4f}")

        all_metrics.append(r2)

    print(f"\nFinal {config['target']} R2 = {np.mean(all_metrics):.4f}")