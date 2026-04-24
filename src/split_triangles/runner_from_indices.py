# runner_from_indices.py
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# ========= 配置区 =========
INPUT_CSV  = r"D:\pmc\tabpfn_project\data_tabpfn\测试集数据20251008.csv"
IND_DIR    = r"D:\pmc\tabpfn_project\data\cv5_indices_by_rebin"   # 与文件1的 OUTPUT_DIR 对齐
RESULTS_DIR = r"D:\pmc\tabpfn_project\results_cv5_by_rebin"

TARGET_COL = "FL"
SELECTED_FEATURES = [
    "底角θ1", "底角θ2", "Re(雷诺数)", "R(曲率半径）", "z/H", "流速(m/s)", "r(粒径)", "X"
]

TABPFN_ARGS = dict(
    max_time=30,
    ignore_pretraining_limits=True,
    # 可加 device / random_state 等
)

# ========= 指标 =========
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100

def smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_pred - y_true) / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-12, None)) * 100

def load_indices(ind_dir: str) -> Dict[int, np.ndarray]:
    ind_dir = Path(ind_dir)
    folds = {}
    for i in range(1, 6):
        p = ind_dir / f"fold_{i}.txt"
        if not p.exists():
            raise FileNotFoundError(f"缺少索引文件：{p}")
        arr = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                arr.append(int(line))
        folds[i] = np.array(arr, dtype=int)
    return folds

def transform_features_targets(df: pd.DataFrame, fit_idx: np.ndarray, apply_idx: np.ndarray):
    """防泄漏变换：在 fit_idx 上拟合 scaler，应用到 apply_idx；目标用 log1p。"""
    X = df[SELECTED_FEATURES].values
    y = df[TARGET_COL].values.reshape(-1, 1)

    sx = MinMaxScaler()
    sy = StandardScaler()

    X_tr = sx.fit_transform(X[fit_idx])
    y_tr_log = np.log1p(np.clip(y[fit_idx], 1e-15, None)).ravel()
    # 应用到测试
    X_te = sx.transform(X[apply_idx])
    y_te = y[apply_idx].ravel()
    y_te_log = np.log1p(np.clip(y[apply_idx], 1e-15, None)).ravel()

    return X_tr, y_tr_log, X_te, y_te, y_te_log, sx, sy  # sy 预留（若想把 log 标准化）

def run_fold(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, fold_id: int) -> dict:
    X_tr, y_tr_log, X_te, y_te, y_te_log, sx, sy = transform_features_targets(df, train_idx, test_idx)

    model = AutoTabPFNRegressor(**TABPFN_ARGS)
    model.fit(X_tr, y_tr_log)

    y_pred_log = model.predict(X_te)
    y_pred = np.expm1(y_pred_log)

    mse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    m   = mape(y_te, y_pred)
    s   = smape(y_te, y_pred)

    print(f"[Fold {fold_id}] ✅ MSE={mse:.3e} | MAE={mae:.3e} | R²={r2:.4f} | MAPE={m:.3f}% | SMAPE={s:.3f}%")
    return {
        "fold": fold_id,
        "n_test": int(len(test_idx)),
        "mse": mse, "mae": mae, "r2": r2, "mape": m, "smape": s
    }

def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    missing = set(SELECTED_FEATURES + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列：{missing}")

    folds = load_indices(IND_DIR)

    # 五折轮转：第 i 折为测试，其余为训练
    all_metrics = []
    n_total = len(df)
    for i in range(1, 6):
        test_idx = folds[i]
        # 训练集 = 其余四折
        train_idx = np.concatenate([folds[j] for j in range(1, 6) if j != i], axis=0)

        # 稳健性断言：索引不重不漏（可注释）
        assert len(np.unique(test_idx)) == len(test_idx)
        assert len(np.unique(train_idx)) == len(train_idx)
        assert set(test_idx.tolist()).isdisjoint(set(train_idx.tolist()))
        assert len(np.unique(np.concatenate([train_idx, test_idx]))) == n_total

        metrics = run_fold(df, train_idx, test_idx, i)
        all_metrics.append(metrics)

        # 每折预测详情（可选导出）
        # 也可把 y_pred 一并导出，这里保持精简
        # 若需要：把 run_fold 改为同时返回 y_pred 细表

    # 汇总
    met_df = pd.DataFrame(all_metrics)
    met_df.to_csv(Path(RESULTS_DIR) / "cv5_metrics_by_rebin.csv", index=False, encoding="utf-8-sig")

    # 按测试集样本量做加权平均
    w = met_df["n_test"] / met_df["n_test"].sum()
    agg = {
        "MSE_weighted": np.average(met_df["mse"], weights=w),
        "MAE_weighted": np.average(met_df["mae"], weights=w),
        "R2_weighted":  np.average(met_df["r2"],  weights=w),
        "MAPE_weighted":np.average(met_df["mape"],weights=w),
        "SMAPE_weighted":np.average(met_df["smape"],weights=w),
    }
    pd.DataFrame([agg]).to_csv(Path(RESULTS_DIR) / "cv5_metrics_weighted_summary.csv",
                               index=False, encoding="utf-8-sig")

    print("\n======== 五折（雷诺桶-索引驱动）加权汇总 ========")
    print(f"✅ Weighted MSE:   {agg['MSE_weighted']:.6e}")
    print(f"✅ Weighted MAE:   {agg['MAE_weighted']:.6e}")
    print(f"✅ Weighted R²:    {agg['R2_weighted']:.4f}")
    print(f"✅ Weighted MAPE:  {agg['MAPE_weighted']:.3f}%")
    print(f"✅ Weighted SMAPE: {agg['SMAPE_weighted']:.3f}%")
    print(f"明细输出目录：{RESULTS_DIR}")

if __name__ == "__main__":
    main()
