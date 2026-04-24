# -*- coding: utf-8 -*-


import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor

# ========= 配置区（按需修改） =========
GROUPS_ROOT  = r"D:\pmc\tabpfn_project\data\triangle_groups_fixed"  # 有 fold1..fold5.csv 的根目录
RESULTS_ROOT = r"D:\pmc\tabpfn_project\results_tabpfn_cv"           # 输出目录
TARGETS      = ["FD", "FL"]                                               # 目标列
SELECTED_FEATURES = [
    "底角θ1", "底角θ2", "Re(雷诺数)", "R(曲率半径）", "z/H", "流速(m/s)", "r(粒径)", "X"
]
TABPFN_MAX_TIME = 30
TABPFN_IGNORE_PRETRAINING_LIMITS = True

# ========= 指标 =========
def mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-12, None))) * 100.0)
def smape(y_true, y_pred):
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / np.clip(np.abs(y_true) + np.abs(y_pred), 1e-12, None)) * 100.0)

# ========= 工具 =========
def discover_buckets(groups_root: Path):
    buckets = []
    for shape_dir in groups_root.iterdir():
        if not shape_dir.is_dir(): 
            continue
        for radius_dir in shape_dir.iterdir():
            if not radius_dir.is_dir():
                continue
            for re_dir in radius_dir.iterdir():
                if not re_dir.is_dir():
                    continue
                # 要求 5 折就绪
                if all((re_dir / f"fold{k}.csv").is_file() for k in range(1, 6)):
                    buckets.append(re_dir)
    return sorted(buckets)

def load_df(fp: Path):
    df = pd.read_csv(fp)
    df["__bucket__"] = str(fp.parent)  # 标记来源桶
    df["__fold__"] = fp.stem           # fold1..fold5
    return df

def load_and_preprocess(df: pd.DataFrame, selected_features, target_col, scaler_X=None, fit=True):
    miss = [c for c in selected_features + [target_col] if c not in df.columns]
    if miss:
        raise ValueError(f"缺少列: {miss}")
    X = df[selected_features].values
    y = df[target_col].values.reshape(-1, 1)
    if fit or scaler_X is None:
        scaler_X = MinMaxScaler()
        Xs = scaler_X.fit_transform(X)
    else:
        Xs = scaler_X.transform(X)
    y_clip = np.clip(y, 1e-15, None)
    y_log  = np.log1p(y_clip).ravel()
    return Xs, y_log, y.ravel(), scaler_X

# ========= 主流程：全桶同步 K=5 =========
def main():
    groups_root = Path(GROUPS_ROOT)
    results_root = Path(RESULTS_ROOT)
    results_root.mkdir(parents=True, exist_ok=True)

    buckets = discover_buckets(groups_root)
    if not buckets:
        print("未发现任何桶（缺少 fold1..5.csv）。请先运行 generate_index.py。")
        return
    print(f"发现 {len(buckets)} 个桶（均含 fold1..5）：")
    for b in buckets[:10]:
        print("  -", b)
    if len(buckets) > 10:
        print("  ...")

    for target in TARGETS:
        print(f"\n==== 目标: {target} | 全桶同步 5 折 ====")
        fold_metrics_all = []   # 每折全局指标
        per_bucket_metrics = [] # 每折下的分桶指标

        for k in range(1, 6):
            # 汇总全局第 k 折：测试集 = 所有桶的 foldk；训练集 = 其余四折
            test_parts, train_parts = [], []
            for bdir in buckets:
                for j in range(1, 6):
                    f = bdir / f"fold{j}.csv"
                    if j == k:
                        test_parts.append(load_df(f))
                    else:
                        train_parts.append(load_df(f))

            test_df  = pd.concat(test_parts,  axis=0, ignore_index=True)
            train_df = pd.concat(train_parts, axis=0, ignore_index=True)

            # 预处理（scaler 仅在训练集拟合）
            X_tr, y_tr_log, y_tr, scaler = load_and_preprocess(train_df, SELECTED_FEATURES, target, None, True)
            X_te, y_te_log, y_te, _      = load_and_preprocess(test_df,  SELECTED_FEATURES, target, scaler, False)

            # 训练 TabPFN
            reg = AutoTabPFNRegressor(max_time=TABPFN_MAX_TIME,
                                      ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS)
            reg.fit(X_tr, y_tr_log)

            # 预测 & 反变换
            y_pred_log = reg.predict(X_te)
            y_pred = np.expm1(y_pred_log)

            # 全局指标
            fold_row = {
                "target": target,
                "fold": f"fold{k}",
                "rows": len(y_te),
                "mse": mean_squared_error(y_te, y_pred),
                "mae": mean_absolute_error(y_te, y_pred),
                "r2":  r2_score(y_te, y_pred),
                "mape": mape(y_te, y_pred),
                "smape": smape(y_te, y_pred),
            }
            fold_metrics_all.append(fold_row)

            # 分桶指标（用于体检每个工况在该折的表现）
            gb = test_df.copy()
            gb[f"{target}_pred"] = y_pred
            # 按桶聚合
            for bucket_id, sub in gb.groupby("__bucket__"):
                y_true_b = sub[target].values
                y_pred_b = sub[f"{target}_pred"].values
                per_bucket_metrics.append({
                    "target": target,
                    "fold": f"fold{k}",
                    "bucket": bucket_id,
                    "rows": len(y_true_b),
                    "mse": mean_squared_error(y_true_b, y_pred_b),
                    "mae": mean_absolute_error(y_true_b, y_pred_b),
                    "r2":  r2_score(y_true_b, y_pred_b) if len(y_true_b) > 1 else np.nan,
                    "mape": mape(y_true_b, y_pred_b),
                    "smape": smape(y_true_b, y_pred_b),
                })

        # 写出全局与分桶明细
        out_dir = results_root / f"global_sync_{target}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df_folds = pd.DataFrame(fold_metrics_all)
        df_folds.to_csv(out_dir / "fold_global_metrics.csv", index=False, encoding="utf-8-sig")

        df_buckets = pd.DataFrame(per_bucket_metrics)
        df_buckets.to_csv(out_dir / "fold_bucket_metrics.csv", index=False, encoding="utf-8-sig")

        # 全局加权汇总（按测试样本数加权）
        N = df_folds["rows"].sum()
        w = df_folds["rows"] / (N if N > 0 else 1)
        global_summary = pd.DataFrame([{
            "target": target,
            "rows_total": int(N),
            "mse_w": float(np.average(df_folds["mse"], weights=w)),
            "mae_w": float(np.average(df_folds["mae"], weights=w)),
            "r2_w":  float(np.average(df_folds["r2"],  weights=w)),
            "mape_w": float(np.average(df_folds["mape"], weights=w)),
            "smape_w": float(np.average(df_folds["smape"], weights=w)),
        }])
        global_summary.to_csv(out_dir / "global_weighted_summary.csv", index=False, encoding="utf-8-sig")
        print(f"[{target}] ✅ 全桶同步 5 折完成，汇总见：{out_dir}")

if __name__ == "__main__":
    main()
