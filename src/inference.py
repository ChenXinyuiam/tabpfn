"""
Inference entry points:
  - predict_target()  : predict a single regression target (FD or FL) on new data
  - predict_focus()   : two-stage pipeline — predict FD+FL, then classify 聚焦判定
  - main()            : CLI for running predictions

Usage (run from project root):
    python -m src.inference --mode focus
    python -m src.inference --mode fd
    python -m src.inference --mode fl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.ensemble import RandomForestClassifier

from .config import (
    MODEL_DIR, TRIANGLE_GROUPS_ROOT, RESULTS_DIR,
    FD_CONFIG, FL_CONFIG,
    BASE_FEATURES, ENGINEERED_FEATURES, N_SPLITS,
    FD_TRANSFORM, FL_TRANSFORM, FD_MAX_TIME, FL_MAX_TIME,
    TABPFN_IGNORE_PRETRAINING_LIMITS, CLASSIFIER_RANDOM_STATE,
)
from utils import (
    setup_offline_tabpfn,
    mape, smape,
    transform_y, inverse_y,
    add_engineered_features,
    discover_leaves, load_fold_df, check_columns,
)


def _get_regressor(max_time: int):
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    return AutoTabPFNRegressor(
        max_time=max_time,
        ignore_pretraining_limits=TABPFN_IGNORE_PRETRAINING_LIMITS,
    )


def predict_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
    transform_mode: str,
    max_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit an AutoTabPFNRegressor on train_df and predict both train and test splits.

    Returns (train_pred, test_pred) in original scale.
    """
    check_columns(train_df, features + [target])
    check_columns(test_df,  features + [target])

    scaler = MinMaxScaler()
    X_tr = scaler.fit_transform(train_df[features].values)
    X_te = scaler.transform(test_df[features].values)

    y_tr = train_df[target].values.astype(float)
    model = _get_regressor(max_time)
    model.fit(X_tr, transform_y(y_tr, transform_mode))

    train_pred = inverse_y(model.predict(X_tr), transform_mode)
    test_pred  = inverse_y(model.predict(X_te), transform_mode)
    return train_pred, test_pred


def predict_focus(data_root: Path, results_dir: Path, use_engineered: bool = True) -> pd.DataFrame:
    """
    Two-stage focus classification:
      Stage 1 — AutoTabPFNRegressor predicts FD and FL
      Stage 2 — RandomForestClassifier predicts 聚焦判定 using [features + FD_pred + FL_pred]

    Saves metrics and prediction CSV files under results_dir.
    Returns the fold-level metrics DataFrame.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    if use_engineered:
        stage1_features = ENGINEERED_FEATURES
    else:
        stage1_features = BASE_FEATURES

    stage2_features = BASE_FEATURES + ["FD_pred", "FL_pred"]

    leaves = discover_leaves(data_root, N_SPLITS)
    if not leaves:
        raise RuntimeError(f"No fold CSVs found under {data_root}")
    print(f"predict_focus: {len(leaves)} leaves")

    fold_metrics   = []
    prediction_rows = []

    for k in range(1, N_SPLITS + 1):
        print(f"\n=== fold{k} ===")
        train_parts, test_parts = [], []

        for leaf in leaves:
            bucket = f"{leaf['shape']}/{leaf['R']}/{leaf['Re']}"
            if k in leaf["fold_paths"]:
                test_parts.append(load_fold_df(leaf["fold_paths"][k], bucket, add_features=use_engineered))
            for j in range(1, N_SPLITS + 1):
                if j != k and j in leaf["fold_paths"]:
                    train_parts.append(load_fold_df(leaf["fold_paths"][j], bucket, add_features=use_engineered))

        if not train_parts or not test_parts:
            print(f"  [warn] fold{k}: empty split, skipping")
            continue

        train_df = pd.concat(train_parts, ignore_index=True)
        test_df  = pd.concat(test_parts,  ignore_index=True)

        required = stage1_features + ["FD", "FL", "聚焦判定"]
        check_columns(train_df, required)
        check_columns(test_df,  required)

        # Stage 1: FD
        fd_tr_pred, fd_te_pred = predict_target(
            train_df, test_df, stage1_features, "FD", FD_TRANSFORM, FD_MAX_TIME
        )
        # Stage 1: FL
        fl_tr_pred, fl_te_pred = predict_target(
            train_df, test_df, stage1_features, "FL", FL_TRANSFORM, FL_MAX_TIME
        )

        train_s2 = train_df.copy()
        test_s2  = test_df.copy()
        train_s2["FD_pred"] = fd_tr_pred;  train_s2["FL_pred"] = fl_tr_pred
        test_s2["FD_pred"]  = fd_te_pred;  test_s2["FL_pred"]  = fl_te_pred

        # Stage 2: classify 聚焦判定
        check_columns(train_s2, stage2_features + ["聚焦判定"])
        check_columns(test_s2,  stage2_features + ["聚焦判定"])

        scaler_cls = MinMaxScaler()
        X_tr_cls = scaler_cls.fit_transform(train_s2[stage2_features].values)
        X_te_cls = scaler_cls.transform(test_s2[stage2_features].values)

        le = LabelEncoder()
        y_tr_cls = le.fit_transform(train_s2["聚焦判定"].astype(str))
        y_te_cls = le.transform(test_s2["聚焦判定"].astype(str))

        clf = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=CLASSIFIER_RANDOM_STATE
        )
        clf.fit(X_tr_cls, y_tr_cls)
        y_pred_cls = clf.predict(X_te_cls)
        y_prob_cls = clf.predict_proba(X_te_cls)

        class_names = list(le.classes_)
        pos_idx = class_names.index("聚焦") if "聚焦" in class_names else 1
        y_prob_pos = y_prob_cls[:, pos_idx]

        acc  = accuracy_score(y_te_cls, y_pred_cls)
        prec = precision_score(y_te_cls, y_pred_cls, average="binary", pos_label=pos_idx, zero_division=0)
        rec  = recall_score(y_te_cls, y_pred_cls, average="binary", pos_label=pos_idx, zero_division=0)
        f1   = f1_score(y_te_cls, y_pred_cls, average="binary", pos_label=pos_idx, zero_division=0)
        try:
            auc = roc_auc_score((y_te_cls == pos_idx).astype(int), y_prob_pos)
        except Exception:
            auc = float("nan")

        print(f"  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  auc={auc:.4f}")
        print(f"  confusion:\n{confusion_matrix(y_te_cls, y_pred_cls)}")

        fold_metrics.append({
            "fold": f"fold{k}", "rows": len(y_te_cls),
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc,
        })

        pred_out = test_s2.copy()
        pred_out["聚焦判定_true"] = test_s2["聚焦判定"].astype(str)
        pred_out["聚焦判定_pred"] = le.inverse_transform(y_pred_cls)
        pred_out["聚焦_prob"]     = y_prob_pos
        pred_out["fold"]          = f"fold{k}"
        prediction_rows.append(pred_out)

    metrics_df = pd.DataFrame(fold_metrics)
    preds_df   = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    metrics_df.to_csv(results_dir / "focus_fold_metrics.csv",       index=False, encoding="utf-8-sig")
    preds_df.to_csv(results_dir   / "focus_fold_predictions.csv",   index=False, encoding="utf-8-sig")

    if not metrics_df.empty:
        N = metrics_df["rows"].sum()
        w = metrics_df["rows"] / N
        summary = pd.DataFrame([{
            "rows_total":   int(N),
            "accuracy_w":  float(np.average(metrics_df["accuracy"],  weights=w)),
            "precision_w": float(np.average(metrics_df["precision"], weights=w)),
            "recall_w":    float(np.average(metrics_df["recall"],    weights=w)),
            "f1_w":        float(np.average(metrics_df["f1"],        weights=w)),
            "roc_auc_w":   float(np.average(metrics_df["roc_auc"].fillna(0), weights=w)),
        }])
        summary.to_csv(results_dir / "focus_weighted_summary.csv", index=False, encoding="utf-8-sig")

    print(f"\nResults saved to {results_dir}")
    return metrics_df


def main():
    parser = argparse.ArgumentParser(description="Run TabPFN inference")
    parser.add_argument("--mode", choices=["fd", "fl", "focus"], default="focus",
                        help="fd: predict FD only (CV report); fl: predict FL only; focus: two-stage focus classification")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--data-root", type=Path, default=TRIANGLE_GROUPS_ROOT)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_offline_tabpfn(args.model_dir)

    if args.mode == "focus":
        results_dir = args.results_dir or (RESULTS_DIR / "focus")
        predict_focus(args.data_root, results_dir)
    else:
        from .train import run_cv_with_metrics
        cfg = dict(FD_CONFIG if args.mode == "fd" else FL_CONFIG, data_root=args.data_root)
        results_dir = args.results_dir or (RESULTS_DIR / f"cv_{cfg['target']}")
        run_cv_with_metrics(cfg, results_dir)


if __name__ == "__main__":
    main()
