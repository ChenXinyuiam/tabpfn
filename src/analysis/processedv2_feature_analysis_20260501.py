"""Feature analysis for data/interim/features/2026-05-01 without lambda features."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LAMBDA_COLS = {
    "lambda_A",
    "lambda_B",
    "lambda_C",
    "lambda_sum",
    "lambda_A_norm",
    "lambda_B_norm",
    "lambda_C_norm",
}

FEATURE_COLUMNS = [
    "base_um",
    "height_um",
    "aspect_ratio",
    "area_um2",
    "perimeter_um",
    "side_ab_um",
    "side_ac_um",
    "left_side_um",
    "right_side_um",
    "angle_1_deg",
    "angle_2_deg",
    "angle_3_deg",
    "Re",
    "Re_norm",
    "x_norm",
    "z_norm",
    "d_left_norm",
    "d_right_norm",
    "d_base_norm",
    "line_ab_norm",
    "line_ac_norm",
    "geometry_valid",
    "is_sym",
]


def geometry_id(filename: str) -> str:
    match = re.search(r"(\d+-\d+)", filename)
    return match.group(1) if match else Path(filename).stem


def as_um(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.abs().median()
    if pd.notna(median) and median < 10:
        return values * 1e6
    return values


def load_feature_workbooks(data_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(data_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df = df.drop(columns=[c for c in LAMBDA_COLS if c in df.columns], errors="ignore")
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        df["source_file"] = path.name
        df["geometry"] = geometry_id(path.name)
        df["is_sym"] = path.name.endswith("-sym.xlsx")
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No feature .xlsx files found under {data_dir}")
    df = pd.concat(parts, ignore_index=True)
    for col in df.columns:
        if col not in {"source_file", "geometry"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["FL", "雷诺数", "X", "Z"]).reset_index(drop=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_um"] = as_um(df["三角形底边"])
    df["side_ab_um"] = as_um(df["AB"])
    df["side_ac_um"] = as_um(df["AC"])
    df["left_side_um"] = as_um(df["左边长度"])
    df["right_side_um"] = as_um(df["右边长度"])
    df["x_um"] = as_um(df["X"])
    df["z_um"] = as_um(df["Z"])
    df["perimeter_um"] = as_um(df["周长"])
    height_raw = pd.to_numeric(df["高"], errors="coerce")
    df["height_um"] = np.where(height_raw.abs() < 10, height_raw * 1e6, height_raw)
    df["area_um2"] = pd.to_numeric(df["面积"], errors="coerce")
    df["area_calc_um2"] = pd.to_numeric(df["面积_calc"], errors="coerce")
    df["d_left_um"] = as_um(df["点到左边距离"])
    df["d_right_um"] = as_um(df["点到右边距离"])
    df["d_base_um"] = as_um(df["点到底边距离"])
    df["line_ab_um"] = as_um(df["到AB直线距离"])
    df["line_ac_um"] = as_um(df["到AC直线距离"])
    df["angle_1_deg"] = df["角度1"]
    df["angle_2_deg"] = df["角度2"]
    df["angle_3_deg"] = df["角度3"]
    df["Re"] = df["雷诺数"]
    df["Re_norm"] = df["Re"] / df["Re"].max()
    df["FL_scaled"] = df["FL"]
    df["abs_FL_scaled"] = df["FL"].abs()
    df["orientation"] = np.where(df["is_sym"].astype(bool), "sym", "FL")
    df["geometry_valid"] = df["geometry_valid"].astype(float)

    eps = 1e-12
    df["aspect_ratio"] = df["height_um"] / np.maximum(df["base_um"], eps)
    df["x_norm"] = df["x_um"] / np.maximum(df["base_um"], eps)
    df["z_norm"] = df["z_um"] / np.maximum(df["height_um"], eps)
    df["d_left_norm"] = df["d_left_um"] / np.maximum(df["height_um"], eps)
    df["d_right_norm"] = df["d_right_um"] / np.maximum(df["height_um"], eps)
    df["d_base_norm"] = df["d_base_um"] / np.maximum(df["height_um"], eps)
    df["line_ab_norm"] = df["line_ab_um"] / np.maximum(df["height_um"], eps)
    df["line_ac_norm"] = df["line_ac_um"] / np.maximum(df["height_um"], eps)
    return df


def savefig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(df: pd.DataFrame, fig_dir: Path) -> None:
    counts = df.groupby(["geometry", "orientation"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_title("Processedv2 data coverage by geometry and orientation")
    ax.set_xlabel("Triangle geometry")
    ax.set_ylabel("Rows")
    ax.legend(title="Orientation")
    savefig(fig, fig_dir / "fig01_processedv2_dataset_coverage.png")


def plot_distributions(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for geom, sub in df.groupby("geometry"):
        axes[0].hist(sub["FL_scaled"], bins=45, histtype="step", linewidth=1.25, label=geom)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Signed FL distribution")
    axes[0].set_xlabel("FL (processed scale)")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=7, ncol=2)

    order = sorted(df["geometry"].unique())
    axes[1].boxplot([df.loc[df["geometry"] == g, "abs_FL_scaled"] for g in order], tick_labels=order, showfliers=False)
    axes[1].set_title("|FL| by geometry")
    axes[1].set_xlabel("Triangle geometry")
    axes[1].set_ylabel("|FL| (processed scale)")
    axes[1].tick_params(axis="x", rotation=35)
    savefig(fig, fig_dir / "fig02_processedv2_fl_distribution_by_geometry.png")


def plot_correlation(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    cols = [c for c in FEATURE_COLUMNS + ["FL_scaled", "abs_FL_scaled"] if c in df.columns]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10.0, 8.3))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    ax.set_title("Spearman correlation heatmap without lambda features")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    savefig(fig, fig_dir / "fig03_processedv2_spearman_correlation_heatmap_no_lambda.png")
    return corr


def plot_pca(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(x_scaled)
    score_df = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "geometry": df["geometry"].values,
            "abs_FL_scaled": df["abs_FL_scaled"].values,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for geom, sub in score_df.groupby("geometry"):
        axes[0].scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.65, label=geom)
    axes[0].set_title("PCA feature space by geometry")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[0].legend(fontsize=7, ncol=2)

    sc = axes[1].scatter(score_df["PC1"], score_df["PC2"], c=score_df["abs_FL_scaled"], s=8, cmap="viridis", alpha=0.75)
    axes[1].set_title("PCA feature space colored by |FL|")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    fig.colorbar(sc, ax=axes[1], label="|FL|")
    savefig(fig, fig_dir / "fig04_processedv2_pca_feature_space_no_lambda.png")

    return pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1_loading", "PC2_loading"])


def plot_importance(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["abs_FL_scaled"].values
    groups = df["geometry"].values
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=3),
    )
    cv = GroupKFold(n_splits=min(5, df["geometry"].nunique()))
    pred = cross_val_predict(model, x, y, cv=cv, groups=groups)
    cv_r2 = r2_score(y, pred)
    model.fit(x, y)
    perm = permutation_importance(model, x, y, n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.DataFrame(
        {
            "feature": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "group_cv_r2_abs_FL": cv_r2,
        }
    ).sort_values("permutation_importance_mean", ascending=False)

    top = imp.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.barh(top["feature"], top["permutation_importance_mean"], xerr=top["permutation_importance_std"], color="#4c78a8")
    ax.set_xlabel("Permutation importance for |FL|")
    ax.set_title(f"Feature hierarchy without lambda features (group-CV R2={cv_r2:.3f})")
    savefig(fig, fig_dir / "fig05_processedv2_feature_importance_abs_FL_no_lambda.png")
    return imp


def plot_response(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
    re_summary = df.groupby("Re", as_index=False).agg(
        mean_abs_FL=("abs_FL_scaled", "mean"),
        sem_abs_FL=("abs_FL_scaled", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    )
    axes[0, 0].errorbar(re_summary["Re"], re_summary["mean_abs_FL"], yerr=re_summary["sem_abs_FL"], marker="o")
    axes[0, 0].set_title("Reynolds-number scaling")
    axes[0, 0].set_xlabel("Re")
    axes[0, 0].set_ylabel("Mean |FL|")

    geom_summary = df.groupby("geometry", as_index=False)["abs_FL_scaled"].mean()
    axes[0, 1].bar(geom_summary["geometry"], geom_summary["abs_FL_scaled"], color="#59a14f")
    axes[0, 1].set_title("Geometry-level loss magnitude")
    axes[0, 1].set_xlabel("Triangle geometry")
    axes[0, 1].set_ylabel("Mean |FL|")
    axes[0, 1].tick_params(axis="x", rotation=35)

    hb = axes[1, 0].hexbin(df["x_norm"], df["z_norm"], C=df["abs_FL_scaled"], gridsize=32, reduce_C_function=np.nanmean, cmap="magma")
    axes[1, 0].set_title("Spatially averaged |FL|")
    axes[1, 0].set_xlabel("x / base")
    axes[1, 0].set_ylabel("z / height")
    fig.colorbar(hb, ax=axes[1, 0], label="Mean |FL|")

    axes[1, 1].scatter(df["aspect_ratio"], df["abs_FL_scaled"], c=df["Re"], s=10, cmap="viridis", alpha=0.45)
    axes[1, 1].set_title("Aspect ratio-Re interaction")
    axes[1, 1].set_xlabel("height / base")
    axes[1, 1].set_ylabel("|FL|")
    savefig(fig, fig_dir / "fig06_processedv2_response_relationships_no_lambda.png")


def write_tables(df: pd.DataFrame, corr: pd.DataFrame, loadings: pd.DataFrame, imp: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "processedv2_feature_matrix_no_lambda.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby(["geometry", "orientation"], as_index=False).agg(
        rows=("FL", "size"),
        Re_min=("Re", "min"),
        Re_max=("Re", "max"),
        mean_FL=("FL_scaled", "mean"),
        mean_abs_FL=("abs_FL_scaled", "mean"),
        std_abs_FL=("abs_FL_scaled", "std"),
        base_um=("base_um", "median"),
        height_um=("height_um", "median"),
        aspect_ratio=("aspect_ratio", "median"),
    )
    summary.to_csv(out_dir / "geometry_orientation_summary_no_lambda.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out_dir / "spearman_correlation_no_lambda.csv", encoding="utf-8-sig")
    loadings.to_csv(out_dir / "pca_loadings_no_lambda.csv", encoding="utf-8-sig")
    imp.to_csv(out_dir / "feature_importance_abs_FL_no_lambda.csv", index=False, encoding="utf-8-sig")
    target_corr = corr[["FL_scaled", "abs_FL_scaled"]].drop(index=["FL_scaled", "abs_FL_scaled"], errors="ignore")
    target_corr.reindex(target_corr["abs_FL_scaled"].abs().sort_values(ascending=False).index).to_csv(
        out_dir / "target_spearman_correlations_no_lambda.csv",
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature workbook analysis without lambda features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/interim/features/2026-05-01"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/feature_analysis/2026-05-01_processedv2_no_lambda"))
    args = parser.parse_args()

    fig_dir = args.out_dir / "figures"
    df = add_features(load_feature_workbooks(args.data_dir))
    plot_coverage(df, fig_dir)
    plot_distributions(df, fig_dir)
    corr = plot_correlation(df, fig_dir)
    loadings = plot_pca(df, fig_dir)
    imp = plot_importance(df, fig_dir)
    plot_response(df, fig_dir)
    write_tables(df, corr, loadings, imp, args.out_dir)
    print(f"Rows: {len(df)}")
    print(f"Geometries: {sorted(df['geometry'].unique())}")
    print(f"Excluded lambda columns: {sorted(LAMBDA_COLS)}")
    print(f"Figures written to: {fig_dir}")
    print(f"Tables written to: {args.out_dir}")


if __name__ == "__main__":
    main()
