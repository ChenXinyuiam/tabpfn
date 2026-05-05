"""Feature analysis for raw 2026-05-01 FL data.

The script reads raw Excel workbooks directly, standardizes the column names,
builds physically interpretable derived features, and writes publication-ready
figures plus summary tables.
"""

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


RENAME = {
    "BC": "base_m",
    "AB": "side_ab_m",
    "AC": "side_ac_m",
    "X": "x_m",
    "Z": "z_m",
    "雷诺数": "Re",
    "周长": "perimeter_m",
    "面积": "area_m2",
    "高": "height_um",
    "角度A": "angle_A_deg",
    "角度B": "angle_B_deg",
    "角度C": "angle_C_deg",
    "到AB垂直距离": "d_to_AB_m",
    "到AC垂直距离": "d_to_AC_m",
    "到BC垂直距离": "d_to_BC_m",
    "点到AB的垂直距离": "d_to_AB_m",
    "点到AC的垂直距离": "d_to_AC_m",
    "点到BC的垂直距离": "d_to_BC_m",
    "到AB直线距离": "line_to_AB_m",
    "到AC直线距离": "line_to_AC_m",
    "点到AB的直线距离": "line_to_AB_m",
    "点到AC的直线距离": "line_to_AC_m",
    "FL": "FL",
}

FEATURE_COLUMNS = [
    "base_um",
    "height_um",
    "aspect_ratio",
    "area_um2",
    "perimeter_um",
    "side_ab_um",
    "side_ac_um",
    "angle_A_deg",
    "angle_B_deg",
    "angle_C_deg",
    "Re",
    "Re_norm",
    "x_norm",
    "z_norm",
    "d_to_AB_norm",
    "d_to_AC_norm",
    "d_to_BC_norm",
    "lambda_A_like",
    "lambda_B_like",
    "lambda_C_like",
    "is_sym",
]


def geometry_id(filename: str) -> str:
    match = re.search(r"(\d+-\d+)", filename)
    return match.group(1) if match else Path(filename).stem


def load_raw_batch(raw_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
        df["source_file"] = path.name
        df["geometry"] = geometry_id(path.name)
        df["is_sym"] = path.name.count("反") > 0
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No raw .xlsx files found under {raw_dir}")
    data = pd.concat(parts, ignore_index=True)
    for column in set(RENAME.values()) & set(data.columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["FL", "Re", "x_m", "z_m"]).reset_index(drop=True)
    return data


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_um"] = df["base_m"] * 1e6
    df["side_ab_um"] = df["side_ab_m"] * 1e6
    df["side_ac_um"] = df["side_ac_m"] * 1e6
    df["perimeter_um"] = df["perimeter_m"] * 1e6
    df["area_um2"] = df["area_m2"] * 1e12
    df["x_um"] = df["x_m"] * 1e6
    df["z_um"] = df["z_m"] * 1e6
    for src, dst in [
        ("d_to_AB_m", "d_to_AB_um"),
        ("d_to_AC_m", "d_to_AC_um"),
        ("d_to_BC_m", "d_to_BC_um"),
        ("line_to_AB_m", "line_to_AB_um"),
        ("line_to_AC_m", "line_to_AC_um"),
    ]:
        if src in df.columns:
            df[dst] = df[src] * 1e6

    df["FL_micro"] = df["FL"] * 1e6
    df["abs_FL_micro"] = df["FL"].abs() * 1e6
    df["signed_orientation"] = np.where(df["is_sym"], "sym", "FL")
    df["height_raw"] = df["height_um"]
    df["height_um"] = np.where(df["height_raw"].abs() < 10, df["height_raw"] * 1e6, df["height_raw"])
    df["height_unit_inferred"] = np.where(df["height_raw"].abs() < 10, "m_to_um", "um")
    df["aspect_ratio"] = df["height_um"] / df["base_um"]
    df["Re_norm"] = df["Re"] / df["Re"].max()
    df["x_norm"] = df["x_um"] / df["base_um"]
    df["z_norm"] = df["z_um"] / df["height_um"]

    eps = 1e-12
    df["d_to_AB_norm"] = df.get("d_to_AB_um", np.nan) / np.maximum(df["height_um"], eps)
    df["d_to_AC_norm"] = df.get("d_to_AC_um", np.nan) / np.maximum(df["height_um"], eps)
    df["d_to_BC_norm"] = df.get("d_to_BC_um", np.nan) / np.maximum(df["height_um"], eps)
    denom = 2.0 * np.maximum(df["area_um2"], eps)
    df["lambda_A_like"] = df["base_um"] * df["d_to_BC_um"] / denom
    df["lambda_B_like"] = df["side_ac_um"] * df["d_to_AC_um"] / denom
    df["lambda_C_like"] = df["side_ab_um"] * df["d_to_AB_um"] / denom
    return df


def savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overview(df: pd.DataFrame, out_dir: Path) -> None:
    counts = df.groupby(["geometry", "signed_orientation"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    counts.plot(kind="bar", ax=ax, color=["#3b6ea8", "#d28f3d"])
    ax.set_ylabel("Rows")
    ax.set_xlabel("Triangle geometry")
    ax.set_title("Raw data coverage by triangle and orientation")
    ax.legend(title="Orientation")
    savefig(fig, out_dir / "fig01_dataset_coverage.png")


def plot_fl_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for geom, sub in df.groupby("geometry"):
        axes[0].hist(sub["FL_micro"], bins=45, histtype="step", linewidth=1.3, label=geom)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Signed FL x 1e6")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Signed FL distribution")
    axes[0].legend(fontsize=7, ncol=2)

    order = sorted(df["geometry"].unique())
    data = [df.loc[df["geometry"] == geom, "abs_FL_micro"].values for geom in order]
    axes[1].boxplot(data, labels=order, showfliers=False)
    axes[1].set_xlabel("Triangle geometry")
    axes[1].set_ylabel("|FL| x 1e6")
    axes[1].set_title("Loss magnitude by geometry")
    axes[1].tick_params(axis="x", rotation=35)
    savefig(fig, out_dir / "fig02_fl_distribution_by_geometry.png")


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    corr_cols = [c for c in FEATURE_COLUMNS + ["FL_micro", "abs_FL_micro"] if c in df.columns]
    corr = df[corr_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10.0, 8.3))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr_cols)), corr_cols, fontsize=7)
    ax.set_title("Spearman feature correlation matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    savefig(fig, out_dir / "fig03_spearman_correlation_heatmap.png")
    return corr


def plot_pca(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
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
            "Re": df["Re"].values,
            "abs_FL_micro": df["abs_FL_micro"].values,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for geom, sub in score_df.groupby("geometry"):
        axes[0].scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.65, label=geom)
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[0].set_title("Feature space structured by geometry")
    axes[0].legend(fontsize=7, ncol=2)

    sc = axes[1].scatter(score_df["PC1"], score_df["PC2"], c=score_df["abs_FL_micro"], s=8, cmap="viridis", alpha=0.75)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[1].set_title("Feature space colored by |FL|")
    fig.colorbar(sc, ax=axes[1], label="|FL| x 1e6")
    savefig(fig, out_dir / "fig04_pca_feature_space.png")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=["PC1_loading", "PC2_loading"],
    )
    return loadings


def plot_feature_importance(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df["abs_FL_micro"].values
    groups = df["geometry"].values

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1, min_samples_leaf=3),
    )
    cv = GroupKFold(n_splits=min(5, df["geometry"].nunique()))
    pred = cross_val_predict(model, x, y, cv=cv, groups=groups, n_jobs=None)
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
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.barh(top["feature"], top["permutation_importance_mean"], xerr=top["permutation_importance_std"], color="#4c78a8")
    ax.set_xlabel("Permutation importance for |FL| x 1e6")
    ax.set_title(f"Predictive feature hierarchy (group-CV R2={cv_r2:.3f})")
    savefig(fig, out_dir / "fig05_feature_importance_abs_FL.png")
    return imp


def plot_response_maps(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))

    re_summary = df.groupby("Re", as_index=False).agg(
        mean_abs_FL=("abs_FL_micro", "mean"),
        sem_abs_FL=("abs_FL_micro", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    )
    axes[0, 0].errorbar(re_summary["Re"], re_summary["mean_abs_FL"], yerr=re_summary["sem_abs_FL"], marker="o")
    axes[0, 0].set_xlabel("Re")
    axes[0, 0].set_ylabel("Mean |FL| x 1e6")
    axes[0, 0].set_title("Reynolds-number scaling")

    geom_summary = df.groupby("geometry", as_index=False)["abs_FL_micro"].mean()
    axes[0, 1].bar(geom_summary["geometry"], geom_summary["abs_FL_micro"], color="#59a14f")
    axes[0, 1].set_xlabel("Triangle geometry")
    axes[0, 1].set_ylabel("Mean |FL| x 1e6")
    axes[0, 1].set_title("Geometry-level loss magnitude")
    axes[0, 1].tick_params(axis="x", rotation=35)

    hb = axes[1, 0].hexbin(df["x_norm"], df["z_norm"], C=df["abs_FL_micro"], gridsize=32, reduce_C_function=np.nanmean, cmap="magma")
    axes[1, 0].set_xlabel("x / base")
    axes[1, 0].set_ylabel("z / height")
    axes[1, 0].set_title("Spatially averaged |FL|")
    fig.colorbar(hb, ax=axes[1, 0], label="Mean |FL| x 1e6")

    axes[1, 1].scatter(df["aspect_ratio"], df["abs_FL_micro"], c=df["Re"], s=10, cmap="viridis", alpha=0.45)
    axes[1, 1].set_xlabel("height / base")
    axes[1, 1].set_ylabel("|FL| x 1e6")
    axes[1, 1].set_title("Aspect ratio-Re interaction")
    savefig(fig, out_dir / "fig06_response_relationships.png")


def write_tables(df: pd.DataFrame, corr: pd.DataFrame, loadings: pd.DataFrame, importance: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "raw_feature_matrix.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby(["geometry", "signed_orientation"], as_index=False).agg(
        rows=("FL", "size"),
        Re_min=("Re", "min"),
        Re_max=("Re", "max"),
        mean_FL_micro=("FL_micro", "mean"),
        mean_abs_FL_micro=("abs_FL_micro", "mean"),
        std_abs_FL_micro=("abs_FL_micro", "std"),
        base_um=("base_um", "median"),
        height_um=("height_um", "median"),
        aspect_ratio=("aspect_ratio", "median"),
    )
    summary.to_csv(out_dir / "geometry_orientation_summary.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out_dir / "spearman_correlation.csv", encoding="utf-8-sig")
    loadings.to_csv(out_dir / "pca_loadings.csv", encoding="utf-8-sig")
    importance.to_csv(out_dir / "feature_importance_abs_FL.csv", index=False, encoding="utf-8-sig")

    target_corr = corr[["FL_micro", "abs_FL_micro"]].drop(index=["FL_micro", "abs_FL_micro"], errors="ignore")
    target_corr.reindex(target_corr["abs_FL_micro"].abs().sort_values(ascending=False).index).to_csv(
        out_dir / "target_spearman_correlations.csv",
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw feature analysis for the 2026-05-01 batch.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rawdata/2026-05-01"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/feature_analysis/2026-05-01_raw"))
    args = parser.parse_args()

    fig_dir = args.out_dir / "figures"
    df = add_derived_features(load_raw_batch(args.raw_dir))
    plot_overview(df, fig_dir)
    plot_fl_distributions(df, fig_dir)
    corr = plot_correlation(df, fig_dir)
    loadings = plot_pca(df, fig_dir)
    importance = plot_feature_importance(df, fig_dir)
    plot_response_maps(df, fig_dir)
    write_tables(df, corr, loadings, importance, args.out_dir)
    print(f"Rows: {len(df)}")
    print(f"Geometries: {sorted(df['geometry'].unique())}")
    print(f"Figures written to: {fig_dir}")
    print(f"Tables written to: {args.out_dir}")


if __name__ == "__main__":
    main()
