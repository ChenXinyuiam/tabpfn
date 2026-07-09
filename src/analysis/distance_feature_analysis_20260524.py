"""Feature-only analysis for data/rawdata/2026-05-24 distance workbooks."""

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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RAW_COLUMNS = [
    "angle_A",
    "angle_B",
    "angle_C",
    "base",
    "height",
    "B_x",
    "point_x",
    "point_y",
    "dist_AB_horizontal",
    "dist_AC_horizontal",
    "dist_BC_vertical",
    "max_distance_to_A_C",
]

FEATURE_COLUMNS = [
    "angle_A",
    "angle_B",
    "angle_C",
    "base_um",
    "height_um",
    "aspect_ratio",
    "B_x_norm",
    "point_x_norm",
    "point_y_norm",
    "dist_AB_norm",
    "dist_AC_norm",
    "dist_BC_norm",
    "max_distance_norm",
    "edge_distance_balance",
    "edge_distance_sum_norm",
    "is_reflected",
]


def geometry_id(filename: str) -> str:
    match = re.search(r"(\d+-\d+)", filename)
    return match.group(1) if match else Path(filename).stem


def infer_orientation(filename: str) -> str:
    return "reflected" if "\u53cd" in filename else "original"


def load_distance_batch(raw_dir: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        df = pd.read_excel(path, engine="openpyxl")
        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        missing = [c for c in RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")
        df = df[RAW_COLUMNS].copy()
        for col in RAW_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["source_file"] = path.name
        df["geometry"] = geometry_id(path.name)
        df["orientation"] = infer_orientation(path.name)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No distance .xlsx files found under {raw_dir}")
    return pd.concat(parts, ignore_index=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-12
    for src, dst in [
        ("base", "base_um"),
        ("height", "height_um"),
        ("B_x", "B_x_um"),
        ("point_x", "point_x_um"),
        ("point_y", "point_y_um"),
        ("dist_AB_horizontal", "dist_AB_um"),
        ("dist_AC_horizontal", "dist_AC_um"),
        ("dist_BC_vertical", "dist_BC_um"),
        ("max_distance_to_A_C", "max_distance_um"),
    ]:
        df[dst] = df[src] * 1e6
    df["aspect_ratio"] = df["height_um"] / np.maximum(df["base_um"], eps)
    df["B_x_norm"] = df["B_x_um"] / np.maximum(df["base_um"], eps)
    df["point_x_norm"] = df["point_x_um"] / np.maximum(df["base_um"], eps)
    df["point_y_norm"] = df["point_y_um"] / np.maximum(df["height_um"], eps)
    df["dist_AB_norm"] = df["dist_AB_um"] / np.maximum(df["height_um"], eps)
    df["dist_AC_norm"] = df["dist_AC_um"] / np.maximum(df["height_um"], eps)
    df["dist_BC_norm"] = df["dist_BC_um"] / np.maximum(df["height_um"], eps)
    df["max_distance_norm"] = df["max_distance_um"] / np.maximum(
        np.sqrt(df["base_um"] ** 2 + df["height_um"] ** 2),
        eps,
    )
    df["edge_distance_balance"] = (df["dist_AB_um"] - df["dist_AC_um"]) / np.maximum(
        df["dist_AB_um"] + df["dist_AC_um"],
        eps,
    )
    df["edge_distance_sum_norm"] = (df["dist_AB_um"] + df["dist_AC_um"]) / np.maximum(df["height_um"], eps)
    df["is_reflected"] = (df["orientation"] == "reflected").astype(float)
    return df


def usable_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan)
        if series.notna().sum() == 0:
            continue
        if series.nunique(dropna=True) <= 1:
            continue
        cols.append(col)
    return cols


def savefig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(df: pd.DataFrame, fig_dir: Path) -> None:
    counts = df.groupby(["geometry", "orientation"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    counts.plot(kind="bar", ax=ax, color=["#4c78a8", "#f58518"])
    ax.set_title("Distance-feature data coverage")
    ax.set_xlabel("Triangle geometry")
    ax.set_ylabel("Rows")
    ax.legend(title="Orientation")
    savefig(fig, fig_dir / "fig01_dataset_coverage.png")


def plot_geometry_summary(df: pd.DataFrame, fig_dir: Path) -> None:
    summary = df.groupby("geometry", as_index=False).agg(
        rows=("source_file", "size"),
        base_um=("base_um", "median"),
        height_um=("height_um", "median"),
        aspect_ratio=("aspect_ratio", "median"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    x = np.arange(len(summary))
    width = 0.38
    axes[0].bar(x - width / 2, summary["base_um"], width, label="base")
    axes[0].bar(x + width / 2, summary["height_um"], width, label="height")
    axes[0].set_xticks(x, summary["geometry"], rotation=35, ha="right")
    axes[0].set_ylabel("Micrometres")
    axes[0].set_title("Geometry dimensions")
    axes[0].legend()

    axes[1].bar(summary["geometry"], summary["rows"], color="#59a14f")
    axes[1].set_title("Sample count by geometry")
    axes[1].set_xlabel("Triangle geometry")
    axes[1].set_ylabel("Rows")
    axes[1].tick_params(axis="x", rotation=35)
    savefig(fig, fig_dir / "fig02_geometry_dimensions_and_counts.png")


def plot_spatial_coverage(df: pd.DataFrame, fig_dir: Path) -> None:
    rng = np.random.default_rng(42)
    max_points = 35000
    plot_df = df
    if len(df) > max_points:
        plot_df = df.iloc[rng.choice(len(df), size=max_points, replace=False)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.4), sharex=True, sharey=True)
    for ax, (geom, sub) in zip(axes.ravel(), plot_df.groupby("geometry")):
        ax.scatter(sub["point_x_norm"], sub["point_y_norm"], s=2, alpha=0.22)
        ax.set_title(geom)
        ax.set_xlim(-0.03, max(1.65, plot_df["point_x_norm"].max() * 1.04))
        ax.set_ylim(-0.03, 1.05)
    for ax in axes[-1, :]:
        ax.set_xlabel("point_x / base")
    for ax in axes[:, 0]:
        ax.set_ylabel("point_y / height")
    savefig(fig, fig_dir / "fig03_normalized_spatial_coverage.png")


def plot_distance_distributions(df: pd.DataFrame, fig_dir: Path) -> None:
    order = sorted(df["geometry"].unique())
    metrics = [
        ("dist_AB_norm", "dist AB / height"),
        ("dist_AC_norm", "dist AC / height"),
        ("max_distance_norm", "max distance / diagonal"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    for ax, (col, label) in zip(axes, metrics):
        data = [df.loc[df["geometry"] == geom, col].dropna().values for geom in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title(label)
        ax.set_xlabel("Triangle geometry")
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Normalized distance")
    savefig(fig, fig_dir / "fig04_distance_feature_distributions.png")


def plot_missingness(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    missing = (
        df[RAW_COLUMNS]
        .isna()
        .mean()
        .rename("missing_fraction")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.bar(missing["column"], missing["missing_fraction"] * 100, color="#e15759")
    ax.set_ylabel("Missing rows (%)")
    ax.set_title("Raw feature missingness")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    savefig(fig, fig_dir / "fig05_missingness_by_column.png")
    return missing


def plot_correlation(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    cols = usable_feature_columns(df)
    corr = df[cols].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.0, 7.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=65, ha="right", fontsize=7)
    ax.set_yticks(range(len(cols)), cols, fontsize=7)
    ax.set_title("Spearman correlation among distance features")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman rho")
    savefig(fig, fig_dir / "fig06_spearman_correlation_heatmap.png")
    return corr


def plot_pca(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    feature_cols = usable_feature_columns(df)
    x = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    x_scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(x_scaled)
    score_df = pd.DataFrame(
        {
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "geometry": df["geometry"].values,
            "orientation": df["orientation"].values,
        }
    )

    rng = np.random.default_rng(42)
    max_points = 35000
    plot_df = score_df
    if len(score_df) > max_points:
        plot_df = score_df.iloc[rng.choice(len(score_df), size=max_points, replace=False)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    for geom, sub in plot_df.groupby("geometry"):
        axes[0].scatter(sub["PC1"], sub["PC2"], s=3, alpha=0.35, label=geom)
    axes[0].set_title("PCA feature space by geometry")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[0].legend(fontsize=7, ncol=2)

    colors = {"original": "#4c78a8", "reflected": "#f58518"}
    for orientation, sub in plot_df.groupby("orientation"):
        axes[1].scatter(sub["PC1"], sub["PC2"], s=3, alpha=0.35, label=orientation, color=colors.get(orientation))
    axes[1].set_title("PCA feature space by orientation")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    axes[1].legend()
    savefig(fig, fig_dir / "fig07_pca_feature_space.png")

    return pd.DataFrame(pca.components_.T, index=feature_cols, columns=["PC1_loading", "PC2_loading"])


def plot_reflection_pairs(df: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    pair_rows: list[dict[str, float | str | int]] = []
    pair_geoms = []
    for geom, sub in df.groupby("geometry"):
        orientations = set(sub["orientation"])
        if {"original", "reflected"}.issubset(orientations):
            pair_geoms.append(geom)
            orig = sub[sub["orientation"] == "original"]
            refl = sub[sub["orientation"] == "reflected"]
            pair_rows.append(
                {
                    "geometry": geom,
                    "original_rows": len(orig),
                    "reflected_rows": len(refl),
                    "mean_x_norm_original": orig["point_x_norm"].mean(),
                    "mean_x_norm_reflected": refl["point_x_norm"].mean(),
                    "mean_y_norm_original": orig["point_y_norm"].mean(),
                    "mean_y_norm_reflected": refl["point_y_norm"].mean(),
                    "mean_dist_AB_norm_original": orig["dist_AB_norm"].mean(),
                    "mean_dist_AB_norm_reflected": refl["dist_AB_norm"].mean(),
                    "mean_dist_AC_norm_original": orig["dist_AC_norm"].mean(),
                    "mean_dist_AC_norm_reflected": refl["dist_AC_norm"].mean(),
                }
            )

    pair_summary = pd.DataFrame(pair_rows)
    if pair_summary.empty:
        return pair_summary

    fig, axes = plt.subplots(1, len(pair_geoms), figsize=(5.2 * len(pair_geoms), 4.2), sharey=True)
    if len(pair_geoms) == 1:
        axes = np.array([axes])
    for ax, geom in zip(axes.ravel(), pair_geoms):
        sub = df[df["geometry"] == geom]
        data = [
            sub.loc[sub["orientation"] == "original", "point_x_norm"].dropna().values,
            sub.loc[sub["orientation"] == "reflected", "point_x_norm"].dropna().values,
        ]
        ax.boxplot(data, tick_labels=["original", "reflected"], showfliers=False)
        ax.set_title(f"{geom}: x coverage")
        ax.set_ylabel("point_x / base")
    savefig(fig, fig_dir / "fig08_reflected_pair_x_coverage.png")
    return pair_summary


def write_tables(
    df: pd.DataFrame,
    missing: pd.DataFrame,
    corr: pd.DataFrame,
    loadings: pd.DataFrame,
    pair_summary: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "distance_feature_matrix.csv", index=False, encoding="utf-8-sig")
    summary = df.groupby(["geometry", "orientation"], as_index=False).agg(
        rows=("source_file", "size"),
        base_um=("base_um", "median"),
        height_um=("height_um", "median"),
        aspect_ratio=("aspect_ratio", "median"),
        x_norm_min=("point_x_norm", "min"),
        x_norm_max=("point_x_norm", "max"),
        y_norm_min=("point_y_norm", "min"),
        y_norm_max=("point_y_norm", "max"),
        mean_dist_AB_norm=("dist_AB_norm", "mean"),
        mean_dist_AC_norm=("dist_AC_norm", "mean"),
        mean_max_distance_norm=("max_distance_norm", "mean"),
    )
    summary.to_csv(out_dir / "geometry_orientation_summary.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(out_dir / "missingness_summary.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(out_dir / "spearman_correlation.csv", encoding="utf-8-sig")
    loadings.to_csv(out_dir / "pca_loadings.csv", encoding="utf-8-sig")
    pair_summary.to_csv(out_dir / "pairwise_reflection_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def write_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    missing: pd.DataFrame,
    corr: pd.DataFrame,
    loadings: pd.DataFrame,
    pair_summary: pd.DataFrame,
    out_dir: Path,
    raw_dir: Path,
) -> None:
    geometry_rows = summary.groupby("geometry")["rows"].sum().sort_index()
    missing_lines = missing.sort_values("missing_fraction", ascending=False).head(5)
    pc1 = loadings["PC1_loading"].abs().sort_values(ascending=False).head(5)
    pc2 = loadings["PC2_loading"].abs().sort_values(ascending=False).head(5)
    corr_pairs = []
    abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool))
    for _ in range(min(5, abs_corr.size)):
        idx = np.nanargmax(abs_corr.values)
        i, j = np.unravel_index(idx, abs_corr.shape)
        corr_pairs.append((abs_corr.index[i], abs_corr.columns[j], corr.iloc[i, j]))
        abs_corr.iloc[i, j] = np.nan
        abs_corr.iloc[j, i] = np.nan

    report = [
        "# Distance Feature Analysis Report: 2026-05-24",
        "",
        f"Input data: `{raw_dir.as_posix()}`",
        "",
        f"Output figures: `{(out_dir / 'figures').as_posix()}`",
        "",
        "## Dataset",
        "",
        f"The batch contains {len(df):,} rows from {df['source_file'].nunique()} Excel workbooks.",
        "",
    ]
    report.extend([f"- {geom}: {rows:,} rows" for geom, rows in geometry_rows.items()])
    report.extend(
        [
            "",
            "The files are feature-only distance workbooks. They do not include FL, FD, or Reynolds-number target columns, so this analysis focuses on feature quality, spatial coverage, correlations, and PCA structure.",
            "",
            "## Key Results",
            "",
            f"1. The batch covers {df['geometry'].nunique()} geometries. Two geometries include reflected counterparts: {', '.join(pair_summary['geometry'].tolist()) if not pair_summary.empty else 'none'}.",
            "",
            f"2. The total normalized x-range is {df['point_x_norm'].min():.3f} to {df['point_x_norm'].max():.3f}; the normalized y-range is {df['point_y_norm'].min():.3f} to {df['point_y_norm'].max():.3f}.",
            "",
            f"3. `dist_BC_vertical` is missing in {missing.loc[missing['column'] == 'dist_BC_vertical', 'missing_fraction'].iloc[0] * 100:.1f}% of rows. Other raw columns have no missing values.",
            "",
            f"4. PC1 is mainly driven by {', '.join(pc1.index.tolist())}; PC2 is mainly driven by {', '.join(pc2.index.tolist())}.",
            "",
            "5. The strongest absolute Spearman feature associations are:",
        ]
    )
    report.extend([f"   - `{a}` vs `{b}`: rho={v:.3f}" for a, b, v in corr_pairs])
    if not pair_summary.empty:
        report.extend(
            [
                "",
                "6. Reflected-pair summaries show different x-coverage centers, especially for the paired 750-series files. Use `pairwise_reflection_summary.csv` when checking whether reflected samples should be balanced before training.",
            ]
        )
    report.extend(
        [
            "",
            "## Figure Captions",
            "",
            "**Figure 1. Dataset coverage.** Row counts by geometry and original/reflected orientation.",
            "",
            "**Figure 2. Geometry dimensions and counts.** Median base/height and sample count for each triangle geometry.",
            "",
            "**Figure 3. Normalized spatial coverage.** Sample locations in `(point_x/base, point_y/height)` space.",
            "",
            "**Figure 4. Distance feature distributions.** Normalized AB, AC, and max-distance distributions by geometry.",
            "",
            "**Figure 5. Missingness by raw column.** Missing fraction for the 12 supplied raw features.",
            "",
            "**Figure 6. Spearman correlation heatmap.** Rank correlations among non-constant usable engineered features.",
            "",
            "**Figure 7. PCA feature space.** First two principal components colored by geometry and orientation.",
            "",
            "**Figure 8. Reflected-pair x coverage.** Original/reflected normalized x distributions for geometries with paired files.",
            "",
            "## Primary Files",
            "",
            "- `distance_feature_matrix.csv`",
            "- `geometry_orientation_summary.csv`",
            "- `missingness_summary.csv`",
            "- `spearman_correlation.csv`",
            "- `pca_loadings.csv`",
            "- `pairwise_reflection_summary.csv`",
        ]
    )
    (out_dir / "FEATURE_ANALYSIS_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distance-feature analysis for the 2026-05-24 batch.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/rawdata/2026-05-24"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/feature_analysis/2026-05-24_distance_features"))
    args = parser.parse_args()

    fig_dir = args.out_dir / "figures"
    df = add_features(load_distance_batch(args.raw_dir))
    plot_coverage(df, fig_dir)
    plot_geometry_summary(df, fig_dir)
    plot_spatial_coverage(df, fig_dir)
    plot_distance_distributions(df, fig_dir)
    missing = plot_missingness(df, fig_dir)
    corr = plot_correlation(df, fig_dir)
    loadings = plot_pca(df, fig_dir)
    pair_summary = plot_reflection_pairs(df, fig_dir)
    summary = write_tables(df, missing, corr, loadings, pair_summary, args.out_dir)
    write_report(df, summary, missing, corr, loadings, pair_summary, args.out_dir, args.raw_dir)
    print(f"Rows: {len(df)}")
    print(f"Workbooks: {df['source_file'].nunique()}")
    print(f"Geometries: {sorted(df['geometry'].unique())}")
    print(f"Figures written to: {fig_dir}")
    print(f"Tables written to: {args.out_dir}")


if __name__ == "__main__":
    main()
