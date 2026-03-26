#!/usr/bin/env python
"""
Cluster sampled h5ad files and render grid-filled cluster maps.

Features:
1) Cluster each sampled h5ad independently
2) Cluster merged h5ad per sample
3) Fill grid by recorded sampling_grid_row/sampling_grid_col with majority cluster label
4) Export annotated h5ad and grid pngs
"""

import argparse
from pathlib import Path
import math
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.decomposition import PCA, TruncatedSVD


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster sampled h5ad files and render grid cluster maps")
    parser.add_argument(
        "--input-dir",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
    )
    parser.add_argument("--input-glob", type=str, default="**/*.h5ad")
    parser.add_argument("--embedding-dim", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-n-neighbors", type=int, default=15)
    parser.add_argument(
        "--normalization",
        type=str,
        default="log1p_cpm",
        choices=["none", "log1p_cpm"],
        help="Normalization before clustering",
    )
    parser.add_argument("--grid-cmap", type=str, default="tab20")
    parser.add_argument("--grid-dpi", type=int, default=300)
    parser.add_argument("--umap-point-size", type=float, default=6.0)
    parser.add_argument("--figure-width", type=float, default=8.0)
    parser.add_argument("--figure-min-height", type=float, default=4.5)
    parser.add_argument("--figure-max-height", type=float, default=9.5)
    parser.add_argument("--limits-padding-ratio", type=float, default=0.03)
    parser.add_argument("--limits-min-pad-um", type=float, default=100.0)
    parser.add_argument("--grid-block-um", type=float, default=None, help="Override sampling block size (um)")
    parser.add_argument("--grid-gap-um", type=float, default=None, help="Override sampling gap size (um)")
    parser.add_argument(
        "--grid-aggregate",
        type=str,
        default="sum",
        choices=["mean", "sum"],
        help="How to aggregate cells within same grid before clustering",
    )
    return parser.parse_args()


def ensure_output_dirs(base_dir: Path):
    dirs = {
        "single_h5ad": base_dir / "single_h5ad",
        "single_grid_png": base_dir / "single_grid_png",
        "single_umap_png": base_dir / "single_umap_png",
        "merged_h5ad": base_dir / "merged_h5ad",
        "merged_umap_png": base_dir / "merged_umap_png",
        "reports": base_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def infer_sample_key_from_path(h5ad_path: Path, input_dir: Path):
    rel_path = h5ad_path.relative_to(input_dir)
    rel_parent = rel_path.parent
    if str(rel_parent) == ".":
        return "ungrouped"
    return rel_parent.as_posix().replace("/", "__")


def extract_sampling_params(adata: ad.AnnData, args):
    uns_sampling = adata.uns.get("sampling", {}) if isinstance(adata.uns, dict) else {}
    block_um = args.grid_block_um if args.grid_block_um is not None else uns_sampling.get("block_um", 20.0)
    gap_um = args.grid_gap_um if args.grid_gap_um is not None else uns_sampling.get("gap_um", 20.0)
    return float(block_um), float(gap_um)


def compute_fixed_limits_for_sample(file_paths: list[Path], args):
    x_mins = []
    x_maxs = []
    y_mins = []
    y_maxs = []

    for path in file_paths:
        adata = ad.read_h5ad(path, backed="r")
        if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
            adata.file.close()
            continue

        x_um = adata.obs["x"].to_numpy(dtype=np.float64) * 1000.0
        y_um = adata.obs["y"].to_numpy(dtype=np.float64) * 1000.0
        adata.file.close()

        if x_um.size == 0:
            continue

        x_mins.append(float(x_um.min()))
        x_maxs.append(float(x_um.max()))
        y_mins.append(float(y_um.min()))
        y_maxs.append(float(y_um.max()))

    if len(x_mins) == 0:
        return None

    x_min = min(x_mins)
    x_max = max(x_maxs)
    y_min = min(y_mins)
    y_max = max(y_maxs)

    x_span = x_max - x_min
    y_span = y_max - y_min
    x_pad = max(args.limits_min_pad_um, x_span * args.limits_padding_ratio)
    y_pad = max(args.limits_min_pad_um, y_span * args.limits_padding_ratio)
    return (x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad)


def normalize_matrix(X, method: str):
    if method == "none":
        return X

    if method != "log1p_cpm":
        raise ValueError(f"Unsupported normalization: {method}")

    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        cell_sum = np.asarray(X.sum(axis=1)).ravel()
        scale = 1e4 / np.maximum(cell_sum, 1.0)
        X = sparse.diags(scale) @ X
        X.data = np.log1p(X.data)
        return X

    X = np.asarray(X, dtype=np.float32)
    cell_sum = X.sum(axis=1)
    scale = 1e4 / np.maximum(cell_sum, 1.0)
    X = X * scale[:, None]
    X = np.log1p(X)
    return X


def build_embedding(X, embedding_dim: int, random_state: int):
    n_obs, n_var = X.shape
    if n_obs <= 1:
        return np.zeros((n_obs, 1), dtype=np.float32)

    dim = max(1, min(embedding_dim, n_obs - 1, n_var - 1 if n_var > 1 else 1))

    if sparse.issparse(X):
        model = TruncatedSVD(n_components=dim, random_state=random_state)
        return model.fit_transform(X)

    model = PCA(n_components=dim, random_state=random_state)
    return model.fit_transform(X)


def cluster_matrix(X, embedding_dim: int, random_state: int, normalization: str, leiden_resolution: float, leiden_n_neighbors: int):
    n_obs = X.shape[0]
    if n_obs == 0:
        return np.array([], dtype=np.int64)
    if n_obs == 1:
        return np.array([0], dtype=np.int64)

    X_work = normalize_matrix(X, normalization)
    X_emb = build_embedding(X_work, embedding_dim, random_state)

    neighbors = max(2, min(int(leiden_n_neighbors), n_obs - 1))
    adata_tmp = ad.AnnData(X=np.zeros((n_obs, 1), dtype=np.float32))
    adata_tmp.obsm["X_emb"] = np.asarray(X_emb, dtype=np.float32)
    sc.pp.neighbors(adata_tmp, use_rep="X_emb", n_neighbors=neighbors, random_state=random_state)
    sc.tl.leiden(
        adata_tmp,
        resolution=float(leiden_resolution),
        random_state=random_state,
        key_added="leiden",
        flavor="igraph",
        directed=False,
        n_iterations=2,
    )
    sc.tl.umap(adata_tmp, random_state=random_state)

    labels = adata_tmp.obs["leiden"].astype(str).to_numpy()
    return pd.Categorical(labels).codes.astype(np.int64), adata_tmp.obsm["X_umap"].astype(np.float32)


def aggregate_expression_by_grid(X, key_df: pd.DataFrame, aggregate: str):

    key_cols = key_df.columns.tolist()
    key_tuples = list(key_df.itertuples(index=False, name=None))
    grid_codes, unique_keys = pd.factorize(pd.Series(key_tuples), sort=False)
    n_cells = grid_codes.shape[0]
    n_grids = len(unique_keys)

    selector = sparse.csr_matrix(
        (np.ones(n_cells, dtype=np.float32), (np.arange(n_cells), grid_codes)),
        shape=(n_cells, n_grids),
    )
    X_sum = selector.T @ X
    counts = np.bincount(grid_codes, minlength=n_grids).astype(np.float32)

    if aggregate == "mean":
        scale = sparse.diags(1.0 / np.maximum(counts, 1.0))
        X_grid = scale @ X_sum
    else:
        X_grid = X_sum

    grid_df = pd.DataFrame(list(unique_keys.to_numpy()), columns=key_cols)
    grid_df["cell_count"] = counts.astype(np.int64)
    return X_grid, grid_df, grid_codes


def render_grid_png(
    grid_df: pd.DataFrame,
    output_png: Path,
    title: str,
    cmap_name: str,
    dpi: int,
    grid_block_um: float,
    grid_gap_um: float,
    figure_width: float,
    figure_min_height: float,
    figure_max_height: float,
    fixed_limits: tuple[float, float, float, float] | None = None,
):
    if len(grid_df) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_title(f"{title}\n(no occupied grids)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    period_um = grid_block_um + grid_gap_um
    max_cluster = int(grid_df["cluster"].max()) if len(grid_df) > 0 else 0
    cmap = plt.get_cmap(cmap_name, max(max_cluster + 1, 2))

    row_min = int(grid_df["grid_row"].min())
    row_max = int(grid_df["grid_row"].max())
    col_min = int(grid_df["grid_col"].min())
    col_max = int(grid_df["grid_col"].max())

    if fixed_limits is None:
        x_min = col_min * period_um
        x_max = col_max * period_um + grid_block_um
        y_min = row_min * period_um
        y_max = row_max * period_um + grid_block_um
        pad_x = max(period_um * 0.5, 5.0)
        pad_y = max(period_um * 0.5, 5.0)
        x_plot_min = x_min - pad_x
        x_plot_max = x_max + pad_x
        y_plot_min = y_min - pad_y
        y_plot_max = y_max + pad_y
    else:
        x_plot_min, x_plot_max, y_plot_min, y_plot_max = fixed_limits

    fig_w = figure_width
    fig_h = fig_w * (y_plot_max - y_plot_min) / (x_plot_max - x_plot_min)
    fig_h = max(figure_min_height, min(figure_max_height, fig_h))
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    for _, record in grid_df.iterrows():
        row = int(record["grid_row"])
        col = int(record["grid_col"])
        cluster = int(record["cluster"])
        x0 = col * period_um
        y0 = row * period_um
        rect = Rectangle(
            (x0, y0),
            grid_block_um,
            grid_block_um,
            facecolor=cmap(cluster),
            edgecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_max, y_plot_min)
    ax.set_aspect("equal")

    ax.set_title(title)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    bounds = np.arange(-0.5, max_cluster + 1.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        boundaries=bounds,
        ticks=np.arange(0, max_cluster + 1, 1, dtype=int),
        spacing="proportional",
    )
    cbar.set_label(f"Cluster (n={max_cluster + 1})")
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cluster_on_grids(adata: ad.AnnData, args, cluster_col: str, group_cols: list[str]):
    required_cols = list(dict.fromkeys(group_cols + ["sampling_grid_row", "sampling_grid_col"]))
    for col in required_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"missing required obs column '{col}'")

    key_df = adata.obs[group_cols].copy()
    if "sampling_grid_row" in key_df.columns:
        key_df["sampling_grid_row"] = key_df["sampling_grid_row"].astype(int)
    if "sampling_grid_col" in key_df.columns:
        key_df["sampling_grid_col"] = key_df["sampling_grid_col"].astype(int)

    X_grid, grid_df, grid_codes = aggregate_expression_by_grid(adata.X, key_df, args.grid_aggregate)

    grid_labels, umap_coords = cluster_matrix(
        X_grid,
        embedding_dim=args.embedding_dim,
        random_state=args.random_state,
        normalization=args.normalization,
        leiden_resolution=args.leiden_resolution,
        leiden_n_neighbors=args.leiden_n_neighbors,
    )

    grid_df["cluster"] = grid_labels
    cell_labels = grid_labels[grid_codes]
    adata.obs[cluster_col] = cell_labels.astype(np.int64)
    adata.obs["umap_coord1"] = umap_coords[grid_codes, 0]
    adata.obs["umap_coord2"] = umap_coords[grid_codes, 1]
    
    return adata, grid_df


def plot_umap(adata: ad.AnnData, output_png: Path,
              title: str, dpi: int, point_size: float, cmap_name: str):
    if "parcellation_substructure" not in adata.obs.columns:
        print(f"Warning: Required columns not found in adata.obs; skipping UMAP plot")
        return
    if "cluster_single" in adata.obs.columns:
        labels = adata.obs["cluster_single"].fillna("Unknown").astype(str)
    elif "cluster_merged" in adata.obs.columns:
        labels = adata.obs["cluster_merged"].fillna("Unknown").astype(str)
    else:
        print(f"Warning: No cluster column found in adata.obs; skipping UMAP plot")
        return
    categories = list(pd.Categorical(labels).categories)
    cat_codes = pd.Categorical(labels, categories=categories).codes
    cmap = plt.get_cmap(cmap_name, max(2, len(categories)))

    umap_xy = adata.obs[["umap_coord1", "umap_coord2"]].to_numpy(dtype=np.float32)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        c=cat_codes,
        cmap=cmap,
        s=point_size,
        linewidths=0,
        alpha=0.9,
    )
    ax[0].set_xlabel("UMAP1")
    ax[0].set_ylabel("UMAP2")

    handles = [
        plt.Line2D([0], [0], marker='o', linestyle='', color=cmap(i), label=cat, markersize=5)
        for i, cat in enumerate(categories)
    ]
    ax[0].legend(handles=handles, title="Cluster", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)

    ax[1].scatter(
        umap_xy[:, 0],
        umap_xy[:, 1],
        c=adata.obs["parcellation_substructure"].astype("category").cat.codes,
        cmap="tab20",
        s=point_size,
        linewidths=0,
        alpha=0.9,
    )
    ax[1].set_xlabel("UMAP1")
    ax[1].set_ylabel("UMAP2")

    handles_sub = [
        plt.Line2D([0], [0], marker='o', linestyle='', 
                   color=plt.get_cmap("tab20", len(adata.obs["parcellation_substructure"].unique()))(i), 
                   label=cat, markersize=5)
        for i, cat in enumerate(adata.obs["parcellation_substructure"].astype("category").cat.categories)
    ]
    if len(adata.obs["parcellation_substructure"].unique()) < 16:       
        ax[1].legend(handles=handles_sub, title="Substructure", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)
    else:
        ax[1].legend(handles=handles_sub, ncol=math.ceil(len(adata.obs["parcellation_substructure"].unique()) / 16), 
                     title="Substructure", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_single_file(h5ad_path: Path, dirs: dict, args, sample_fixed_limits: dict):
    adata = ad.read_h5ad(h5ad_path)
    grid_block_um, grid_gap_um = extract_sampling_params(adata, args)
    sample_key = infer_sample_key_from_path(h5ad_path, args.input_dir)

    group_cols = ["sampling_grid_row", "sampling_grid_col"]
    adata, grid_df = cluster_on_grids(adata, args, cluster_col="cluster_single", group_cols=group_cols)
    adata.uns["grid_clustering"] = {
        "level": "grid",
        "aggregate": "sum",
        "method": "leiden",
        "leiden_resolution": args.leiden_resolution,
        "leiden_n_neighbors": args.leiden_n_neighbors,
        "grid_block_um": grid_block_um,
        "grid_gap_um": grid_gap_um,
    }

    out_h5ad = dirs["single_h5ad"] / f"{h5ad_path.stem}_clustered.h5ad"
    adata.write_h5ad(out_h5ad, compression="gzip")

    out_png = dirs["single_grid_png"] / f"{h5ad_path.stem}_grid_cluster.png"
    grid_df_for_plot = grid_df.rename(
        columns={"sampling_grid_row": "grid_row", "sampling_grid_col": "grid_col"}
    )
    render_grid_png(
        grid_df_for_plot,
        output_png=out_png,
        title=f"Single-file grid cluster {h5ad_path.stem}",
        cmap_name=args.grid_cmap,
        dpi=args.grid_dpi,
        grid_block_um=grid_block_um,
        grid_gap_um=grid_gap_um,
        figure_width=args.figure_width,
        figure_min_height=args.figure_min_height,
        figure_max_height=args.figure_max_height,
        fixed_limits=sample_fixed_limits.get(sample_key),
    )

    out_umap_png = dirs["single_umap_png"] / f"{h5ad_path.stem}_umap_cluster.png"
    plot_umap(
        adata,
        output_png=out_umap_png,
        title=f"Single-file UMAP {h5ad_path.stem}",
        dpi=args.grid_dpi,
        point_size=args.umap_point_size,
        cmap_name=args.grid_cmap,
    )

    return {
        "file": h5ad_path.name,
        "sample_key": sample_key,
        "cells": int(adata.n_obs),
        "grids": int(len(grid_df)),
        "n_clusters_observed": int(grid_df["cluster"].nunique()),
        "single_h5ad": str(out_h5ad),
        "single_grid_png": str(out_png),
        "single_umap_png": str(out_umap_png),
    }


def process_merged_sample(sample_key: str, file_paths: list[Path], dirs: dict, args, sample_fixed_limits: dict):
    adatas = []
    sampling_params = []
    for path in file_paths:
        adata = ad.read_h5ad(path)
        adata.obs = adata.obs.copy()
        sampling_params.append(extract_sampling_params(adata, args))
        adatas.append(adata)

    merged = ad.concat(adatas, join="inner", merge="same", index_unique="-")
    grid_block_um, grid_gap_um = sampling_params[0]
    inconsistent = [p for p in sampling_params if not np.isclose(p[0], grid_block_um) or not np.isclose(p[1], grid_gap_um)]
    if len(inconsistent) > 0:
        print(f"Warning: {sample_key} has mixed sampling params across files; using first one ({grid_block_um}, {grid_gap_um})")

    group_cols = ["brain_section_label", "sampling_grid_row", "sampling_grid_col"]
    merged, grid_df = cluster_on_grids(merged, args, cluster_col="cluster_merged", group_cols=group_cols)
    merged.uns["grid_clustering"] = {
        "level": "grid",
        "aggregate": "sum",
        "method": "leiden",
        "leiden_resolution": args.leiden_resolution,
        "leiden_n_neighbors": args.leiden_n_neighbors,
        "grid_block_um": grid_block_um,
        "grid_gap_um": grid_gap_um,
    }

    out_h5ad = dirs["merged_h5ad"] / f"{sample_key}_merged_clustered.h5ad"
    merged.write_h5ad(out_h5ad, compression="gzip")

    out_png = dirs["merged_umap_png"] / f"{sample_key}_merged_umap.png"
    plot_umap(
        merged,
        output_png=out_png,
        title=f"Merged UMAP {sample_key}",
        dpi=args.grid_dpi,
        point_size=args.umap_point_size,
        cmap_name=args.grid_cmap,
    )

    return {
        "sample_key": sample_key,
        "files": len(file_paths),
        "cells": int(merged.n_obs),
        "n_clusters_observed": int(grid_df["cluster"].nunique()),
        "merged_h5ad": str(out_h5ad),
        "merged_umap_png": str(out_png),
    }


def main():
    args = parse_args()

    input_files = sorted([p for p in args.input_dir.glob(args.input_glob) if p.is_file()]) if args.input_dir.is_dir() else []
    if len(input_files) == 0:
        raise FileNotFoundError(f"No files found under {args.input_dir} with pattern {args.input_glob}")

    dirs = ensure_output_dirs(args.output_dir)

    grouped = {}
    for file_path in input_files:
        sample_key = infer_sample_key_from_path(file_path, args.input_dir)
        grouped.setdefault(sample_key, []).append(file_path)

    sample_fixed_limits = {}
    for sample_key, file_paths in grouped.items():
        sample_fixed_limits[sample_key] = compute_fixed_limits_for_sample(file_paths, args)
        if sample_fixed_limits[sample_key] is not None:
            xmin, xmax, ymin, ymax = sample_fixed_limits[sample_key]
            print(f"{sample_key}: fixed limits X=({xmin:.0f},{xmax:.0f}) µm Y=({ymin:.0f},{ymax:.0f}) µm")

    single_records = []
    for sample_key, file_paths in grouped.items():
        for idx, file_path in enumerate(file_paths, start=1):
            print(f"[{sample_key}] [{idx}/{len(file_paths)}] Single clustering: {file_path.name}")
            record = process_single_file(file_path, dirs, args, sample_fixed_limits)
            single_records.append(record)

    merged_records = []
    sample_keys = sorted(grouped)
    for idx, sample_key in enumerate(sample_keys, start=1):
        print(f"[{idx}/{len(sample_keys)}] Merged clustering: {sample_key} ({len(grouped[sample_key])} files)")
        record = process_merged_sample(sample_key, grouped[sample_key], dirs, args, sample_fixed_limits)
        merged_records.append(record)

    single_df = pd.DataFrame(single_records)
    merged_df = pd.DataFrame(merged_records)

    single_report = dirs["reports"] / "single_clustering_report.csv"
    merged_report = dirs["reports"] / "merged_clustering_report.csv"
    single_df.to_csv(single_report, index=False)
    merged_df.to_csv(merged_report, index=False)

    print("\nDone.")
    print(f"Single report: {single_report}")
    print(f"Merged report: {merged_report}")


if __name__ == "__main__":
    main()
