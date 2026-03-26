#!/usr/bin/env python
"""
Integrate MERFISH datasets into scRNA-seq reference.

Workflow:
1) Load reference expression matrices (e.g. WMB-10Xv2, WMB-10Xv3, WMB-10XMulti)
2) Add reference cluster labels from WMB-10X cell metadata and merge references
3) Load and merge MERFISH matrices from output/cluster_results/merged_h5ad
4) Integrate reference and MERFISH with SeuratIntegration and transfer cluster labels
5) Plot MERFISH cells using stored x/y coordinates colored by transferred cluster labels
"""

import anndata as ad
import os
import math
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
from ALLCools.integration.seurat_class import SeuratIntegration


def parse_args():
    parser = argparse.ArgumentParser(description='Get expression matrix for a dataset.')
    parser.add_argument('--download-base', type=Path, help='Base directory for downloading data.')
    parser.add_argument('--reference-datasets', '--reference-dataset', dest='reference_datasets', type=str, help='Reference datasets, comma-separated.')
    parser.add_argument('--expression-matrix-kind', type=str, default='raw', help='Kind of expression matrix to get (e.g. raw, log2).')
    parser.add_argument('--cell-metadata-file', type=str, default='WMB-10X', help='File name for cell metadata.')
    parser.add_argument('--division', type=str, help='Use ALL to disable division filter')
    parser.add_argument('--cluster-col', type=str, default='cluster_alias', help='Reference cluster column to transfer.')
    parser.add_argument('--integration-pcs', type=int, default=50, help='Number of PCs used for integration and transfer.')
    parser.add_argument('--integration-features', type=int, default=2000, help='Number of features for CCA anchor finding.')
    parser.add_argument('--max-reference-cells', type=int, default=30000, help='Max number of reference cells used for integration (stratified by cluster).')
    parser.add_argument('--enable-reference-downsampling', dest='enable_reference_downsampling', action='store_true', default=True,
                        help='Enable stratified reference downsampling before integration (default: enabled).')
    parser.add_argument('--disable-reference-downsampling', dest='enable_reference_downsampling', action='store_false',
                        help='Disable reference downsampling and use all reference cells.')
    parser.add_argument('--label-transfer-k-weight', type=int, default=50, help='k_weight for Seurat label transfer.')
    parser.add_argument('--plot-dpi', type=int, default=300, help='DPI for coordinate plots.')
    parser.add_argument('--grid-cmap', type=str, default='tab20', help='Colormap for grid/scatter plots.')
    parser.add_argument('--figure-width', type=float, default=8.0, help='Figure width in inches.')
    parser.add_argument('--figure-min-height', type=float, default=4.5, help='Minimum figure height in inches.')
    parser.add_argument('--figure-max-height', type=float, default=9.5, help='Maximum figure height in inches.')
    args = parser.parse_args()
    return args


def get_expression_matrix(dataset, expression_matrix_kind, abc_cache, division):
    """Get expression matrix for a dataset."""
    try:
        print(f"Trying to get file with division: {dataset}-{division}")
        file = abc_cache.get_file_path(directory=dataset, file_name=f'{dataset}-{division}/{expression_matrix_kind}')
    except:
        print(f"Failed to get file with division: {dataset}-{division}, trying without division: {dataset}")
        file = abc_cache.get_file_path(directory=dataset, file_name=f'{dataset}/{expression_matrix_kind}')
    adata = ad.read_h5ad(file)
    return adata


def get_cell_metadata(cell_metadata_file, abc_cache):
    """Get cell metadata."""
    cell_metadata = abc_cache.get_metadata_dataframe(directory=cell_metadata_file, file_name='cell_metadata', dtype={'cell_label': str})
    cell_metadata.set_index('cell_label', inplace=True)
    return cell_metadata


def add_cluster(adata, cluster_dataframe, output_file, cluster_col):
    """Add cluster information to an AnnData object."""
    if cluster_col not in cluster_dataframe.columns:
        raise KeyError(f"'{cluster_col}' not found in metadata columns: {list(cluster_dataframe.columns)}")
    cluster_map = {c[0]: c[1] for c in cluster_dataframe[["cell_label", cluster_col]].values}
    adata.obs[cluster_col] = adata.obs_names.map(cluster_map)
    adata.write_h5ad(output_file, compression='gzip')
    return adata


def combined_adata(adata_list, output_path, label=None):
    """Combine multiple AnnData objects into one."""
    combined_adata = ad.concat(adata_list, join='inner', merge='same', index_unique='-')
    if label is not None:
        combined_adata.uns['source_label'] = label
    combined_adata.write_h5ad(output_path, compression='gzip')
    return combined_adata


def read_and_merge_merfish(merged_h5ad_dir: Path, output_h5ad: Path):
    files = sorted(merged_h5ad_dir.glob('*.h5ad'))
    if len(files) == 0:
        raise FileNotFoundError(f'No MERFISH merged h5ad found in {merged_h5ad_dir}')

    adata_list = []
    for f in files:
        adata = ad.read_h5ad(f)
        adata.obs = adata.obs.copy()
        adata_list.append(adata)
        print(f"Loaded MERFISH: {f.name}, shape={adata.shape}")

    merged = ad.concat(adata_list, join='inner', merge='same', index_unique='-')
    merged.write_h5ad(output_h5ad, compression='gzip')
    return merged


def preprocess_for_integration(adata, n_pcs):
    if not isinstance(adata.X, np.ndarray):
        adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    pcs = min(n_pcs, max(2, adata.n_obs - 1), max(2, adata.n_vars - 1))
    sc.tl.pca(adata, svd_solver='arpack', n_comps=pcs, zero_center=False)
    return pcs


def downsample_reference_by_cluster(ref_adata, cluster_col, max_cells, random_state=0):
    if max_cells is None or ref_adata.n_obs <= max_cells:
        return ref_adata

    rng = np.random.default_rng(random_state)
    obs = ref_adata.obs[[cluster_col]].copy()
    group_sizes = obs.groupby(cluster_col).size().sort_values(ascending=False)
    frac = max_cells / float(ref_adata.n_obs)

    keep_index = []
    for cluster_value, size in group_sizes.items():
        n_keep = max(1, int(np.floor(size * frac)))
        members = obs.index[obs[cluster_col] == cluster_value].to_numpy()
        if n_keep >= members.shape[0]:
            keep_index.extend(members.tolist())
        else:
            selected = rng.choice(members, size=n_keep, replace=False)
            keep_index.extend(selected.tolist())

    if len(keep_index) > max_cells:
        keep_index = rng.choice(np.array(keep_index), size=max_cells, replace=False).tolist()

    return ref_adata[keep_index].copy()


def run_seurat_integration_and_transfer(ref_adata, merfish_adata, cluster_col, n_pcs, n_features, label_transfer_k_weight):
    n_pcs_ref = preprocess_for_integration(ref_adata, n_pcs)
    n_pcs_qry = preprocess_for_integration(merfish_adata, n_pcs)
    n_pcs_use = int(min(n_pcs_ref, n_pcs_qry))

    integrator = SeuratIntegration()
    adata_list = [ref_adata, merfish_adata]
    integrator.find_anchor(
        adata_list,
        k_local=None,
        key_local='X_pca',
        k_anchor=5,
        key_anchor='X',
        dim_red='cca',
        max_cc_cells=100000,
        k_score=30,
        k_filter=None,
        scale1=False,
        scale2=False,
        n_components=n_pcs_use,
        n_features=n_features,
        alignments=[[[0], [1]]],
    )

    transfer_results = integrator.label_transfer(
        ref=[0],
        qry=[1],
        categorical_key=[cluster_col],
        key_dist='X_pca',
        k_weight=label_transfer_k_weight,
        npc=n_pcs_use,
    )

    transfer_df = transfer_results[cluster_col]
    predicted = transfer_df.idxmax(axis=1)
    confidence = transfer_df.max(axis=1)

    merfish_out = merfish_adata.copy()
    merfish_out.obs[f'{cluster_col}_transfer'] = predicted.reindex(merfish_out.obs_names).astype(str).to_numpy()
    merfish_out.obs[f'{cluster_col}_transfer_score'] = confidence.reindex(merfish_out.obs_names).astype(np.float32).to_numpy()

    integrated_obs_ref = ref_adata.obs[[cluster_col]].copy()
    integrated_obs_ref['dataset'] = 'reference'
    integrated_obs_ref[f'{cluster_col}_transfer'] = integrated_obs_ref[cluster_col].astype(str)

    integrated_obs_qry = merfish_out.obs.copy()
    integrated_obs_qry['dataset'] = 'merfish'
    integrated_obs_qry[cluster_col] = np.nan

    integrated_obs = pd.concat([integrated_obs_ref, integrated_obs_qry], axis=0)
    integrated_meta_adata = ad.AnnData(obs=integrated_obs)
    return merfish_out, integrated_meta_adata


def sanitize_filename(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in str(text)).strip('_') or 'unknown'


def make_adaptive_cluster_ticks(max_cluster: int, max_ticks: int = 18):
    if max_cluster <= 0:
        return np.array([0], dtype=int)
    step = max(1, math.ceil((max_cluster + 1) / max_ticks))
    ticks = np.arange(0, max_cluster + 1, step, dtype=int)
    if ticks[-1] != max_cluster:
        ticks = np.append(ticks, max_cluster)
    return ticks


def plot_grid_with_colorbar(
    grid_df: pd.DataFrame,
    output_png: Path,
    title: str,
    cmap_name: str,
    dpi: int,
    grid_block_um: float,
    grid_gap_um: float,
    figure_width: float = 8.0,
    figure_min_height: float = 4.5,
    figure_max_height: float = 9.5,
    fixed_limits: tuple[float, float, float, float] | None = None,
):
    """Plot grid-based cluster visualization with colorbar, similar to render_grid_png."""
    if len(grid_df) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.set_title(f"{title}\n(no occupied grids)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return

    period_um = grid_block_um + grid_gap_um
    clusters_present = sorted(pd.Series(grid_df["cluster"]).astype(int).unique().tolist())
    cluster_to_local = {cluster_id: idx for idx, cluster_id in enumerate(clusters_present)}
    n_clusters_present = len(clusters_present)
    cmap = plt.get_cmap(cmap_name, max(n_clusters_present, 2))

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
        cluster_local = cluster_to_local[cluster]
        x0 = col * period_um
        y0 = row * period_um
        rect = Rectangle(
            (x0, y0),
            grid_block_um,
            grid_block_um,
            facecolor=cmap(cluster_local),
            edgecolor="none",
        )
        ax.add_patch(rect)

    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_max, y_plot_min)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")

    max_local_cluster = max(0, n_clusters_present - 1)
    bounds = np.arange(-0.5, max_local_cluster + 1.5, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ticks = make_adaptive_cluster_ticks(max_local_cluster)
    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        boundaries=bounds,
        ticks=cbar_ticks,
        spacing="proportional",
    )
    cbar.set_label(f"Cluster (present n={n_clusters_present})", fontsize=10)
    cbar.ax.tick_params(labelsize=7)
    if len(cbar_ticks) > 0:
        cbar.ax.set_yticklabels([str(clusters_present[int(t)]) for t in cbar_ticks])

    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_merfish_by_coordinates(
    merfish_merged_adata,
    cluster_col,
    output_dir: Path,
    dpi: int = 300,
    figure_width: float = 8.0,
    figure_min_height: float = 4.5,
    figure_max_height: float = 9.5,
    grid_cmap: str = 'tab20',
):
    """
    Plot MERFISH data by coordinates.
    Prefers grid-based visualization if sampling_grid_row/col exist, falls back to scatter plot.
    """
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    transfer_col = f'{cluster_col}_transfer'
    if transfer_col in merfish_merged_adata.obs.columns:
        color_col = transfer_col
    elif cluster_col in merfish_merged_adata.obs.columns:
        color_col = cluster_col
    else:
        raise KeyError(f"Neither '{transfer_col}' nor '{cluster_col}' found in MERFISH obs.")

    if 'x' not in merfish_merged_adata.obs.columns or 'y' not in merfish_merged_adata.obs.columns:
        raise KeyError("MERFISH obs must contain 'x' and 'y' columns for coordinate plotting.")

    obs = merfish_merged_adata.obs.copy()
    obs = obs.dropna(subset=['x', 'y'])
    obs[color_col] = obs[color_col].astype(str)

    has_grid = 'sampling_grid_row' in obs.columns and 'sampling_grid_col' in obs.columns
    
    if has_grid:
        obs = obs.dropna(subset=['sampling_grid_row', 'sampling_grid_col'])
        obs['cluster_int'] = pd.Categorical(obs[color_col]).codes.astype(np.int64)
        
        grid_df_all = obs.groupby(['sampling_grid_row', 'sampling_grid_col']).agg({
            'cluster_int': lambda x: x.mode()[0] if len(x.mode()) > 0 else int(x.iloc[0])
        }).reset_index()
        grid_df_all.columns = ['grid_row', 'grid_col', 'cluster']
        
        grid_block_um = 20.0
        grid_gap_um = 20.0
        if 'sampling' in merfish_merged_adata.uns and isinstance(merfish_merged_adata.uns['sampling'], dict):
            grid_block_um = merfish_merged_adata.uns['sampling'].get('block_um', 20.0)
            grid_gap_um = merfish_merged_adata.uns['sampling'].get('gap_um', 20.0)

        period_um = float(grid_block_um) + float(grid_gap_um)
        global_row_min = int(obs['sampling_grid_row'].astype(int).min())
        global_row_max = int(obs['sampling_grid_row'].astype(int).max())
        global_col_min = int(obs['sampling_grid_col'].astype(int).min())
        global_col_max = int(obs['sampling_grid_col'].astype(int).max())
        global_x_min = global_col_min * period_um
        global_x_max = global_col_max * period_um + float(grid_block_um)
        global_y_min = global_row_min * period_um
        global_y_max = global_row_max * period_um + float(grid_block_um)
        global_pad_x = max(period_um * 0.5, 5.0)
        global_pad_y = max(period_um * 0.5, 5.0)
        global_limits = (
            global_x_min - global_pad_x,
            global_x_max + global_pad_x,
            global_y_min - global_pad_y,
            global_y_max + global_pad_y,
        )

        if 'brain_section_label' in obs.columns:
            section_groups = obs.dropna(subset=['brain_section_label']).groupby('brain_section_label', observed=True)
            for section_label, section_df in section_groups:
                section_grid_df = section_df.groupby(['sampling_grid_row', 'sampling_grid_col'], observed=True).agg({
                    'cluster_int': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else int(x.iloc[0])
                }).reset_index()
                section_grid_df.columns = ['grid_row', 'grid_col', 'cluster']
                section_name = sanitize_filename(section_label)
                plot_grid_with_colorbar(
                    section_grid_df,
                    output_png=plot_dir / f'brain_section_{section_name}_grid_cluster.png',
                    title=f'Brain section {section_label} grid clusters by {color_col}',
                    cmap_name=grid_cmap,
                    dpi=dpi,
                    grid_block_um=float(grid_block_um),
                    grid_gap_um=float(grid_gap_um),
                    figure_width=figure_width,
                    figure_min_height=figure_min_height,
                    figure_max_height=figure_max_height,
                    fixed_limits=global_limits,
                )


def main():
    args = parse_args()
    if not args.reference_datasets:
        raise ValueError('Please provide --reference-datasets (or --reference-dataset), comma-separated.')

    output_path = args.download_base.parent / f'output_{args.expression_matrix_kind}'
    embedding_output_path = output_path / 'embedding'
    embedding_output_path.mkdir(parents=True, exist_ok=True)
    cluster_results_dir = output_path / 'cluster_results' / 'merged_h5ad'
    reference_datasets = [x.strip() for x in args.reference_datasets.split(',') if x.strip()]
    print('Loading AbcProjectCache...')
    abc_cache = AbcProjectCache.from_cache_dir(args.download_base)

    if os.path.exists(embedding_output_path / 'combined_reference.h5ad'):
        print("Combined reference data already exists, loading from file...")
        ref_adata = ad.read_h5ad(embedding_output_path / 'combined_reference.h5ad')
    else:
        print("Loading cell metadata...")
        cluster_dataframe = get_cell_metadata(args.cell_metadata_file, abc_cache)
        cluster_dataframe = cluster_dataframe.reset_index()
        if args.cluster_col not in cluster_dataframe.columns:
            raise KeyError(f"Cluster column '{args.cluster_col}' not in cell metadata. Available: {list(cluster_dataframe.columns)}")

        adata_list = []
        for rd in reference_datasets:
            adata = get_expression_matrix(rd, args.expression_matrix_kind, abc_cache, args.division)
            print(f"Loaded expression matrix for dataset {rd} with shape {adata.shape}")
            adata = add_cluster(adata, cluster_dataframe, embedding_output_path / f'{rd}_with_cluster.h5ad', args.cluster_col)
            adata.obs['reference_dataset'] = rd
            print(f"Added cluster information for dataset {rd}")
            adata_list.append(adata)
        print(f"Loaded and processed {len(adata_list)} reference datasets.")
        ref_adata = combined_adata(adata_list, embedding_output_path / 'combined_reference.h5ad', label='reference')

    merfish_adata = read_and_merge_merfish(cluster_results_dir, embedding_output_path / 'combined_merfish.h5ad')
    print(f"Combined MERFISH datasets into single AnnData object with shape {merfish_adata.shape}")

    common_genes = np.array(ref_adata.var_names.intersection(merfish_adata.var_names))
    if common_genes.size == 0:
        raise ValueError('No common genes between reference and MERFISH matrices.')
    ref_adata = ref_adata[:, common_genes].copy()
    merfish_adata = merfish_adata[:, common_genes].copy()

    ref_adata = ref_adata[~ref_adata.obs[args.cluster_col].isna()].copy()
    print(f'Reference cells with non-null {args.cluster_col}: {ref_adata.n_obs}')

    if args.enable_reference_downsampling:
        ref_adata = downsample_reference_by_cluster(
            ref_adata,
            cluster_col=args.cluster_col,
            max_cells=args.max_reference_cells,
            random_state=0,
        )
        print(f'Reference cells after stratified downsampling: {ref_adata.n_obs}')
    else:
        print('Reference downsampling is disabled.')

    merfish_integrated, integrated_meta = run_seurat_integration_and_transfer(
        ref_adata=ref_adata,
        merfish_adata=merfish_adata,
        cluster_col=args.cluster_col,
        n_pcs=args.integration_pcs,
        n_features=args.integration_features,
        label_transfer_k_weight=args.label_transfer_k_weight,
    )
    integrated_path = embedding_output_path / 'reference_merfish_integrated_obs.h5ad'
    integrated_meta.write_h5ad(integrated_path, compression='gzip')
    print(f'Saved integrated metadata: {integrated_path}')

    merfish_integrated_path = embedding_output_path / 'merfish_with_transferred_cluster.h5ad'
    merfish_integrated.write_h5ad(merfish_integrated_path, compression='gzip')
    print(f'Saved MERFISH with transferred labels: {merfish_integrated_path}')

    plot_merfish_by_coordinates(
        merfish_integrated,
        cluster_col=args.cluster_col,
        output_dir=embedding_output_path,
        dpi=args.plot_dpi,
        figure_width=args.figure_width,
        figure_min_height=args.figure_min_height,
        figure_max_height=args.figure_max_height,
        grid_cmap=args.grid_cmap,
    )
    print(f"Saved coordinate plots under: {embedding_output_path / 'plots'}")

if __name__ == "__main__":
    main()