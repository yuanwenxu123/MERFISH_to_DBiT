#!/usr/bin/env python
"""
CCF Registration to Image Script
Loads MERFISH data and CCF registration results, applies grid sampling,
and saves visualizations plus sampled expression matrices.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import anndata as ad
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import pandas as pd
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

CCF_PIXEL_SIZE_UM = 10.0
MM_TO_UM = 1000.0
MM_TO_PX = MM_TO_UM / CCF_PIXEL_SIZE_UM
SECTION_XY_MM_TO_UM = 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description='CCF registration visualization and grid sampling export')
    parser.add_argument('--download-base', type=Path)
    parser.add_argument('--datasets', type=str)
    parser.add_argument('--division', type=str, help='Use ALL to disable division filter')
    parser.add_argument('--grid-block-um', type=float, default=20.0)
    parser.add_argument('--grid-gap-um', type=float, default=20.0)
    parser.add_argument('--expression-matrix-kind', type=str, default='raw', choices=['raw', 'log2'])
    parser.add_argument('--export-sampled-h5ad', action='store_true', default=True)
    parser.add_argument('--no-export-sampled-h5ad', dest='export_sampled_h5ad', action='store_false')
    parser.add_argument('--export-sampling-mask', action='store_true', default=True)
    parser.add_argument('--no-export-sampling-mask', dest='export_sampling_mask', action='store_false')
    parser.add_argument('--show-sampling-grid', action='store_true', default=True)
    parser.add_argument('--hide-sampling-grid', dest='show_sampling_grid', action='store_false')
    parser.add_argument('--point-size', type=float, default=0.5)
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--figure-width', type=float, default=8.0)
    parser.add_argument('--figure-min-height', type=float, default=4.5)
    parser.add_argument('--figure-max-height', type=float, default=9.5)
    parser.add_argument('--grid-linewidth', type=float, default=0.25)
    parser.add_argument('--grid-alpha', type=float, default=0.7)
    parser.add_argument('--mask-preview-max-points', type=int, default=500000)
    return parser.parse_args()

# ============================================================================
# Helper function to plot and save sections
# ============================================================================


def clean_color_values(color_values, default_color='#B0B0B0'):
    """Ensure colors are valid hex strings for matplotlib."""
    cleaned = []
    for color in color_values:
        if not isinstance(color, str):
            cleaned.append(default_color)
            continue

        color = color.strip()
        if color.startswith('#') and len(color) in (7, 9):
            cleaned.append(color)
        elif len(color) == 6 or len(color) == 8:
            cleaned.append(f'#{color}')
        else:
            cleaned.append(default_color)
    return cleaned


def build_substructure_color_map(substructure_values):
    """Build a deterministic high-contrast color map for substructures."""
    unique_substructures = sorted(
        {
            value.strip()
            for value in substructure_values
            if isinstance(value, str) and value.strip()
        }
    )

    if len(unique_substructures) == 0:
        return {}

    def _collect_candidate_colors():
        candidates = []

        for cmap_name in ['tab20', 'tab20b', 'tab20c', 'Set3', 'Paired', 'Accent', 'Dark2']:
            cmap = plt.get_cmap(cmap_name)
            n = getattr(cmap, 'N', 256)
            for i in range(n):
                rgba = cmap(i / max(1, n - 1))
                candidates.append(np.asarray(rgba[:3], dtype=np.float64))

        for h in np.linspace(0.0, 1.0, 120, endpoint=False):
            rgb = mcolors.hsv_to_rgb((h, 0.75, 0.92))
            candidates.append(np.asarray(rgb, dtype=np.float64))

        uniq = []
        seen = set()
        for rgb in candidates:
            key = tuple(np.round(rgb, 4))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(rgb)
        return np.vstack(uniq)

    def _select_max_separation_colors(candidate_rgb, n_colors):
        if n_colors <= 0:
            return np.empty((0, 3), dtype=np.float64)

        if n_colors >= len(candidate_rgb):
            return candidate_rgb

        luminance = candidate_rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
        first_idx = int(np.argmin(np.abs(luminance - 0.5)))
        selected = [first_idx]

        min_dist_sq = np.sum((candidate_rgb - candidate_rgb[first_idx]) ** 2, axis=1)
        min_dist_sq[first_idx] = -1.0

        for _ in range(1, n_colors):
            next_idx = int(np.argmax(min_dist_sq))
            selected.append(next_idx)
            dist_sq = np.sum((candidate_rgb - candidate_rgb[next_idx]) ** 2, axis=1)
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
            min_dist_sq[selected] = -1.0

        return candidate_rgb[selected]

    candidate_rgb = _collect_candidate_colors()
    picked_rgb = _select_max_separation_colors(candidate_rgb, len(unique_substructures))

    color_map = {}
    for substructure, rgb in zip(unique_substructures, picked_rgb):
        color_map[substructure] = mcolors.to_hex(rgb)

    return color_map


def compute_keep_mask_from_xy(x_mm, y_mm, grid_block_um, grid_gap_um):
    period_um = grid_block_um + grid_gap_um
    x_um = x_mm.to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
    y_um = y_mm.to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
    x_phase = np.mod(x_um, period_um)
    y_phase = np.mod(y_um, period_um)
    return (x_phase < grid_block_um) & (y_phase < grid_block_um)


def compute_grid_indices_from_xy(x_mm, y_mm, grid_block_um, grid_gap_um):
    """Compute grid row/col index from coordinates using the same sampling period."""
    period_um = grid_block_um + grid_gap_um
    x_um = x_mm.to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
    y_um = y_mm.to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
    grid_col = np.floor(x_um / period_um).astype(np.int64)
    grid_row = np.floor(y_um / period_um).astype(np.int64)
    return grid_row, grid_col


def save_sampling_mask(section_cells, keep_mask, output_path, grid_block_um, grid_gap_um):
    """Save per-dataset sampling mask aligned to section_cells order."""
    period_um = grid_block_um + grid_gap_um
    np.savez_compressed(
        output_path,
        cell_label=section_cells.index.astype(str).to_numpy(),
        keep_mask=keep_mask.astype(np.bool_),
        grid_block_um=np.float64(grid_block_um),
        grid_gap_um=np.float64(grid_gap_um),
        grid_period_um=np.float64(period_um),
    )


def save_sampling_mask_png(section_cells, keep_mask, fixed_limits, output_path, args):
    x_vals_um = section_cells['x'].to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
    y_vals_um = section_cells['y'].to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
    x_plot_min, x_plot_max, y_plot_min, y_plot_max = fixed_limits

    total_points = len(section_cells)
    if total_points > args.mask_preview_max_points:
        rng = np.random.default_rng(0)
        pick_idx = rng.choice(total_points, size=args.mask_preview_max_points, replace=False)
        x_vals_um = x_vals_um[pick_idx]
        y_vals_um = y_vals_um[pick_idx]
        keep_mask = keep_mask[pick_idx]

    fig, ax = plt.subplots(1, 1, figsize=(args.figure_width, max(args.figure_min_height, args.figure_width * (y_plot_max - y_plot_min) / (x_plot_max - x_plot_min))))
    ax.scatter(x_vals_um[~keep_mask], y_vals_um[~keep_mask], s=args.point_size, c='#7a7a7a', marker='.', alpha=0.25)
    ax.scatter(x_vals_um[keep_mask], y_vals_um[keep_mask], s=args.point_size, c='#ff3030', marker='.', alpha=0.6)
    ax.set_aspect('equal')
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_max, y_plot_min)
    ax.set_xlabel('X (µm)', fontsize=10)
    ax.set_ylabel('Y (µm)', fontsize=10)
    ax.set_title('Sampling Mask Preview', fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()


def get_expression_matrix(dataset, expression_matrix_kind, abc_cache):
    """Get expression matrix for a dataset."""
    file = abc_cache.get_file_path(directory=dataset, file_name=f'{dataset}/{expression_matrix_kind}')
    adata = ad.read_h5ad(file, backed='r')

    return adata


def export_sampled_section_h5ad(backed_adata, sampled_cells, output_path, args):
    """Export sampled cells as a new h5ad matrix."""
    sampled_labels = sampled_cells.index.astype(str)
    row_indexer = backed_adata.obs_names.get_indexer(sampled_labels)
    valid_mask = row_indexer >= 0

    if not np.any(valid_mask):
        print(f"  Warning: no sampled labels found in expression matrix: {output_path.name}")
        return

    if not np.all(valid_mask):
        missing_count = int((~valid_mask).sum())
        print(f"  Warning: {missing_count} sampled labels missing in expression matrix")

    valid_labels = sampled_labels[valid_mask]
    valid_rows = row_indexer[valid_mask]

    sampled_adata = backed_adata[valid_rows, :].to_memory()
    sampled_adata.obs = sampled_adata.obs.copy()

    sampled_obs = sampled_cells.loc[valid_labels].copy()
    sampled_obs.index = valid_labels

    keep_columns = [
        'brain_section_label',
        'x',
        'y',
        'parcellation_division',
        'parcellation_substructure',
    ]
    available_columns = [col for col in keep_columns if col in sampled_obs.columns]

    if len(available_columns) > 0:
        aligned_obs = sampled_obs[available_columns].reindex(sampled_adata.obs_names)
        for column in available_columns:
            sampled_adata.obs[column] = aligned_obs[column].to_numpy()

    grid_row, grid_col = compute_grid_indices_from_xy(
        sampled_obs['x'],
        sampled_obs['y'],
        args.grid_block_um,
        args.grid_gap_um,
    )
    sampled_adata.obs['sampling_grid_row'] = grid_row
    sampled_adata.obs['sampling_grid_col'] = grid_col
    sampled_adata.obs['sampling_grid_id'] = np.array(
        [f'r{row}_c{col}' for row, col in zip(grid_row, grid_col)],
        dtype=object,
    )

    sampled_adata.uns['sampling'] = {
        'method': 'grid_block_gap',
        'block_um': args.grid_block_um,
        'gap_um': args.grid_gap_um,
        'period_um': args.grid_block_um + args.grid_gap_um,
        'description': 'Keep cells where x%period<block and y%period<block',
        'grid_id_format': 'r{row}_c{col}',
    }

    sampled_adata.write_h5ad(output_path, compression='lzf')


def add_sampling_grid(ax, x_min, x_max, y_min, y_max, args):
    """Overlay only sampled 20 µm squares (skip gap squares)."""
    period_um = args.grid_block_um + args.grid_gap_um
    x_base_start = np.floor(x_min / period_um) * period_um
    x_base_end = np.ceil(x_max / period_um) * period_um
    y_base_start = np.floor(y_min / period_um) * period_um
    y_base_end = np.ceil(y_max / period_um) * period_um

    x_base_lines = np.arange(x_base_start, x_base_end + period_um, period_um)
    y_base_lines = np.arange(y_base_start, y_base_end + period_um, period_um)

    x_samples = [x0 for x0 in x_base_lines if (x0 + args.grid_block_um) >= x_min and x0 <= x_max]
    y_samples = [y0 for y0 in y_base_lines if (y0 + args.grid_block_um) >= y_min and y0 <= y_max]

    segments = []
    for x0 in x_samples:
        x1 = x0 + args.grid_block_um
        for y0 in y_samples:
            y1 = y0 + args.grid_block_um
            segments.append([(x0, y0), (x1, y0)])
            segments.append([(x0, y1), (x1, y1)])
            segments.append([(x0, y0), (x0, y1)])
            segments.append([(x1, y0), (x1, y1)])

    if len(segments) > 0:
        line_collection = LineCollection(
            segments,
            colors='red',
            linewidths=args.grid_linewidth,
            alpha=args.grid_alpha,
            zorder=3,
        )
        ax.add_collection(line_collection)


def plot_dataset_parcellation(section_cells, output_path, substructure_color_map, args, fixed_limits=None):
    """Plot CCF parcellation colors (substructure only)."""
    # Follow tutorial style, but convert section x/y from mm to µm for physical axes.
    x_vals_um = section_cells['x'].values * SECTION_XY_MM_TO_UM
    y_vals_um = section_cells['y'].values * SECTION_XY_MM_TO_UM

    x_min = float(x_vals_um.min())
    x_max = float(x_vals_um.max())
    y_min = float(y_vals_um.min())
    y_max = float(y_vals_um.max())

    x_span_um = x_max - x_min
    y_span_um = y_max - y_min

    x_pad = max(100.0, x_span_um * 0.03)
    y_pad = max(100.0, y_span_um * 0.03)

    if fixed_limits is not None:
        x_plot_min, x_plot_max, y_plot_min, y_plot_max = fixed_limits
    else:
        x_plot_min = x_min - x_pad
        x_plot_max = x_max + x_pad
        y_plot_min = y_min - y_pad
        y_plot_max = y_max + y_pad

    width = args.figure_width
    height = width * (y_plot_max - y_plot_min) / (x_plot_max - x_plot_min)
    height = max(args.figure_min_height, min(args.figure_max_height, height))

    if len(substructure_color_map) > 0 and 'parcellation_substructure' in section_cells.columns:
        substructure_labels = [
            value.strip() if isinstance(value, str) and value.strip() else 'Unknown'
            for value in section_cells['parcellation_substructure'].values
        ]
        colors = [substructure_color_map.get(label, '#B0B0B0') for label in substructure_labels]
    elif 'parcellation_substructure_color' in section_cells.columns:
        colors = clean_color_values(section_cells['parcellation_substructure_color'].values)
    else:
        colors = ['#B0B0B0'] * len(section_cells)

    fig, (ax, legend_ax) = plt.subplots(
        1,
        2,
        figsize=(width + 3.0, height),
        gridspec_kw={'width_ratios': [4.5, 1.8]},
    )
    ax.scatter(x_vals_um, y_vals_um, s=args.point_size, c=colors, marker='.', alpha=0.9)
    ax.set_aspect('equal')
    ax.set_xlabel('X (µm)', fontsize=10)
    ax.set_ylabel('Y (µm)', fontsize=10)
    ax.set_xlim(x_plot_min, x_plot_max)
    ax.set_ylim(y_plot_max, y_plot_min)

    if args.show_sampling_grid:
        add_sampling_grid(ax, x_plot_min, x_plot_max, y_plot_min, y_plot_max, args)

    legend_ax.axis('off')
    if len(substructure_color_map) > 0 and 'parcellation_substructure' in section_cells.columns:
        section_substructures = sorted(
            {
                value.strip()
                for value in section_cells['parcellation_substructure'].values
                if isinstance(value, str) and value.strip()
            }
        )
        legend_handles = [
            Patch(facecolor=substructure_color_map.get(name, '#B0B0B0'), edgecolor='none', label=name)
            for name in section_substructures
        ]
        if len(legend_handles) > 0:
            legend_ax.legend(
                handles=legend_handles,
                title='Substructure',
                loc='upper left',
                frameon=False,
                fontsize=6,
                title_fontsize=8,
                handlelength=1.2,
                handleheight=1.2,
                borderaxespad=0.0,
            )

    ax.set_title('CCF Substructure', fontsize=11)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f'CCF Parcellation Substructure ({len(section_cells)} cells)\n'
        f'Physical span: {x_span_um:.0f} × {y_span_um:.0f} µm',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def write_sampling_stats_txt(stats_path, stats_records, dataset_summaries):
    lines = []
    lines.append('Sampling Statistics')
    lines.append('=' * 100)
    lines.append('')

    for summary in dataset_summaries:
        lines.append(f"Dataset: {summary['dataset']}")
        lines.append(f"Division: {summary['division']}")
        lines.append(f"Grid block/gap (um): {summary['grid_block_um']}/{summary['grid_gap_um']}")
        lines.append(
            f"Whole-sample summary: total={summary['total_cells']}, sampled={summary['sampled_cells']}, "
            f"discarded={summary['discarded_cells']}, sampling_rate={summary['sampling_rate']:.6f}"
        )
        lines.append(f"Images generated: {summary['image_count']}")
        lines.append('-' * 100)
        lines.append('section\ttotal_cells\tsampled_cells\tdiscarded_cells\tsampling_rate\timage_path')

        dataset_records = [r for r in stats_records if r['dataset'] == summary['dataset']]
        for record in dataset_records:
            lines.append(
                f"{record['section']}\t{record['total_cells']}\t{record['sampled_cells']}\t"
                f"{record['discarded_cells']}\t{record['sampling_rate']:.6f}\t{record['image_path']}"
            )
        lines.append('')

    stats_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    args = parse_args()
    output_dir = args.download_base.parent / 'output'
    output_dir.mkdir(exist_ok=True, parents=True)
    sampling_image_dir = output_dir / 'sampling_images'
    sampled_h5ad_dir = output_dir / 'sampled_h5ad'
    sampling_mask_dir = output_dir / 'sampling_mask'
    sampling_image_dir.mkdir(exist_ok=True, parents=True)
    sampled_h5ad_dir.mkdir(exist_ok=True, parents=True)
    sampling_mask_dir.mkdir(exist_ok=True, parents=True)

    datasets = [x.strip() for x in args.datasets.split(',') if x.strip()]
    selected_division = args.division.strip()
    division_suffix = selected_division if selected_division.upper() != 'ALL' else 'ALL'
    stats_records = []
    dataset_summaries = []

    print('Loading AbcProjectCache...')
    abc_cache = AbcProjectCache.from_cache_dir(args.download_base)

    print('Loading cell metadata...')
    cell_extended = {}
    for d in datasets:
        cell_extended[d] = abc_cache.get_metadata_dataframe(directory=d, file_name='cell_metadata')
        cell_extended[d].set_index('cell_label', inplace=True)

    print('Loading CCF coordinates...')
    ccf_coordinates = {}
    for d in datasets:
        try:
            ccf_coordinates[d] = abc_cache.get_metadata_dataframe(directory=f'{d}-CCF', file_name='ccf_coordinates')
            ccf_coordinates[d].set_index('cell_label', inplace=True)
            ccf_coordinates[d].rename(columns={'x': 'x_ccf_mm', 'y': 'y_ccf_mm', 'z': 'z_ccf_mm'}, inplace=True)
            ccf_coordinates[d]['x_um'] = ccf_coordinates[d]['x_ccf_mm'] * MM_TO_UM
            ccf_coordinates[d]['y_um'] = ccf_coordinates[d]['y_ccf_mm'] * MM_TO_UM
            ccf_coordinates[d]['z_um'] = ccf_coordinates[d]['z_ccf_mm'] * MM_TO_UM
            ccf_coordinates[d]['x_px'] = ccf_coordinates[d]['x_ccf_mm'] * MM_TO_PX
            ccf_coordinates[d]['y_px'] = ccf_coordinates[d]['y_ccf_mm'] * MM_TO_PX
            ccf_coordinates[d]['z_px'] = ccf_coordinates[d]['z_ccf_mm'] * MM_TO_PX
            cell_extended[d] = cell_extended[d].join(ccf_coordinates[d], how='inner')
            print(f'  {d}: {len(cell_extended[d])} cells with CCF registration')
        except Exception as e:
            print(f'  Warning: Could not load CCF data for {d}: {e}')

    print('Loading parcellation annotations...')
    try:
        parcellation_annotation = abc_cache.get_metadata_dataframe(
            directory='Allen-CCF-2020',
            file_name='parcellation_to_parcellation_term_membership_acronym',
        )
        parcellation_annotation.set_index('parcellation_index', inplace=True)
        parcellation_annotation.columns = [f'parcellation_{x}' for x in parcellation_annotation.columns]

        parcellation_color = abc_cache.get_metadata_dataframe(
            directory='Allen-CCF-2020',
            file_name='parcellation_to_parcellation_term_membership_color',
        )
        parcellation_color.set_index('parcellation_index', inplace=True)
        parcellation_color.columns = [f'parcellation_{x}' for x in parcellation_color.columns]

        for d in datasets:
            cell_extended[d] = cell_extended[d].join(parcellation_annotation, on='parcellation_index')
            cell_extended[d] = cell_extended[d].join(parcellation_color, on='parcellation_index')
    except Exception as e:
        print(f'  Warning: could not load parcellation tables: {e}')

    print('\nGenerating parcellation color visualizations (tutorial style)...')
    for dataset in datasets:
        dataset_cells = cell_extended[dataset]

        if selected_division.upper() != 'ALL':
            if 'parcellation_division' not in dataset_cells.columns:
                print(f'\n{dataset}: missing parcellation_division column, skip')
                continue
            dataset_cells = dataset_cells[dataset_cells['parcellation_division'] == selected_division]

        if len(dataset_cells) == 0:
            print(f'\n{dataset}: no cells after division filter ({selected_division}), skip')
            continue

        dataset_x_um = dataset_cells['x'].to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
        dataset_y_um = dataset_cells['y'].to_numpy(dtype=np.float64) * SECTION_XY_MM_TO_UM
        dataset_x_min = float(dataset_x_um.min())
        dataset_x_max = float(dataset_x_um.max())
        dataset_y_min = float(dataset_y_um.min())
        dataset_y_max = float(dataset_y_um.max())
        dataset_x_span = dataset_x_max - dataset_x_min
        dataset_y_span = dataset_y_max - dataset_y_min
        dataset_x_pad = max(100.0, dataset_x_span * 0.03)
        dataset_y_pad = max(100.0, dataset_y_span * 0.03)
        dataset_fixed_limits = (
            dataset_x_min - dataset_x_pad,
            dataset_x_max + dataset_x_pad,
            dataset_y_min - dataset_y_pad,
            dataset_y_max + dataset_y_pad,
        )
        print(
            f'{dataset}: fixed range X=({dataset_fixed_limits[0]:.0f}, {dataset_fixed_limits[1]:.0f}) µm, '
            f'Y=({dataset_fixed_limits[2]:.0f}, {dataset_fixed_limits[3]:.0f}) µm'
        )

        dataset_keep_mask = compute_keep_mask_from_xy(dataset_cells['x'], dataset_cells['y'], args.grid_block_um, args.grid_gap_um)
        dataset_total_cells = int(len(dataset_cells))
        dataset_sampled_cells = int(dataset_keep_mask.sum())
        dataset_discarded_cells = dataset_total_cells - dataset_sampled_cells
        dataset_sampling_rate = (dataset_sampled_cells / dataset_total_cells) if dataset_total_cells > 0 else 0.0

        dataset_prefix = f"{dataset.lower().replace('-', '_')}_{division_suffix}_sampled{int(args.grid_block_um)}um_gap{int(args.grid_gap_um)}um"
        if args.export_sampling_mask:
            mask_npz_path = sampling_mask_dir / f'{dataset_prefix}_mask.npz'
            save_sampling_mask(dataset_cells, dataset_keep_mask, mask_npz_path, args.grid_block_um, args.grid_gap_um)
            mask_png_path = sampling_mask_dir / f'{dataset_prefix}_mask.png'
            save_sampling_mask_png(dataset_cells, dataset_keep_mask, dataset_fixed_limits, mask_png_path, args)
            print(f'Saved: {mask_npz_path}')
            print(f'Saved: {mask_png_path}')

        if 'parcellation_substructure' in dataset_cells.columns:
            substructure_color_map = build_substructure_color_map(dataset_cells['parcellation_substructure'].values)
            substructure_color_map_df = pd.DataFrame({
                'parcellation_substructure': list(substructure_color_map.keys()),
                'color': list(substructure_color_map.values()),
            })
            substructure_color_map_df.to_csv(sampling_image_dir / f'{dataset_prefix}_substructure_color_map.csv', index=False)
            print(f'{dataset}: reassigned colors for {len(substructure_color_map)} substructures')
        else:
            substructure_color_map = {}
            print(f'{dataset}: missing parcellation_substructure, fallback to provided colors')

        backed_expr = None
        if args.export_sampled_h5ad:
            backed_expr = get_expression_matrix(dataset, args.expression_matrix_kind, abc_cache)
            if backed_expr is None:
                print(f'{dataset}: expression matrix not found, skip h5ad export')
            else:
                print(f'{dataset}: loading expression matrix in backed mode')

        unique_sections = dataset_cells['brain_section_label'].unique()
        print(f'\n{dataset}: Found {len(unique_sections)} sections after division filter')
        dataset_image_count = 0

        for section_label in sorted(unique_sections):
            section_cells = dataset_cells[dataset_cells['brain_section_label'] == section_label]
            if len(section_cells) == 0:
                continue

            section_keep_mask = dataset_keep_mask[np.where(dataset_cells['brain_section_label'].to_numpy() == section_label)[0]]
            sampled_cells = section_cells.iloc[np.where(section_keep_mask)[0]]
            if len(sampled_cells) == 0:
                print(f'  {section_label}: no cells after sampling, skip')
                continue

            section_total_cells = int(len(section_cells))
            section_sampled_cells = int(len(sampled_cells))
            section_discarded_cells = section_total_cells - section_sampled_cells
            section_sampling_rate = (section_sampled_cells / section_total_cells) if section_total_cells > 0 else 0.0

            safe_section = section_label.replace('.', '_').replace('/', '_')
            out_prefix = f"{safe_section}_{division_suffix}_sampled{int(args.grid_block_um)}um_gap{int(args.grid_gap_um)}um"
            out_name = f'{out_prefix}_ccf_parcellation_substructure.png'
            sample_dir = sampling_image_dir / dataset
            sample_dir.mkdir(parents=True, exist_ok=True)
            image_path = sample_dir / out_name

            plot_dataset_parcellation(
                sampled_cells,
                image_path,
                substructure_color_map,
                args,
                fixed_limits=dataset_fixed_limits,
            )
            dataset_image_count += 1

            stats_records.append(
                {
                    'dataset': dataset,
                    'section': section_label,
                    'total_cells': section_total_cells,
                    'sampled_cells': section_sampled_cells,
                    'discarded_cells': section_discarded_cells,
                    'sampling_rate': section_sampling_rate,
                    'image_path': str(image_path),
                }
            )

            if backed_expr is not None:
                h5ad_dir = sampled_h5ad_dir / dataset
                h5ad_dir.mkdir(parents=True, exist_ok=True)
                sampled_h5ad_path = h5ad_dir / f'{out_prefix}_{args.expression_matrix_kind}.h5ad'
                export_sampled_section_h5ad(backed_expr, sampled_cells, sampled_h5ad_path, args)

        if backed_expr is not None:
            backed_expr.file.close()

        dataset_summaries.append(
            {
                'dataset': dataset,
                'division': selected_division,
                'grid_block_um': args.grid_block_um,
                'grid_gap_um': args.grid_gap_um,
                'total_cells': dataset_total_cells,
                'sampled_cells': dataset_sampled_cells,
                'discarded_cells': dataset_discarded_cells,
                'sampling_rate': dataset_sampling_rate,
                'image_count': dataset_image_count,
            }
        )

    stats_name = (
        f"sampling_stats_{division_suffix}_"
        f"sampled{int(args.grid_block_um)}um_gap{int(args.grid_gap_um)}um.txt"
    )
    stats_path = output_dir / stats_name
    write_sampling_stats_txt(stats_path, stats_records, dataset_summaries)
    print(f'Saved: {stats_path}')

    print('\n' + '=' * 70)
    print('CCF Registration Summary')
    print('=' * 70)
    for d in datasets:
        if d not in cell_extended:
            continue
        df = cell_extended[d]
        if 'x_um' not in df.columns:
            continue
        x_range = (df['x_um'].min(), df['x_um'].max())
        y_range = (df['y_um'].min(), df['y_um'].max())
        z_range = (df['z_um'].min(), df['z_um'].max())
        print(f'\n{d}:')
        print(f'  Cells: {len(df)}')
        print(f'  X range (μm): {x_range[0]:.1f} - {x_range[1]:.1f} (span: {x_range[1] - x_range[0]:.1f})')
        print(f'  Y range (μm): {y_range[0]:.1f} - {y_range[1]:.1f} (span: {y_range[1] - y_range[0]:.1f})')
        print(f'  Z range (μm): {z_range[0]:.1f} - {z_range[1]:.1f} (span: {z_range[1] - z_range[0]:.1f})')

    print('\n' + '=' * 70)
    print(f'Output directory: {output_dir.absolute()}')
    print('=' * 70)


if __name__ == '__main__':
    main()
