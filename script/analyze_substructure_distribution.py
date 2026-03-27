#!/usr/bin/env python
"""
Analyze Step 1 output: count the number of grids and cells for each parcellation_substructure
within each brain_section_label.

Usage:
  python analyze_substructure_distribution.py --input-dir <sampled_h5ad_dir> --output-dir <output_dir>

Input directory layout:
    <input-dir>/<dataset-name>/**/*.h5ad
    (dataset names are provided by --datasets)

Generated outputs:
    1. individual_substructure_plots_raw/<dataset>/*.png - one raw plot per substructure per dataset
    2. individual_substructure_plots_interp/<dataset>/*.png - one interpolated plot per substructure per dataset
    3. substructure_span_summary.csv - per-substructure span summary

X-axis:
    distance between brain sections (µm), controlled by --section-spacing-um
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze parcellation_substructure distribution across brain sections'
    )
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory containing sampled h5ad files from Step 1')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for results. Default: input-dir/../analysis_substructure_distribution')
    parser.add_argument('--datasets', type=str, default='Zhuang-ABCA-1,Zhuang-ABCA-2',
                        help='Comma-separated dataset names, e.g. Zhuang-ABCA-1,Zhuang-ABCA-2')
    parser.add_argument('--section-spacing-um', type=str, default='100,200',
                        help='Comma-separated section spacing (µm), aligned to --datasets order')
    parser.add_argument('--interp-spacing-um', type=float, default=20.0,
                        help='Interpolation interval on distance axis (µm)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='DPI for output figures')
    parser.add_argument('--figure-width-1-5', type=float, default=12.0,
                        help='Figure width in inches')
    parser.add_argument('--figure-height-1-5', type=float, default=6.0,
                        help='Figure height in inches')
    parser.add_argument('--line-width', type=float, default=2.0,
                        help='Line width for plots')
    parser.add_argument('--marker-size', type=int, default=8,
                        help='Marker size for plots')
    return parser.parse_args()


def parse_csv_values(text):
    return [value.strip() for value in str(text).split(',') if value.strip()]


def build_dataset_spacing_map(args):
    datasets = parse_csv_values(args.datasets)
    spacing_values = parse_csv_values(args.section_spacing_um)

    if len(datasets) == 0:
        raise ValueError('--datasets must include at least one dataset name')
    if len(datasets) != len(spacing_values):
        raise ValueError('--datasets and --section-spacing-um must have the same number of items')

    try:
        spacings = [float(value) for value in spacing_values]
    except ValueError as exc:
        raise ValueError(f'Invalid spacing value in --section-spacing-um: {spacing_values}') from exc

    return datasets, dict(zip(datasets, spacings))


def extract_section_order(brain_section_label):
    """Extract numeric suffix/index from brain_section_label for sorting."""
    match = re.search(r'(\d+)$', str(brain_section_label))
    if match:
        return int(match.group(1))
    return float('inf')  # Non-numbered sections go to the end


def get_section_spacing_um(dataset_name, dataset_spacing_map):
    if dataset_name in dataset_spacing_map:
        return float(dataset_spacing_map[dataset_name])
    return float(next(iter(dataset_spacing_map.values())))


def collect_h5ad_files_in_dataset_dir(dataset_dir):
    """Collect h5ad files under dataset folder recursively without glob."""
    h5ad_files = []
    for root, _, files in os.walk(dataset_dir):
        root_path = Path(root)
        for file_name in files:
            if file_name.lower().endswith('.h5ad'):
                h5ad_files.append(root_path / file_name)
    return sorted(h5ad_files)


def aggregate_substructure_stats(input_dir, datasets):
    """Aggregate substructure statistics from per-dataset folders under input_dir.
    
    Returns:
        pd.DataFrame: columns [dataset, section_label, section_order, substructure, n_cells, n_grids]
    """
    stats_records = []
    total_h5ad_count = 0

    for dataset_name in datasets:
        dataset_dir = input_dir / dataset_name
        if not dataset_dir.exists() or not dataset_dir.is_dir():
            print(f"  Warning: dataset folder not found: {dataset_dir}")
            continue

        h5ad_files = collect_h5ad_files_in_dataset_dir(dataset_dir)
        total_h5ad_count += len(h5ad_files)
        print(f"Dataset {dataset_name}: found {len(h5ad_files)} h5ad files")

        for h5ad_path in h5ad_files:
            print(f"Processing [{dataset_name}]: {h5ad_path.name}")
            try:
                adata = ad.read_h5ad(h5ad_path)
            except Exception as e:
                print(f"  Warning: Failed to read {h5ad_path}: {e}")
                continue

            if len(adata) == 0:
                print(f"  Warning: Empty h5ad file: {h5ad_path.name}")
                continue

            if 'brain_section_label' not in adata.obs.columns:
                print(f"  Warning: missing 'brain_section_label' column in {h5ad_path.name}")
                continue

            if 'parcellation_substructure' not in adata.obs.columns:
                print(f"  Warning: missing 'parcellation_substructure' column in {h5ad_path.name}")
                continue

            if 'sampling_grid_row' not in adata.obs.columns or 'sampling_grid_col' not in adata.obs.columns:
                print(f"  Warning: missing grid index columns in {h5ad_path.name}")
                continue

            grouped = adata.obs.groupby(['brain_section_label', 'parcellation_substructure'],
                                        observed=True, dropna=False)

            for (section_label, substructure), group_df in grouped:
                if pd.isna(section_label) or str(section_label).strip() == '':
                    continue

                section_label = str(section_label).strip()
                substructure = str(substructure).strip() if not pd.isna(substructure) else 'Unknown'

                grid_coords = group_df[['sampling_grid_row', 'sampling_grid_col']].drop_duplicates()
                n_grids = len(grid_coords)
                n_cells = len(group_df)
                section_order = extract_section_order(section_label)

                stats_records.append({
                    'dataset': dataset_name,
                    'section_order': section_order,
                    'substructure': substructure,
                    'n_cells': n_cells,
                    'n_grids': n_grids,
                })

    print(f"Total h5ad files processed: {total_h5ad_count}")
    
    if not stats_records:
        raise ValueError(f"No valid statistics collected from {input_dir}")
    
    stats_df = pd.DataFrame(stats_records)
    return stats_df


def _plot_substructure_curve(x_vals, cells_vals, grids_vals, title, output_path, args):
    fig, ax1 = plt.subplots(figsize=(args.figure_width_1_5, args.figure_height_1_5))

    color_cells = '#1f77b4'
    ax1.set_xlabel('Distance between brain sections (µm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Cells', fontsize=11, fontweight='bold', color=color_cells)
    line1 = ax1.plot(
        x_vals,
        cells_vals,
        marker='o',
        markersize=args.marker_size,
        linewidth=args.line_width,
        label='Cell Count',
        color=color_cells,
        alpha=0.8,
    )
    ax1.tick_params(axis='y', labelcolor=color_cells)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = ax1.twinx()
    color_grids = '#ff7f0e'
    line2 = ax2.plot(
        x_vals,
        grids_vals,
        marker='s',
        markersize=args.marker_size,
        linewidth=args.line_width,
        label='Grid Count',
        color=color_grids,
        alpha=0.8,
    )
    ax2.set_ylabel('Number of Grids (20µm×20µm)', fontsize=11, fontweight='bold', color=color_grids)
    ax2.tick_params(axis='y', labelcolor=color_grids)

    all_y = np.concatenate([
        np.asarray(cells_vals, dtype=np.float64),
        np.asarray(grids_vals, dtype=np.float64),
    ])
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    if np.isclose(y_min, y_max):
        margin = 1.0 if np.isclose(y_max, 0.0) else abs(y_max) * 0.05
    else:
        margin = (y_max - y_min) * 0.05
    shared_ylim = (y_min - margin, y_max + margin)
    ax1.set_ylim(*shared_ylim)
    ax2.set_ylim(*shared_ylim)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)

    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)


def create_line_plots(stats_df, output_dir, args, dataset_spacing_map):
    """Create per-substructure raw and interpolated plots for each dataset."""
    output_dir.mkdir(exist_ok=True, parents=True)

    substructure_raw_dir = output_dir / f'individual_substructure_plots_raw_{args.interp_spacing_um}um'
    substructure_interp_dir = output_dir / f'individual_substructure_plots_interp_{args.interp_spacing_um}um'
    substructure_raw_dir.mkdir(exist_ok=True, parents=True)
    substructure_interp_dir.mkdir(exist_ok=True, parents=True)
    
    # Aggregate stats by dataset/section/substructure (sum across files if multiple)
    agg_stats = stats_df.groupby(
        ['dataset', 'section_order', 'substructure'],
        as_index=False
    ).agg({
        'n_cells': 'sum',
        'n_grids': 'sum',
    })
    
    # Get unique substructures and sort
    substructures = sorted(agg_stats['substructure'].unique())
    datasets = sorted(agg_stats['dataset'].unique())
    
    saved_raw_count = 0
    saved_interp_count = 0
    
    # ========== Create individual plot for each substructure and each dataset ==========
    summary_records = []
    for dataset_name in datasets:
        dataset_raw_dir = substructure_raw_dir / dataset_name
        dataset_interp_dir = substructure_interp_dir / dataset_name
        dataset_raw_dir.mkdir(exist_ok=True, parents=True)
        dataset_interp_dir.mkdir(exist_ok=True, parents=True)

        dataset_df = agg_stats[agg_stats['dataset'] == dataset_name].copy()
        spacing_um = get_section_spacing_um(dataset_name, dataset_spacing_map)

        section_info = dataset_df[['section_order']].drop_duplicates().copy()
        section_info = section_info.sort_values('section_order')
        if len(section_info) == 0:
            continue

        section_info['distance_um'] = section_info['section_order'] * spacing_um
        distance_map = dict(zip(section_info['section_order'], section_info['distance_um']))

        for substructure in substructures:
            sub_data = dataset_df[dataset_df['substructure'] == substructure].copy()
            if len(sub_data) == 0:
                continue

            sub_data = sub_data.sort_values('section_order')
            x_vals = [distance_map[label] for label in sub_data['section_order']]
            cells_vals = sub_data['n_cells'].values
            grids_vals = sub_data['n_grids'].values

            safe_name = substructure.replace('/', '_').replace('\\', '_').replace(' ', '_')
            raw_plot_path = dataset_raw_dir / f'substructure_{safe_name}.png'
            _plot_substructure_curve(
                x_vals,
                cells_vals,
                grids_vals,
                title=f'Substructure: {substructure} | Dataset: {dataset_name} | Raw | Section spacing: {spacing_um:.0f} µm',
                output_path=raw_plot_path,
                args=args,
            )
            saved_raw_count += 1
            start_um = x_vals[0] if len(x_vals) > 0 else 0
            end_um = x_vals[-1] if len(x_vals) > 0 else 0
            start_slice = sub_data['section_order'].min()
            end_slice = sub_data['section_order'].max()
            raw_cell_number = cells_vals.sum()
            raw_grid_number = grids_vals.sum()

            x_raw = np.asarray(x_vals, dtype=np.float64)
            y_cells_raw = np.asarray(cells_vals, dtype=np.float64)
            y_grids_raw = np.asarray(grids_vals, dtype=np.float64)
            if len(x_raw) == 1:
                interp_x = np.array([x_raw[0]], dtype=np.float64)
                interp_cells = np.array([y_cells_raw[0]], dtype=np.float64)
                interp_grids = np.array([y_grids_raw[0]], dtype=np.float64)
            else:
                interp_x = np.arange(x_raw.min(), x_raw.max() + args.interp_spacing_um * 0.5, args.interp_spacing_um, dtype=np.float64)
                interp_cells = np.interp(interp_x, x_raw, y_cells_raw)
                interp_grids = np.interp(interp_x, x_raw, y_grids_raw)
            
            interp_cell_number = interp_cells.sum()
            interp_grid_number = interp_grids.sum()
            summary_records.append({
                'dataset': dataset_name,
                'substructure': substructure,
                'start_slice': start_slice,
                'end_slice': end_slice,
                'start_um': start_um,
                'end_um': end_um,
                'raw_cell_number': raw_cell_number,
                'raw_grid_number': raw_grid_number,
                'interp_cell_number': interp_cell_number,
                'interp_grid_number': interp_grid_number,
            })

            interp_plot_path = dataset_interp_dir / f'substructure_{safe_name}.png'
            _plot_substructure_curve(
                interp_x,
                interp_cells,
                interp_grids,
                title=(
                    f'Substructure: {substructure} | Dataset: {dataset_name} | Interpolated '
                    f'({args.interp_spacing_um:.0f} µm) | Section spacing: {spacing_um:.0f} µm'
                ),
                output_path=interp_plot_path,
                args=args,
            )
            saved_interp_count += 1

    print(f"\n✓ Created {saved_raw_count} raw substructure plots")
    print(f"✓ Created {saved_interp_count} interpolated substructure plots")
    
    return summary_records


def plot_bar(stats_df, output_path, args):
    """Create a bar plot showing the number of cells and grids for each substructure across datasets."""
    agg_df = stats_df.groupby(['dataset', 'substructure'], as_index=False).agg({
        'raw_cell_number': 'sum',
        'raw_grid_number': 'sum',
        'interp_cell_number': 'sum',
        'interp_grid_number': 'sum',
    })

    substructures = sorted(agg_df['substructure'].unique())
    datasets = sorted(agg_df['dataset'].unique())
    
    x = np.arange(len(substructures))
    
    for _, dataset_name in enumerate(datasets):
        dataset_data = agg_df[agg_df['dataset'] == dataset_name]
        raw_cells_vals = [dataset_data[dataset_data['substructure'] == sub]['raw_cell_number'].values[0] if sub in dataset_data['substructure'].values else 0 for sub in substructures]
        raw_grids_vals = [dataset_data[dataset_data['substructure'] == sub]['raw_grid_number'].values[0] if sub in dataset_data['substructure'].values else 0 for sub in substructures]
        inter_cells_vals = [dataset_data[dataset_data['substructure'] == sub]['interp_cell_number'].values[0] if sub in dataset_data['substructure'].values else 0 for sub in substructures]
        interp_grids_vals = [dataset_data[dataset_data['substructure'] == sub]['interp_grid_number'].values[0] if sub in dataset_data['substructure'].values else 0 for sub in substructures]
        fig, ax = plt.subplots(2, 1, figsize=(max(10, len(substructures) * 0.3), 8))

        ax[0].bar(x, raw_cells_vals, label=f'{dataset_name} - Raw Cells', alpha=0.7)
        ax[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(substructures, rotation=45, ha='right', fontsize=12)
        ax[0].legend(fontsize=10)
        ax[0].grid(True, alpha=0.3, linestyle='--')

        ax[1].bar(x, raw_grids_vals, label=f'{dataset_name} - Raw Grids', alpha=0.7)
        ax[1].set_xlabel('Substructure', fontsize=14, fontweight='bold')
        ax[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(substructures, rotation=45, ha='right', fontsize=12)
        ax[1].legend(fontsize=10)
        ax[1].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path / f'{dataset_name}_cell_grid_counts_{args.interp_spacing_um}um.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(2, 1, figsize=(max(10, len(substructures) * 0.3), 8))
        ax[0].bar(x, inter_cells_vals, label=f'{dataset_name} - Cells (Interp)', alpha=0.7)
        ax[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(substructures, rotation=45, ha='right', fontsize=12)
        ax[0].legend(fontsize=10)
        ax[0].grid(True, alpha=0.3, linestyle='--')

        ax[1].bar(x, interp_grids_vals, label=f'{dataset_name} - Grids (Interp)', alpha=0.7)
        ax[1].set_xlabel('Substructure', fontsize=14, fontweight='bold')
        ax[1].set_ylabel('Count', fontsize=12, fontweight='bold')
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(substructures, rotation=45, ha='right', fontsize=12)
        ax[1].legend(fontsize=10)
        ax[1].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(output_path / f'{dataset_name}_interp_cell_grid_counts_{args.interp_spacing_um}um.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    args = parse_args()
    datasets_arg, dataset_spacing_map = build_dataset_spacing_map(args)
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / 'analysis_substructure_distribution'
    
    args.output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {datasets_arg}")
    print(f"Section spacing map (µm): {dataset_spacing_map}")
    print()
    
    # Aggregate statistics from all h5ad files
    print("\n=== Aggregating substructure statistics ===")
    stats_df = aggregate_substructure_stats(args.input_dir, datasets_arg)
    if len(stats_df) == 0:
        raise ValueError('No rows left after filtering by --datasets')

    print(f"Collected {len(stats_df)} section-substructure combinations")
    
    # Create line plots
    print("\n=== Creating line plots ===")
    summary_records = create_line_plots(stats_df, args.output_dir, args, dataset_spacing_map)

    summary_csv_path = args.output_dir / f'substructure_span_summary_{args.interp_spacing_um}um.csv'
    pd.DataFrame(summary_records).to_csv(summary_csv_path, index=False)
    print(f"Saved span summary: {summary_csv_path}")

    print("\n=== Creating bar plots ===")
    plot_bar(pd.DataFrame(summary_records), args.output_dir, args)
    print(f"Saved bar plots for each dataset in: {args.output_dir}")
    
    print("\n=== Analysis complete ===")


if __name__ == '__main__':
    main()
