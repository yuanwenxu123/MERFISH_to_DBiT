#!/usr/bin/env python
"""
Master controller for MERFISH sampling + clustering pipeline.

Step 1: ccf_registration_to_image.py
Step 2: cluster_sampled_h5ad.py
Step 3 (optional): embedding_merfish.py
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def add_legacy_alias(parser, old_flag: str, dest: str):
    parser.add_argument(old_flag, dest=dest, help=argparse.SUPPRESS)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MERFISH step-1 + step-2 (+ optional step-3 embedding) pipeline")

    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to run sub-scripts")
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)

    parser.add_argument("--skip-step1", action="store_true", help="Skip step 1 (sampling/export)")
    parser.add_argument("--skip-step1-5", action="store_true", help="Skip step 1.5 (substructure distribution analysis)")
    parser.add_argument("--skip-step2", action="store_true", help="Skip step 2 (Leiden clustering)")
    parser.add_argument("--run-step3", action="store_true", help="Run step 3 (reference integration + label transfer + plotting)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")

    parser.add_argument("--download-base", type=Path)
    parser.add_argument("--datasets", type=str, default="Zhuang-ABCA-1, Zhuang-ABCA-2", help="Comma-separated dataset names")
    parser.add_argument("--division", type=str)
    parser.add_argument("--grid-block-um", type=float, default=20.0)
    parser.add_argument("--grid-gap-um", type=float, default=20.0)
    parser.add_argument("--expression-matrix-kind", type=str, default="raw", choices=["raw", "log2"])

    parser.add_argument("--step1-show-sampling-grid", action="store_true", default=True)
    parser.add_argument("--step1-hide-sampling-grid", dest="step1_show_sampling_grid", action="store_false")
    parser.add_argument("--step1-export-sampled-h5ad", action="store_true", default=True)
    parser.add_argument("--step1-no-export-sampled-h5ad", dest="step1_export_sampled_h5ad", action="store_false")
    parser.add_argument("--step1-export-sampling-mask", action="store_true", default=True)
    parser.add_argument("--step1-no-export-sampling-mask", dest="step1_export_sampling_mask", action="store_false")

    parser.add_argument("--point-size", type=float, default=3.0,
                        help="Unified plot point size")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Unified plot dpi for step1/step2/step3")
    parser.add_argument("--figure-width", type=float, default=8.0,
                        help="Unified figure width for step1/step2/step3")
    parser.add_argument("--figure-min-height", type=float, default=4.5,
                        help="Unified figure min height for step1/step2/step3")
    parser.add_argument("--figure-max-height", type=float, default=9.5,
                        help="Unified figure max height for step1/step2/step3")

    parser.add_argument("--step1-grid-linewidth", type=float, default=0.25)
    parser.add_argument("--step1-grid-alpha", type=float, default=0.3)
    parser.add_argument("--step1-mask-preview-max-points", type=int, default=500000)
    
    parser.add_argument('--section-spacing-um', type=str, default='100,200',
                        help='Comma-separated section spacing (µm), aligned to --datasets order')
    parser.add_argument('--interp-spacing-um', type=float, default=20.0,
                        help='Interpolation interval on distance axis (µm)')
    parser.add_argument('--figure-width-1-5', type=float, default=12.0,
                        help='Figure width in inches')
    parser.add_argument('--figure-height-1-5', type=float, default=6.0,
                        help='Figure height in inches')
    parser.add_argument('--line-width', type=float, default=2.0,
                        help='Line width for plots')
    parser.add_argument('--marker-size', type=int, default=8,
                        help='Marker size for plots')
    
    parser.add_argument('--RA', type=float, default=3.0, help='The number of RA per grids')
    parser.add_argument('--TA', type=float, default=4.5, help='The number of TA per grids')
    parser.add_argument('--informative_UMI', type=float, default=0.2, help='Proportion of informative UMI')
    parser.add_argument('--cutoff', type=int, default=200, help='Cutoff for informative UMI number')

    parser.add_argument("--cluster-input-glob", type=str, default="**/*.h5ad")
    parser.add_argument("--embedding-dim", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-n-neighbors", type=int, default=15)
    parser.add_argument("--normalization", type=str, default="log1p_cpm", choices=["none", "log1p_cpm"])
    parser.add_argument("--grid-cmap", type=str, default="tab20")
    parser.add_argument("--limits-padding-ratio", type=float, default=0.03)
    parser.add_argument("--limits-min-pad-um", type=float, default=100.0)
    parser.add_argument("--grid-aggregate", type=str, default="sum", choices=["mean", "sum"])

    parser.add_argument("--reference-datasets", "--reference-dataset", dest="reference_datasets", type=str, 
                        default="WMB-10Xv2,WMB-10Xv3,WMB-10XMulti",
                        help="Reference datasets for step 3, comma-separated (e.g. WMB-10Xv2,WMB-10Xv3,WMB-10XMulti)")
    parser.add_argument("--cell-metadata-file", type=str, default="WMB-10X")
    parser.add_argument("--cluster-col", type=str, default="cluster_alias")
    parser.add_argument("--integration-pcs", type=int, default=50)
    parser.add_argument("--integration-features", type=int, default=2000)
    parser.add_argument("--max-reference-cells", type=int, default=30000)
    parser.add_argument("--enable-reference-downsampling", dest="enable_reference_downsampling", action="store_true", default=True)
    parser.add_argument("--disable-reference-downsampling", dest="enable_reference_downsampling", action="store_false")
    parser.add_argument("--label-transfer-k-weight", type=int, default=50)

    return parser.parse_args()


def run_cmd(cmd, dry_run=False):
    print("\n>>>", " ".join(shlex.quote(str(x)) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_step1_cmd(args, step1_script: Path):
    cmd = [
        args.python_exe,
        str(step1_script),
        "--download-base",
        str(args.download_base),
        "--datasets",
        args.datasets,
        "--division",
        args.division,
        "--grid-block-um",
        str(args.grid_block_um),
        "--grid-gap-um",
        str(args.grid_gap_um),
        "--expression-matrix-kind",
        args.expression_matrix_kind,
        "--point-size",
        str(args.point_size),
        "--dpi",
        str(args.dpi),
        "--figure-width",
        str(args.figure_width),
        "--figure-min-height",
        str(args.figure_min_height),
        "--figure-max-height",
        str(args.figure_max_height),
        "--grid-linewidth",
        str(args.step1_grid_linewidth),
        "--grid-alpha",
        str(args.step1_grid_alpha),
        "--mask-preview-max-points",
        str(args.step1_mask_preview_max_points),
    ]

    cmd.append("--show-sampling-grid" if args.step1_show_sampling_grid else "--hide-sampling-grid")
    cmd.append("--export-sampled-h5ad" if args.step1_export_sampled_h5ad else "--no-export-sampled-h5ad")
    cmd.append("--export-sampling-mask" if args.step1_export_sampling_mask else "--no-export-sampling-mask")
    return cmd


def build_step2_cmd(args, step2_script: Path, cluster_input_dir: Path, cluster_output_dir: Path):
    cmd = [
        args.python_exe,
        str(step2_script),
        "--input-dir",
        str(cluster_input_dir),
        "--output-dir",
        str(cluster_output_dir),
        "--input-glob",
        args.cluster_input_glob,
        "--embedding-dim",
        str(args.embedding_dim),
        "--random-state",
        str(args.random_state),
        "--leiden-resolution",
        str(args.leiden_resolution),
        "--leiden-n-neighbors",
        str(args.leiden_n_neighbors),
        "--normalization",
        args.normalization,
        "--grid-cmap",
        args.grid_cmap,
        "--grid-dpi",
        str(args.dpi),
        "--figure-width",
        str(args.figure_width),
        "--figure-min-height",
        str(args.figure_min_height),
        "--figure-max-height",
        str(args.figure_max_height),
        "--limits-padding-ratio",
        str(args.limits_padding_ratio),
        "--limits-min-pad-um",
        str(args.limits_min_pad_um),
        "--grid-block-um",
        str(args.grid_block_um),
        "--grid-gap-um",
        str(args.grid_gap_um),
        "--grid-aggregate",
        args.grid_aggregate,
    ]
    return cmd

def build_step1_5_cmd(args, step1_5_script: Path, input_dir: Path, output_dir: Path):
    cmd = [
        args.python_exe,
        str(step1_5_script),
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--datasets", args.datasets,
        "--section-spacing-um", str(args.section_spacing_um),
        "--interp-spacing-um", str(args.interp_spacing_um),
        "--dpi", str(args.dpi),
        "--figure-width-1-5", str(args.figure_width_1_5),
        "--figure-height-1-5", str(args.figure_height_1_5),
        "--line-width", str(args.line_width),
        "--marker-size", str(args.marker_size),
    ]
    return cmd


def build_step_simulation_cmd(args, simulation_script: Path, input_file: Path):
    cmd = [
        args.python_exe,
        str(simulation_script),
        "--input", str(input_file),
        "--RA", str(args.RA),
        "--TA", str(args.TA),
        "--informative_UMI", str(args.informative_UMI),
        "--cutoff", str(args.cutoff),
        "--dpi", str(args.dpi),
    ]
    return cmd


def build_step3_cmd(args, step3_script: Path):
    cmd = [
        args.python_exe,
        str(step3_script),
        "--download-base",
        str(args.download_base),
        "--reference-datasets",
        args.reference_datasets,
        "--expression-matrix-kind",
        args.expression_matrix_kind,
        "--division",
        args.division,
        "--cell-metadata-file",
        args.cell_metadata_file,
        "--cluster-col",
        args.cluster_col,
        "--integration-pcs",
        str(args.integration_pcs),
        "--integration-features",
        str(args.integration_features),
        "--max-reference-cells",
        str(args.max_reference_cells),
        "--label-transfer-k-weight",
        str(args.label_transfer_k_weight),
        "--plot-dpi",
        str(args.dpi),
        "--grid-cmap",
        args.grid_cmap,
        "--figure-width",
        str(args.figure_width),
        "--figure-min-height",
        str(args.figure_min_height),
        "--figure-max-height",
        str(args.figure_max_height),
    ]
    cmd.append("--enable-reference-downsampling" if args.enable_reference_downsampling else "--disable-reference-downsampling")
    return cmd


def main():
    args = parse_args()

    scripts_dir = args.scripts_dir.resolve()
    step1_script = scripts_dir / "ccf_registration_to_image.py"
    step1_5_script = scripts_dir / "analyze_substructure_distribution.py"
    step_simulation_script = scripts_dir / "DARLIN_simulation.py"
    step2_script = scripts_dir / "cluster_sampled_h5ad.py"
    step3_script = scripts_dir / "embedding_merfish.py"

    if not step1_script.exists():
        raise FileNotFoundError(f"Missing step1 script: {step1_script}")
    if not step1_5_script.exists():
        raise FileNotFoundError(f"Missing step1.5 script: {step1_5_script}")
    if not step2_script.exists():
        raise FileNotFoundError(f"Missing step2 script: {step2_script}")
    if args.run_step3 and not step3_script.exists():
        raise FileNotFoundError(f"Missing step3 script: {step3_script}")

    output_root = args.download_base.parent / f"output_{args.expression_matrix_kind}"
    cluster_input_dir = output_root / "sampled_h5ad"
    cluster_output_dir = output_root / "cluster_results"

    print("=" * 72)
    print("MERFISH Pipeline Controller")
    print("=" * 72)
    print(f"scripts_dir        : {scripts_dir}")
    print(f"python_exe         : {args.python_exe}")
    print(f"download_base      : {args.download_base}")
    print(f"cluster_input_dir  : {cluster_input_dir}")
    print(f"cluster_output_dir : {cluster_output_dir}")
    print(f"skip_step1         : {args.skip_step1}")
    print(f"skip_step2         : {args.skip_step2}")
    print(f"run_step3          : {args.run_step3}")
    print(f"dry_run            : {args.dry_run}")

    if args.skip_step1 and args.skip_step2 and (not args.run_step3) and args.skip_step1_5:
        print("All selected steps are skipped. Nothing to run.")
        return

    if not args.skip_step1:
        step1_cmd = build_step1_cmd(args, step1_script)
        run_cmd(step1_cmd, dry_run=args.dry_run)

    # Step 1.5: Analyze substructure distribution (optional, but recommended if step 1 is run)
    if not args.skip_step1_5:
        step1_5_input_dir = args.download_base.parent / f"output_{args.expression_matrix_kind}" / "sampled_h5ad"
        step1_5_output_dir = args.download_base.parent / f"output_{args.expression_matrix_kind}" / "analysis_substructure_distribution"
        step1_5_cmd = build_step1_5_cmd(args, step1_5_script, step1_5_input_dir, step1_5_output_dir)
        run_cmd(step1_5_cmd, dry_run=args.dry_run)

        step_simulation_input = step1_5_output_dir / f'substructure_span_summary_{args.interp_spacing_um}um.csv'
        step_simulation_cmd = build_step_simulation_cmd(args, step_simulation_script, step_simulation_input)
        run_cmd(step_simulation_cmd, dry_run=args.dry_run)


    if not args.skip_step2:
        step2_cmd = build_step2_cmd(args, step2_script, cluster_input_dir, cluster_output_dir)
        run_cmd(step2_cmd, dry_run=args.dry_run)

    if args.run_step3:
        if not args.reference_datasets:
            raise ValueError("--run-step3 requires --reference-datasets (or --reference-dataset).")
        step3_cmd = build_step3_cmd(args, step3_script)
        run_cmd(step3_cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
