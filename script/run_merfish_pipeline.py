#!/usr/bin/env python
"""
Master controller for MERFISH sampling + clustering pipeline.

Step 1: ccf_registration_to_image.py
Step 2: cluster_sampled_h5ad.py
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run MERFISH step-1 + step-2 pipeline")

    parser.add_argument("--python-exe", type=str, default=sys.executable, help="Python executable to run sub-scripts")
    parser.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)

    parser.add_argument("--skip-step1", action="store_true", help="Skip step 1 (sampling/export)")
    parser.add_argument("--skip-step2", action="store_true", help="Skip step 2 (Leiden clustering)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only, do not execute")

    parser.add_argument("--download-base", type=Path)
    parser.add_argument("--datasets", type=str,)
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

    parser.add_argument("--step1-point-size", type=float, default=3)
    parser.add_argument("--step1-dpi", type=int, default=300)
    parser.add_argument("--step1-figure-width", type=float, default=8.0)
    parser.add_argument("--step1-figure-min-height", type=float, default=4.5)
    parser.add_argument("--step1-figure-max-height", type=float, default=9.5)
    parser.add_argument("--step1-grid-linewidth", type=float, default=0.25)
    parser.add_argument("--step1-grid-alpha", type=float, default=0.3)
    parser.add_argument("--step1-mask-preview-max-points", type=int, default=500000)

    parser.add_argument("--cluster-input-dir", type=Path, default=None, help="Default: <download-base>/../output/sampled_h5ad")
    parser.add_argument("--cluster-output-dir", type=Path, default=None, help="Default: <download-base>/../output/cluster_results")
    parser.add_argument("--cluster-input-glob", type=str, default="**/*.h5ad")
    parser.add_argument("--embedding-dim", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-n-neighbors", type=int, default=15)
    parser.add_argument("--normalization", type=str, default="log1p_cpm", choices=["none", "log1p_cpm"])
    parser.add_argument("--grid-cmap", type=str, default="tab20")
    parser.add_argument("--grid-dpi", type=int, default=300)
    parser.add_argument("--figure-width", type=float, default=8.0)
    parser.add_argument("--figure-min-height", type=float, default=4.5)
    parser.add_argument("--figure-max-height", type=float, default=9.5)
    parser.add_argument("--limits-padding-ratio", type=float, default=0.03)
    parser.add_argument("--limits-min-pad-um", type=float, default=100.0)
    parser.add_argument("--grid-aggregate", type=str, default="sum", choices=["mean", "sum"])

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
        str(args.step1_point_size),
        "--dpi",
        str(args.step1_dpi),
        "--figure-width",
        str(args.step1_figure_width),
        "--figure-min-height",
        str(args.step1_figure_min_height),
        "--figure-max-height",
        str(args.step1_figure_max_height),
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
        str(args.grid_dpi),
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


def main():
    args = parse_args()

    scripts_dir = args.scripts_dir.resolve()
    step1_script = scripts_dir / "ccf_registration_to_image.py"
    step2_script = scripts_dir / "cluster_sampled_h5ad.py"

    if not step1_script.exists():
        raise FileNotFoundError(f"Missing step1 script: {step1_script}")
    if not step2_script.exists():
        raise FileNotFoundError(f"Missing step2 script: {step2_script}")

    output_root = args.download_base.parent / "output"
    cluster_input_dir = args.cluster_input_dir if args.cluster_input_dir is not None else (output_root / "sampled_h5ad")
    cluster_output_dir = args.cluster_output_dir if args.cluster_output_dir is not None else (output_root / "cluster_results")

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
    print(f"dry_run            : {args.dry_run}")

    if args.skip_step1 and args.skip_step2:
        print("Both steps are skipped. Nothing to run.")
        return

    if not args.skip_step1:
        step1_cmd = build_step1_cmd(args, step1_script)
        run_cmd(step1_cmd, dry_run=args.dry_run)

    if not args.skip_step2:
        step2_cmd = build_step2_cmd(args, step2_script, cluster_input_dir, cluster_output_dir)
        run_cmd(step2_cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
