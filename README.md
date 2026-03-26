# MERFISH to DBiT Pipeline

This folder provides a master controller to run the full pipeline end-to-end:

1. Step 1: `ccf_registration_to_image.py`  
   (CCF visualization + grid sampling + sampled h5ad export)
2. Step 1.5: `analyze_substructure_distribution.py`  
   (Quantity analysis + interpolation)
3. Step 2: `cluster_sampled_h5ad.py`  
   (grid-level aggregation + Leiden clustering + single/merged outputs)
4. Step 3 (optional): `embedding_merfish.py`  
  (reference integration with SeuratIntegration + label transfer + section-wise grid plots)

Master entry point: `run_merfish_pipeline.py`

---

## 1. Environment Setup


### Conda (recommended for this workspace)

```bash
conda create -n merfish-dbit python=3.12 -y
conda activate merfish-dbit
pip install numpy pandas matplotlib scipy scikit-learn anndata scanpy abc-atlas-access h5py pybedtools allcools opentsne dask tpot-imblearn
```

### Quick dependency check

```bash
python -c "import anndata, scanpy, abc_atlas_access, sklearn, matplotlib, pandas, numpy, scipy, allcools; print('OK')"
```

---

## 2. Quick Start

From script directory:

```bash
python run_merfish_pipeline.py \
  --download-base /path/to/download \
  --datasets MERFISH_dataset \
  --run-step3 --division brain_region
```

By default, this runs Step 1 + Step 1.5 + Step 2 with the built-in defaults.

Use `--run-step3` to additionally run embedding/integration.

`--run-step3` needs to be performed under the specified `--division`

---

## 3. Input/Output

### Step 1 (default)
- Input base: `<download-base>` If not downloaded in advance, it will be automatically downloaded to the folder provided.
- Output root: `<download-base>/../output_<expression-matrix-kind>`
- Main outputs:
  - `sampling_images/`
  - `sampling_mask/`
  - `sampled_h5ad/`
  - `sampling_stats_*.txt`

### step 1.5 (default)
- Input: `<download-base>/../output_<expression-matrix-kind>/sampled_h5ad`
- Output: `<download-base>/../output_<expression-matrix-kind>/analysis_substructure_distribution`
- Main outputs:
  - `individual_substructure_plots_interp/`
  - `individual_substructure_plots_raw/`
  - `substructure_span_summary.csv`

### Step 2 (default)
- Input: `<download-base>/../output_<expression-matrix-kind>/sampled_h5ad`
- Output: `<download-base>/../output_<expression-matrix-kind>/cluster_results`
- Main outputs:
  - `single_h5ad/`
  - `single_grid_png/`
  - `merged_h5ad/`
  - `merged_grid_png/`
  - `reports/single_clustering_report.csv`
  - `reports/merged_clustering_report.csv`

### Results
<p align="center">
    <a href="docs/image/cell.png">
        <img src="docs/image/cell.png" height="300" />
    </a>
    <a href="docs/image/grid.png">
        <img src="docs/image/grid.png" height="300" />
    </a>
</p>

<p align="center">
    <sub>Left: Sampling results | Right: Cluster (click images for full size)</sub>
</p>

### Step 3 outputs (when `--run-step3` is enabled)
- Input: `<download-base>/../output_<expression-matrix-kind>/cluster_results/merged_h5ad`
- Output: `<download-base>/../output_<expression-matrix-kind>/embedding`
- Main outputs:
  - `combined_reference.h5ad`
  - `combined_merfish.h5ad`
  - `reference_merfish_integrated_obs.h5ad`
  - `merfish_with_transferred_cluster.h5ad`
  - `plots/merfish_all_datasets_grid_cluster.png`
  - `plots/brain_section_*_grid_cluster.png`

---

## 4. Common Commands

### 4.1 Run full pipeline (HY, 20/20 sampling)

```bash
python run_merfish_pipeline.py \
  --download-base /path/to/download
  --datasets MERFISH_dataset
  --division HY \
  --grid-block-um 20 \
  --grid-gap-um 20 \
  --leiden-resolution 1.0 \
  --leiden-n-neighbors 15
```

### 4.2 Run full pipeline + Step 3 integration

```bash
python run_merfish_pipeline.py \
  --download-base /path/to/download \
  --datasets Zhuang-ABCA-1,Zhuang-ABCA-2 \
  --division HY \
  --expression-matrix-kind log2 \
  --run-step3 \
  --reference-dataset WMB-10Xv2,WMB-10Xv3,WMB-10XMulti
```

### 4.3 Run only Step 2 (reuse existing sampled_h5ad)

```bash
python run_merfish_pipeline.py \
  --skip-step1 \
  --download-base /path/to/download
```

### 4.4 Run only Step 1

```bash
python run_merfish_pipeline.py \
  --skip-step2 \
  --download-base /path/to/download
```

### 4.5 Dry-run (print commands only)

```bash
python run_merfish_pipeline.py \
  --download-base /path/to/download \
  --dry-run
```

---

## 5. Key Parameters (Master Script)

### Global
- `--python-exe`: Python executable used to launch sub-scripts
- `--scripts-dir`: script directory (defaults to current file location)
- `--skip-step1` / `--skip-step1-5` / `--skip-step2`: skip a step
- `--run-step3`: enable optional embedding/integration step
- `--dry-run`: print commands without execution
- Unified plotting args:
  - `--dpi` (used by Step1/Step1.5/Step2/Step3)
  - `--figure-width` / `--figure-min-height` / `--figure-max-height`

### Step 1
- `--download-base`
- `--datasets`
- `--division` (e.g. `HY` or `ALL`)
- `--grid-block-um` / `--grid-gap-um`
- `--expression-matrix-kind` (`raw` / `log2`)
- `--step1-show-sampling-grid` / `--step1-hide-sampling-grid`
- `--step1-export-sampled-h5ad` / `--step1-no-export-sampled-h5ad`
- `--step1-export-sampling-mask` / `--step1-no-export-sampling-mask`

### Step 1.5
- `--datasets`
- `--section-spacing-um`
- `--interp-spacing-um`
- `-figure-width-1-5`
- `--figure-height-1-5`
- `--line-width`
- `-marker-size`

### Step 2
- `--cluster-input-dir` / `--cluster-output-dir`
- `--cluster-input-glob`
- `--leiden-resolution`
- `--leiden-n-neighbors`
- `--embedding-dim`
- `--normalization` (`none` / `log1p_cpm`)
- `--grid-aggregate` (`sum` / `mean`)

### Step 3 (optional)
- `--reference-datasets` / `--reference-dataset`
- `--cell-metadata-file`
- `--cluster-col`
- `--integration-pcs`
- `--integration-features`
- `--max-reference-cells`
- `--enable-reference-downsampling` / `--disable-reference-downsampling`
- `--label-transfer-k-weight`
- `--embedding-script` (override path to `embedding_merfish.py`)


---


## 6. Notes

- Step 2 groups samples automatically using relative paths under `--cluster-input-dir`.
- If you changed Step 1 output locations, explicitly set `--cluster-input-dir`.
- Keep `grid-block-um / grid-gap-um` consistent across both steps to ensure geometric alignment.
- Step 3 reads merged clustered h5ad from Step 2 output; run Step 2 first (or provide existing outputs).
- For memory-limited environments, keep reference downsampling enabled (default) and tune `--max-reference-cells`.

---

## Attribution

This project utilizes data from the **Allen Brain Cell Atlas (ABC Atlas)**. 

- **Data Source:** Allen Institute for Brain Science ([Link](https://alleninstitute.github.io/abc_atlas_access/intro.html)).
- **Dataset Used in This Workflow:** Whole Mouse Brain (WMB) dataset ([Link](https://alleninstitute.github.io/abc_atlas_access/descriptions/WMB_dataset.html)). Zhuang-ABCA dataset ([Link](https://alleninstitute.github.io/abc_atlas_access/descriptions/Zhuang_dataset.html)).
- **Access Tool:** We use the [abc_atlas_access](https://github.com/AllenInstitute/abc_atlas_access) library. (manifest version = 20260228)
- **License Note:** Data is provided under the Allen Institute Terms of Use. Please cite the primary ABC Atlas publication when using this pipeline.