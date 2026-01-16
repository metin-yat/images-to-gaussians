# COLMAP Preparation for 3D Gaussian Splatting

This repository provides a complete pipeline for preparing COLMAP data for 3D Gaussian Splatting initialization. The pipeline extracts camera parameters, sparse point clouds, and initializes Gaussian parameters from a set of input images.

## Project Structure

```
colmap_3dgaussian/
├── images/                 # Input images (ignored in .gitignore)
├── database.db            # COLMAP database (generated, ignored)
├── sparse/                # COLMAP sparse reconstruction (generated, ignored)
│   └── 0/
│       ├── cameras.bin
│       ├── frames.bin
│       ├── images.bin
│       ├── points3D.bin
│       └── points3D.ply   # Converted PLY file
├── outputs/               # Processed outputs (generated, ignored)
│   ├── analysis/          # Analysis plots
│   ├── cameras.json       # Extracted camera parameters
│   ├── gaussian_params.npz # Combined Gaussian parameters
│   ├── scales.npz         # Computed scales
│   └── stats.txt          # Statistics summary
├── scripts/               # Python processing scripts
└── requirements.txt       # Python dependencies
```

## Prerequisites

- [COLMAP](https://colmap.github.io/) installed and available in PATH
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) or Miniconda

## Installation

1. Create and activate conda environment:
```bash
conda create -n 3dgs_init python=3.10
conda activate 3dgs_init
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Place Input Images

Put your input images in the `images/` folder. The pipeline assumes all images come from a single camera (single-camera setup).

### Step 2: COLMAP Feature Extraction

Extract features from images and store them in the database:

```bash
colmap feature_extractor \
    --database_path database.db \
    --image_path images \
    --ImageReader.single_camera 1
```

### Step 3: COLMAP Feature Matching

Perform exhaustive matching between all image pairs:

```bash
colmap exhaustive_matcher \
    --database_path database.db
```

### Step 4: COLMAP Sparse Reconstruction

Create sparse directory and perform Structure-from-Motion:

```bash
mkdir sparse

colmap mapper \
    --database_path database.db \
    --image_path images \
    --output_path sparse
```

### Step 5: Convert Binary to PLY

Convert the sparse point cloud from COLMAP's binary format to PLY:

```bash
colmap model_converter \
    --input_path sparse/0 \
    --output_path sparse/0/points3D.ply \
    --output_type PLY
```

### Step 6: Process Data with Python Scripts

Run the Python scripts in sequence to process the COLMAP data:

1. **Extract Camera Parameters:**
```bash
python scripts/extract_cameras.py
```

2. **Compute Scales:**
```bash
python scripts/compute_scales.py
```

3. **Initialize Gaussians:**
```bash
python scripts/initialize_gaussians.py
```

4. **Analyze Results (Optional):**
```bash
python scripts/analyze_ply.py
```

## Output Files

After running the complete pipeline, you'll find:

- **`outputs/cameras.json`**: Camera intrinsics and extrinsics in JSON format
- **`outputs/scales.npz`**: Computed scales for each 3D point
- **`outputs/gaussian_params.npz`**: Combined Gaussian parameters ready for 3DGS training
- **`outputs/stats.txt`**: Summary statistics of the generated parameters
- **`outputs/analysis/`**: Visualization plots (point cloud view, scale histogram)


## Pipeline Overview

```
Images → COLMAP → Sparse Reconstruction → Python Processing → Gaussian Parameters
    ↓         ↓              ↓                      ↓               ↓
 images/   database.db    sparse/0/points3D.ply   scripts/     gaussian_params.npz
```

## Troubleshooting

- **COLMAP commands fail**: Ensure COLMAP is properly installed and in your PATH
- **Python scripts fail**: Make sure the conda environment is activated and all dependencies are installed
- **Missing PLY file**: Ensure Step 5 (model_converter) completed successfully
- **Empty outputs**: Check that your images have sufficient overlap and quality for SfM

## Next Steps

The generated `gaussian_params.npz` file contains initialized Gaussian parameters ready for 3D Gaussian Splatting training in subsequent projects.