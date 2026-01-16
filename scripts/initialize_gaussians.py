"""
Initialize Gaussian Parameters
Combines all parameters into a single NPZ file for 3D Gaussian Splatting.

This script creates the initial Gaussian parameters from COLMAP data:
- means: XYZ positions from the point cloud
- scales: computed from nearest neighbor distances
- rotations: identity quaternions (no rotation initially)
- opacities: constant starting value
- colors: RGB values from the point cloud
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    project_root = Path(__file__).parent.parent
    sparse_dir = project_root / "sparse" / "0"
    output_dir = project_root / "outputs"

    ply_path = sparse_dir / "points3D.ply"
    scales_path = output_dir / "scales.npz"
    output_npz = output_dir / "gaussian_params.npz"
    stats_path = output_dir / "stats.txt"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing Gaussian parameters...")
    print(f"Working in: {project_root}")
    print()

    print("Loading point cloud data...")

    if not ply_path.exists():
        print(f"PLY file not found: {ply_path}")
        print("Make sure COLMAP reconstruction is complete")
        return 1

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        means = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
    except Exception as e:
        print(f"Failed to load PLY file: {e}")
        return 1

    n_points = len(means)
    if n_points == 0:
        print("Point cloud is empty!")
        return 1

    print(f"Loaded {n_points:,} points")

    if len(colors) == 0:
        print("No colors found, using default gray")
        colors = np.full((n_points, 3), 0.5, dtype=np.float32)
    else:
        print("Colors loaded successfully")

    print()

    print("Loading scales...")

    if not scales_path.exists():
        print(f"Scales file not found: {scales_path}")
        print("Run compute_scales.py first")
        return 1

    try:
        scales_data = np.load(scales_path)
        scales = scales_data['scales']
    except Exception as e:
        print(f"Failed to load scales: {e}")
        return 1

    if len(scales) != n_points:
        print(f"Scale count mismatch: got {len(scales)}, expected {n_points}")
        return 1

    print(f"Loaded {len(scales)} scales")
    print()

    print("Setting up Gaussian parameters...")

    # Start with identity rotations (no rotation)
    rotations = np.zeros((n_points, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1 for identity quaternion

    opacities = np.full(n_points, 0.1, dtype=np.float32)

    # Make sure all arrays are float32
    means = means.astype(np.float32)
    scales = scales.astype(np.float32)
    colors = colors.astype(np.float32)

    print("Parameters created:")
    print(f"  positions: {means.shape}")
    print(f"  scales:    {scales.shape}")
    print(f"  rotations: {rotations.shape}")
    print(f"  opacities: {opacities.shape}")
    print(f"  colors:    {colors.shape}")
    print()

    print("Validating parameters...")

    errors = []

    # Check scales are positive
    if np.any(scales <= 0):
        errors.append(f"Found {np.sum(scales <= 0)} non-positive scales")

    # Check opacities are in valid range
    if np.any((opacities < 0) | (opacities > 1)):
        errors.append("Opacities outside [0, 1] range")

    # Check colors are in valid range
    if np.any((colors < 0) | (colors > 1)):
        errors.append("Colors outside [0, 1] range")

    # Check rotations are identity
    if not (np.allclose(rotations[:, 0], 1.0) and np.allclose(rotations[:, 1:], 0.0)):
        errors.append("Rotations are not identity quaternions")

    # Check for NaN values
    for name, array in [('means', means), ('scales', scales), ('rotations', rotations),
                       ('opacities', opacities), ('colors', colors)]:
        if np.any(np.isnan(array)):
            errors.append(f"NaN values in {name}")

    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("All checks passed")
    print()

    print("Saving results...")

    try:
        np.savez_compressed(
            output_npz,
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors=colors
        )
        print(f"Saved parameters to: {output_npz}")
    except Exception as e:
        print(f"Failed to save parameters: {e}")
        return 1

    summary = f"""
            Gaussian Parameters Summary
            ==========================

            Total Gaussians: {n_points:,}

            Positions:
            X range: [{means[:, 0].min():.3f}, {means[:, 0].max():.3f}]
            Y range: [{means[:, 1].min():.3f}, {means[:, 1].max():.3f}]
            Z range: [{means[:, 2].min():.3f}, {means[:, 2].max():.3f}]
            Center: [{means.mean(axis=0)[0]:.3f}, {means.mean(axis=0)[1]:.3f}, {means.mean(axis=0)[2]:.3f}]

            Scales:
            Range: [{scales.min():.6f}, {scales.max():.6f}]
            Mean: {scales.mean():.6f}
            Median: {np.median(scales):.6f}

            Colors (RGB):
            Mean: [{colors.mean(axis=0)[0]:.3f}, {colors.mean(axis=0)[1]:.3f}, {colors.mean(axis=0)[2]:.3f}]

            Files created:
            - {output_npz.name}
            - {stats_path.name}
    """

    try:
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Saved summary to: {stats_path}")
    except Exception as e:
        print(f"Warning: Could not save summary file: {e}")

    print(summary)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
