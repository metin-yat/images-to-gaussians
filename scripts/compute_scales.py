"""
Compute initial scales for 3D Gaussians
Uses K-Nearest Neighbors (K=3) approach from 3D Gaussian Splatting paper
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


def main():
    project_root = Path(__file__).parent.parent
    ply_path = project_root / "sparse" / "0" / "points3D.ply"
    output_dir = project_root / "outputs"
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    output_npz = output_dir / "scales.npz"
    histogram_path = analysis_dir / "scale_histogram.png"

    print("Computing Gaussian scales...")
    print(f"Input: {ply_path}")
    print()

    if not ply_path.exists():
        print(f"PLY file not found: {ply_path}")
        return 1

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
    except Exception as e:
        print(f"Failed to load PLY: {e}")
        return 1

    N = len(points)

    if N == 0:
        print("Point cloud is empty")
        return 1

    print(f"Loaded {N:,} points")
    print()

    print("Computing K-Nearest Neighbors (K=3)...")

    K = 3

    try:
        nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto').fit(points)
        distances, indices = nbrs.kneighbors(points)
        neighbor_distances = distances[:, 1:]  # Skip self-distance

    except Exception as e:
        print(f"KNN computation failed: {e}")
        return 1

    print(f"KNN computed for {N:,} points")
    print()
    
    
    # Scale = mean distance to K nearest neighbors
    scales = neighbor_distances.mean(axis=1)

    # Statistics
    scale_min = scales.min()
    scale_max = scales.max()
    scale_mean = scales.mean()
    scale_median = np.median(scales)
    scale_std = scales.std()

    print("Scale statistics:")
    print(f"  Min:    {scale_min:.6f}")
    print(f"  Max:    {scale_max:.6f}")
    print(f"  Mean:   {scale_mean:.6f}")
    print(f"  Median: {scale_median:.6f}")
    print(f"  Std:    {scale_std:.6f}")

    non_positive = np.sum(scales <= 0)
    if non_positive > 0:
        print(f"{non_positive} points have non-positive scale")
        return 1

    if scale_max / scale_mean > 100:
        print(f"Warning: Large scale variation (max/mean = {scale_max/scale_mean:.1f})")
        print("  May indicate outliers in point cloud")

    if np.any(np.isnan(scales)):
        print("NaN values found in scales")
        return 1

    print("Validation passed")

    try:
        np.savez_compressed(output_npz, scales=scales)
        print(f"Saved scales to: {output_npz}")
    except Exception as e:
        print(f"Failed to save scales: {e}")
        return 1

    try:
        plt.figure(figsize=(10, 6))
        plt.hist(scales, bins=50, edgecolor='black', alpha=0.7)
        plt.axvline(scale_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {scale_mean:.6f}')
        plt.axvline(scale_median, color='green', linestyle='--', linewidth=2, label=f'Median: {scale_median:.6f}')
        plt.xlabel('Scale (σ)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Gaussian Scale Distribution (K=3 Nearest Neighbors)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(histogram_path, dpi=150)
        plt.close()

        print(f"Saved histogram to: {histogram_path}")
    except Exception as e:
        print(f"Could not create histogram: {e}")

    print(f"\nComputed scales for {N:,} points")
    return 0
    

if __name__ == "__main__":
    sys.exit(main())