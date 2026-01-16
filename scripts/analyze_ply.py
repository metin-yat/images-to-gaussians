"""
Analyze COLMAP point cloud data
Basic quality checks and visualization of sparse reconstruction results
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    project_root = Path(__file__).parent.parent
    ply_path = project_root / "sparse" / "0" / "points3D.ply"
    output_dir = project_root / "outputs" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Analyzing COLMAP point cloud...")
    print(f"Input: {ply_path}")
    print()

    if not ply_path.exists():
        print(f"PLY file not found: {ply_path}")
        print("Run COLMAP model_converter first")
        return 1

    try:
        pcd = o3d.io.read_point_cloud(str(ply_path))
    except Exception as e:
        print(f"Failed to read PLY file: {e}")
        return 1

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if len(points) == 0:
        print("PLY file contains no points!")
        return 1

    print(f"Loaded {len(points):,} points")
    print()

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"Point count: {len(points)}")
    print("Bounding box:")
    print(f"  X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] (size: {bbox_size[0]:.2f})")
    print(f"  Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] (size: {bbox_size[1]:.2f})")
    print(f"  Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}] (size: {bbox_size[2]:.2f})")
    print()

    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print("Point cloud contains invalid values (NaN/Inf)")
        return 1
    else:
        print("No invalid values found")

    if len(colors) > 0:
        color_range = colors.min(), colors.max()
        if not (0 <= color_range[0] <= 1 and 0 <= color_range[1] <= 1):
            print("Warning: Color values outside expected [0,1] range")
        else:
            print("Color values are valid")
    else:
        print("No color information found")

    if len(points) < 1000:
        print(f"Warning: Only {len(points)} points - might need more images")

    print()
    
    # Save a screenshot for documentation
    screenshot_path = output_dir / "point_cloud_view.png"
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(screenshot_path))
        vis.destroy_window()
        print(f"Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"Could not save screenshot: {e}")

    # Show interactive viewer
    print("\nOpening point cloud viewer...")
    print("Close the window when done")

    try:
        print("Viewer opened successfully")
        o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
    except Exception as e:
        print(f"Viewer failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
