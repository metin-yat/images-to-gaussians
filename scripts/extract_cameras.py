"""
Extract camera parameters from COLMAP reconstruction
Convert intrinsics and extrinsics to JSON format
"""

import sys
import json
from pathlib import Path
import numpy as np
import pycolmap


def quaternion_to_rotation_matrix(qvec):
    """
    Convert quaternion to 3x3 rotation matrix.
    
    Args:
        qvec: [qw, qx, qy, qz]
    
    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = qvec
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def main():
    project_root = Path(__file__).parent.parent
    sparse_dir = project_root / "sparse" / "0"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / "cameras.json"

    print("Extracting camera parameters from COLMAP...")
    print(f"Input: {sparse_dir}")
    print()
    
    # Load COLMAP reconstruction
    if not sparse_dir.exists():
        print(f"Sparse directory not found: {sparse_dir}")
        return 1

    try:
        reconstruction = pycolmap.Reconstruction(str(sparse_dir))
    except Exception as e:
        print(f"Failed to load COLMAP reconstruction: {e}")
        return 1

    num_cameras = len(reconstruction.cameras)
    num_images = len(reconstruction.images)

    print(f"Loaded reconstruction with {num_cameras} cameras and {num_images} images")
    print()

    if num_cameras == 0 or num_images == 0:
        print("Reconstruction contains no cameras or images")
        return 1
    
    # Extract camera intrinsics
    camera_id = list(reconstruction.cameras.keys())[0]
    camera = reconstruction.cameras[camera_id]

    # Assuming PINHOLE model (common for phone cameras)
    # Extract PINHOLE parameters: [fx, fy, cx, cy]
    fx, fy, cx, cy = camera.params

    intrinsics = {
        "model": "PINHOLE",
        "width": camera.width,
        "height": camera.height,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy)
    }

    print("Camera intrinsics:")
    print(f"  Model: PINHOLE")
    print(f"  Size: {camera.width} x {camera.height}")
    print(f"  Focal: fx={fx:.2f}, fy={fy:.2f}")
    print(f"  Center: cx={cx:.2f}, cy={cy:.2f}\n")

    extrinsics = []

    for image_id, image in reconstruction.images.items():
        pose = image.cam_from_world()

        tvec = pose.translation  # [tx, ty, tz]
        rotation = pose.rotation # Rotation3d object
        
        # Get rotation matrix directly from pycolmap
        R = rotation.matrix()
        
        # PyCOLMAP uses [x, y, z, w]. Reorder to [w, x, y, z] for Hamilton convention
        xyzw = rotation.quat
        qvec_hamilton = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        extrinsics.append({
            "image_id": int(image_id),
            "image_name": image.name,
            "R": R.tolist(),
            "t": tvec.tolist(),
            "qvec": qvec_hamilton.tolist()
        })

    print(f"Extracted {len(extrinsics)} camera poses")
    print()

    output_data = {
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "metadata": {
            "num_cameras": num_cameras,
            "num_images": num_images,
            "num_points3D": len(reconstruction.points3D)
        }
    }

    try:
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved camera data to: {output_json}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")
        return 1

    invalid_rotations = 0
    for ext in extrinsics:
        R = np.array(ext["R"])
        det = np.linalg.det(R)
        if not np.isclose(det, 1.0, atol=1e-3):
            invalid_rotations += 1

    if invalid_rotations > 0:
        print(f"Warning: {invalid_rotations} rotation matrices may be invalid")
    else:
        print("All rotation matrices valid")

    tvec_norms = [np.linalg.norm(ext["t"]) for ext in extrinsics]
    mean_norm = np.mean(tvec_norms)
    print(f"Translation vector norms: mean={mean_norm:.3f}")

    print(f"\nCamera data saved to: {output_json}")
    return 0
    

if __name__ == "__main__":
    sys.exit(main())
