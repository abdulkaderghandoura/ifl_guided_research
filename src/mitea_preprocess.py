from pathlib import Path
from tqdm import tqdm
import numpy as np
import nibabel as nib
import open3d as o3d
import trimesh
from skimage import measure
from skimage.morphology import binary_dilation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay

import pyvista as pv

repo_dir_path = Path(__file__).resolve().parent.parent


def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine


def extract_lv_contours(volume_data):
    """
    Extract LV contours from the MRI volume data where the mask equals 1.
    """
    contours = []
    for i in range(volume_data.shape[0]):
        slice_data = volume_data[i, :, :]
        # Only consider the slice data where mask == 1
        binary_slice = (slice_data == 1)
        # Extract contours from this binary mask
        contour = measure.find_contours(binary_slice, 0.5)

        # Only append if contours are found
        if contour:
            contours.append(contour)

    return contours


def save_wireframes_to_obj(contours, output_file, slice_spacing, affine):
    """
    Save wireframes as points in an OBJ file, ensuring horizontal alignment
    and applying the affine transformation.
    """
    with open(output_file, 'w') as obj_file:
        for i in range(0, len(contours), slice_spacing):
            slice_contours = contours[i]
            z = i  # Slice index used as the depth coordinate (Z in the horizontal plane)
            for contour in slice_contours:
                for point in contour:
                    # Apply affine transformation
                    x, y = point[1], point[0]
                    coord = np.array([x, y, z, 1.0])
                    transformed_coord = np.dot(affine, coord)[:3]
                    obj_file.write(f"v {transformed_coord[0]} {transformed_coord[1]} {transformed_coord[2]}\n")


def extract_points(data_scan, mask, affine, cnt_threshold=None):
    # Find the indices of the voxels in the mask that are non-zero
    mask_indices = np.array(np.nonzero(mask == 1)).T

    # Convert these indices to physical coordinates
    physical_points = nib.affines.apply_affine(affine, mask_indices)

    # Optionally, retrieve the corresponding intensity values from the CT scan
    intensities = data_scan[mask_indices[:, 0], mask_indices[:, 1], mask_indices[:, 2]]

    # Limit to 5,000 points for visualization
    if cnt_threshold is not None and physical_points.shape[0] > cnt_threshold:
        idx_mask = np.random.choice(physical_points.shape[0], cnt_threshold, replace=False)
        physical_points = physical_points[idx_mask]
        intensities = intensities[idx_mask]
    
    return physical_points, intensities


def create_point_cloud(points, intensities=None):
    # Create an open3d point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign points
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Optionally, assign intensities as colors (gray scale)
    if intensities is not None:
        colors = np.tile(intensities[:, np.newaxis], (1, 3))  # Repeat intensity values for R, G, B
        colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))  # Normalize to [0, 1]
    
    else:
        default_color = [0.83, 0.83, 0.83]  # RGB for light gray
        colors = np.tile(default_color, (len(points), 1))
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def extract_mesh(mask, affine, threshold=0.5):
    # Extract the surface mesh using marching cubes
    verts, faces, _, _ = measure.marching_cubes(mask == 1, level=threshold)
    
    # Convert voxel indices to physical coordinates
    physical_points = nib.affines.apply_affine(affine, verts)
    
    return physical_points, faces


def create_mesh(vertices, faces):
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh


def save_point_cloud(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)


def save_mesh(mesh, file_path):
    mesh.export(file_path)


def generate_pcd(raw_data_dir_path, generated_data_dir_path):
    image_paths = list((raw_data_dir_path / "images").glob("*.nii.gz"))
    for image_path in tqdm(image_paths):
        label_path = raw_data_dir_path / "labels" / image_path.name

        # Load 3DE scan and mask files
        MITEA_scan, MITEA_affine = load_nifti(str(image_path))
        mask, mask_affine = load_nifti(str(label_path))

        # Ensure both files have the same shape
        assert MITEA_scan.shape == mask.shape
        # Assert that the affine matrices are approximately equal
        assert np.allclose(MITEA_affine, mask_affine)

        # Extract point cloud from the mask
        points, intensities = extract_points(MITEA_scan, mask, MITEA_affine, cnt_threshold=5000)

        # Create the point cloud
        MITEA_gt_pcd = create_point_cloud(points, intensities)

        pcd_dir_path = generated_data_dir_path / "points"
        label_file_name = Path(label_path.stem).stem

        # Save the point cloud to a file
        save_point_cloud(MITEA_gt_pcd, str(pcd_dir_path / (label_file_name + ".ply")))

        # Load the PLY file
        pcd = trimesh.load(str(pcd_dir_path / (label_file_name + ".ply")), file_type='ply')

        # Write to OBJ file
        pcd.export(str(pcd_dir_path / (label_file_name + ".obj")), file_type='obj')


def generate_pcd_and_mesh(raw_data_dir_path, generated_data_dir_path):
    image_paths = list((raw_data_dir_path / "images").glob("*.nii.gz"))
    for image_path in tqdm(image_paths):
        label_path = raw_data_dir_path / "labels" / image_path.name

        # Load 3DE scan and mask files
        MITEA_scan, MITEA_affine = load_nifti(str(image_path))
        mask, mask_affine = load_nifti(str(label_path))

        # Ensure both files have the same shape
        assert MITEA_scan.shape == mask.shape
        # Assert that the affine matrices are approximately equal
        assert np.allclose(MITEA_affine, mask_affine)

        # Extract point cloud from the mask
        points, intensities = extract_points(MITEA_scan, mask, MITEA_affine, cnt_threshold=5000)
        MITEA_gt_pcd = create_point_cloud(points, intensities)

        # Extract mesh from the mask
        vertices, faces = extract_mesh(mask, MITEA_affine)
        MITEA_gt_mesh = create_mesh(vertices, faces)

        # Define directories and filenames
        pcd_dir_path = generated_data_dir_path / "points"
        mesh_dir_path = generated_data_dir_path / "meshes"
        label_file_name = Path(label_path.stem).stem

        # Save the point cloud
        save_point_cloud(MITEA_gt_pcd, str(pcd_dir_path / (label_file_name + ".ply")))

        # Save the mesh
        save_mesh(MITEA_gt_mesh, str(mesh_dir_path / (label_file_name + ".ply")))

        # Optionally, save the mesh in .obj format
        save_mesh(MITEA_gt_mesh, str(mesh_dir_path / (label_file_name + ".obj")))


if __name__ == "__main__":
    data_dir_path = repo_dir_path / "data" / "mitea"
    raw_data_dir_path = data_dir_path / "raw"
    generated_data_dir_path = data_dir_path / "generated"
    # generate_pcd(raw_data_dir_path, generated_data_dir_path)
    # print("Generating point cloud is done!")



    # generate_pcd_and_mesh(raw_data_dir_path, generated_data_dir_path)
    # print("Generating point clouds and meshes is done!")

    file_path = com_dir = '../notebooks/MITEA_001_scan1_ED_label.nii.gz'
    volume_data, affine = load_nifti(file_path)
    lv_contours = extract_lv_contours(volume_data)
    # generate_wireframes(lv_contours)
    
    save_wireframes_to_obj(lv_contours, 'lv_wireframes.obj', slice_spacing=4, affine=affine)

