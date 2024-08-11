from pathlib import Path
import nibabel as nib
import numpy as np
import open3d as o3d
import trimesh


def visualize_mesh(ply_file_path):
    # Load the mesh
    mesh = o3d.io.read_triangle_mesh(ply_file_path)

    print(mesh)
    
    # Visualize the mesh using Open3D's plotly viewer
    o3d.visualization.draw_plotly([mesh])


def visualize_pcd(ply_file_path):
    # Load the point cloud
    mesh = o3d.io.read_point_cloud(ply_file_path)

    print(mesh)
    
    # Visualize the point cloud using Open3D's plotly viewer
    o3d.visualization.draw_plotly([mesh])


def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine


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


# def extract_points(data_scan, mask, affine):
#     # Find the indices of the voxels in the mask that are non-zero
#     mask_indices = np.array(np.nonzero(mask == 1)).T
    
#     # Convert these indices to physical coordinates
#     physical_points = nib.affines.apply_affine(affine, mask_indices)
    
#     return physical_points


# def create_point_cloud(points):
#     # Create an open3d point cloud object
#     pcd = o3d.geometry.PointCloud()
    
#     points = points[np.random.choice(points.shape[0], 5000, replace=False)]

#     # Assign points
#     pcd.points = o3d.utility.Vector3dVector(points)
    
#     return pcd


def save_point_cloud(pcd, file_path):
    o3d.io.write_point_cloud(file_path, pcd)


if __name__ == "__main__":
    com_dir = '../../notebooks/'

    # # Load the OBJ file
    # gt_mesh = trimesh.load(com_dir + 'meshes/020_00_gt.obj')
    # def_pcd = trimesh.load(com_dir + 'points/020_00_def.obj')
    # gen_pcd = trimesh.load(com_dir + 'points/020_00_gen.obj')

    # # Save the mesh as a PLY file
    # gt_mesh.export(com_dir + 'meshes/020_00_gt.ply')

    # # Save the point cloud as a PLY file
    # def_pcd.export(com_dir + 'points/020_00_def.ply')
    # gen_pcd.export(com_dir + 'points/020_00_gen.ply')

    # visualize_mesh(com_dir + 'meshes/020_00_gt.ply')
    # visualize_pcd(com_dir + 'points/020_00_gen.ply')
    # visualize_pcd(com_dir + 'points/020_00_def.ply')
    # visualize_mesh(com_dir + 'meshes/020_reconstruction/020_00.ply')
    
    # # Load CT scan and mask files
    # ct_scan, ct_affine = load_nifti(com_dir + 'MITEA_001_scan1_ED_image.nii.gz')
    # mask, mask_affine = load_nifti(com_dir + 'MITEA_001_scan1_ED_label.nii.gz')

    # print(ct_scan.shape)

    # # Ensure both files have the same shape and affine
    # assert ct_scan.shape == mask.shape
    # assert np.allclose(ct_affine, mask_affine)

    # # Extract point cloud from the mask
    # points, intensities = extract_colored_points(ct_scan, mask, ct_affine)

    # # Create the point cloud
    # MITEA_gt_pcd_colored = create_colored_point_cloud(points, intensities)

    # # Save the point cloud to a file
    # save_point_cloud(MITEA_gt_pcd_colored, com_dir + 'points/MITEA_001_scan1_ED_colored.ply')

    # # visualize_pcd(com_dir + 'points/MITEA_001_scan1_ED_colored.ply')

    # Load 3DE scan and mask files
    MITEA_scan, MITEA_affine = load_nifti(com_dir + 'MITEA_001_scan1_ED_image.nii.gz')
    mask, mask_affine = load_nifti(com_dir + 'MITEA_001_scan1_ED_label.nii.gz')

    # Ensure both files have the same shape
    assert MITEA_scan.shape == mask.shape
    # Assert that the affine matrices are approximately equal
    assert np.allclose(MITEA_affine, mask_affine)

    # Extract point cloud from the mask
    points, intensities = extract_points(MITEA_scan, mask, MITEA_affine)

    # Create the point cloud
    MITEA_gt_pcd = create_point_cloud(points, intensities, cnt_threshold=None)

    # Save the point cloud to a file
    save_point_cloud(MITEA_gt_pcd, com_dir + 'points/MITEA_001_scan1_ED.ply')
    
    # Optionally, you may want to visualize the point cloud
    visualize_pcd(com_dir + 'points/MITEA_001_scan1_ED.ply')

