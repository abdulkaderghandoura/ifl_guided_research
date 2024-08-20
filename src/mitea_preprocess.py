from pathlib import Path
from tqdm import tqdm
import numpy as np
import nibabel as nib
from skimage import measure
import cv2


def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine


def extract_contours(volume_data):
    """
    Extract LV contours from the MRI volume data where the mask equals 1.
    """
    inner_contours = []
    outer_contours = []
    for i in range(volume_data.shape[0]):
        slice_data = volume_data[i, :, :]
        # Only consider the slice data where mask == 1
        binary_slice = (slice_data == 1)
        # Extract contours from this binary mask
        slice_contours = measure.find_contours(binary_slice, 0.5)

        if len(slice_contours) == 1:
            outer_contours.append(slice_contours[0])
            inner_contours.append(np.array([]))
        elif len(slice_contours) == 2:
            slice_contours_0 = np.array(slice_contours[0], dtype=np.float32)
            slice_contours_1 = np.array(slice_contours[1], dtype=np.float32)
            if cv2.contourArea(slice_contours_0) > cv2.contourArea(slice_contours_1):
                outer_contours.append(slice_contours[0])
                inner_contours.append(slice_contours[1])
            else:
                inner_contours.append(slice_contours[0])            
                outer_contours.append(slice_contours[1])

    return inner_contours, outer_contours


def save_wireframes(contours, output_file, affine, slice_spacing, format='obj'):
    """
    Save wireframes as points in an OBJ file, ensuring horizontal alignment
    and applying the affine transformation.
    """
    with open(output_file, 'w') as obj_file:
        for slice_idx in range(0, len(contours), slice_spacing):
            contour = contours[slice_idx]
            z = slice_idx  # Slice index used as the depth coordinate (Z in the horizontal plane)
            for point in contour:
                # Apply affine transformation
                x, y = point[1], point[0]
                coord = np.array([x, y, z, 1.0])
                transformed_coord = np.dot(affine, coord)[:3]
                obj_file.write(f"v {transformed_coord[0]} {transformed_coord[1]} {transformed_coord[2]}\n")


def generate_pcd(raw_data_dir_path, generated_data_dir_path):
    label_paths = list((raw_data_dir_path / "labels").glob("*.nii.gz"))
    for label_path in tqdm(label_paths):
        # Load mask files of 3DE scans
        volume_data, affine = load_nifti(label_path)
        inner_contours, outer_contours = extract_contours(volume_data)

        pcd_dir_path = generated_data_dir_path / "points"
        label_file_name = Path(label_path.stem).stem
        inner_lv_wireframes_file_path = pcd_dir_path / (label_file_name + "_inner_" + ".obj")
        outer_lv_wireframes_file_path = pcd_dir_path / (label_file_name + "_outer_" + ".obj")

        save_wireframes(inner_contours, inner_lv_wireframes_file_path, affine, slice_spacing=4)
        save_wireframes(outer_contours, outer_lv_wireframes_file_path, affine, slice_spacing=4)


def generate_mesh(raw_data_dir_path, generated_data_dir_path, smoothing_itr):
    label_paths = list((raw_data_dir_path / "labels").glob("*.nii.gz"))
    for label_path in tqdm(label_paths):
        # Load mask files of 3DE scans
        volume_data, affine = load_nifti(label_path)
        
        # Create a binary mask where the label is 1
        lv_mask = (volume_data == 1)

        # Use the marching cubes algorithm to extract the surface mesh
        verts, faces, normals, _ = measure.marching_cubes(lv_mask, level=0.5)
        
        # Apply affine transformation to the vertices
        verts = np.dot(affine[:3, :3], verts.T).T + affine[:3, 3]

        # Create the mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

        # Perform Laplacian smoothing
        trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_itr)

        mesh_dir_path = generated_data_dir_path / "meshes"
        label_file_name = Path(label_path.stem).stem

        # Save the mesh to a file
        mesh.export(str(mesh_dir_path / (label_file_name + ".obj")))


if __name__ == "__main__":
    repo_dir_path = Path(__file__).resolve().parent.parent
    data_dir_path = repo_dir_path / "data" / "mitea"
    raw_data_dir_path = data_dir_path / "raw"
    generated_data_dir_path = data_dir_path / "generated"

    generate_pcd(raw_data_dir_path, generated_data_dir_path)
    generate_mesh(raw_data_dir_path, generated_data_dir_path, smoothing_itr=50)
    print("Generating point cloud is done!")
