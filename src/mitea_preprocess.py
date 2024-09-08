import time
from pathlib import Path
from tqdm import tqdm
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh
import cv2


class Timer:
    """Class for measuring execution time."""
    def __init__(self):
        self.start_time = None

    def start(self):
        """Starts the timer."""
        self.start_time = time.time()

    def report(self):
        """Ends the timer and report the elapsed time in seconds.

        Raises:
            RuntimeError: If the timer has not been started.
        """
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        end_time = time.time()
        execution_time = end_time - self.start_time
        print(f"Execution time: {execution_time} seconds")


def load_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    affine = nii.affine
    return data, affine


def extract_contours(volume_data, inner_outer_sep):
    """
    Extract LV contours from the MRI volume data where the mask equals 1.
    """
    inner_contours = []
    outer_contours = []
    contours = []

    for i in range(volume_data.shape[0]):
        slice_data = volume_data[i, :, :]
        # Only consider the slice data where mask == 1
        binary_slice = (slice_data == 1)
        # Extract contours from this binary mask
        slice_contours = measure.find_contours(binary_slice, 0.5)

        # Add inner and outer contours together
        if not inner_outer_sep:
            contours.append(slice_contours)
        
        else:
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
            
            else:  # TODO: Check if this step is necessary
                inner_contours.append(np.array([]))
                outer_contours.append(np.array([]))

    if not inner_outer_sep:
        return contours

    return inner_contours, outer_contours


def save_wireframes(contours, output_file, affine, slice_spacing, inner_outer_sep, format='obj'):
    """
    Save wireframes as points in an OBJ file, ensuring horizontal alignment
    and applying the affine transformation.
    """
    with open(output_file, 'w') as obj_file:
        for slice_idx in range(0, len(contours), slice_spacing):
            slice_contours = contours[slice_idx]
            z = slice_idx  # Slice index used as the depth coordinate (Z in the horizontal plane)

            if inner_outer_sep:
                slice_contours = [slice_contours]

            for contour in slice_contours:
                for point in contour:
                    # Apply affine transformation
                    x, y = point[1], point[0]
                    coord = np.array([x, y, z, 1.0])
                    transformed_coord = np.dot(affine, coord)[:3]
                    obj_file.write(f"v {transformed_coord[0]} {transformed_coord[1]} {transformed_coord[2]}\n")


def generate_pcd(raw_data_dir_path, generated_data_dir_path, slice_spacing=4, inner_outer_sep=False):
    label_paths = list((raw_data_dir_path / "labels").glob("*.nii.gz"))
    # label_paths = [raw_data_dir_path / "labels" / "MITEA_001_scan1_ED.nii.gz"]
    for label_path in tqdm(label_paths):
        # Load mask files of 3DE scans
        volume_data, affine = load_nifti(label_path)
        contours = extract_contours(volume_data, inner_outer_sep)

        label_file_name = Path(label_path.stem).stem
        subject_id = label_file_name.split('_')[1]
        scan_id = label_file_name.split('_')[2]
        phase_id = "00" if label_file_name.split('_')[3] == "ED" else "01"

        pcd_dir_path = generated_data_dir_path / scan_id / subject_id / "points"
        pcd_dir_path.mkdir(parents=True, exist_ok=True)

        if not inner_outer_sep:
            lv_wireframes_file_path = pcd_dir_path / f"{phase_id}.obj"
            save_wireframes(contours, lv_wireframes_file_path, affine, slice_spacing, False)

        else:
            inner_contours, outer_contours = contours
            inner_lv_wireframes_file_path = pcd_dir_path / f"{phase_id}_inner.obj"
            outer_lv_wireframes_file_path = pcd_dir_path / f"{phase_id}_outer.obj"

            save_wireframes(inner_contours, inner_lv_wireframes_file_path, affine, slice_spacing, True)
            save_wireframes(outer_contours, outer_lv_wireframes_file_path, affine, slice_spacing, True)


def generate_mesh(raw_data_dir_path, generated_data_dir_path, smoothing_itr):
    label_paths = list((raw_data_dir_path / "labels").glob("*.nii.gz"))
    # label_paths = [raw_data_dir_path / "labels" / "MITEA_001_scan1_ED.nii.gz"]
    for label_path in tqdm(label_paths):
        # Load mask files of 3DE scans
        volume_data, affine = load_nifti(label_path)
        
        # Create a binary mask where the label is 1
        lv_mask = (volume_data == 1)

        # Use the marching cubes algorithm to extract the surface mesh
        vertices, faces, normals, _ = measure.marching_cubes(lv_mask, level=0.5)

        # Modify vertices to match the coordinate system used in save_wireframes
        vertices = vertices[:, [2, 1, 0]]  # (z, y, x) -> (x, y, z)

        # # Apply affine transformation to the vertices
        # verts = np.dot(affine[:3, :3], verts.T).T + affine[:3, 3]

        # Add a column of ones to the vertices to make them homogeneous coordinates
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

        # Apply the transformation to each vertex and remove the homogeneous coordinate (the last column)
        transformed_vertices = (vertices_homogeneous @ affine.T)[:, :3]

        # Create the mesh
        mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=faces, vertex_normals=normals)

        # Perform Laplacian smoothing
        trimesh.smoothing.filter_laplacian(mesh, iterations=smoothing_itr)

        label_file_name = Path(label_path.stem).stem
        subject_id = label_file_name.split('_')[1]
        scan_id = label_file_name.split('_')[2]
        phase_id = "00" if label_file_name.split('_')[3] == "ED" else "01"

        mesh_dir_path = generated_data_dir_path / scan_id / subject_id / "mesh"
        mesh_dir_path.mkdir(parents=True, exist_ok=True)

        # Save the mesh to a file
        mesh.export(str(mesh_dir_path / f"{phase_id}.obj"))


if __name__ == "__main__":
    repo_dir_path = Path(__file__).resolve().parent.parent
    data_dir_path = repo_dir_path / "data" / "mitea"
    raw_data_dir_path = data_dir_path / "raw"
    generated_data_dir_path = data_dir_path / "generated"

    timer = Timer()
    timer.start()

    generate_pcd(raw_data_dir_path, generated_data_dir_path, slice_spacing=4)
    print("Generating points is done!")

    generate_mesh(raw_data_dir_path, generated_data_dir_path, smoothing_itr=50)
    print("Generating meshes is done!")

    timer.report()
