from pathlib import Path
import trimesh
import numpy as np


def ply_to_obj(source_file_path, target_file_path):
    # Load the PLY file
    mesh = trimesh.load(str(source_file_path), file_type='ply')

    # Write to OBJ file
    mesh.export(str(target_file_path), file_type='obj')


def apply_transform_mat(source_file_path, target_file_path, mat_file_path):
    mesh = trimesh.load(source_file_path)
    vertices = np.array(mesh.vertices)
    transform_mat = np.loadtxt(mat_file_path)

    # Add a column of ones to the vertices to make them homogeneous coordinates
    vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    
    # Apply the transformation matrix to each vertex
    transformed_vertices = vertices_homogeneous @ transform_mat.T

    # Remove the homogeneous coordinate (the last column)
    transformed_vertices = transformed_vertices[:, :3]

    mesh.vertices = transformed_vertices
    mesh.export(target_file_path)


if __name__ == "__main__":
    repo_dir_path = Path(__file__).resolve().parent.parent.parent
    gen_mitea_dir_path = repo_dir_path / "data" / "mitea" / "generated"

    # source_file_path = repo_dir_path / "notebooks" / "020_00.ply"
    # target_file_path = repo_dir_path / "notebooks" / "020_00.obj"
    # ply_to_obj(source_file_path, target_file_path)

    source_file_path = repo_dir_path / "path/to/source.obj"
    target_file_path = repo_dir_path / "path/to/target.obj"
    mat_file_path = repo_dir_path / "path/to/transformation.txt"

    apply_transform_mat(source_file_path, target_file_path, mat_file_path)
