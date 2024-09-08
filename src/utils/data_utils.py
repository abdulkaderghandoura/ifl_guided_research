from pathlib import Path
import trimesh
import numpy as np


def ply_to_obj(source_file_path, target_file_path):
    # Load the PLY file
    pcd = trimesh.load(str(source_file_path), file_type='ply')

    # Write to OBJ file
    pcd.export(str(target_file_path), file_type='obj')


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
    mesh.export(target_file_path)  # TODO: Test also for point cloud

    # # Write the transformed vertices to a new .obj file manually
    # with open(target_file_path, 'w') as file:
    #     for vertex in transformed_vertices:
    #         file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")


if __name__ == "__main__":
    repo_dir_path = Path(__file__).resolve().parent.parent.parent
    gen_mitea_dir_path = repo_dir_path / "data" / "mitea" / "generated"

    # source_file_path = "path/to/source/file"
    # target_file_path = "path/to/target/file" 
    # ply_to_obj(source_file_path, target_file_path)

    seq_id = "005"
    phase_id = "00"
    source_file_path = gen_mitea_dir_path / "scan1" / seq_id / "points" / f"{phase_id}.obj"
    target_file_path = repo_dir_path / "notebooks" / f"mitea_{seq_id}_{phase_id}_transformed.obj"
    mat_file_path = gen_mitea_dir_path / "scan1" / seq_id / "P.txt"
    apply_transform_mat(source_file_path, target_file_path, mat_file_path)
