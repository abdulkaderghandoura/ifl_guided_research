from pathlib import Path
import numpy as np
import trimesh


def get_max_distance(gen_mitea_dir_path):
    max_distance = 0
    # Iterate over all directories at depth 2
    for directory in gen_mitea_dir_path.glob('*/' * 2):
        if directory.is_dir():
            for obj_file_path in list((directory / "points").glob("*.obj")):
                
                # Load the point cloud and translate it to the origin
                mesh = trimesh.load(obj_file_path)
                vertices = np.array(mesh.vertices)
                vertices -= np.mean(vertices, axis=0)

                centroid = np.mean(vertices, axis=0)
                assert np.all(np.abs(centroid) < 1e-9), f"The centroid {centroid} is not (0, 0, 0)"

                # Calculate the maximum distance from the origin
                max_distance = max(max_distance, np.max(np.linalg.norm(vertices, axis=1)))
    
    return max_distance


def generate_transformation_mat(gen_mitea_dir_path, max_distance):
    # Iterate over all directories at depth 2
    for directory in gen_mitea_dir_path.glob('*/' * 2):
        if directory.is_dir():
            ED_obj_file_path = directory / "points" / "00.obj"
            mat_file_path = directory / "P.txt"
            transform_point_cloud(ED_obj_file_path, mat_file_path, max_distance)


def transform_point_cloud(obj_file_path, mat_file_path, max_distance, transformed_obj_file_path=None):    
    # Load the point cloud using trimesh
    mesh = trimesh.load(obj_file_path)
    
    # Ensure it's a point cloud (vertices only, no faces)
    if not mesh.is_empty and mesh.vertices is not None:
        vertices = np.array(mesh.vertices)

        # Rotation matrix (180 degrees around x-axis)
        R_x = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

        # Compute the centroid of the point cloud
        centroid = np.mean(vertices, axis=0)
        
        # Translation matrix to center the point cloud at the origin
        T = np.array([
            [1, 0, 0, -centroid[0]],
            [0, 1, 0, -centroid[1]],
            [0, 0, 1, -centroid[2]],
            [0, 0, 0, 1]
        ])

        # Add a column of ones to the vertices to make them homogeneous coordinates
        vertices_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

        # Apply the translation and rotation to each vertex
        transformed_vertices = vertices_homogeneous @ (R_x @ T).T
        
        # Remove the homogeneous coordinate (the last column)
        transformed_vertices = transformed_vertices[:, :3]

        # Scaling matrix to fit vertices within a unit ball
        S = np.array([
            [1/max_distance, 0, 0, 0],
            [0, 1/max_distance, 0, 0],
            [0, 0, 1/max_distance, 0],
            [0, 0, 0, 1]
        ])
        
        # Final transformation matrix
        P = S @ R_x @ T
        
        # Save the transformation matrix to a file
        np.savetxt(mat_file_path, P, fmt='%.6f')

        print(f"Transformation matrix saved to {mat_file_path}")

        if transformed_obj_file_path is not None:
            # Apply the transformation matrix to each vertex
            final_transformed_vertices = vertices_homogeneous @ P.T

            # Remove the homogeneous coordinate (the last column)
            final_transformed_vertices = final_transformed_vertices[:, :3]

            # Write the transformed vertices to a new .obj file manually
            with open(transformed_obj_file_path, 'w') as file:
                for vertex in final_transformed_vertices:
                    file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            print(f"Transformed point cloud saved to {transformed_obj_file_path}")
    
    else:
        print("The provided OBJ file does not contain any vertices.")


if __name__ == "__main__":
    # obj_file_path = 'mitea_010_00.obj'
    # mat_file_path = 'output_p.txt'
    # transformed_obj_file_path = 'transformed_mitea_010_00_new.obj'
    # transform_point_cloud(obj_file_path, mat_file_path, transformed_obj_file_path)

    repo_dir_path = Path(__file__).resolve().parent.parent
    gen_mitea_dir_path = repo_dir_path / "data" / "mitea" / "generated"
    max_distance = get_max_distance(gen_mitea_dir_path)
    generate_transformation_mat(gen_mitea_dir_path, max_distance)
