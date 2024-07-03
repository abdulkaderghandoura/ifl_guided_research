from pathlib import Path
import trimesh
import numpy as np

repo_dir_path = Path(__file__).resolve().parent.parent


def generate_points(raw_data_dir_path):
    for subdir in raw_data_dir_path.iterdir():
        if subdir.is_dir():
            points_dir = subdir / 'points'
            points_dir.mkdir(exist_ok=True)
            mesh_paths = list((subdir / "mesh").glob("*.obj"))
            
            for mesh_path in mesh_paths:
                # Load the mesh
                mesh = trimesh.load(mesh_path)
                # Sample points from the surface
                num_points = 50000
                points, _ = trimesh.sample.sample_surface(mesh, num_points)

                # Save points to OBJ file
                with open(points_dir / (mesh_path.stem + '.obj'), 'w') as f:
                    for point in points:
                        f.write(f'v {point[0]} {point[1]} {point[2]}\n')


if __name__ == "__main__":
    data_dir_path = repo_dir_path / "data" / "4DM_Dataset"
    raw_data_dir_path = data_dir_path / "raw" / "v1.0"
    generate_points(raw_data_dir_path)

    print("Generating points is done!")
