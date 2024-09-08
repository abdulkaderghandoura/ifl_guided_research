from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import numpy as np
# from scipy.ndimage import zoom


# # Resample the volume data
# resampled_volume_data = zoom(volume_data, resampling_factors, order=1)  # Linear interpolation

# # Resample the label data (use order=0 for nearest-neighbor interpolation for labels)
# resampled_label_data = zoom(label_data, resampling_factors, order=0)

# # Update the affine matrix for the new voxel size
# new_affine = np.copy(volume_affine)
# new_affine[:3, :3] = volume_affine[:3, :3] / resampling_factors


# resampled_volume_img = nib.Nifti1Image(resampled_volume_data, new_affine)
# resampled_label_img = nib.Nifti1Image(resampled_label_data, new_affine)

# nib.save(resampled_volume_img, f'{file_stem}_resampled.nii.gz')
# nib.save(resampled_label_img, f'{file_stem}_resampled.nii.gz')


def get_spacing(affine):
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def check_affines(raw_data_dir_path, reference_affine, target_voxel_spacing):

    for file_path in tqdm(list(raw_data_dir_path.rglob("*.nii.gz"))):
        nii = nib.load(file_path)
        affine = nii.affine

        assert np.all(affine == reference_affine), \
            f"Different affine in:\n{file_path}"

        voxel_spacing = get_spacing(affine)
        
        assert np.all(voxel_spacing == target_voxel_spacing), \
            f"{voxel_spacing} affine voxel spacing in:\n{file_path}"
    
    print(f"All affine matrices are the same, and voxel spacing from all affine matrices is {target_voxel_spacing}!")


def check_headers(raw_data_dir_path, target_voxel_spacing):

    for file_path in tqdm(list(raw_data_dir_path.rglob("*.nii.gz"))):
        nii = nib.load(file_path)
        voxel_spacing = nii.header.get_zooms()
        
        assert np.all(voxel_spacing == target_voxel_spacing), \
            f"{voxel_spacing} header voxel spacing in:\n{file_path}"
    
    print(f'Voxel spacing from all headers is {target_voxel_spacing}!')


if __name__ == "__main__":
    repo_dir_path = Path(__file__).resolve().parent.parent
    data_dir_path = repo_dir_path / "data" / "mitea"
    raw_data_dir_path = data_dir_path / "raw"

    target_voxel_spacing = np.array([1.0, 1.0, 1.0])

    reference_affine = np.array([[-1.,  0.,  0.,  0.],
                                 [ 0., -1.,  0.,  0.],
                                 [ 0.,  0.,  1.,  0.],
                                 [ 0.,  0.,  0.,  1.]])
    
    check_affines(raw_data_dir_path, reference_affine, target_voxel_spacing)
    check_headers(raw_data_dir_path, target_voxel_spacing)
