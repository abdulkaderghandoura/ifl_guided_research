# MR-Informed 4D Echocardiographic Motion Interpolation for Reconstructing Left Ventricle Sparse Sequences

Guided Research

### Supervisor

Prof. Dr. Nassir Navab

### Author

Abdulkader Ghandoura

### Advisors

- Magdalena Wysocki
- Mohammad Farid Azampour

### Note

The scripts we implemented are not limited to the `src` directory. We had to create and modify scripts in the `baseline` directory to facilitate access to the existing packages due to certain path requirements.

### Request Access and Install the Datasets

1. To access the [baseline dataset](https://drive.google.com/drive/folders/1027CUnLNoGiAiqBNI65f7pbAM5wn9Rih), please download the [4DM Data Access Agreement.pdf](https://github.com/yuan-xiaohan/4D-Myocardium-Reconstruction-with-Decoupled-Motion-and-Shape-Model/blob/main/4DM%20Data%20Access%20Agreement.pdf). It should be printed, signed, and scanned as a single .pdf document. Then, send the signed e-copy to xiaohan_yuan@163.com and CC to yangangwang@seu.edu.cn. If approved, the authors will add your Google Drive email address to the sharing list.
  - The default destination is `Path/to/repo/data/4DM_Dataset/raw`

2. To access the [MITEA dataset](https://www.cardiacatlas.org/mitea/), please submit this form on the website.
  - The default destination is `Path/to/repo/data/mitea/raw`

### Set up the Environment

1. Create and activate conda environment 
```
conda create --name ifl
conda activate ifl
```
2. Install the requirements 
```
pip install -r requiremnets.txt 
```

### Data Preprocessing
1. Baseline data
```
python baseline/preprocess_data.py --data_dir data/4DM_Dataset/processed/v1.0/<run_id>
                                   --source_dir data/4DM_Dataset/raw/v1.0 --source_name LVV
                                   --class_name train --split baseline/train_baseline.json
```
```
python baseline/preprocess_data.py --data_dir data/4DM_Dataset/processed/v1.0/<run_id>
                                   --source_dir data/4DM_Dataset/raw/v1.0 --source_name LVV
                                   --class_name test --split baseline/test_baseline.json --test
```

2. MITEA data
```
python src/mitea_preprocess.py
```
```
python src/mitea_gen_transform_mat.py
```
```
python baseline/preprocess_data.py --data_dir data/mitea/processed/ --source_dir data/mitea/generated/scan1/
                                   --source_name LVV --class_name train --split baseline/train_mitea.json
```
```
python baseline/preprocess_data.py --data_dir data/mitea/processed/ --source_dir data/mitea/generated/scan1/
                                   --source_name LVV --class_name test --split baseline/test_mitea.json --test
```

### Baseline pre-training and fine-tuning
1. Open baseline/experiments/specs.json
  - Modify the experiment name in the paths
  - If needed, modify the batch size and number of snapshots
  - Check other configurations (if needed)

2. Execute the following commands:
  - Baseline
```
python baseline/train.py --experiment baseline/experiments
                         --data data/4DM_Dataset/processed/v1.0/<run_id>
                         --experiment_name <mandatory_experiment_name>
```
```
python baseline/reconstruct.py --experiment baseline/experiments/<same_pretrain_experiment_name>
                               --data data/4DM_Dataset/processed/v1.0/<run_id>
                               --split baseline/experiments/<same_pretrain_experiment_name>/test.json
                               --checkpoint <snapshot_num> --do_not_transform
                               # --subsequence baseline/experiments/<same_pretrain_experiment_name>/subsequence.json
```
```
python baseline/interpolate_reconstruct.py --experiment baseline/experiments/<same_pretrain_experiment_name>
                                           --data data/4DM_Dataset/processed/v1.0/<run_id>
                                           --split baseline/experiments/<same_pretrain_experiment_name>/test.json
                                           --checkpoint <snapshot_num>
```
  - MITEA
```
python baseline/train.py --experiment baseline/experiments 
                         --data data/mitea/processed/ 
                         --experiment_name <mandatory_experiment_name> 
                         --continue <snapshot_num>
```
```
python baseline/reconstruct.py --experiment baseline/experiments/<same_finetune_experiment_name>
                               --data data/mitea/processed/ --split baseline/experiments/<same_finetune_experiment_name>/test.json
                               --checkpoint <snapshot_num> --do_not_transform
                               # --subsequence baseline/experiments/<same_finetune_experiment_name>/subsequence.json
```
```
python baseline/interpolate_reconstruct.py --experiment baseline/experiments/<same_finetune_experiment_name>
                                           --pre_train_experiment baseline/experiments/<same_pretrain_experiment_name>
                                           --pre_train_data data/4DM_Dataset/processed/v1.0/<run_id>
                                           --data data/mitea/processed
                                           --split baseline/experiments/<same_finetune_experiment_name>/test.json
                                           --checkpoint <snapshot_num> --pre_train_checkpoint <snapshot_num> --mitea
```

## References
Baseline dataset and paper:
```
Yuan, X., Liu, C. and Wang, Y., 2023. 4D Myocardium Reconstruction with Decoupled Motion and Shape Model.
In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 21252-21262).
```

Target dataset:
```
Zhao, D., Ferdian, E., Maso Talou, G.D., Quill, G.M., Gilbert, K., Wang, V.Y., Babarenda Gamage, T.P., Pedrosa, J.,
D’hooge, J., Sutton, T.M. and Lowe, B.S., 2023. MITEA: A dataset for machine learning segmentation of the left
ventricle in 3D echocardiography using subject-specific labels from cardiac magnetic resonance imaging.
Frontiers in Cardiovascular Medicine, 9, p.1016703.
```
