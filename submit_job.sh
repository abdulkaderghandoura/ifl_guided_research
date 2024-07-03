#!/bin/sh
 
#SBATCH --job-name=baseline
#SBATCH --output=jobs/baseline.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get different logging files
#SBATCH --error=jobs/baseline.err  # Standard error of the script
#SBATCH --time=1-00:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=2  # Number of CPUs (Don't use more than 12/6 per GPU)
#SBATCH --mem=24G  # Memory in GB (Don't use more than 48/24 per GPU unless you absolutely need it and know what you are doing)
#SBATCH --nodelist=ithor

# run the program
ml cuda  # load default CUDA module
ml miniconda3  # load default miniconda and python module
source activate ifl  # Activate the conda environment

# python baseline/preprocess_data.py \
#     --data_dir data/4DM_Dataset/processed \
#     --source_dir data/4DM_Dataset/raw/v1.0/ \
#     --source_name LVV \
#     --class_name train \
#     --split baseline/train.json

# python src/generate_points.py

# python train.py \
#     --experiment examples/demo \
#     --data examples/demo/data

python baseline/train.py \
    --experiment baseline/experiment \
    --data data/4DM_Dataset/processed/

conda deactivate  # Deactivate the conda environment
ml -cuda -miniconda3  # unload all modules