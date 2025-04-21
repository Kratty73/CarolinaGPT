#!/bin/bash

# Parameters
#SBATCH --error=%A_%a_0_log.err
#SBATCH --gres=gpu:1
#SBATCH --job-name=data_gen
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --output=%A_%a_0_log.out
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --time=8639
#SBATCH --cpus-per-task=8

# setup
module load cuda/12.8
module load anaconda/2023.03
nvidia-smi
export PATH="/nas/longleaf/home/kratty73/.conda/envs/carolinagpt"
conda deactivate
conda info --envs
conda activate carolinagpt
# python=~/.conda/envs/carolinagpt/bin/python
# which python
python3 dpo_data.py