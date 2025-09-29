#!/bin/bash
#SBATCH --job-name=football
#SBATCH --output=/scratch/izar/cizinsky/multiply-output/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/multiply-output/slurm/%x.%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8           
#SBATCH --mem=32G                   
#SBATCH --gres=gpu:1 
#SBATCH --account=master 

# Conda init and activate
source /home/cizinsky/miniconda3/etc/profile.d/conda.sh
conda activate multiply

# Change to the directory where the script is located
cd /home/cizinsky/MultiPly

# python code/train.py model.is_continue=true
python code/train.py  --config-name=football_base model.is_continue=true