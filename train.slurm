#!/bin/bash
#SBATCH --job-name=hi4d_train
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

# -------- Custom videos
# python code/train.py model.is_continue=false
# python code/train.py  --config-name=football_base model.is_continue=false

# -------- Hi4D videos
# hi4d_pair00_dance00_cam76, total frames = 140
# python code/train.py  --config-name=hi4d_base model.is_continue=false scene_name=hi4d_pair00_dance00_cam76 end_frame=140 model.num_training_frames=140
# scheduled

# hi4d_pair01_hug01_cam76, total frames = 120
# python code/train.py  --config-name=hi4d_base model.is_continue=false scene_name=hi4d_pair01_hug01_cam76 end_frame=120 model.num_training_frames=120
# scheduled

# hi4d_pair15_fight15_cam4, total frames = 150  
# python code/train.py  --config-name=hi4d_base model.is_continue=false scene_name=hi4d_pair15_fight15_cam4 end_frame=150 model.num_training_frames=150
# scheduled

# hi4d_pair16_jump16_cam4, total frames = 90  
# python code/train.py  --config-name=hi4d_base model.is_continue=false scene_name=hi4d_pair16_jump16_cam4 end_frame=90 model.num_training_frames=90
# scheduled

# hi4d_pair17_dance17_cam28, total frames = 125  
# python code/train.py  --config-name=hi4d_base model.is_continue=false scene_name=hi4d_pair17_dance17_cam28 end_frame=125 model.num_training_frames=125
# scheduled

# hi4d_pair19_piggyback19_cam4, total frames = 150  
# python code/train.py  --config-name=hi4d_base model.is_continue=false scene_name=hi4d_pair19_piggyback19_cam4 end_frame=150 model.num_training_frames=150
# scheduled