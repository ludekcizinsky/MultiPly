#!/bin/bash
#SBATCH --job-name=viser
#SBATCH --output=/scratch/izar/cizinsky/multiply-output/slurm/%x.%j.out
#SBATCH --error=/scratch/izar/cizinsky/multiply-output/slurm/%x.%j.err
#SBATCH --time=01:00:00
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

# Pick inputs
INPUT_ROOT=/scratch/izar/cizinsky/multiply-output/preprocessing/data/taichi01_vitpose_openpose
OUTPUT_ROOT=/scratch/izar/cizinsky/multiply-output/training/taichi01/visualisations

# Where to write the share URL
SHARE_FILE="/home/cizinsky/MultiPly/viser/share_url.txt"

echo "[Slurm] Launching Viser demo..."
python -u viser/show_meshes.py \
  --input_root "$INPUT_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --share \
  --share-url-file "$SHARE_FILE"

# If your script runs indefinitely, Slurm will keep it alive until time limit.
