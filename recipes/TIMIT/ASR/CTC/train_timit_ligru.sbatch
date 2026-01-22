#!/bin/bash
# SBATCH script to run the TIMIT training with baseline LiGRU (CRDNN) encoder
# Adjust `--mail-user` or remove the mail lines if you don't want emails.

# Job name and output
#SBATCH --job-name=timit_train_ligru
#SBATCH --output=/idiap/temp/evida/speechbrain/recipes/TIMIT/ASR/CTC/results/liGRU/%x.%j.out
#SBATCH --error=/idiap/temp/evida/speechbrain/recipes/TIMIT/ASR/CTC/results/liGRU/%x.%j.err
#SBATCH --account=spider

# Resources
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus 1

# Email notifications (set your email or remove these lines)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elija.vida@epfl.ch

set -euo pipefail

echo "Starting job: $SLURM_JOB_NAME (id: $SLURM_JOB_ID) on $(hostname)"

# Optional: make sure cluster helper tools are in PATH (your site doc)
# export PATH=/idiap/resource/cluster/bin:$PATH

# Activate conda environment
source /idiap/temp/evida/miniconda3/etc/profile.d/conda.sh
conda activate venv-speechbrain

# Ensure Python uses the requested number of threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Move to recipe directory and run training
cd /idiap/temp/evida/speechbrain/recipes/TIMIT/ASR/CTC
exec python train.py hparams/train.yaml --data_folder /idiap/temp/evida/data/TIMIT/TIMIT\ dataset/data
