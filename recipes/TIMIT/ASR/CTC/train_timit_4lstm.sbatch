#!/bin/bash
# SBATCH script to run the TIMIT training with LSTM encoder
# Adjust `--mail-user` or remove the mail lines if you don't want emails.

# Job name and output
#SBATCH --job-name=timit_train_4lstm
#SBATCH --output=/idiap/temp/evida/speechbrain/recipes/TIMIT/ASR/CTC/results/4lstm/%x.%j.out
#SBATCH --error=/idiap/temp/evida/speechbrain/recipes/TIMIT/ASR/CTC/results/4lstm/%x.%j.err
#SBATCH --account=spider

# Resources
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gpus rtx3090:2

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
exec python run_training_lstm.py hparams/timit_train_4lstm.yaml
