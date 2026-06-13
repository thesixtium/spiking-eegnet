#!/bin/bash
#SBATCH --job-name=snn_eeg_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aleksander.berezowsk@ucalgary.ca

# -------------------------------------------------------
# CHANGE THE EMAIL ADDRESS ABOVE BEFORE SUBMITTING
# -------------------------------------------------------

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "Directory: $(pwd)"

# Put raw MOABB downloads on scratch (fast, auto-cleaned after 5 days)
export MOABB_DOWNLOAD_DIR=/scratch/${SLURM_JOB_ID}/mne_data
mkdir -p "$MOABB_DOWNLOAD_DIR"

# Cache preprocessed arrays somewhere persistent across jobs
export SPIKING_EEGNET_CACHE=/home/aleksander.berezowsk/.cache/spiking_eegnet
mkdir -p "$SPIKING_EEGNET_CACHE"

# Activate the virtual environment
module load cuda/11.3.0
source ~/venv/bin/activate

# setuptools must come first — pkg_resources depends on it
pip install "setuptools==69.5.1"

# Install everything except torch
grep -v "^torch\|^torchvision\|^torchaudio" requirements.txt > ~/requirements_notorch.txt
pip install --no-deps -r ~/requirements_notorch.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 torchmetrics
pip install optuna snntorch

# Thread counts to match --cpus-per-task
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd /home/aleksander.berezowsk/spiking-eegnet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

python3 src/main.py

echo "Finished : $(date)"
echo "Exit code: $?"