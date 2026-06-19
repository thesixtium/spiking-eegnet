#!/bin/bash
#SBATCH --job-name=snn_eeg_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
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

# Record this job's place in the chain (for bug-checking chained submissions)
echo "job_id    : $SLURM_JOB_ID"  >> chain_status.txt
echo "started   : $(date)"        >> chain_status.txt
echo "----"                        >> chain_status.txt

# Shared persistent MOABB download cache -- all parallel jobs reuse the same
# data instead of each job re-downloading to its own /scratch/${SLURM_JOB_ID}.
# MOABB itself is safe to read from concurrently once the data is on disk.
export MOABB_DOWNLOAD_DIR=/scratch/mne_data
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
echo "finished  : $(date), exit=$?" >> chain_status.txt
echo "===="                          >> chain_status.txt
