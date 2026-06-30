#!/bin/bash
#SBATCH --job-name=snn_eeg_ablation
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
#
# Runs src/ablation.py, which:
#   Phase 1  - trains a true LOSO model per subject (9x for BNCI2014_001)
#              using the best hyperparameters straight out of
#              optuna_study.db (no best_params.json needed -- the script
#              auto-detects the study and reads study.best_params), and
#              reports per-subject accuracy/confusion matrices/curves.
#   Phase 2a - re-runs LOSO with one design choice flipped at a time
#              (z-score on/off, bandpass on/off, readout mode) to see how
#              much each one matters.
#   Phase 2b - reuses the Phase-1 models to measure per-channel importance
#              via occlusion (zero one channel, see how much accuracy drops).
# -------------------------------------------------------

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "Directory: $(pwd)"

echo "job_id    : $SLURM_JOB_ID"  >> chain_status.txt
echo "started   : $(date)"        >> chain_status.txt
echo "----"                        >> chain_status.txt

# Shared persistent MOABB download cache -- reused across jobs
export MOABB_DOWNLOAD_DIR=/scratch/mne_data
mkdir -p "$MOABB_DOWNLOAD_DIR"

# Cache preprocessed arrays somewhere persistent across jobs
export SPIKING_EEGNET_CACHE=/home/aleksander.berezowsk/.cache/spiking_eegnet
mkdir -p "$SPIKING_EEGNET_CACHE"

# Activate the virtual environment
module load cuda/11.3.0
source ~/venv/bin/activate

# setuptools must come first -- pkg_resources depends on it
pip install "setuptools==69.5.1"

# Install everything except torch
grep -v "^torch\|^torchvision\|^torchaudio" requirements.txt > ~/requirements_notorch.txt
pip install --no-deps -r ~/requirements_notorch.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 torchmetrics
pip install optuna snntorch seaborn scikit-learn

# Thread counts to match --cpus-per-task
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd /home/aleksander.berezowsk/spiking-eegnet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# optuna_study.db lives at the repo root (same place main.py writes it),
# so the script's default --optuna-db sqlite:///optuna_study.db just works
# as long as we run from here. Study name is auto-detected.
python3 src/ablation.py \
    --out-dir results_FinalTest/BNCI2014_001/ablation_study \
    --epochs 50

# To restrict the (expensive) Phase 2a ablation sweep to a few representative
# subjects instead of all 9, add e.g.:
#     --ablation-subjects 0,4,8
# To skip a phase entirely:
#     --skip-ablation   /   --skip-channels
# To point at a specific study (if the DB ever has more than one):
#     --study-name snn_eegnet_v3_200_20

echo "Finished : $(date)"
echo "Exit code: $?"
echo "finished  : $(date), exit=$?" >> chain_status.txt
echo "===="                          >> chain_status.txt
