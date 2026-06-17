#!/bin/bash
#SBATCH --job-name=snn_eeg_quant
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

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "Directory: $(pwd)"

echo "job_id    : $SLURM_JOB_ID"  >> chain_status.txt
echo "started   : $(date)"        >> chain_status.txt
echo "----"                        >> chain_status.txt

export MOABB_DOWNLOAD_DIR=/scratch/${SLURM_JOB_ID}/mne_data
mkdir -p "$MOABB_DOWNLOAD_DIR"

export SPIKING_EEGNET_CACHE=/home/aleksander.berezowsk/.cache/spiking_eegnet
mkdir -p "$SPIKING_EEGNET_CACHE"

module load cuda/11.3.0
source ~/venv/bin/activate

pip install "setuptools==69.5.1"

grep -v "^torch\|^torchvision\|^torchaudio" requirements.txt > ~/requirements_notorch.txt
pip install --no-deps -r ~/requirements_notorch.txt
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 torchmetrics
pip install optuna snntorch

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd /home/aleksander.berezowsk/spiking-eegnet
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Pass --load-checkpoint if you have a saved model and want to skip retraining
python3 -m quantization_testing.run_quantization
# python3 -m quantization_testing.run_quantization --load-checkpoint results/BNCI2014_001/model.pt

echo "Finished : $(date)"
echo "Exit code: $?"
echo "finished  : $(date), exit=$?" >> chain_status.txt
echo "===="                          >> chain_status.txt