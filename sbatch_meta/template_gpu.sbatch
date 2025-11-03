#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=gpu_v100
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log



# =====================================================================================
# ============================= Load And Activate =====================================
# =====================================================================================

module load miniconda3/3.13.25

source activate base
conda activate fer-thesis




# =====================================================================================
# ================================= Set Path ==========================================
# =====================================================================================

## Ensure CUDA / libdevice visible to XLA and Python
# Prefer conda env libs but make sure CUDA toolkit lib paths come first
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"



# =====================================================================================
# =================================== Run Code ========================================
# =====================================================================================

echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
echo "=========================================================================="

cd /home/dbuhnila/models/FER-Thesis-NNs
pwd




# =====================================================================================
# ============================= Copy Logs Outside =====================================
# =====================================================================================
DEST_ROOT="$HOME/fer-thesis-logs-to-push"
mkdir -p "$DEST_ROOT"

TS=$(date +%Y%m%d-%H%M%S)
DEST_DIR="$DEST_ROOT/job_${SLURM_JOB_ID}_$TS"
mkdir -p "$DEST_DIR"

# Primary log (SLURM --output goes here)
MAIN_LOG="/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_${SLURM_JOB_ID}.log"
# If your job uses timestamped names, adjust accordingly:
if [ -f "$MAIN_LOG" ]; then
  cp "$MAIN_LOG" "$DEST_DIR/" || true
fi

# Also copy the sbatch script used (helpful for provenance)
cp "${SLURM_SUBMIT_DIR:-/home/dbuhnila/models/FER-Thesis-NNs}/sbatch/evaluate_model_gpu.sbatch" "$DEST_DIR/" 2>/dev/null || true

# Optional: create a small tarball of the current working tree (no .git)
# Useful if you want the exact code snapshot present at runtime.
SNAP_TAR="$DEST_DIR/code_snapshot_${TS}.tar.gz"
tar --exclude='.git' -czf "$SNAP_TAR" -C /home/dbuhnila/models/FER-Thesis-NNs . || true

# Create a provenance file
PROV="$DEST_DIR/provenance.txt"
{
  echo "JOBID: ${SLURM_JOB_ID}"
  echo "USER: ${USER}"
  echo "HOST: $(hostname)"
  echo "DATE: $(date --utc)"
  echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-unknown}"
  echo "PYTHON_EXECUTABLE: $(which python || echo unknown)"
  echo "GIT_COMMIT: $(git -C /home/dbuhnila/models/FER-Thesis-NNs rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
} > "$PROV"

# Done - show where files are
echo "Saved logs and snapshot to $DEST_DIR"