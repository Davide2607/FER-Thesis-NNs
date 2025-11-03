#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=gpu_v100
#SBATCH --time=00:30:00
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

# Try to discover CUDA root (where nvcc lives) and set CUDA_HOME
CUDA_HOME=""
if command -v nvcc >/dev/null 2>&1; then
	# resolve nvcc location and go two levels up to get CUDA root
	nvcc_path=$(readlink -f "$(command -v nvcc)")
	CUDA_HOME=$(dirname "$(dirname "$nvcc_path")")
fi

# Fallback to common system locations if nvcc not in PATH
if [ -z "$CUDA_HOME" ]; then
	for cand in /usr/local/cuda-12.2 /usr/local/cuda /share/apps/hpc_sdk/*/cuda /opt/cuda; do
		if [ -d "$cand" ]; then
			CUDA_HOME="$cand"
			break
		fi
	done
fi

if [ -n "$CUDA_HOME" ]; then
	export CUDA_HOME
	export PATH="$CUDA_HOME/bin:$PATH"
	export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Find libdevice (.bc) used by XLA and point XLA_FLAGS to it so JIT can compile
LIBDEVICE_DIR=""
for maybe in "$CUDA_HOME"/nvvm/libdevice "$CUDA_HOME"/nvvm/libdevice/* "$CONDA_PREFIX"/lib/python*/site-packages/tensorflow/**/cuda_nvcc; do
	if compgen -G "$maybe/libdevice*.bc" >/dev/null 2>&1; then
		LIBDEVICE_DIR=$(dirname "$(compgen -G "$maybe/libdevice*.bc" | head -n1)")
		break
	fi
done

if [ -n "$LIBDEVICE_DIR" ]; then
	export XLA_FLAGS="--xla_gpu_cuda_data_dir=$LIBDEVICE_DIR"
	echo "Set XLA_FLAGS=$XLA_FLAGS"
else
	echo "Warning: libdevice not found; XLA JIT may fail. Searched CUDA_HOME=$CUDA_HOME and CONDA_PREFIX=$CONDA_PREFIX"
fi



# =====================================================================================
# =================================== Run Code ========================================
# =====================================================================================

echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
python -c "import tensorflow as tf; print('tf',tf.__version__); print(tf.sysconfig.get_build_info()); print('gpus:', tf.config.list_physical_devices('GPU'))"
echo "=========================================================================="

echo 'running ./scripts/evaluate_model.py "ADELE"'
python ./scripts/evaluate_model.py "ADELE"
echo 'running ./scripts/evaluate_model.py "OCCLUDED"'
python ./scripts/evaluate_model.py "OCCLUDED"



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

# Create a zip with all the code inside of scripts and modules
# Prefer zip if available, otherwise fall back to tar.gz
CODE_ARCHIVE="$DEST_DIR/code_$(date +%Y%m%d-%H%M%S).zip"
ROOT_DIR="/home/dbuhnila/models/FER-Thesis-NNs"
if command -v zip >/dev/null 2>&1; then
  echo "Creating ZIP archive $CODE_ARCHIVE (scripts/ + modules/)"
  (cd "$ROOT_DIR" && zip -r "$CODE_ARCHIVE" scripts modules -x "*.git*" "*/__pycache__/*") >/dev/null 2>&1 || echo "zip failed"
else
  # fallback to tar.gz
  CODE_TAR="${CODE_ARCHIVE%.zip}.tar.gz"
  echo "zip not found, creating tarball $CODE_TAR"
  tar -C "$ROOT_DIR" -czf "$CODE_TAR" scripts modules --exclude='.git' --exclude='*/__pycache__' || echo "tar failed"
fi

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