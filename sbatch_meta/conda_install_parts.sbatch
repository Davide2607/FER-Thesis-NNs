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
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.err

module load miniconda3/3.13.25

# confirm conda is found
which conda
conda --version

# accept Anaconda channels' ToS (replace CHANNEL with the shown URLs)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r


conda create -n fer-thesis python=3.11 -y
echo "=========================================================================="
echo "Created venv!"
echo "=========================================================================="
