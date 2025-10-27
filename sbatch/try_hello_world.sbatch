#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=gpu_v100
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log
echo "running python ./scripts/hello_world.py"
python ./scripts/hello_world.py
