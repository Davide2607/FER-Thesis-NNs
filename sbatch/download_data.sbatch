#!/bin/bash
#SBATCH --job-name=dbuhnila/tesi
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ppp36213@gmail.com
#SBATCH --partition=cpu_skylake
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=1G
#SBATCH --ntasks-per-node=1
#SBATCH --output=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log
#SBATCH --error=/home/dbuhnila/models/FER-Thesis-NNs/out_err/train_%j.log

module load miniconda3/3.13.25

source activate base
conda activate fer-thesis
echo "=========================================================================="
echo "Activated venv!"
python -c "import sys; print('Python executable:', sys.executable); print('Environment:', sys.prefix)"
echo "=========================================================================="

cd /home/dbuhnila/models/FER-Thesis-NNs
pwd

# Define the Google Drive file ID and output file name
FILE_ID="1tkWnzQVDiWGN3Ofjf33IbL-ZxSr-2ncE"
OUTPUT="data.zip"

# Download the file using curl
curl -L -o "${OUTPUT}" "https://drive.google.com/uc?export=download&id=${FILE_ID}"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful. Extracting ${OUTPUT}..."
    unzip "${OUTPUT}" -d "data"
    echo "Extraction complete."
else
    echo "Download failed. Please check the file permissions or your internet connection."
fi