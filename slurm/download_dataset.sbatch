#!/bin/bash
#SBATCH --job-name=looped_nanochat_download_data
#SBATCH --output=logs/download_data/%A.out
#SBATCH --error=logs/download_data/%A.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH -p lrz-cpu
#SBATCH -q cpu
#SBATCH --mem=32G
#SBATCH --ntasks=1

cd /dss/dsshome1/0D/go68tos2/looped_nanochat
uv sync
source .venv/bin/activate

BASE_DIR="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000"
# Set cache directories (datasets as subdirectory of HF_HOME)
export HF_HOME="$BASE_DIR/LLMs/.cache/huggingface"
export HF_DATASETS_CACHE="$BASE_DIR/LLMs/.cache/huggingface/datasets"

# Set nanochat directory
export NANOCHAT_BASE_DIR="$BASE_DIR/kristian/.cache/nanochat"

python -m nanochat.dataset -n 700