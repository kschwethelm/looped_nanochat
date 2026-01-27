#!/bin/bash
#SBATCH --job-name=looped_nanochat_download
#SBATCH --output=logs/download/%A.out
#SBATCH --error=logs/download/%A.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH -p lrz-cpu
#SBATCH -q cpu
#SBATCH --mem=32G
#SBATCH --ntasks=1

cd /dss/dsshome1/0D/go68tos2/looped_nanochat
source .venv/bin/activate

BASE_DIR="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000"
# Set cache directories (datasets as subdirectory of HF_HOME)
export HF_HOME="$BASE_DIR/LLMs/.cache/huggingface"
export HF_DATASETS_CACHE="$BASE_DIR/LLMs/.cache/huggingface/datasets"

# Set nanochat directory
export NANOCHAT_BASE_DIR="$BASE_DIR/kristian/.cache/nanochat"

# Download specific checkpoint(s) from HF
hf download Trelis/nanochat-recursive \
  --include "base/d20/model_021292.pt" "base/d20/meta_021292.json" "mid/*" "sft/*" "tokenizer/*" \
  --local-dir "$NANOCHAT_BASE_DIR" --repo-type model

echo "Download complete to $NANOCHAT_BASE_DIR"
