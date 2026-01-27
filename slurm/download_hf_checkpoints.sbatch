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
uv sync
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

rm -r $BASE_DIR/kristian/.cache/nanochat/.cache

echo "Download complete to $NANOCHAT_BASE_DIR"

# Rename folders
cd "$NANOCHAT_BASE_DIR"
[ -d "base" ] && mv base base_checkpoints && echo "Renamed base -> base_checkpoints"
[ -d "mid" ] && mv mid mid_checkpoints && echo "Renamed mid -> mid_checkpoints"
[ -d "sft" ] && mv sft chatsft_checkpoints && echo "Renamed sft -> chatsft_checkpoints"
[ -d "rl" ] && mv rl chatrl_checkpoints && echo "Renamed rl -> chatrl_checkpoints"

# Move tokenizer contents from latest subdirectory to parent
[ -d "tokenizer/latest" ] && mv tokenizer/latest/* tokenizer/ && rmdir tokenizer/latest && echo "Moved tokenizer/latest/* to tokenizer/"

echo "All renaming complete"