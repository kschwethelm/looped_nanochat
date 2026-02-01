#!/bin/bash
#SBATCH --job-name=looped_nanochat_tokenizer
#SBATCH --output=logs/tokenizer/%A.out
#SBATCH --error=logs/tokenizer/%A.err
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

# train the tokenizer with vocab size 2**16 = 65536 on ~10B characters of data
# With value embeddings, we could half the vocab size (see nanochat)
python -m scripts.tok_train --vocab-size 65536 --max-chars 10_000_000_000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval