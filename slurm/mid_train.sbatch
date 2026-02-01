#!/bin/bash
#SBATCH --job-name=looped_nanochat_mid_train
#SBATCH --output=logs/mid_train/%A.out
#SBATCH --error=logs/mid_train/%A.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=k.schwethelm@tum.de

cd /dss/dsshome1/0D/go68tos2/looped_nanochat
uv sync
source .venv/bin/activate

BASE_DIR="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000"
# Set cache directories (datasets as subdirectory of HF_HOME)
export HF_HOME="$BASE_DIR/LLMs/.cache/huggingface"
export HF_DATASETS_CACHE="$BASE_DIR/LLMs/.cache/huggingface/datasets"

# Set nanochat directory
export NANOCHAT_BASE_DIR="$BASE_DIR/kristian/.cache/nanochat"

# Number of processes/GPUs to use
NPROC_PER_NODE=1

# Run chat evaluation
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train \
    -- --model-tag d20 --output-tag d20_def \
    --device-batch-size 32 \
    --run default_cfg

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval \
    -- -i mid -g d20_def \
    --batch-size 32 \
    --kv-budget 1 \
    --use-rec-warm-start \
    --num-recur "2,4,10,16"