#!/bin/bash
#SBATCH --job-name=looped_nanochat_chat_eval
#SBATCH --output=logs/chat_eval/%A_%a.out
#SBATCH --error=logs/chat_eval/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH -p mcml-hgx-a100-80x4
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
# Modify the arguments as needed:
#   -i/--source: sft|mid|rl (required)
#   -a/--task-name: ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval|SpellingBee (optional, defaults to all)
#   -d/--dtype: float32|bfloat16 (default: bfloat16)
#   -b/--batch-size: batch size for categorical evaluation (default: 8)
#   -x/--max-problems: limit number of problems (optional)
#   -g/--model-tag: Model tag to load (optional)
#   -s/--step: Step to load (optional)
#   -r/--num-recur: number of recurrences for recursive transformer (optional)
#   -rws/--use-rec-warm-start: Use recurrent warm-start (carry recurrent state when decoding tokens)
#   -kv/--kv-cache-mode: KV cache mode: 'final' (only cache final recurrence, default) or 'all' (cache all recurrences)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval \
    -- -i sft -g d20