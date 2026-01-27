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

cd /dss/dsshome1/0D/go68tos2/looped_nanochat
source .venv/bin/activate

# Set cache directories (datasets as subdirectory of HF_HOME)
export HF_HOME="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/LLMs/.cache/huggingface"
export HF_DATASETS_CACHE="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/LLMs/.cache/huggingface/datasets"

# Set nanochat directory
export NANOCHAT_BASE_DIR="/dss/dssmcmlfs01/pn73mu/pn73mu-dss-0000/kristian/.cache/nanochat"

# Number of processes/GPUs to use
NPROC_PER_NODE=1

# Run chat evaluation
# Modify the arguments as needed:
#   -i/--source: sft|mid|rl (required)
#   -a/--task-name: ARC-Easy|ARC-Challenge|MMLU|GSM8K|HumanEval|SpellingBee (optional, defaults to all)
#   -d/--dtype: float32|bfloat16 (default: bfloat16)
#   -b/--batch-size: batch size for categorical evaluation (default: 8)
#   -x/--max-problems: limit number of problems (optional)
#   --num-recur: number of recurrences for recursive transformer (optional)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft