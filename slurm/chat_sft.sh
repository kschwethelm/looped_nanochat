#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source slurm/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${SLURM_GPUS:-1}

# Run chat evaluation
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft \
    -- --model-tag d20 --output-tag d20_def \
    --device-batch-size 16 \
    --run default_cfg

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval \
    -- -i sft -g d20_def \
    --batch-size 32 \
    --kv-budget 1 \
    --use-rec-warm-start \
    --num-recur "2,4,10,16"
