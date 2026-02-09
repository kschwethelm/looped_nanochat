#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source slurm/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${SLURM_GPUS:-1}

# Run chat SFT
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft \
    -- --model-tag s20 --output-tag s20 \
    --device-batch-size 16 \
    --run s20
