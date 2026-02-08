#!/bin/bash

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source slurm/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${SLURM_GPUS:-1}

EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)

# Run base training with detailed gradient tracking
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --eval-tokens=$EVAL_TOKENS \
    --target-flops 2.15e18 \
    --model-tag r4_init_random_2.15e18_s12 \
    --run r4_init_random_2.15e18_s12 \
    --device-batch-size 32 \
    --core-metric-every=-1 \
    --core-metric-max-per-task=-1 \
    --target-param-data-ratio=-1 \
    --sample-every=-1 \
    --no-sample-recur \
    --save-every=-1 \
    --input-injection=inject_init_random \
    --size 12
