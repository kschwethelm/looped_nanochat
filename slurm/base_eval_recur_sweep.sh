#!/bin/bash
#
# Evaluate a trained model sweeping over different numbers of loop iterations (num_recur).
# This script demonstrates test-time compute scaling by varying the recurrence depth.
#

set -e  # Exit on error

cd ~/looped_nanochat
uv sync
source .venv/bin/activate

source slurm/machine_config.sh
validate_config || exit 1

# Number of processes/GPUs to use (from machine_config.sh, defaults to 1)
NPROC_PER_NODE=${SLURM_GPUS:-1}

# Parse arguments
# Model tags to evaluate:
MODEL_TAGS=(
    "r4_2.15e18_s12"                      # not sampled
    "r4_sample_2.15e18_s12"               # bs16 sampled
    "r4_sample_init_random_2.15e18_s12"   # bs16 sampled with random init
    "r4_init_random_2.15e18_s12"          # not sampled with random init
)

# Allow single model override via command line
if [ -n "$1" ]; then
    MODEL_TAGS=("$1")
fi

STEP_ARG=""
if [ -n "$2" ]; then
    STEP_ARG="--step=$2"
fi

# Evaluation settings
EVAL_MODES="bpb"  # Can be "core", "bpb", "sample", or any combination
MAX_PER_TASK=500       # Number of examples per CORE task (-1 for all)
SPLIT_TOKENS=$((100 * 524288))  # ~10M tokens for BPB evaluation
DEVICE_BATCH_SIZE=16

# Recurrence values to sweep over
# Typical values: 1 (no recurrence), 2, 4 (training default), 8, 16, 32 (more test-time compute)
RECUR_VALUES=(2 4 8 16 32)

echo "=========================================="
echo "Recurrence Sweep Evaluation"
echo "=========================================="
echo "Model tags: ${MODEL_TAGS[@]}"
echo "Step: ${2:-latest}"
echo "Evaluation modes: $EVAL_MODES"
echo "Recurrence values: ${RECUR_VALUES[@]}"
echo "Max per task: $MAX_PER_TASK"
echo "Split tokens: $SPLIT_TOKENS"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Number of GPUs: $NPROC_PER_NODE"
echo "=========================================="
echo ""

# Outer loop over model tags
for MODEL_TAG in "${MODEL_TAGS[@]}"; do
    echo ""
    echo "######################################"
    echo "### Evaluating Model: $MODEL_TAG"
    echo "######################################"
    echo ""

    # Inner loop over different num_recur values
    for NUM_RECUR in "${RECUR_VALUES[@]}"; do
        echo ""
        echo "=========================================="
        echo "Model: $MODEL_TAG | num_recur=$NUM_RECUR"
        echo "=========================================="

        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval \
            --model-tag="$MODEL_TAG" \
            $STEP_ARG \
            --eval="$EVAL_MODES" \
            --num-recur=$NUM_RECUR \
            --max-per-task=$MAX_PER_TASK \
            --split-tokens=$SPLIT_TOKENS \
            --device-batch-size=$DEVICE_BATCH_SIZE

        echo ""
        echo "Completed: $MODEL_TAG with num_recur=$NUM_RECUR"
        echo ""
    done

    echo ""
    echo "### Completed all recur values for $MODEL_TAG"
    echo ""
done

echo ""
echo "=========================================="
echo "Sweep Complete!"
echo "=========================================="
echo "Results saved in ~/nanochat_data/base_eval/"
echo ""
echo "To analyze results, check:"
echo "  - CSV files: ~/nanochat_data/base_eval/*.csv"
echo "  - Logs: Check wandb or terminal output"
