"""
Print model structure and parameter dimensions for a randomly initialized GPT model.

Usage:
    python -m dev.print_model_structure --size=20
"""

import argparse

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


def print_model_structure(config: GPTConfig) -> None:
    """Print detailed model structure with parameter dimensions."""

    # Initialize model with random weights
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()

    print("=" * 80)
    print("GPT Model Configuration")
    print("=" * 80)
    print(f"Sequence length:     {config.sequence_len}")
    print(f"Vocab size:          {config.vocab_size}")
    print(f"Size (width knob):   {config.size}")
    print(f"Number of heads:     {config.n_head}")
    print(f"Number of KV heads:  {config.n_kv_head}")
    print(f"Embedding dim:       {config.n_embd}")
    print(f"Window pattern:      {config.window_pattern}")
    print()
    print("Looped Transformer Configuration:")
    print(f"  Prelude layers:    {config.n_prelude}")
    print(f"  Recurrent block:   {config.n_recur_block}")
    print(f"  Coda layers:       {config.n_coda}")
    print(f"  Train recur mean:  {config.train_recur_mean}")
    print(f"  Train recur max:   {config.train_recur_max}")
    print(f"  BPTT-k:            {config.bptt_k}")
    print()

    # Print parameter counts
    param_counts = model.num_scaling_params()
    print("=" * 80)
    print("Parameter Counts")
    print("=" * 80)
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    print(f"Estimated FLOPs:     {model.estimate_flops():e} per token")
    print()

    # Print detailed parameter structure
    print("=" * 80)
    print("Detailed Parameter Structure")
    print("=" * 80)

    for name, param in model.named_parameters():
        print(f"{name:60s} {str(tuple(param.shape)):30s} {param.numel():>15,}")

    print("=" * 80)

    # Print module structure summary
    print()
    print("=" * 80)
    print("Module Structure Summary")
    print("=" * 80)
    print(model)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print GPT model structure")
    parser.add_argument("--size", type=int, default=20, help="model size (model_dim = size * aspect_ratio)")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = size * aspect_ratio")
    parser.add_argument("--head-dim", type=int, default=128, help="Target head dimension")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Max context length")
    parser.add_argument("--window-pattern", type=str, default="LLSSSLLL", help="Sliding window pattern")
    # Looped Transformer config
    parser.add_argument("--n-prelude", type=int, default=2, help="Number of prelude layers")
    parser.add_argument(
        "--n-recur-block",
        type=int,
        default=4,
        help="Number of layers in the recurrent block",
    )
    parser.add_argument("--n-coda", type=int, default=2, help="Number of coda layers")
    parser.add_argument(
        "--train-recur-mean",
        type=float,
        default=4.0,
        help="Mean recurrences during training (also default r at inference); r=4 gives 20 effective layers",
    )
    parser.add_argument(
        "--train-recur-max",
        type=int,
        default=16,
        help="Max recurrences sampled during training",
    )
    parser.add_argument(
        "--bptt-k",
        type=int,
        default=4,
        help="Truncate backprop to last k recurrences (limits gradient depth)",
    )
    args = parser.parse_args()

    # Tokenizer (for vocab size)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Model dimensions (same logic as base_train.py)
    size = args.size
    base_dim = args.size * args.aspect_ratio
    n_embd = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    n_head = n_embd // args.head_dim
    n_kv_head = n_head  # default is 1:1 GQA ratio (i.e. GQA is disabled)
    head_dim = n_embd // n_head

    print("Calculated dimensions:")
    print(f"  size: {size}")
    print(f"  n_embd: {n_embd} (base: {base_dim}, nudge: {n_embd - base_dim:+d})")
    print(f"  n_head: {n_head}")
    print(f"  head_dim: {head_dim}")
    print(f"  n_kv_head: {n_kv_head}")
    print()

    # Create config
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        size=size,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=args.window_pattern,
        # Looped Transformer config
        n_prelude=args.n_prelude,
        n_recur_block=args.n_recur_block,
        n_coda=args.n_coda,
        train_recur_mean=args.train_recur_mean,
        train_recur_max=args.train_recur_max,
        bptt_k=args.bptt_k,
    )

    # Print structure
    print_model_structure(config)
