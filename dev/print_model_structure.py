"""
Print model structure and parameter dimensions for a randomly initialized GPT model.

Usage:
    python -m dev.print_model_structure --depth=20
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
    print(f"Number of layers:    {config.n_layer}")
    print(f"Number of heads:     {config.n_head}")
    print(f"Number of KV heads:  {config.n_kv_head}")
    print(f"Embedding dim:       {config.n_embd}")
    print(f"Window pattern:      {config.window_pattern}")
    print()

    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    print("=" * 80)
    print("Parameter Counts")
    print("=" * 80)
    print(f"Total parameters:    {total_params:,}")
    print(f"Scaling parameters:  {model.num_scaling_params():,}")
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
    parser.add_argument(
        "--depth", type=int, default=20, help="Number of transformer layers"
    )
    parser.add_argument(
        "--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio"
    )
    parser.add_argument(
        "--head-dim", type=int, default=128, help="Target head dimension"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=2048, help="Max context length"
    )
    parser.add_argument(
        "--window-pattern", type=str, default="SSSL", help="Sliding window pattern"
    )
    args = parser.parse_args()

    # Tokenizer (for vocab size)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    # Model dimensions (same logic as base_train.py)
    n_layer = args.depth
    base_dim = args.depth * args.aspect_ratio
    n_embd = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    n_head = n_embd // args.head_dim
    n_kv_head = n_head  # default is 1:1 GQA ratio (i.e. GQA is disabled)
    head_dim = n_embd // n_head

    print("Calculated dimensions:")
    print(f"  n_layer: {n_layer}")
    print(f"  n_embd: {n_embd} (base: {base_dim}, nudge: {n_embd - base_dim:+d})")
    print(f"  n_head: {n_head}")
    print(f"  head_dim: {head_dim}")
    print(f"  n_kv_head: {n_kv_head}")
    print()

    # Create config
    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embd=n_embd,
        window_pattern=args.window_pattern,
    )

    # Print structure
    print_model_structure(config)
