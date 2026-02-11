"""
Print model structure and parameter dimensions for a randomly initialized GPT model.

Usage:
    python -m dev.print_model_structure --size=20
"""

import argparse
import math

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


def print_model_structure(config: GPTConfig, *, verbose: bool = False) -> None:
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

    # Group parameters by execution pattern
    print("Embedding & Output:")
    print(f"  {'wte':24s}: {param_counts['wte']:>15,}")
    print(f"  {'lm_head':24s}: {param_counts['lm_head']:>15,}")

    print("\nTransformer Layers (by execution pattern):")
    print(f"  {'prelude (1× per fwd)':24s}: {param_counts['prelude']:>15,}")
    print(f"  {'recur_block (r× per fwd)':24s}: {param_counts['recur_block']:>15,}")
    print(f"  {'coda (1× per fwd)':24s}: {param_counts['coda']:>15,}")
    print(f"  {'inject (r× per fwd)':24s}: {param_counts['inject']:>15,}")

    print("\nOther:")
    print(f"  {'scalars (norms)':24s}: {param_counts['scalars']:>15,}")
    print(f"  {'value_embeds':24s}: {param_counts['value_embeds']:>15,}")

    print("\nTotals:")
    print(f"  {'total':24s}: {param_counts['total']:>15,}")

    # Show effective parameters for different recurrence depths
    print("\nEffective Parameters (accounting for reuse):")
    num_recur = int(config.train_recur_mean)
    eff_params = model.effective_params(num_recur=num_recur)
    multiplier = eff_params / param_counts['total']
    print(f"  At r={num_recur} (train mean):   {eff_params:>15,}  ({multiplier:.2f}× total)")

    # Show a few other recurrence depths for reference
    for r in [1, 8, 16]:
        if r != num_recur:
            eff = model.effective_params(num_recur=r)
            mult = eff / param_counts['total']
            print(f"  At r={r:2d}:              {eff:>15,}  ({mult:.2f}× total)")

    print(f"\nEstimated FLOPs:     {model.estimate_flops():e} per token (at r={int(config.train_recur_mean)})")
    print()

    if verbose:
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

    return model


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
    # Training horizon (same as base_train.py)
    parser.add_argument("--num-iterations", type=int, default=-1, help="explicit number of optimization steps (-1 = disable)")
    parser.add_argument("--target-flops", type=float, default=-1.0, help="calculate num_iterations to reach target_flops (-1 = disable)")
    parser.add_argument(
        "--target-param-data-ratio",
        type=int,
        default=7,
        help="calculate num_iterations to maintain data:param ratio (accounts for parameter reuse + slight overtrain), -1 = disable)",
    )
    parser.add_argument("--total-batch-size", type=int, default=-1, help="total batch size in tokens (-1 = auto-compute via Power Lines paper)")
    parser.add_argument("--verbose", action="store_true", help="print detailed parameter structure and module summary")
    parser.add_argument("--weight-decay", type=float, default=0.2, help="reference weight decay (before scaling)")
    parser.add_argument("--embedding-lr", type=float, default=0.3, help="reference embedding LR (before scaling)")
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="reference unembedding LR (before scaling)")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="reference matrix/Muon LR (before scaling)")
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
    model = print_model_structure(config, verbose=args.verbose)

    # Training horizon, batch size, LR, and weight decay scaling (mirrors base_train.py logic)
    # Refs: Power Lines (https://arxiv.org/abs/2505.13738), T_epoch (https://arxiv.org/abs/2405.13698)
    num_flops_per_token = model.estimate_flops()
    num_scaling_params = model.effective_params(num_recur=int(config.train_recur_mean))

    # Reference s12 model
    B_REF = 2**19
    ref_dim = 12 * args.aspect_ratio
    ref_dim = ((ref_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    config_kwargs = dict(
        sequence_len=args.max_seq_len, vocab_size=vocab_size, size=12,
        n_head=ref_dim // args.head_dim, n_kv_head=ref_dim // args.head_dim, n_embd=ref_dim,
        window_pattern=args.window_pattern, n_prelude=args.n_prelude, n_recur_block=args.n_recur_block,
        n_coda=args.n_coda, train_recur_mean=args.train_recur_mean, train_recur_max=args.train_recur_max,
        bptt_k=args.bptt_k,
    )
    with torch.device("meta"):
        ref_model = GPT(GPTConfig(**config_kwargs))
    ref_scaling_params = ref_model.effective_params(num_recur=int(config.train_recur_mean))
    del ref_model

    # 1) Calculate target training tokens
    if args.num_iterations > 0:
        assert args.total_batch_size > 0, "Must specify --total-batch-size when using --num-iterations"
        total_batch_size = args.total_batch_size
        target_tokens = args.num_iterations * total_batch_size
        num_iterations = args.num_iterations
        source = "explicit"
    elif args.target_flops > 0:
        target_tokens = round(args.target_flops / num_flops_per_token)
        source = f"target_flops={args.target_flops:e}"
    elif args.target_param_data_ratio > 0:
        target_tokens = round(args.target_param_data_ratio * num_scaling_params)
        source = f"target_param_data_ratio={args.target_param_data_ratio}"
    else:
        num_iterations = None
        source = None

    if source is not None:
        D_REF = (target_tokens / num_scaling_params) * ref_scaling_params

        # 2) Auto-compute optimal batch size: Bopt ∝ D^0.383 (Power Lines paper)
        if args.num_iterations <= 0:
            total_batch_size = args.total_batch_size
            if total_batch_size == -1:
                batch_size_ratio = target_tokens / D_REF
                predicted_batch_size = B_REF * batch_size_ratio ** 0.383
                total_batch_size = 2 ** round(math.log2(predicted_batch_size))
            num_iterations = round(target_tokens / total_batch_size)

        # 3) LR scaling: η ∝ √(B/B_ref)
        batch_ratio = total_batch_size / B_REF
        batch_lr_scale = batch_ratio ** 0.5 if batch_ratio != 1.0 else 1.0

        # 4) Weight decay scaling via T_epoch: λ = λ_ref · √(B/B_ref) · (D_ref/D)
        weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)

        total_tokens = total_batch_size * num_iterations
        param_data_ratio = total_tokens / num_scaling_params
        total_flops = num_flops_per_token * total_tokens

        print("=" * 80)
        print(f"Training Horizon & Scaling ({source})")
        print("=" * 80)
        print(f"  Total batch size:      {total_batch_size:,}  (ref: {B_REF:,})")
        print(f"  Num iterations:        {num_iterations:,}")
        print(f"  Total tokens:          {total_tokens:,}")
        print(f"  Total FLOPs:           {total_flops:e}")
        print(f"  Tokens:Params ratio:   {param_data_ratio:.2f}")
        print()
        print(f"  LR scale factor:       {batch_lr_scale:.4f}")
        print(f"    Embedding LR:        {args.embedding_lr} -> {args.embedding_lr * batch_lr_scale:.6f}")
        print(f"    Unembedding LR:      {args.unembedding_lr} -> {args.unembedding_lr * batch_lr_scale:.6f}")
        print(f"    Matrix/Muon LR:      {args.matrix_lr} -> {args.matrix_lr * batch_lr_scale:.6f}")
        print(f"  Weight decay:          {args.weight_decay} -> {weight_decay_scaled:.6f}")
        print("=" * 80)
