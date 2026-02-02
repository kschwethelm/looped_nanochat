"""
Track latent states during GSM8K response generation and visualize L2 distances.

This script:
1. Loads one GSM8K test case
2. Generates a response using the looped transformer
3. Captures the recurrent state after each loop iteration for ALL tokens
4. Plots a heatmap showing L2 distance between consecutive loop steps per token

The latent state tracking is done via PyTorch hooks on the final recurrent block.
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K


class LatentStateHook:
    """Hook to capture latent states during recurrent block execution."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset state tracking for a new forward pass."""
        self.states = []

    def __call__(self, module, input, output):  # noqa: ARG002
        """Hook called after each recurrent block execution."""
        # output is the activations after the last recurrent block: shape (B, T, hidden_dim)
        # We capture states for the most recently generated token (final position)
        final_token_state = output[:, -1, :].detach().cpu().clone()
        self.states.append(final_token_state)


def generate_with_latent_tracking(
    engine: Engine,
    tokenizer,
    prompt_tokens,
    num_recur,
    max_tokens=512,
    temperature=0.7,
    top_k=50,
    seed=42,
    kv_budget=1,
    use_warm_start=False,
):
    """
    Generate response while tracking latent states for ALL tokens.

    Returns:
        tokens: list of generated token ids
        latent_states_per_token: list of lists, where each inner list contains
                                num_recur tensors (one per loop iteration) for that token
    """
    hook = LatentStateHook()

    # Register hook on the last recurrent block
    last_recur_block = engine.model.transformer.recur[-1]
    handle = last_recur_block.register_forward_hook(hook)

    try:
        generated_tokens = []
        latent_states_per_token = []
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        bos = tokenizer.get_bos_token_id()

        states_before_token = 0

        for token_column, _token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            num_recur=num_recur,
            kv_budget=kv_budget,
            use_warm_start=use_warm_start,
        ):
            token = token_column[0]  # batch size is 1

            # Stop if we hit a terminal token
            if token in (assistant_end, bos):
                break

            generated_tokens.append(token)

            # Capture latent states for this token
            states_after_token = len(hook.states)
            new_states = hook.states[states_before_token:states_after_token]

            # Store all num_recur states for this token
            if len(new_states) >= num_recur:
                latent_states_per_token.append(new_states[-num_recur:])

            states_before_token = states_after_token

    finally:
        handle.remove()

    return generated_tokens, latent_states_per_token


def plot_latent_state_distances(
    tokens: list[int],
    latent_states_per_token: list[list[torch.Tensor]],
    tokenizer,
    num_recur: int,
    question: str,
    response: str,
    kv_budget: int = 1,
    use_warm_start: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Plot heatmap of L2 distances between consecutive loop steps for each token.

    Args:
        tokens: List of generated token IDs
        latent_states_per_token: List of lists, each containing num_recur states
        tokenizer: Tokenizer for decoding tokens
        num_recur: Number of recurrence iterations
        question: The input question
        response: The decoded response text
        kv_budget: KV-cache budget used during generation
        use_warm_start: Whether warm start was used
        vmin: Minimum value for colorbar (in log scale). If None, auto-scales.
        vmax: Maximum value for colorbar (in log scale). If None, auto-scales.
    """
    num_tokens = len(latent_states_per_token)

    # Compute L2 distances between consecutive loop steps
    # Shape: (num_tokens, num_recur - 1)
    distances = np.zeros((num_tokens, num_recur - 1))

    for token_idx, states in enumerate(latent_states_per_token):
        for recur_idx in range(num_recur - 1):
            state1 = states[recur_idx].flatten()
            state2 = states[recur_idx + 1].flatten()
            l2_dist = torch.norm(state2 - state1, p=2).item()
            distances[token_idx, recur_idx] = l2_dist

    # Apply log transform (add small epsilon to avoid log(0))
    log_distances = np.log10(distances + 1e-10)

    # Create plot with extra space for response text
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 5], hspace=0.3)

    # Top panel: show the decoded response
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis("off")
    response_preview = response[:500] + "..." if len(response) > 500 else response
    ax_text.text(
        0.5,
        0.5,
        f"Response: {response_preview}",
        ha="center",
        va="center",
        fontsize=9,
        wrap=True,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.3},
    )

    # Bottom panel: heatmap
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(
        log_distances,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    # Set labels
    ax.set_xlabel("Recursion Transition", fontsize=12)
    ax.set_title(
        f"Log L2 Distance Between Consecutive Loop Steps (num_recur={num_recur}, kv_budget={kv_budget})\n{question[:80]}...",
        fontsize=13,
    )

    # Set x-axis ticks
    ax.set_xticks(range(num_recur - 1))
    ax.set_xticklabels([f"{i}→{i + 1}" for i in range(num_recur - 1)])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log₁₀(L2 Distance)", fontsize=11)

    # Add token text as y-axis labels
    token_texts = [tokenizer.decode([t]) for t in tokens]
    ax.set_yticks(range(num_tokens))

    if num_tokens <= 50:
        # Show full token text for small responses
        display_texts = [t[:20] for t in token_texts]
        ax.set_yticklabels(display_texts, fontsize=8)
    elif num_tokens <= 100:
        # Show every other token for medium responses
        ax.set_yticks(range(0, num_tokens, 2))
        display_texts = [token_texts[i][:15] for i in range(0, num_tokens, 2)]
        ax.set_yticklabels(display_texts, fontsize=7)
    else:
        # Show every 10th token for large responses
        stride = max(1, num_tokens // 20)
        ax.set_yticks(range(0, num_tokens, stride))
        display_texts = [token_texts[i][:15] for i in range(0, num_tokens, stride)]
        ax.set_yticklabels(display_texts, fontsize=7)

    ax.set_ylabel("Token", fontsize=12)

    plt.tight_layout()

    # Save to plots directory
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    warmstart_suffix = "_warmstart" if use_warm_start else ""
    output_path = (
        plots_dir
        / f"gsm8k_latent_state_distances_recur{num_recur}_kvbudget{kv_budget}{warmstart_suffix}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Track and plot latent states for one GSM8K sample"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sft",
        help="Checkpoint to load (default: base)",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="GSM8K sample index to process (default: 0)",
    )
    parser.add_argument(
        "--num-recur",
        type=int,
        default=16,
        help="Number of recurrences (default: use model's train_recur_mean)",
    )
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        help="Use teacher forcing with ground truth tokens instead of free generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 512, ignored for teacher forcing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0, ignored for teacher forcing)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50, ignored for teacher forcing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42, ignored for teacher forcing)",
    )
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=1,
        help="Fixed KV-cache budget for recurrences (default: 1)",
    )
    parser.add_argument(
        "--use-rec-warm-start",
        action="store_true",
        help="Use recurrent warm-start (carry recurrent state when decoding tokens)",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Minimum value for colorbar (log scale). If not set, auto-scales.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Maximum value for colorbar (log scale). If not set, auto-scales.",
    )
    args = parser.parse_args()

    # Initialize compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    if ddp:
        print("Warning: Running with DDP, but this script is designed for single GPU")

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, tokenizer, meta = load_model(args.checkpoint, device, phase="eval")
    engine = Engine(model, tokenizer)

    # Determine num_recur
    num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
    print(f"Using num_recur={num_recur}")

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    gsm8k = GSM8K(subset="main", split="test")

    # Get single sample
    print(f"\nProcessing sample {args.sample_idx}")
    conversation = gsm8k.get_example(args.sample_idx)
    question = conversation["messages"][0]["content"]

    # Render prompt for completion
    prompt_tokens = tokenizer.render_for_completion(conversation)

    print(f"Question: {question}")
    print(f"Prompt length: {len(prompt_tokens)} tokens")

    # Generate with latent state tracking
    with autocast_ctx:
        generated_tokens, latent_states_per_token = generate_with_latent_tracking(
            engine=engine,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            num_recur=num_recur,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            kv_budget=args.kv_budget,
            use_warm_start=args.use_rec_warm_start,
        )

    # Decode response
    response = tokenizer.decode(generated_tokens)
    print(f"\nGenerated {len(generated_tokens)} tokens")
    print(f"Captured latent states for {len(latent_states_per_token)} tokens")
    print(f"Response: {response}")

    # Plot the results
    if latent_states_per_token:
        plot_latent_state_distances(
            tokens=generated_tokens,
            latent_states_per_token=latent_states_per_token,
            tokenizer=tokenizer,
            num_recur=num_recur,
            question=question,
            response=response,
            kv_budget=args.kv_budget,
            use_warm_start=args.use_rec_warm_start,
            vmin=args.vmin,
            vmax=args.vmax,
        )
    else:
        print("No latent states captured for plotting")


if __name__ == "__main__":
    main()
