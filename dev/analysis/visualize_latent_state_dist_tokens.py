"""
Track latent states during response generation and visualize L2 distances.

This script:
1. Loads one test case from a chosen evaluation dataset
2. Generates a response using the looped transformer
3. Captures the recurrent state after each loop iteration for ALL tokens
4. Plots a heatmap showing L2 distance between consecutive loop steps per token

The latent state tracking is done via PyTorch hooks on the final recurrent block.

Usage:
    uv run python dev/analysis/visualize_latent_state_dist_tokens.py -i sft --num-recur 16 --sample-idx 0
    uv run python dev/analysis/visualize_latent_state_dist_tokens.py -i sft --num-recur 16 --task-name ARC-Easy --sample-idx 5
"""

import argparse
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from dev.analysis.common import generate_with_latent_tracking
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

TASK_MODULES = {
    "GSM8K": partial(GSM8K, subset="main", split="test"),
    "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "MMLU": partial(MMLU, subset="all", split="test"),
    "HumanEval": HumanEval,
    "SpellingBee": partial(SpellingBee, size=256, split="test"),
}


def plot_latent_state_distances(
    input_tokens: list[int],
    output_tokens: list[int],
    input_latent_states: list[list[torch.Tensor]],
    output_latent_states: list[list[torch.Tensor]],
    tokenizer,
    num_recur: int,
    task_name: str = "GSM8K",
    kv_budget: int = 1,
    use_warm_start: bool = False,
    model_tag: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Plot heatmaps of L2 distances between consecutive loop steps for input and output tokens.

    Args:
        input_tokens: List of input token IDs
        output_tokens: List of generated token IDs
        input_latent_states: List of lists, each containing num_recur states for input tokens
        output_latent_states: List of lists, each containing num_recur states for output tokens
        tokenizer: Tokenizer for decoding tokens
        num_recur: Number of recurrence iterations
        kv_budget: KV-cache budget used during generation
        use_warm_start: Whether warm start was used
        vmin: Minimum value for colorbar (default: 1e-6). If None, auto-scales.
        vmax: Maximum value for colorbar. If None, auto-scales.
    """

    def compute_distances(latent_states_per_token):
        """Compute L2 distances for a list of token states."""
        num_tokens = len(latent_states_per_token)
        distances = np.zeros((num_tokens, num_recur - 1))

        for token_idx, states in enumerate(latent_states_per_token):
            for recur_idx in range(num_recur - 1):
                state1 = states[recur_idx].flatten()
                state2 = states[recur_idx + 1].flatten()
                l2_dist = torch.norm(state2 - state1, p=2).item()
                distances[token_idx, recur_idx] = l2_dist

        return distances

    # Compute distances for input and output tokens
    input_distances = compute_distances(input_latent_states) if input_latent_states else None
    output_distances = compute_distances(output_latent_states) if output_latent_states else None

    # Determine shared color scale
    all_distances = []
    if input_distances is not None:
        all_distances.append(input_distances)
    if output_distances is not None:
        all_distances.append(output_distances)
    shared_vmin = vmin or min(d[d > 0].min() for d in all_distances if d[d > 0].size > 0)
    shared_vmax = vmax or max(d.max() for d in all_distances)
    shared_norm = LogNorm(vmin=shared_vmin, vmax=shared_vmax)

    # Scale figure height to the larger token count so cells stay ~square
    n_input = len(input_latent_states) if input_latent_states else 0
    n_output = len(output_latent_states) if output_latent_states else 0
    max_tokens = max(n_input, n_output, 1)
    cell_h = 0.18
    fig_height = max(6, min(24, max_tokens * cell_h + 2.5))

    # Grid: two heatmap columns + thin colorbar column
    fig = plt.figure(figsize=(28, fig_height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.03], wspace=0.12)

    axes = []
    last_im = None

    # Left panel: Input tokens heatmap
    if input_distances is not None and n_input > 0:
        ax_input = fig.add_subplot(gs[0, 0])
        im_input = ax_input.imshow(
            input_distances,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            norm=shared_norm,
        )

        ax_input.set_xlabel("Recursion Transition", fontsize=11)
        ax_input.set_ylabel("Input Token Position", fontsize=11)
        ax_input.set_title(
            f"Input Tokens (n={n_input})\nL2 Distance Between Loop Steps",
            fontsize=12,
        )

        ax_input.set_xticks(range(num_recur - 1))
        ax_input.set_xticklabels([f"{i}→{i + 1}" for i in range(num_recur - 1)])

        input_token_texts = [tokenizer.decode([t]) for t in input_tokens]
        ax_input.set_yticks(range(n_input))
        display_texts = [t[:15] for t in input_token_texts]
        tick_fontsize = max(3, min(7, 200 // n_input))
        ax_input.set_yticklabels(display_texts, fontsize=tick_fontsize)

        axes.append(ax_input)
        last_im = im_input

    # Right panel: Output tokens heatmap
    if output_distances is not None and n_output > 0:
        ax_output = fig.add_subplot(gs[0, 1])
        im_output = ax_output.imshow(
            output_distances,
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
            norm=shared_norm,
        )

        ax_output.set_xlabel("Recursion Transition", fontsize=11)
        ax_output.set_ylabel("Output Token Position", fontsize=11)
        ax_output.set_title(
            f"Output Tokens (n={n_output})\nL2 Distance Between Loop Steps",
            fontsize=12,
        )

        ax_output.set_xticks(range(num_recur - 1))
        ax_output.set_xticklabels([f"{i}→{i + 1}" for i in range(num_recur - 1)])

        output_token_texts = [tokenizer.decode([t]) for t in output_tokens]
        ax_output.set_yticks(range(n_output))
        display_texts = [t[:15] for t in output_token_texts]
        tick_fontsize = max(3, min(7, 200 // n_output))
        ax_output.set_yticklabels(display_texts, fontsize=tick_fontsize)

        axes.append(ax_output)
        last_im = im_output

    # Shared colorbar in the third column
    if last_im is not None:
        cbar_ax = fig.add_subplot(gs[0, 2])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("L2 Distance", fontsize=10)

    plt.suptitle(
        f"Latent State L2 Distances (num_recur={num_recur}, kv_budget={kv_budget})",
        fontsize=14,
        y=0.995,
    )

    # Save to plots directory
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    model_tag_suffix = f"_{model_tag}" if model_tag else ""
    warmstart_suffix = "_warmstart" if use_warm_start else ""
    task_slug = task_name.lower().replace("-", "_")
    output_path = (
        plots_dir
        / f"{task_slug}{model_tag_suffix}_latent_state_distances_recur{num_recur}_kvbudget{kv_budget}{warmstart_suffix}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Track and plot latent states for one evaluation sample"
    )
    parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: base|sft|rl (default: sft)")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., s12). If not specified, uses largest model.")
    parser.add_argument("-a", "--task-name", type=str, default="GSM8K", choices=list(TASK_MODULES.keys()), help="Evaluation task to load (default: GSM8K)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to process (default: 0)")
    parser.add_argument("--num-recur", type=int, default=16, help="Number of recurrences (default: 16)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum tokens to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--kv-budget", type=int, default=1, help="Fixed KV-cache budget for recurrences (default: 1)")
    parser.add_argument("--use-rec-warm-start", action="store_true", help="Use recurrent warm-start (carry recurrent state when decoding tokens)")
    parser.add_argument("--vmin", type=float, default=1e-2, help="Minimum value for colorbar (default: 1e-2). If not set, auto-scales.")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum value for colorbar. If not set, auto-scales.")
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

    print(f"Loading model from source: {args.source}" + (f", model_tag: {args.model_tag}" if args.model_tag else ""))
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)

    # Determine num_recur
    num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
    print(f"Using num_recur={num_recur}")

    # Load dataset
    print(f"Loading {args.task_name} dataset...")
    task_object = TASK_MODULES[args.task_name]()

    # Get single sample
    print(f"\nProcessing sample {args.sample_idx}")
    conversation = task_object.get_example(args.sample_idx)
    question = conversation["messages"][0]["content"]

    # Render prompt for completion
    prompt_tokens = tokenizer.render_for_completion(conversation)

    print(f"Question: {question}")
    print(f"Prompt length: {len(prompt_tokens)} tokens")

    # Generate with latent state tracking
    with autocast_ctx:
        generated_tokens, input_latent_states, output_latent_states, _ = generate_with_latent_tracking(
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
    print(f"Captured latent states for {len(input_latent_states)} input tokens + {len(output_latent_states)} output tokens")
    print(f"Response: {response}")

    # Plot the results
    if input_latent_states or output_latent_states:
        plot_latent_state_distances(
            input_tokens=prompt_tokens,
            output_tokens=generated_tokens,
            input_latent_states=input_latent_states,
            output_latent_states=output_latent_states,
            tokenizer=tokenizer,
            num_recur=num_recur,
            task_name=args.task_name,
            kv_budget=args.kv_budget,
            use_warm_start=args.use_rec_warm_start,
            model_tag=args.model_tag,
            vmin=args.vmin,
            vmax=args.vmax,
        )
    else:
        print("No latent states captured for plotting")


if __name__ == "__main__":
    main()
