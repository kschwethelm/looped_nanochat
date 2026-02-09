"""
PCA trajectory visualization for looped transformer latent states.

Shows how each token's latent state evolves through loop iterations in PCA space.
Blue = input tokens, orange = output tokens. Circle = start, square = end.

Example:
    uv run python dev/analysis/visualize_latent_pca_trajectories.py -i sft --num-recur 16
    uv run python dev/analysis/visualize_latent_pca_trajectories.py -i sft --num-recur 4 --sample-idx 3
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from dev.analysis.common import generate_with_latent_tracking
from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K


def plot_pca_trajectories(
    input_latent_states: list[list[torch.Tensor]],
    output_latent_states: list[list[torch.Tensor]],
    num_recur: int,
    plots_dir: Path,
    suffix: str,
    max_trajectories: int = 80,
    seed: int = 42,
):
    """Plot token trajectories in PCA space with arrows showing loop-step evolution."""
    n_input = len(input_latent_states)
    n_output = len(output_latent_states)
    n_total = n_input + n_output

    # Collect all vectors for PCA: iterate tokens, then steps within each token
    all_vectors = []
    for states in input_latent_states:
        for state in states:
            all_vectors.append(state.flatten().float().numpy())
    for states in output_latent_states:
        for state in states:
            all_vectors.append(state.flatten().float().numpy())

    all_vectors = np.stack(all_vectors)

    # Fit PCA on all states jointly
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(all_vectors)
    var1, var2 = pca.explained_variance_ratio_[:2] * 100

    # Reshape: (n_total_tokens, num_recur, 2)
    trajectories = projected.reshape(n_total, num_recur, 2)

    # Subsample tokens if needed, preserving input/output ratio
    rng = np.random.default_rng(seed)
    input_indices = list(range(n_input))
    output_indices = list(range(n_input, n_total))

    if n_total > max_trajectories:
        input_budget = max(1, int(max_trajectories * n_input / n_total))
        output_budget = max(1, max_trajectories - input_budget)
        if len(input_indices) > input_budget:
            input_indices = sorted(
                rng.choice(input_indices, input_budget, replace=False)
            )
        if len(output_indices) > output_budget:
            output_indices = sorted(
                rng.choice(output_indices, output_budget, replace=False)
            )

    show_indices = input_indices + output_indices

    # Colorblind-safe: blue for input, orange for output
    input_cmap = plt.get_cmap("Blues")
    output_cmap = plt.get_cmap("Oranges")

    fig, ax = plt.subplots(figsize=(14, 10))

    for idx in show_indices:
        traj = trajectories[idx]
        is_output = idx >= n_input

        if is_output:
            pos_frac = (idx - n_input) / max(1, n_output - 1)
            color = output_cmap(0.3 + 0.6 * pos_frac)
        else:
            pos_frac = idx / max(1, n_input - 1)
            color = input_cmap(0.3 + 0.6 * pos_frac)

        # Draw arrows between consecutive loop steps
        for step in range(num_recur - 1):
            ax.annotate(
                "",
                xy=(traj[step + 1, 0], traj[step + 1, 1]),
                xytext=(traj[step, 0], traj[step, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.7, alpha=0.5),
            )

        # Mark start (circle) and end (square)
        ax.scatter(
            traj[0, 0], traj[0, 1],
            color=color, s=15, marker="o", zorder=5, alpha=0.7,
        )
        ax.scatter(
            traj[-1, 0], traj[-1, 1],
            color=color, s=25, marker="s", zorder=5, alpha=0.9,
        )

    legend_elements = [
        Line2D([0], [0], color=input_cmap(0.6), lw=2,
               label=f"Input tokens (n={n_input})"),
        Line2D([0], [0], color=output_cmap(0.6), lw=2,
               label=f"Output tokens (n={n_output})"),
        Line2D([0], [0], marker="o", color="gray", markersize=6,
               linestyle="None", label="Start (step 0)"),
        Line2D([0], [0], marker="s", color="gray", markersize=6,
               linestyle="None", label=f"End (step {num_recur - 1})"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    ax.set_xlabel(f"PC 1 ({var1:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC 2 ({var2:.1f}% var)", fontsize=11)
    ax.set_title(
        f"Token Trajectories Through Loop Iterations (PCA)\n"
        f"num_recur={num_recur}, showing {len(show_indices)}/{n_total} tokens",
        fontsize=13,
    )
    ax.grid(True, alpha=0.2)

    output_path = plots_dir / f"gsm8k_latent_pca_trajectories{suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="PCA trajectory visualization for looped transformer latent states"
    )
    parser.add_argument(
        "-i", "--source", type=str, default="sft",
        help="Model source: base|sft|rl (default: sft)",
    )
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., d12)")
    parser.add_argument("--sample-idx", type=int, default=0, help="GSM8K sample index (default: 0)")
    parser.add_argument("--num-recur", type=int, default=16, help="Number of recurrences (default: 16)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--kv-budget", type=int, default=1, help="KV-cache budget (default: 1)")
    parser.add_argument("--use-rec-warm-start", action="store_true", help="Use recurrent warm-start")
    parser.add_argument(
        "--max-trajectories", type=int, default=80,
        help="Max token trajectories to show (subsamples if exceeded, default: 80)",
    )
    args = parser.parse_args()

    # Initialize compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    # Load model
    print(f"Loading model: source={args.source}, tag={args.model_tag}")
    model, tokenizer, meta = load_model(
        args.source, device, phase="eval", model_tag=args.model_tag
    )
    engine = Engine(model, tokenizer)
    num_recur = (
        args.num_recur
        if args.num_recur is not None
        else int(model.config.train_recur_mean)
    )
    print(f"Using num_recur={num_recur}")

    # Load GSM8K and get single sample
    gsm8k = GSM8K(subset="main", split="test")
    conversation = gsm8k.get_example(args.sample_idx)
    question = conversation["messages"][0]["content"]
    prompt_tokens = tokenizer.render_for_completion(conversation)
    print(f"Question: {question[:100]}...")
    print(f"Prompt length: {len(prompt_tokens)} tokens")

    # Generate with latent state tracking
    with autocast_ctx:
        generated_tokens, input_latent_states, output_latent_states = (
            generate_with_latent_tracking(
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
        )

    response = tokenizer.decode(generated_tokens)
    print(f"Generated {len(generated_tokens)} tokens: {response[:100]}...")

    if not input_latent_states and not output_latent_states:
        print("No latent states captured")
        return

    # Generate PCA trajectory plot
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    warmstart = "_warmstart" if args.use_rec_warm_start else ""
    suffix = f"_recur{num_recur}_sample{args.sample_idx}{warmstart}"

    plot_pca_trajectories(
        input_latent_states, output_latent_states,
        num_recur, plots_dir, suffix,
        max_trajectories=args.max_trajectories, seed=args.seed,
    )


if __name__ == "__main__":
    main()
