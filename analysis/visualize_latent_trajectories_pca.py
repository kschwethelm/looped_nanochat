"""
Visualize latent state trajectories using PCA.

This script implements the trajectory visualization described in the Huginn paper:
- Convergence: tokens reaching fixed points
- Orbits: circular/elliptical paths (especially for math tokens)
- Sliders: linear drift patterns
- Path independence: different initializations converging to same paths

Based on: arXiv:2502.05171 "Scaling up Test-Time Compute with Latent Reasoning"
"""

import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d
from sklearn.decomposition import PCA

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init, get_base_dir
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):  # noqa: ARG002
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class LatentStateHook:
    """Hook to capture latent states during recurrent block execution."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset state tracking for a new forward pass."""
        self.states = []

    def __call__(self, module, input, output):  # noqa: ARG002
        """Hook called after each recurrent block execution."""
        # Capture the final token's state after each recurrence
        final_token_state = output[:, -1, :].detach().cpu().clone()
        self.states.append(final_token_state)


def generate_with_tracking(
    engine: Engine,
    tokenizer,
    prompt_tokens: list[int],
    num_recur: int,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_k: int = 50,
    seed: int = 42,
    kv_budget: int = 1,
    use_warm_start: bool = False,
) -> tuple[list[int], list[torch.Tensor], str]:
    """
    Generate response while tracking the final token's latent state trajectory.

    Returns:
        tokens: List of generated token IDs
        latent_states: List of tensors, one per recurrence iteration for the FINAL token
        response: Decoded response text
    """
    hook = LatentStateHook()

    # Register hook on the last recurrent block
    last_recur_block = engine.model.transformer.recur[-1]
    handle = last_recur_block.register_forward_hook(hook)

    try:
        generated_tokens = []
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        bos = tokenizer.get_bos_token_id()

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

            # Stop if we hit terminal token
            if token in (assistant_end, bos):
                break

            generated_tokens.append(token)

        # After generation completes, extract the final token's trajectory
        # The last num_recur states correspond to the final token's evolution
        final_token_states = hook.states[-num_recur:] if len(hook.states) >= num_recur else []

    finally:
        handle.remove()

    response = tokenizer.decode(generated_tokens)
    return generated_tokens, final_token_states, response


def compute_trajectory_metrics(states_pca: np.ndarray) -> dict[str, float]:
    """
    Compute metrics to characterize trajectory behavior.

    Args:
        states_pca: Array of shape (num_recur, n_components) - PCA-reduced states

    Returns:
        Dictionary of metrics
    """
    num_states = len(states_pca)
    if num_states < 2:
        return {"convergence_rate": 0.0, "circularity": 0.0, "path_length": 0.0}

    # Convergence rate: average distance between consecutive states
    dists = np.linalg.norm(np.diff(states_pca, axis=0), axis=1)
    convergence_rate = np.mean(dists)

    # Path length: total trajectory distance
    path_length = np.sum(dists)

    # Circularity: ratio of end-to-end distance to path length
    # Low ratio (~0) suggests circular motion, high ratio (~1) suggests linear drift
    end_to_end = np.linalg.norm(states_pca[-1] - states_pca[0])
    circularity = end_to_end / path_length if path_length > 0 else 1.0

    # Final convergence: distance of last few steps
    final_dists = dists[-3:] if len(dists) >= 3 else dists
    final_convergence = np.mean(final_dists)

    return {
        "convergence_rate": float(convergence_rate),
        "circularity": float(circularity),
        "path_length": float(path_length),
        "final_convergence": float(final_convergence),
        "end_to_end_distance": float(end_to_end),
    }


def classify_trajectory(metrics: dict[str, float]) -> str:
    """
    Classify trajectory based on metrics.

    Returns one of: "converged", "orbit", "slider", "exploring"
    """
    circularity = metrics["circularity"]
    final_conv = metrics["final_convergence"]
    conv_rate = metrics["convergence_rate"]

    # Converged: very small final movements
    if final_conv < 0.05 * conv_rate:
        return "converged"

    # Orbit: low circularity (end close to start) with continued movement
    if circularity < 0.3 and final_conv > 0.02:
        return "orbit"

    # Slider: high circularity (straight-ish path) with continued movement
    if circularity > 0.7:
        return "slider"

    return "exploring"


def plot_trajectory_2d(
    ax: plt.Axes,
    states_pca: np.ndarray,
    title: str,
    metrics: dict[str, float] | None = None,
    show_arrows: bool = True,
):
    """
    Plot a single trajectory in 2D.

    Args:
        ax: Matplotlib axes
        states_pca: Array of shape (num_recur, 2)
        title: Plot title
        metrics: Optional trajectory metrics to display
        show_arrows: Whether to draw arrows between points
    """
    num_states = len(states_pca)
    colors = plt.cm.viridis(np.linspace(0, 1, num_states))

    # Plot trajectory points
    ax.scatter(states_pca[:, 0], states_pca[:, 1], c=colors, s=50, alpha=0.7, zorder=3)

    # Mark start and end
    ax.scatter(
        states_pca[0, 0],
        states_pca[0, 1],
        c="green",
        s=200,
        marker="o",
        label="Start",
        edgecolors="black",
        linewidths=2,
        zorder=4,
    )
    ax.scatter(
        states_pca[-1, 0],
        states_pca[-1, 1],
        c="red",
        s=200,
        marker="X",
        label="End",
        edgecolors="black",
        linewidths=2,
        zorder=4,
    )

    # Draw arrows to show direction
    if show_arrows:
        for i in range(num_states - 1):
            ax.annotate(
                "",
                xy=states_pca[i + 1],
                xytext=states_pca[i],
                arrowprops={
                    "arrowstyle": "->",
                    "color": colors[i],
                    "alpha": 0.4,
                    "lw": 1.5,
                    "shrinkA": 5,
                    "shrinkB": 5,
                },
                zorder=2,
            )

    ax.set_xlabel("PC1", fontsize=11)
    ax.set_ylabel("PC2", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add metrics text if provided
    if metrics:
        pattern = classify_trajectory(metrics)
        text = (
            f"Pattern: {pattern}\n"
            f"Circularity: {metrics['circularity']:.3f}\n"
            f"Path length: {metrics['path_length']:.2f}\n"
            f"Final conv: {metrics['final_convergence']:.4f}"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )


def plot_trajectory_3d(
    ax: Axes3D,
    states_pca: np.ndarray,
    title: str,
    metrics: dict[str, float] | None = None,
    show_arrows: bool = True,
):
    """
    Plot a single trajectory in 3D.

    Args:
        ax: Matplotlib 3D axes
        states_pca: Array of shape (num_recur, 3)
        title: Plot title
        metrics: Optional trajectory metrics to display
        show_arrows: Whether to draw arrows between points
    """
    num_states = len(states_pca)
    colors = plt.cm.viridis(np.linspace(0, 1, num_states))

    # Plot trajectory points
    for i in range(num_states):
        ax.scatter(
            states_pca[i, 0],
            states_pca[i, 1],
            states_pca[i, 2],
            c=[colors[i]],
            s=50,
            alpha=0.7,
        )

    # Mark start and end
    ax.scatter(
        states_pca[0, 0],
        states_pca[0, 1],
        states_pca[0, 2],
        c="green",
        s=200,
        marker="o",
        label="Start",
        edgecolors="black",
        linewidths=2,
    )
    ax.scatter(
        states_pca[-1, 0],
        states_pca[-1, 1],
        states_pca[-1, 2],
        c="red",
        s=200,
        marker="X",
        label="End",
        edgecolors="black",
        linewidths=2,
    )

    # Draw line connecting points
    ax.plot(
        states_pca[:, 0],
        states_pca[:, 1],
        states_pca[:, 2],
        c="gray",
        alpha=0.3,
        linewidth=1,
    )

    # Draw arrows (simpler in 3D)
    if show_arrows and num_states > 1:
        # Only show arrows for every few steps to avoid clutter
        step = max(1, num_states // 5)
        for i in range(0, num_states - 1, step):
            arrow = Arrow3D(
                [states_pca[i, 0], states_pca[i + 1, 0]],
                [states_pca[i, 1], states_pca[i + 1, 1]],
                [states_pca[i, 2], states_pca[i + 1, 2]],
                mutation_scale=20,
                lw=2,
                arrowstyle="->",
                color=colors[i],
                alpha=0.5,
            )
            ax.add_artist(arrow)

    ax.set_xlabel("PC1", fontsize=10)
    ax.set_ylabel("PC2", fontsize=10)
    ax.set_zlabel("PC3", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8)

    # Add metrics text if provided
    if metrics:
        pattern = classify_trajectory(metrics)
        text = f"Pattern: {pattern}\nCirc: {metrics['circularity']:.2f}"
        ax.text2D(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )


def main():
    parser = argparse.ArgumentParser(description="Visualize latent state trajectories using PCA")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="base",
        help="Checkpoint to load (default: base)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of GSM8K samples to process and visualize (default: 6)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in GSM8K dataset (default: 0)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=2,
        choices=[2, 3],
        help="Number of PCA components (2 or 3, default: 2)",
    )
    parser.add_argument(
        "--num-recur",
        type=int,
        default=None,
        help="Number of recurrences (default: use model's train_recur_mean)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-arrows",
        action="store_true",
        help="Don't draw arrows on trajectories",
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
    num_samples = min(args.num_samples, gsm8k.num_examples() - args.start_idx)
    print(f"Processing {num_samples} samples starting from index {args.start_idx}\n")

    # Collect trajectories
    trajectories = []
    questions = []
    responses = []

    with autocast_ctx:
        for i in range(num_samples):
            idx = args.start_idx + i
            print(f"Processing example {i + 1}/{num_samples} (GSM8K index {idx})")

            # Get conversation
            conversation = gsm8k.get_example(idx)
            question = conversation["messages"][0]["content"]

            # Render prompt
            prompt_tokens = tokenizer.render_for_completion(conversation)

            print(f"  Question: {question[:80]}...")
            print(f"  Prompt length: {len(prompt_tokens)} tokens")

            # Generate with tracking
            generated_tokens, latent_states, response = generate_with_tracking(
                engine=engine,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                num_recur=num_recur,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + idx,
                kv_budget=args.kv_budget,
                use_warm_start=args.use_rec_warm_start,
            )

            print(f"  Generated: {len(generated_tokens)} tokens")
            print(f"  Captured: {len(latent_states)} recurrence states")
            print(f"  Response: {response[:80]}...")

            if latent_states and len(latent_states) > 1:
                trajectories.append(latent_states)
                questions.append(question)
                responses.append(response)
            else:
                print("  WARNING: No latent states captured!")

            print()

    if not trajectories:
        print("No trajectories collected! Exiting.")
        return

    print(f"Successfully collected {len(trajectories)} trajectories\n")

    # Determine grid layout
    num_examples = len(trajectories)
    if num_examples <= 2:
        rows, cols = 1, num_examples
    elif num_examples <= 4:
        rows, cols = 2, 2
    elif num_examples <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    # Create figure
    if args.n_components == 3:
        fig = plt.figure(figsize=(6 * cols, 5 * rows))
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if num_examples == 1:
            axes = np.array([axes])
        axes = axes.flatten()

    all_metrics = {}

    print("Generating visualizations...")
    for idx, (latent_states, question, response) in enumerate(
        zip(trajectories, questions, responses)
    ):
        # Stack latent states: shape (num_recur, hidden_dim)
        states_tensor = torch.stack(latent_states)  # (num_recur, batch=1, hidden_dim)
        states_np = states_tensor.squeeze(1).float().numpy()  # (num_recur, hidden_dim)

        # Apply PCA
        pca = PCA(n_components=args.n_components)
        states_pca = pca.fit_transform(states_np)

        # Compute metrics
        metrics = compute_trajectory_metrics(states_pca)
        all_metrics[idx] = metrics

        # Create plot title
        question_preview = question[:50].replace("\n", " ")
        response_preview = response[:30].replace("\n", " ")
        title = f"Ex {idx}: {question_preview}...\nResp: {response_preview}..."

        # Plot trajectory
        if args.n_components == 3:
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            plot_trajectory_3d(ax, states_pca, title, metrics, not args.no_arrows)
        else:
            ax = axes[idx]
            plot_trajectory_2d(ax, states_pca, title, metrics, not args.no_arrows)

        # Print metrics
        pattern = classify_trajectory(metrics)
        print(f"Example {idx}: {pattern}")
        print(f"  Question: {question_preview}...")
        print(f"  Response: {response_preview}...")
        print(f"  Circularity: {metrics['circularity']:.3f}")
        print(f"  Path length: {metrics['path_length']:.2f}")
        print(f"  Final convergence: {metrics['final_convergence']:.4f}")
        print(
            f"  Explained variance: {pca.explained_variance_ratio_[: args.n_components].sum():.3f}"
        )
        print()

    # Remove extra subplots
    if args.n_components == 2:
        for idx in range(num_examples, len(axes)):
            fig.delaxes(axes[idx])

    plt.suptitle(
        f"Latent State Trajectories ({args.n_components}D PCA Projection, num_recur={num_recur}, kv_budget={args.kv_budget})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    # Summary statistics
    print("=" * 80)
    print("TRAJECTORY PATTERN SUMMARY")
    print("=" * 80)

    patterns = [classify_trajectory(m) for m in all_metrics.values()]
    pattern_counts = {p: patterns.count(p) for p in set(patterns)}
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")

    avg_circularity = np.mean([m["circularity"] for m in all_metrics.values()])
    avg_path_length = np.mean([m["path_length"] for m in all_metrics.values()])
    print(f"\nAverage circularity: {avg_circularity:.3f}")
    print(f"Average path length: {avg_path_length:.2f}")

    # Save figure
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    warmstart_suffix = "_warmstart" if args.use_rec_warm_start else ""
    output_path = plots_dir / f"latent_trajectories_{args.n_components}d_recur{num_recur}_kvbudget{args.kv_budget}{warmstart_suffix}.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
