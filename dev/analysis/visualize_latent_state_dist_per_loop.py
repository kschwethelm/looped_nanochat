"""
Track latent states during response generation and visualize average distances per loop step.

This script:
1. Loads one or more test cases from a chosen evaluation dataset
2. Generates responses using the looped transformer
3. Captures the recurrent state after each loop iteration for ALL tokens (input + output)
   - Input tokens: captured during prefill pass
   - Output tokens: captured during generation
4. Plots average L2 distance, cosine similarity, and KL divergence per loop step for:
   - Full sequence
   - Input sequence only (actual prompt tokens)
   - Output sequence only (generated tokens)

The KL divergence measures how the output distribution (after coda + lm_head) changes
between consecutive loop steps: KL(p_{i+1} || p_i).

The latent state tracking is done via PyTorch hooks on the final recurrent block.

Example:
    uv run python dev/analysis/visualize_latent_state_dist_per_loop.py -i sft --num-recur 16 --num-samples 5
    uv run python dev/analysis/visualize_latent_state_dist_per_loop.py -i sft --num-recur 16 --task-name ARC-Easy --num-samples 5
"""

import argparse
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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


def compute_distance_metrics(
    latent_states_per_token: list[list[torch.Tensor]],
    num_recur: int,
    num_input_tokens: int,
):
    """
    Compute average L2 distance and cosine similarity per loop step for a single sample.

    Returns:
        Dict with keys 'l2', 'cosine', 'relative_l2', 'state_norms', each containing:
            'full': (means, stds, medians, q25, q75) for full sequence
            'input': (means, stds, medians, q25, q75) for input sequence only
            'output': (means, stds, medians, q25, q75) for output sequence only
    """
    num_tokens = len(latent_states_per_token)

    # Initialize arrays to store distances for each transition
    # Shape: (num_tokens, num_recur - 1) for distances/similarities
    # Shape: (num_tokens, num_recur) for state norms
    l2_distances = np.zeros((num_tokens, num_recur - 1))
    cosine_sims = np.zeros((num_tokens, num_recur - 1))
    relative_l2_distances = np.zeros((num_tokens, num_recur - 1))
    state_norms = np.zeros((num_tokens, num_recur))

    # Compute distances for each token and each loop transition
    for token_idx, states in enumerate(latent_states_per_token):
        for recur_idx in range(num_recur):
            state = states[recur_idx].flatten()

            # State norm
            state_norms[token_idx, recur_idx] = torch.norm(state, p=2).item()

            if recur_idx < num_recur - 1:
                state2 = states[recur_idx + 1].flatten()

                # L2 distance
                l2_dist = torch.norm(state2 - state, p=2).item()
                l2_distances[token_idx, recur_idx] = l2_dist

                # Relative L2 distance (normalized by current state norm)
                state_norm = state_norms[token_idx, recur_idx]
                relative_l2_distances[token_idx, recur_idx] = l2_dist / state_norm if state_norm > 0 else 0

                # Cosine similarity
                cosine_sim = F.cosine_similarity(state.unsqueeze(0), state2.unsqueeze(0)).item()
                cosine_sims[token_idx, recur_idx] = cosine_sim

    def compute_stats(data, axis=0):
        """Compute mean, std, median, and IQR."""
        means = np.mean(data, axis=axis)
        stds = np.std(data, axis=axis)
        medians = np.median(data, axis=axis)
        q25 = np.percentile(data, 25, axis=axis)
        q75 = np.percentile(data, 75, axis=axis)
        return means, stds, medians, q25, q75

    # Compute statistics for full sequence
    l2_full = compute_stats(l2_distances)
    cos_full = compute_stats(cosine_sims)
    rel_l2_full = compute_stats(relative_l2_distances)
    norms_full = compute_stats(state_norms)

    # Compute statistics for input sequence only (first num_input_tokens)
    if num_input_tokens > 0:
        l2_input = compute_stats(l2_distances[:num_input_tokens])
        cos_input = compute_stats(cosine_sims[:num_input_tokens])
        rel_l2_input = compute_stats(relative_l2_distances[:num_input_tokens])
        norms_input = compute_stats(state_norms[:num_input_tokens])
    else:
        l2_input = (None,) * 5
        cos_input = (None,) * 5
        rel_l2_input = (None,) * 5
        norms_input = (None,) * 5

    # Compute statistics for output sequence only (after input tokens)
    if num_input_tokens < num_tokens:
        l2_output = compute_stats(l2_distances[num_input_tokens:])
        cos_output = compute_stats(cosine_sims[num_input_tokens:])
        rel_l2_output = compute_stats(relative_l2_distances[num_input_tokens:])
        norms_output = compute_stats(state_norms[num_input_tokens:])
    else:
        l2_output = (None,) * 5
        cos_output = (None,) * 5
        rel_l2_output = (None,) * 5
        norms_output = (None,) * 5

    return {
        "l2": {
            "full": l2_full,
            "input": l2_input,
            "output": l2_output,
        },
        "cosine": {
            "full": cos_full,
            "input": cos_input,
            "output": cos_output,
        },
        "relative_l2": {
            "full": rel_l2_full,
            "input": rel_l2_input,
            "output": rel_l2_output,
        },
        "state_norms": {
            "full": norms_full,
            "input": norms_input,
            "output": norms_output,
        },
    }


def compute_output_kl_divergence(
    intermediate_logits: list[torch.Tensor],
    num_input_tokens: int,
) -> dict:
    """
    Compute KL divergence between output distributions at consecutive loop steps.

    Takes pre-computed intermediate logits (from forward with return_intermediate_logits=True)
    and computes KL(p_{i+1} || p_i) per token per transition.

    Args:
        intermediate_logits: list of num_recur tensors, each (total_tokens, vocab_size).
            Produced by generate_with_latent_tracking with return_intermediate_logits=True.
        num_input_tokens: number of prompt tokens (for input/output split).

    Returns:
        Dict with 'full', 'input', 'output' keys, each containing
        (means, stds, medians, q25, q75) arrays of shape (num_recur - 1,).
    """
    num_recur = len(intermediate_logits)
    num_tokens = intermediate_logits[0].shape[0]

    kl_divergences = np.zeros((num_tokens, num_recur - 1))
    prev_log_probs = None

    for recur_idx in range(num_recur):
        log_probs = F.log_softmax(intermediate_logits[recur_idx], dim=-1)  # (num_tokens, vocab_size)

        if prev_log_probs is not None:
            # KL(p_current || p_prev) using log_target=True for numerical stability
            kl = F.kl_div(prev_log_probs, log_probs, reduction="none", log_target=True).sum(dim=-1)
            kl_divergences[:, recur_idx - 1] = kl.numpy()

        prev_log_probs = log_probs

    # Compute statistics for full / input / output splits
    def compute_stats(data, axis=0):
        means = np.mean(data, axis=axis)
        stds = np.std(data, axis=axis)
        medians = np.median(data, axis=axis)
        q25 = np.percentile(data, 25, axis=axis)
        q75 = np.percentile(data, 75, axis=axis)
        return means, stds, medians, q25, q75

    kl_full = compute_stats(kl_divergences)

    if num_input_tokens > 0:
        kl_input = compute_stats(kl_divergences[:num_input_tokens])
    else:
        kl_input = (None,) * 5

    if num_input_tokens < num_tokens:
        kl_output = compute_stats(kl_divergences[num_input_tokens:])
    else:
        kl_output = (None,) * 5

    return {
        "full": kl_full,
        "input": kl_input,
        "output": kl_output,
    }


def aggregate_metrics_across_samples(all_metrics: list[dict]) -> dict:
    """
    Aggregate metrics from multiple samples.

    For each sequence type (full/input/output), we average the means and stds,
    and compute median of medians and IQR bounds across all samples.

    Args:
        all_metrics: List of metric dicts from individual samples

    Returns:
        Aggregated metrics with same structure as single-sample metrics
    """
    if not all_metrics:
        return None

    # Collect all statistics for each metric type and sequence type
    def collect_and_aggregate(metric_key: str, seq_type: str):
        means_list = []
        stds_list = []
        medians_list = []
        q25_list = []
        q75_list = []

        for metrics in all_metrics:
            stats = metrics[metric_key][seq_type]
            if stats[0] is not None:  # Check if means is not None
                means, stds, medians, q25, q75 = stats
                means_list.append(means)
                stds_list.append(stds)
                medians_list.append(medians)
                q25_list.append(q25)
                q75_list.append(q75)

        if not means_list:
            return (None,) * 5

        # Average the means across samples
        avg_means = np.mean(np.array(means_list), axis=0)

        # For stds, we combine them properly: sqrt(mean of variances)
        variances = np.array([s**2 for s in stds_list])
        avg_variance = np.mean(variances, axis=0)
        avg_stds = np.sqrt(avg_variance)

        # Median of medians
        avg_medians = np.median(np.array(medians_list), axis=0)

        # Average quartiles
        avg_q25 = np.mean(np.array(q25_list), axis=0)
        avg_q75 = np.mean(np.array(q75_list), axis=0)

        return avg_means, avg_stds, avg_medians, avg_q25, avg_q75

    return {
        "l2": {
            "full": collect_and_aggregate("l2", "full"),
            "input": collect_and_aggregate("l2", "input"),
            "output": collect_and_aggregate("l2", "output"),
        },
        "cosine": {
            "full": collect_and_aggregate("cosine", "full"),
            "input": collect_and_aggregate("cosine", "input"),
            "output": collect_and_aggregate("cosine", "output"),
        },
        "relative_l2": {
            "full": collect_and_aggregate("relative_l2", "full"),
            "input": collect_and_aggregate("relative_l2", "input"),
            "output": collect_and_aggregate("relative_l2", "output"),
        },
        "state_norms": {
            "full": collect_and_aggregate("state_norms", "full"),
            "input": collect_and_aggregate("state_norms", "input"),
            "output": collect_and_aggregate("state_norms", "output"),
        },
        "kl_divergence": {
            "full": collect_and_aggregate("kl_divergence", "full"),
            "input": collect_and_aggregate("kl_divergence", "input"),
            "output": collect_and_aggregate("kl_divergence", "output"),
        },
    }


def plot_distance_curves(
    metrics: dict,
    num_recur: int,
    num_input_tokens: int,
    num_output_tokens: int,
    question: str,
    task_name: str = "GSM8K",
    kv_budget: int = 1,
    use_warm_start: bool = False,
    model_tag: str | None = None,
    num_samples: int = 1,
    ylim: tuple[float, float, float, float] | None = None,
    log_scale_l2: bool = False,
):
    """
    Plot comprehensive distance metrics per loop step.

    Creates eight subplots:
    1. L2 distance (mean ± std) - log scale
    2. L2 distance (median + IQR) - log scale
    3. Relative L2 distance (median + IQR)
    4. State norms (median + IQR)
    5. Cosine similarity (mean ± std)
    6. Cosine similarity (median + IQR)
    7. KL divergence (mean ± std) - log scale
    8. KL divergence (median + IQR) - log scale

    Args:
        num_samples: Number of samples averaged (for plot title/filename)
        ylim: Y-axis limits (l2_ymin, l2_ymax, cosine_ymin, cosine_ymax)
        log_scale_l2: Whether to use log scale for L2 distance y-axis
    """
    loop_steps_transitions = np.arange(1, num_recur)  # For distances between states
    loop_steps_states = np.arange(0, num_recur)  # For state norms

    fig = plt.figure(figsize=(18, 21))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    # Colorblind-safe palette (Wong) with distinct linestyles and markers
    colors = {"full": "#332288", "input": "#E69F00", "output": "#009E73"}
    linestyles = {"full": "-", "input": "--", "output": "-."}
    markers = {"full": "o", "input": "s", "output": "D"}

    samples_text = f", averaged over {num_samples} samples" if num_samples > 1 else ""

    # Plot 1: L2 Distance (mean ± std) - log scale
    ax1 = fig.add_subplot(gs[0, 0])
    for seq_type in ["input", "output"]:
        means, stds, _, _, _ = metrics["l2"][seq_type]
        if means is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax1.plot(loop_steps_transitions, means, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax1.fill_between(
                loop_steps_transitions,
                means - stds,
                means + stds,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax1.set_xlabel("Loop Step", fontsize=11)
    ax1.set_ylabel("L2 Distance (mean ± std)", fontsize=11)
    ax1.set_title(
        f"L2 Distance: Mean ± Std (Log Scale)\n(num_recur={num_recur}{samples_text})",
        fontsize=12,
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    if log_scale_l2:
        ax1.set_yscale("log")
    if ylim is not None:
        ax1.set_ylim(ylim[0], ylim[1])

    # Plot 2: L2 Distance (median + IQR) - log scale
    ax2 = fig.add_subplot(gs[0, 1])
    for seq_type in ["input", "output"]:
        _, _, medians, q25, q75 = metrics["l2"][seq_type]
        if medians is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax2.plot(loop_steps_transitions, medians, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax2.fill_between(
                loop_steps_transitions,
                q25,
                q75,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax2.set_xlabel("Loop Step", fontsize=11)
    ax2.set_ylabel("L2 Distance (median, IQR)", fontsize=11)
    ax2.set_title(
        "L2 Distance: Median + IQR (Log Scale)\n(more robust to outliers)",
        fontsize=12,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    if log_scale_l2:
        ax2.set_yscale("log")
    if ylim is not None:
        ax2.set_ylim(ylim[0], ylim[1])

    # Plot 3: Relative L2 Distance (median + IQR)
    ax3 = fig.add_subplot(gs[1, 0])
    for seq_type in ["input", "output"]:
        _, _, medians, q25, q75 = metrics["relative_l2"][seq_type]
        if medians is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax3.plot(loop_steps_transitions, medians, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax3.fill_between(
                loop_steps_transitions,
                q25,
                q75,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax3.set_xlabel("Loop Step", fontsize=11)
    ax3.set_ylabel("Relative L2 Distance", fontsize=11)
    ax3.set_title(
        "Relative L2 Distance: ||s_{t+1} - s_t|| / ||s_t||\n(normalized by state magnitude)",
        fontsize=12,
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: State Norms (median + IQR)
    ax4 = fig.add_subplot(gs[1, 1])
    for seq_type in ["input", "output"]:
        _, _, medians, q25, q75 = metrics["state_norms"][seq_type]
        if medians is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax4.plot(loop_steps_states, medians, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax4.fill_between(
                loop_steps_states,
                q25,
                q75,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax4.set_xlabel("Loop Step", fontsize=11)
    ax4.set_ylabel("State Norm ||s_t||", fontsize=11)
    ax4.set_title(
        "State Norms Across Loop Steps (Log Scale)\n(magnitude of hidden states)",
        fontsize=12,
    )
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale("log")

    # Plot 5: Cosine Similarity (mean ± std)
    ax5 = fig.add_subplot(gs[2, 0])
    for seq_type in ["input", "output"]:
        means, stds, _, _, _ = metrics["cosine"][seq_type]
        if means is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax5.plot(loop_steps_transitions, means, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax5.fill_between(
                loop_steps_transitions,
                means - stds,
                means + stds,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax5.set_xlabel("Loop Step", fontsize=11)
    ax5.set_ylabel("Cosine Similarity (mean ± std)", fontsize=11)
    ax5.set_title(
        f"Cosine Similarity: Mean ± Std\nQuestion: {question[:50]}...",
        fontsize=12,
    )
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    if ylim is not None:
        ax5.set_ylim(ylim[2], ylim[3])

    # Plot 6: Cosine Similarity (median + IQR)
    ax6 = fig.add_subplot(gs[2, 1])
    for seq_type in ["input", "output"]:
        _, _, medians, q25, q75 = metrics["cosine"][seq_type]
        if medians is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax6.plot(loop_steps_transitions, medians, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax6.fill_between(
                loop_steps_transitions,
                q25,
                q75,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax6.set_xlabel("Loop Step", fontsize=11)
    ax6.set_ylabel("Cosine Similarity (median, IQR)", fontsize=11)
    ax6.set_title(
        "Cosine Similarity: Median + IQR\n(more robust to outliers)",
        fontsize=12,
    )
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    if ylim is not None:
        ax6.set_ylim(ylim[2], ylim[3])

    # Plot 7: KL Divergence (mean ± std) - log scale
    ax7 = fig.add_subplot(gs[3, 0])
    for seq_type in ["input", "output"]:
        means, stds, _, _, _ = metrics["kl_divergence"][seq_type]
        if means is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax7.plot(loop_steps_transitions, means, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax7.fill_between(
                loop_steps_transitions,
                np.maximum(means - stds, 1e-8),
                means + stds,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax7.set_xlabel("Loop Step", fontsize=11)
    ax7.set_ylabel("KL Divergence (mean ± std)", fontsize=11)
    ax7.set_title(
        f"KL(p_{{i+1}} || p_i): Mean ± Std (Log Scale)\n(output distribution change per loop{samples_text})",
        fontsize=12,
    )
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale("log")

    # Plot 8: KL Divergence (median + IQR) - log scale
    ax8 = fig.add_subplot(gs[3, 1])
    for seq_type in ["input", "output"]:
        _, _, medians, q25, q75 = metrics["kl_divergence"][seq_type]
        if medians is not None:
            label = f"{seq_type.capitalize()}"
            if seq_type == "input":
                label += f" (n={num_input_tokens})"
            else:
                label += f" (n={num_output_tokens})"

            ax8.plot(loop_steps_transitions, medians, label=label, color=colors[seq_type], marker=markers[seq_type], linestyle=linestyles[seq_type], markersize=4)
            ax8.fill_between(
                loop_steps_transitions,
                np.maximum(q25, 1e-8),
                q75,
                alpha=0.2,
                color=colors[seq_type],
            )

    ax8.set_xlabel("Loop Step", fontsize=11)
    ax8.set_ylabel("KL Divergence (median, IQR)", fontsize=11)
    ax8.set_title(
        "KL(p_{i+1} || p_i): Median + IQR (Log Scale)\n(more robust to outliers)",
        fontsize=12,
    )
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.set_yscale("log")

    # Save to plots directory
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    model_tag_suffix = f"_{model_tag}" if model_tag else ""
    warmstart_suffix = "_warmstart" if use_warm_start else ""
    samples_suffix = f"_avg{num_samples}" if num_samples > 1 else ""
    logscale_suffix = "_logscale" if log_scale_l2 else ""
    task_slug = task_name.lower().replace("-", "_")
    output_path = (
        plots_dir
        / f"{task_slug}{model_tag_suffix}_latent_distances_per_loop_recur{num_recur}_kvbudget{kv_budget}{samples_suffix}{warmstart_suffix}{logscale_suffix}.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nComprehensive plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Track and plot average latent distances per loop step for evaluation samples")
    parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: base|sft|rl (default: sft)")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., d12). If not specified, uses largest model.")
    parser.add_argument("-a", "--task-name", type=str, default="GSM8K", choices=list(TASK_MODULES.keys()), help="Evaluation task to load (default: GSM8K)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to start from (default: 0)")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to average over (default: 1)")
    parser.add_argument("--num-recur", type=int, default=16, help="Number of recurrences (default: 16)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--kv-budget", type=int, default=1, help="Fixed KV-cache budget for recurrences (default: 1)")
    parser.add_argument(
        "--use-rec-warm-start", action="store_true", help="Use recurrent warm-start (carry recurrent state when decoding tokens)"
    )
    parser.add_argument(
        "--ylim",
        default=None,
        type=float,
        nargs=4,
        metavar=("L2_YMIN", "L2_YMAX", "COS_YMIN", "COS_YMAX"),
        help="Y-axis limits (l2_ymin l2_ymax cosine_ymin cosine_ymax)",
    )
    parser.add_argument("--log-scale-l2", action="store_true", help="Use log scale for L2 distance y-axis")
    args = parser.parse_args()

    # Initialize compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    if ddp:
        print("Warning: Running with DDP, but this script is designed for single GPU")

    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    print(f"Loading model from source: {args.source}, model_tag: {args.model_tag}")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag)
    engine = Engine(model, tokenizer)

    # Determine num_recur
    num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
    print(f"Using num_recur={num_recur}")

    # Load dataset
    print(f"Loading {args.task_name} dataset...")
    task_object = TASK_MODULES[args.task_name]()

    # Process multiple samples
    all_metrics = []
    total_input_tokens = 0
    total_output_tokens = 0
    first_question = None

    for sample_offset in range(args.num_samples):
        sample_idx = args.sample_idx + sample_offset
        print(f"\nProcessing sample {sample_idx} ({sample_offset + 1}/{args.num_samples})")

        conversation = task_object.get_example(sample_idx)
        question = conversation["messages"][0]["content"]

        if sample_offset == 0:
            first_question = question

        # Render prompt for completion
        prompt_tokens = tokenizer.render_for_completion(conversation)

        print(f"Question: {question[:80]}...")
        print(f"Prompt length: {len(prompt_tokens)} tokens")

        # Generate with latent state tracking
        with autocast_ctx:
            generated_tokens, input_latent_states, output_latent_states, intermediate_logits = generate_with_latent_tracking(
                engine=engine,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                num_recur=num_recur,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + sample_offset,  # Different seed per sample
                kv_budget=args.kv_budget,
                use_warm_start=args.use_rec_warm_start,
                return_intermediate_logits=True,
            )

        # Combine input and output latent states
        latent_states_per_token = input_latent_states + output_latent_states
        num_input_tokens = len(input_latent_states)
        num_output_tokens = len(generated_tokens)
        print(f"Generated {num_output_tokens} tokens")
        print(f"Captured latent states for {len(latent_states_per_token)} tokens ({num_input_tokens} input + {len(output_latent_states)} output)")

        total_input_tokens += num_input_tokens
        total_output_tokens += num_output_tokens

        # Compute distance metrics for this sample
        if latent_states_per_token:
            metrics = compute_distance_metrics(latent_states_per_token, num_recur, num_input_tokens)

            # Compute KL divergence from intermediate logits (captured during generation)
            print("Computing KL divergence of output distributions per loop step...")
            kl_metrics = compute_output_kl_divergence(intermediate_logits, num_input_tokens)
            metrics["kl_divergence"] = kl_metrics

            all_metrics.append(metrics)
        else:
            print(f"Warning: No latent states captured for sample {sample_idx}")

    # Aggregate metrics and plot
    if all_metrics:
        if args.num_samples > 1:
            print(f"\nAggregating metrics across {len(all_metrics)} samples...")
            aggregated_metrics = aggregate_metrics_across_samples(all_metrics)
            avg_input_tokens = total_input_tokens // args.num_samples
            avg_output_tokens = total_output_tokens // args.num_samples
        else:
            aggregated_metrics = all_metrics[0]
            avg_input_tokens = total_input_tokens
            avg_output_tokens = total_output_tokens

        # Plot the results
        ylim = tuple(args.ylim) if args.ylim is not None else None
        plot_distance_curves(
            metrics=aggregated_metrics,
            num_recur=num_recur,
            num_input_tokens=avg_input_tokens,
            num_output_tokens=avg_output_tokens,
            question=first_question,
            task_name=args.task_name,
            kv_budget=args.kv_budget,
            use_warm_start=args.use_rec_warm_start,
            model_tag=args.model_tag,
            num_samples=args.num_samples,
            ylim=ylim,
            log_scale_l2=args.log_scale_l2,
        )
    else:
        print("No latent states captured for plotting")


if __name__ == "__main__":
    main()
