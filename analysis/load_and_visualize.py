"""
Example script showing how to load and analyze the latent state tracking results.
"""

import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.track_gsm8k_latent_states import GenerationResult  # noqa: F401 - needed for pickle


def main():
    parser = argparse.ArgumentParser(description="Load and visualize latent state results")
    parser.add_argument(
        "input_file",
        type=str,
        help="Input pickle file (e.g., gsm8k_latent_states.pkl)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=5,
        help="Number of examples to display (default: 5)",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.input_file}")
    with open(args.input_file, "rb") as f:
        results = pickle.load(f)

    print(f"\nLoaded {len(results)} results")
    print(f"First result type: {type(results[0])}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_tokens = sum(len(r.tokens) for r in results)
    avg_tokens = total_tokens / len(results)
    print(f"Total examples: {len(results)}")
    print(f"Average tokens per response: {avg_tokens:.1f}")

    if results and results[0].latent_states:
        example_state = results[0].latent_states[0]
        print(f"Latent state shape: {example_state.shape}")
        print(f"Latent state dtype: {example_state.dtype}")
        print(f"Number of recurrences: {results[0].num_recur}")
        print(f"Latent states per example: {len(results[0].latent_states)}")

    # Show individual examples
    print("\n" + "=" * 80)
    print(f"EXAMPLE GENERATIONS (showing first {args.num_examples})")
    print("=" * 80)

    for i, result in enumerate(results[: args.num_examples]):
        print(f"\n--- Example {i + 1} ---")
        print(f"Question: {result.question[:200]}...")
        print(f"Response: {result.response[:200]}...")
        print(f"Tokens generated: {len(result.tokens)}")
        print(f"Latent states captured: {len(result.latent_states)}")

        if result.latent_states:
            print(f"State shapes: {[s.shape for s in result.latent_states]}")

            # Compute some statistics on the latent states
            print("\nLatent state statistics per loop:")
            for loop_idx, state in enumerate(result.latent_states):
                mean = state.mean().item()
                std = state.std().item()
                norm = torch.norm(state).item()
                print(f"  Loop {loop_idx}: mean={mean:.4f}, std={std:.4f}, norm={norm:.4f}")

    # Analyze latent state evolution across loops
    if results and results[0].latent_states and len(results[0].latent_states) > 1:
        print("\n" + "=" * 80)
        print("LATENT STATE EVOLUTION ANALYSIS")
        print("=" * 80)

        for i, result in enumerate(results[: args.num_examples]):
            if len(result.latent_states) < 2:
                continue

            print(f"\nExample {i + 1}:")

            # Compute cosine similarity between consecutive loop iterations
            print("  Cosine similarity between consecutive loops:")
            for loop_idx in range(len(result.latent_states) - 1):
                s1 = result.latent_states[loop_idx].flatten()
                s2 = result.latent_states[loop_idx + 1].flatten()
                cos_sim = torch.nn.functional.cosine_similarity(s1, s2, dim=0).item()
                print(f"    Loop {loop_idx} -> {loop_idx + 1}: {cos_sim:.4f}")

            # Compute change in norm across loops
            print("  Norm change between consecutive loops:")
            for loop_idx in range(len(result.latent_states) - 1):
                norm1 = torch.norm(result.latent_states[loop_idx]).item()
                norm2 = torch.norm(result.latent_states[loop_idx + 1]).item()
                change = norm2 - norm1
                pct_change = (change / norm1) * 100
                print(f"    Loop {loop_idx} -> {loop_idx + 1}: {change:+.4f} ({pct_change:+.2f}%)")

    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Filter results with multiple latent states
    multi_loop_results = [r for r in results if len(r.latent_states) > 1]

    if multi_loop_results:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Latent State Evolution Analysis", fontsize=16)

        # Plot 1: Cosine similarity across loops (aggregated)
        ax = axes[0, 0]
        num_loops = max(len(r.latent_states) for r in multi_loop_results)
        cos_sim_by_transition = [[] for _ in range(num_loops - 1)]

        for result in multi_loop_results:
            for loop_idx in range(len(result.latent_states) - 1):
                s1 = result.latent_states[loop_idx].flatten()
                s2 = result.latent_states[loop_idx + 1].flatten()
                cos_sim = torch.nn.functional.cosine_similarity(s1, s2, dim=0).item()
                cos_sim_by_transition[loop_idx].append(cos_sim)

        transitions = [f"{i}→{i + 1}" for i in range(num_loops - 1)]
        means = [np.mean(sims) if sims else 0 for sims in cos_sim_by_transition]
        stds = [np.std(sims) if sims else 0 for sims in cos_sim_by_transition]

        ax.errorbar(transitions, means, yerr=stds, marker="o", capsize=5, linewidth=2)
        ax.set_xlabel("Loop Transition", fontsize=12)
        ax.set_ylabel("Cosine Similarity", fontsize=12)
        ax.set_title("Cosine Similarity Between Consecutive Loops", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Plot 2: Norm evolution across loops
        ax = axes[0, 1]
        norm_by_loop = [[] for _ in range(num_loops)]

        for result in multi_loop_results:
            for loop_idx, state in enumerate(result.latent_states):
                norm = torch.norm(state).item()
                norm_by_loop[loop_idx].append(norm)

        loop_indices = list(range(num_loops))
        means = [np.mean(norms) if norms else 0 for norms in norm_by_loop]
        stds = [np.std(norms) if norms else 0 for norms in norm_by_loop]

        ax.errorbar(loop_indices, means, yerr=stds, marker="s", capsize=5, linewidth=2)
        ax.set_xlabel("Loop Index", fontsize=12)
        ax.set_ylabel("L2 Norm", fontsize=12)
        ax.set_title("Latent State Norm Across Loops", fontsize=13)
        ax.grid(True, alpha=0.3)

        # Plot 3: Distribution of response lengths
        ax = axes[1, 0]
        token_counts = [len(r.tokens) for r in results]
        ax.hist(token_counts, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(
            np.mean(token_counts),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(token_counts):.1f}",
        )
        ax.set_xlabel("Number of Tokens", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Response Lengths", fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 4: Mean activation values across loops
        ax = axes[1, 1]
        mean_by_loop = [[] for _ in range(num_loops)]

        for result in multi_loop_results:
            for loop_idx, state in enumerate(result.latent_states):
                mean_val = state.mean().item()
                mean_by_loop[loop_idx].append(mean_val)

        means = [np.mean(vals) if vals else 0 for vals in mean_by_loop]
        stds = [np.std(vals) if vals else 0 for vals in mean_by_loop]

        ax.errorbar(
            loop_indices, means, yerr=stds, marker="^", capsize=5, linewidth=2, color="green"
        )
        ax.set_xlabel("Loop Index", fontsize=12)
        ax.set_ylabel("Mean Activation", fontsize=12)
        ax.set_title("Mean Latent State Activation Across Loops", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        output_file = args.input_file.replace(".pkl", "_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"\nVisualization saved to: {output_file}")
        plt.show()
    else:
        print("No results with multiple loops found for visualization")


if __name__ == "__main__":
    main()
