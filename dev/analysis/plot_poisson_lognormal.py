"""
Plot the Poisson log-normal distribution for recurrence depth sampling.

Usage:
    uv run python -m dev.plot_poisson_lognormal --mean-recur 4.0
    uv run python -m dev.plot_poisson_lognormal --mean-recur 32.0 --sigma 0.5 --max-recur 64
"""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanochat.common import get_base_dir, sample_poisson_lognormal_recurrence


def plot_distribution(
    mean_recur: float,
    sigma: float = 0.5,
    min_recur: int = 1,
    max_recur: int | None = None,
    num_samples: int = 1000000,
):
    """Plot histogram of Poisson log-normal recurrence samples."""
    print(
        f"Plotting with mean_recur={mean_recur}, sigma={sigma}, min_recur={min_recur}, max_recur={max_recur}, num_samples={num_samples}"
    )

    # Generate samples
    samples = [
        sample_poisson_lognormal_recurrence(mean_recur, sigma, min_recur, max_recur)
        for _ in range(num_samples)
    ]

    # Calculate statistics
    sample_mean = np.mean(samples)
    sample_median = np.median(samples)

    # Count frequency
    counter = Counter(samples)
    values = sorted(counter.keys())
    probabilities = [counter[v] / num_samples for v in values]

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot probability mass function - bar chart with overlaid line
    ax.bar(values, probabilities, alpha=0.5, color="steelblue", edgecolor="navy", linewidth=1)
    ax.plot(
        values,
        probabilities,
        linewidth=2.5,
        color="darkgreen",
        marker="o",
        markersize=5,
        label="Continuous trend",
    )
    ax.axvline(
        sample_mean, color="red", linestyle="--", linewidth=2, label=f"Mean: {sample_mean:.2f}"
    )
    ax.axvline(
        sample_median,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {sample_median:.0f}",
    )
    ax.set_xlabel("Number of Recurrences (r)", fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(
        f"Poisson Log-Normal Distribution\n(μ={mean_recur}, σ={sigma}, min={min_recur}, max={max_recur}, n={num_samples:,})",
        fontsize=14,
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()

    # Save the figure
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate filename from parameters
    min_str = f"min{min_recur}" if min_recur > 1 else ""
    max_str = f"max{max_recur}" if max_recur else "unclamped"
    clamp_str = "_".join(filter(None, [min_str, max_str]))
    filename = f"poisson_lognormal_mean{mean_recur}_sigma{sigma}_{clamp_str}.png"
    save_path = plots_dir / filename

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")


def main():
    """Entry point for console script."""
    parser = argparse.ArgumentParser(description="Plot Poisson log-normal distribution")
    parser.add_argument(
        "--mean-recur",
        type=float,
        default=4.0,
        help="Mean number of recurrences (r̄), default: 4.0 for nanochat, 32.0 for Huginn",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.5,
        help="Standard deviation of log-normal component (default: 0.5)",
    )
    parser.add_argument(
        "--min-recur",
        type=int,
        default=1,
        help="Minimum recurrence value for clamping (default: 1)",
    )
    parser.add_argument(
        "--max-recur",
        type=int,
        default=None,
        help="Maximum recurrence value for clamping (default: None, no clamping)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000000,
        help="Number of samples to generate (default: 1000000)",
    )

    args = parser.parse_args()

    plot_distribution(
        mean_recur=args.mean_recur,
        sigma=args.sigma,
        min_recur=args.min_recur,
        max_recur=args.max_recur,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
