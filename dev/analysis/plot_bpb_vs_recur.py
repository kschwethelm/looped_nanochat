"""
Plot validation BPB vs number of recurrences for different models.

Reads BPB CSV files from base_eval directory and creates a comparison plot
showing how validation loss changes with test-time compute (num_recur).

Usage:
    uv run python dev/plot_bpb_vs_recur.py [--step STEP] [--model-tags TAG1 TAG2 ...]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanochat.common import get_base_dir


# Colorblind-safe palette (dark blues/purples + bright accents)
COLORS = [
    "#8cc5e3",  # Bright Mint
    "#0d7d87",  # Bright Cyan
    "#4a2377",  # Deep Navy Blue
    "#f55f74",  # Dusky Purple
    "#FFD166",  # Bright Gold
    "#FF8C42",  # Bright Orange
    "#FF66C4",  # Bright Pink
    "#7EC8E3",  # Bright Sky Blue
]

# Distinct marker styles
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X"]


def parse_bpb_csv(csv_path: Path) -> dict[str, float]:
    """Parse a BPB CSV file and return dict of split -> bpb."""
    results = {}
    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip():
                split_name, bpb = line.strip().split(",")
                results[split_name] = float(bpb)
    return results


def parse_filename(filename: str) -> tuple[str, int, int] | None:
    """
    Parse filename to extract model_tag, step, and num_recur.

    Expected format: {model_tag}_step{step:06d}_recur{num_recur}_bpb.csv
    Returns (model_tag, step, num_recur) or None if parsing fails.
    """
    # Pattern: anything_step######_recur##_bpb.csv
    pattern = r"^(.+)_step(\d+)_recur(\d+)_bpb\.csv$"
    match = re.match(pattern, filename)
    if match:
        model_tag = match.group(1)
        step = int(match.group(2))
        num_recur = int(match.group(3))
        return (model_tag, step, num_recur)
    return None


def collect_data(base_eval_dir: Path, step_filter: int | None = None, model_tags_filter: list[str] | None = None) -> dict:
    """
    Collect all BPB results from CSV files.

    Returns dict: {model_tag: {step: {num_recur: {"train": float, "val": float}}}}
    """
    data = {}

    for csv_file in base_eval_dir.glob("*_bpb.csv"):
        parsed = parse_filename(csv_file.name)
        if parsed is None:
            continue

        model_tag, step, num_recur = parsed

        # Apply filters
        if step_filter is not None and step != step_filter:
            continue
        if model_tags_filter is not None and model_tag not in model_tags_filter:
            continue

        # Parse BPB values
        bpb_values = parse_bpb_csv(csv_file)

        # Store in nested dict
        if model_tag not in data:
            data[model_tag] = {}
        if step not in data[model_tag]:
            data[model_tag][step] = {}
        data[model_tag][step][num_recur] = bpb_values

    return data


def create_short_label(model_tag: str) -> str:
    """Create a shorter, more readable label from model_tag."""
    # Remove common prefixes/suffixes for cleaner labels
    label = model_tag
    # Example: "r4_sample_init_random_2.15e18_s12" -> "r4_sample_init_random"
    # Remove FLOPs count and seed suffix
    label = re.sub(r"_[\d.]+e\d+_s\d+$", "", label)
    return label


def plot_bpb_vs_recur(data: dict, output_path: Path):
    """Create plot of validation BPB vs num_recur for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect all model_tags and steps
    model_step_pairs = []
    for model_tag, steps_dict in sorted(data.items()):
        for step in sorted(steps_dict.keys()):
            model_step_pairs.append((model_tag, step))

    # Plot each model-step combination
    for idx, (model_tag, step) in enumerate(model_step_pairs):
        recur_dict = data[model_tag][step]

        # Extract and sort by num_recur
        recur_vals = sorted(recur_dict.keys())
        val_bpbs = [recur_dict[r]["val"] for r in recur_vals]

        # Create label
        short_label = create_short_label(model_tag)
        if len(data[model_tag]) > 1:  # Multiple steps for this model
            label = f"{short_label} (step {step})"
        else:
            label = short_label

        # Plot with colorblind-safe colors and distinct markers
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(
            recur_vals,
            val_bpbs,
            marker=marker,
            markersize=8,
            linewidth=2,
            linestyle="-",
            color=color,
            label=label,
        )

    # Styling
    ax.set_xlabel("Number of Recurrences", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation BPB", fontsize=12, fontweight="bold")
    ax.set_title("Test-Time Compute Scaling: Validation BPB vs Recurrence Depth", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)

    # Set x-axis to show integer recurrence values
    if len(model_step_pairs) > 0:
        all_recur_vals = set()
        for model_tag, step in model_step_pairs:
            all_recur_vals.update(data[model_tag][step].keys())
        ax.set_xticks(sorted(all_recur_vals))

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for papers
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot validation BPB vs number of recurrences")
    parser.add_argument("--step", type=int, default=None, help="Filter by specific checkpoint step")
    parser.add_argument("--model-tags", nargs="+", default=None, help="Filter by specific model tags")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: plots/bpb_vs_recur.png)")
    args = parser.parse_args()

    # Get base directory
    base_dir = Path(get_base_dir())
    base_eval_dir = base_dir / "base_eval"

    if not base_eval_dir.exists():
        print(f"Error: base_eval directory not found at {base_eval_dir}")
        return

    # Collect data
    print("Collecting BPB results from CSV files...")
    data = collect_data(base_eval_dir, step_filter=args.step, model_tags_filter=args.model_tags)

    if not data:
        print("No BPB CSV files found matching the filters.")
        print(f"Looking in: {base_eval_dir}")
        if args.step:
            print(f"Step filter: {args.step}")
        if args.model_tags:
            print(f"Model tags filter: {args.model_tags}")
        return

    # Print summary
    print(f"\nFound results for {len(data)} model(s):")
    for model_tag, steps_dict in sorted(data.items()):
        for step, recur_dict in sorted(steps_dict.items()):
            recur_vals = sorted(recur_dict.keys())
            print(f"  {model_tag} (step {step}): {len(recur_vals)} recurrence values {recur_vals}")

    # Create plot
    output_path = Path(args.output) if args.output else (base_dir / "plots" / "bpb_vs_recur.png")
    print(f"\nCreating plot...")
    plot_bpb_vs_recur(data, output_path)


if __name__ == "__main__":
    main()
