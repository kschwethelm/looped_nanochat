"""
Plot ChatCORE metric vs number of recurrences for SFT models.

Reads summary CSV files from chat_eval directory (produced by chat_eval.py)
and creates a comparison plot showing how performance changes with test-time
compute (num_recur).

CSV format (from chat_eval.py):
    # source=sft, model_tag=..., step=..., ...
    num_recur,kv_budget,ARC-Easy,ARC-Challenge,...,ChatCORE
    2,1,0.461,...,0.253

Usage:
    uv run python dev/analysis/plot_chatcore_vs_recur.py
    uv run python dev/analysis/plot_chatcore_vs_recur.py --model-tags r4_sample_2.15e18_s12
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

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


def parse_filename(filename: str) -> tuple[str, int] | None:
    """
    Parse filename to extract model_tag and step.

    Expected format: {source}_{model_tag}_step{step:06d}.csv
    Returns (model_tag, step) or None if parsing fails.
    """
    pattern = r"^(.+)_step(\d+)\.csv$"
    match = re.match(pattern, filename)
    if match:
        return (match.group(1), int(match.group(2)))
    return None


def parse_summary_csv(csv_path: Path) -> list[dict[str, float]]:
    """
    Parse a chat_eval summary CSV file.

    Skips comment lines (starting with #) and reads the header + data rows.
    Returns list of dicts with at least 'num_recur' and 'ChatCORE'.
    """
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        non_comment_lines = [line for line in f if not line.startswith("#")]
    if not non_comment_lines:
        return rows

    reader = csv.DictReader(non_comment_lines)
    for row in reader:
        if "ChatCORE" not in row or "num_recur" not in row:
            continue
        try:
            parsed = {
                "num_recur": int(row["num_recur"]),
                "chatcore": float(row["ChatCORE"]),
            }
            rows.append(parsed)
        except (ValueError, KeyError):
            continue
    return rows


def collect_data(chat_eval_dir: Path, model_tags_filter: list[str] | None = None) -> dict:
    """
    Collect all ChatCORE results from summary CSV files.

    Returns dict: {model_tag: {num_recur: chatcore_value}}
    """
    data: dict = {}

    for csv_file in chat_eval_dir.glob("*.csv"):
        parsed = parse_filename(csv_file.name)
        if parsed is None:
            continue

        model_tag, _step = parsed

        # Apply filter
        if model_tags_filter is not None and model_tag not in model_tags_filter:
            continue

        rows = parse_summary_csv(csv_file)
        if not rows:
            continue

        if model_tag not in data:
            data[model_tag] = {}

        for row in rows:
            data[model_tag][row["num_recur"]] = row["chatcore"]

    return data


def parse_mean_recur(model_tag: str) -> int | None:
    """Extract mean training recurrence from model tag prefix (e.g., 'r4_...' -> 4)."""
    match = re.match(r"^r(\d+)_", model_tag)
    if match:
        return int(match.group(1))
    return None


def create_short_label(model_tag: str) -> str:
    """Create a shorter, more readable label from model_tag."""
    # Remove FLOPs count and seed suffix
    # Example: "r4_sample_2.15e18_s12" -> "r4_sample"
    label = re.sub(r"_[\d.]+e\d+_s\d+$", "", model_tag)
    return label


def plot_chatcore_vs_recur(data: dict, output_path: Path, y_min: float | None = None, y_max: float | None = None):
    """Create plot of ChatCORE vs num_recur for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for idx, model_tag in enumerate(sorted(data.keys())):
        recur_dict = data[model_tag]

        # Extract and sort by num_recur
        recur_vals = sorted(recur_dict.keys())
        chatcore_vals = [recur_dict[r] for r in recur_vals]

        # Create label
        label = create_short_label(model_tag)

        # Plot with colorblind-safe colors and distinct markers
        color = COLORS[idx % len(COLORS)]
        marker = MARKERS[idx % len(MARKERS)]

        ax.plot(
            recur_vals,
            chatcore_vals,
            marker=marker,
            markersize=8,
            linewidth=2,
            linestyle="-",
            color=color,
            label=label,
        )

    # Add visual marks for specific recurrence ranges
    # Shaded region for num_recur=2-16
    ax.axvspan(2, 16, alpha=0.15, color="gray", zorder=0, label="sample range")

    # Vertical lines for each unique mean training recurrence
    seen_recur: set[int] = set()
    for model_tag in data:
        mean_recur = parse_mean_recur(model_tag)
        if mean_recur is not None and mean_recur not in seen_recur:
            seen_recur.add(mean_recur)
            ax.axvline(x=mean_recur, color="darkred", linestyle="--", linewidth=2, alpha=0.7, zorder=1, label=f"mean train recur={mean_recur}")

    # Styling
    ax.set_xlabel("Number of Recurrences", fontsize=12, fontweight="bold")
    ax.set_ylabel("ChatCORE Metric\n(0=random baseline, 1=perfect)", fontsize=12, fontweight="bold")
    ax.set_title("Test-Time Compute Scaling: ChatCORE vs Recurrence Depth", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", framealpha=0.9)

    # Set x-axis to show integer recurrence values
    if data:
        all_recur_vals = set()
        for model_tag in data:
            all_recur_vals.update(data[model_tag].keys())
        ax.set_xticks(sorted(all_recur_vals))

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

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
    parser = argparse.ArgumentParser(description="Plot ChatCORE vs number of recurrences")
    parser.add_argument("--model-tags", nargs="+", default=None, help="Filter by specific model tags")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: plots/chatcore_vs_recur.png)")
    parser.add_argument("--y-min", type=float, default=None, help="Minimum y-axis value")
    parser.add_argument("--y-max", type=float, default=None, help="Maximum y-axis value")
    args = parser.parse_args()

    # Get base directory
    base_dir = Path(get_base_dir())
    chat_eval_dir = base_dir / "chat_eval"

    if not chat_eval_dir.exists():
        print(f"Error: chat_eval directory not found at {chat_eval_dir}")
        return

    # Collect data
    print("Collecting ChatCORE results from CSV files...")
    data = collect_data(chat_eval_dir, model_tags_filter=args.model_tags)

    if not data:
        print("No ChatCORE CSV files found matching the filters.")
        print(f"Looking in: {chat_eval_dir}")
        if args.model_tags:
            print(f"Model tags filter: {args.model_tags}")
        return

    # Print summary
    print(f"\nFound results for {len(data)} model(s):")
    for model_tag, recur_dict in sorted(data.items()):
        recur_vals = sorted(recur_dict.keys())
        chatcore_vals = [recur_dict[r] for r in recur_vals]
        print(f"  {model_tag}: {len(recur_vals)} recurrence values {recur_vals}")
        for r, c in zip(recur_vals, chatcore_vals):
            print(f"    r={r}: ChatCORE={c:.4f}")

    # Create plot
    output_path = Path(args.output) if args.output else (base_dir / "plots" / "chatcore_vs_recur.png")
    print(f"\nCreating plot...")
    plot_chatcore_vs_recur(data, output_path, y_min=args.y_min, y_max=args.y_max)


if __name__ == "__main__":
    main()
