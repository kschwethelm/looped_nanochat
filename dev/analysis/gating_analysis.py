"""
Early-exit gating analysis for looped transformers.

Evaluates training-free early-exit gates by running a single max-r forward
pass per model, then simulating all gates x thresholds post-hoc. Produces
efficiency vs accuracy Pareto curves comparing across multiple models.

Gate functions (all zero-shot, no trained params):
  1. Acceleration (Pappone et al.): second-order step-size ratio
  2. Relative L2: ||s_i - s_{i-1}|| / ||s_i||
  3. KL divergence on intermediate logits

Example:
    uv run python dev/analysis/gating_analysis.py -i sft -a ARC-Easy -g d12
    uv run python dev/analysis/gating_analysis.py -i sft -a "ARC-Easy|ARC-Challenge|MMLU" -g "d12|d20" --max-problems 200
"""

import argparse
import csv
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from tasks.arc import ARC
from tasks.mmlu import MMLU

# ---------------------------------------------------------------------------
# Task registry (categorical tasks only — gate analysis uses logit probing)

TASK_MODULES = {
    "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
    "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
    "MMLU": partial(MMLU, subset="all", split="test"),
}

BASELINE_ACCURACIES = {
    "ARC-Easy": 0.25,
    "ARC-Challenge": 0.25,
    "MMLU": 0.25,
}

# ---------------------------------------------------------------------------
# Gate functions
#
# Each gate takes intermediate_states and/or intermediate_logits and returns
# a dict mapping threshold -> per-token exit depths (1-indexed recurrence).
# States: list of num_recur tensors, each (B, T, D).
# Logits: list of num_recur tensors, each (B, T, V).


def gate_acceleration(
    intermediate_states: list[torch.Tensor],
    thresholds: list[float],
    **_kwargs,
) -> dict[float, torch.Tensor]:
    """
    Acceleration gate (Pappone et al.): exit when step-size ratio ≈ 1.

    ratio_i = ||s_i - s_{i-1}|| / ||s_{i-1} - s_{i-2}||
    Exit when ratio <= threshold (refinement plateau).
    Needs ≥ 3 states; tokens exit at the earliest qualifying recurrence.
    """
    num_recur = len(intermediate_states)
    B, T, _D = intermediate_states[0].shape

    # Precompute step norms: step_norms[i] = ||s_{i+1} - s_i||, shape (B, T)
    step_norms = []
    for i in range(num_recur - 1):
        diff = intermediate_states[i + 1] - intermediate_states[i]
        step_norms.append(torch.norm(diff, dim=-1))  # (B, T)

    max_depth = torch.full((B, T), num_recur, dtype=torch.long, device=intermediate_states[0].device)
    results = {}
    for thr in thresholds:
        exit_depth = max_depth.clone()
        # Ratio defined from i=2 (needs steps i-1 and i-2)
        for i in range(1, len(step_norms)):
            ratio = step_norms[i] / (step_norms[i - 1] + 1e-10)
            # recurrence index i+1 corresponds to state index i+1 (1-indexed: i+2)
            # Exit at recurrence i+1 (0-indexed state) means we use logits[i+1]
            qualifies = ratio <= thr
            # Only update tokens that haven't exited yet
            mask = qualifies & (exit_depth == num_recur)
            exit_depth[mask] = i + 2  # 1-indexed recurrence depth
        results[thr] = exit_depth
    return results


def gate_relative_l2(
    intermediate_states: list[torch.Tensor],
    thresholds: list[float],
    **_kwargs,
) -> dict[float, torch.Tensor]:
    """
    Relative L2 gate: exit when ||s_i - s_{i-1}|| / ||s_i|| < threshold.
    """
    num_recur = len(intermediate_states)
    B, T, _D = intermediate_states[0].shape
    max_depth = torch.full((B, T), num_recur, dtype=torch.long, device=intermediate_states[0].device)

    results = {}
    for thr in thresholds:
        exit_depth = max_depth.clone()
        for i in range(1, num_recur):
            diff_norm = torch.norm(intermediate_states[i] - intermediate_states[i - 1], dim=-1)
            state_norm = torch.norm(intermediate_states[i], dim=-1)
            rel_l2 = diff_norm / (state_norm + 1e-10)
            qualifies = rel_l2 < thr
            mask = qualifies & (exit_depth == num_recur)
            exit_depth[mask] = i + 1  # 1-indexed
        results[thr] = exit_depth
    return results


def gate_kl_divergence(
    intermediate_logits: list[torch.Tensor],
    thresholds: list[float],
    **_kwargs,
) -> dict[float, torch.Tensor]:
    """
    KL divergence gate: exit when KL(p_i || p_{i-1}) < threshold.
    """
    num_recur = len(intermediate_logits)
    B, T, _V = intermediate_logits[0].shape
    max_depth = torch.full((B, T), num_recur, dtype=torch.long, device=intermediate_logits[0].device)

    # Precompute log probs
    log_probs = [F.log_softmax(il, dim=-1) for il in intermediate_logits]

    results = {}
    for thr in thresholds:
        exit_depth = max_depth.clone()
        for i in range(1, num_recur):
            # KL(p_i || p_{i-1}), summed over vocab, shape (B, T)
            kl = F.kl_div(log_probs[i - 1], log_probs[i], reduction="none", log_target=True).sum(dim=-1)
            qualifies = kl < thr
            mask = qualifies & (exit_depth == num_recur)
            exit_depth[mask] = i + 1  # 1-indexed
        results[thr] = exit_depth
    return results


# Gate registry: (name, function, default thresholds)
GATES = [
    ("acceleration", gate_acceleration, [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]),
    ("relative_l2", gate_relative_l2, [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]),
    ("kl_divergence", gate_kl_divergence, [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]),
]

# ---------------------------------------------------------------------------
# Batched categorical evaluation with gating


@torch.no_grad()
def evaluate_with_gates(
    model,
    tokenizer,
    task_object,
    batch_size: int = 8,
    num_recur: int | None = None,
    max_problems: int | None = None,
) -> dict:
    """
    Run categorical eval at max_r, collecting intermediate logits and states.

    For each batch, applies all gates x thresholds post-hoc by indexing into
    intermediate_logits[exit_depth] at the answer position.

    Returns a dict with:
        - gate_results: {gate_name: {threshold: {"correct": int, "total": int, "total_depth": int}}}
        - max_r_correct: int (baseline at full recurrence)
        - r1_correct: int (baseline at r=1)
        - train_r_correct: int (baseline at native training recurrence)
        - total: int
        - num_recur: int
        - train_recur: int
    """
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()
    train_recur = int(model.config.train_recur_mean)

    if num_recur is None:
        num_recur = train_recur

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Initialize gate result accumulators
    gate_results = {}
    for gate_name, _gate_fn, thresholds in GATES:
        gate_results[gate_name] = {}
        for thr in thresholds:
            gate_results[gate_name][thr] = {"correct": 0, "total": 0, "total_depth": 0}

    max_r_correct = 0
    r1_correct = 0
    train_r_correct = 0
    total = 0

    letter_to_id_cache = {}

    def ceil_div(x, y):
        return -(-x // y)

    # Distributed: stride batches across ranks
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    num_batches = ceil_div(num_problems, batch_size)

    for batch_idx in range(rank, num_batches, world_size):
        i0 = batch_idx * batch_size
        i1 = min((batch_idx + 1) * batch_size, num_problems)

        # Prepare batch
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conv) for conv in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_positions = [len(ids) - 1 for ids in prompt_ids]
        padded = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)

        # Forward with intermediate logits AND states
        logits, _s, intermediate_logits, intermediate_states = model(
            input_ids,
            num_recur=num_recur,
            return_intermediate_logits=True,
            return_intermediate_states=True,
        )
        # intermediate_logits: list of num_recur tensors, each (B, T, V)
        # intermediate_states: list of num_recur tensors, each (B, T, D)

        # Apply gates
        gate_exit_depths = {}
        for gate_name, gate_fn, thresholds in GATES:
            # Gates that use states
            if gate_name in ("acceleration", "relative_l2"):
                gate_exit_depths[gate_name] = gate_fn(
                    intermediate_states=intermediate_states,
                    thresholds=thresholds,
                )
            # Gates that use logits
            else:
                gate_exit_depths[gate_name] = gate_fn(
                    intermediate_logits=intermediate_logits,
                    thresholds=thresholds,
                )

        # Evaluate each example
        for idx, conversation in enumerate(conversations):
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded = tokenizer.encode(letter)
                    assert len(encoded) == 1
                    letter_to_id_cache[letter] = encoded[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_positions[idx]
            correct_letter = conversation["messages"][-1]["content"]
            correct_idx_in_letters = letters.index(correct_letter)

            # Max-r baseline
            focus_logits = logits[idx, answer_pos, letter_ids]
            predicted_idx = focus_logits.argmax(dim=-1).item()
            is_max_r_correct = predicted_idx == correct_idx_in_letters
            max_r_correct += int(is_max_r_correct)

            # r=1 baseline
            r1_focus = intermediate_logits[0][idx, answer_pos, letter_ids]
            r1_predicted = r1_focus.argmax(dim=-1).item()
            r1_correct += int(r1_predicted == correct_idx_in_letters)

            # train_r baseline (native training recurrence)
            train_r_focus = intermediate_logits[train_recur - 1][idx, answer_pos, letter_ids]
            train_r_predicted = train_r_focus.argmax(dim=-1).item()
            train_r_correct += int(train_r_predicted == correct_idx_in_letters)

            # Gate evaluations
            for gate_name, _gate_fn, thresholds in GATES:
                for thr in thresholds:
                    exit_depth_tensor = gate_exit_depths[gate_name][thr]
                    # Exit depth for this token (1-indexed)
                    token_exit = exit_depth_tensor[idx, answer_pos].item()
                    # Clamp to valid range
                    token_exit = max(1, min(token_exit, num_recur))
                    # Check correctness at exit depth
                    r_focus = intermediate_logits[token_exit - 1][idx, answer_pos, letter_ids]
                    r_predicted = r_focus.argmax(dim=-1).item()
                    is_correct = r_predicted == correct_idx_in_letters

                    gate_results[gate_name][thr]["correct"] += int(is_correct)
                    gate_results[gate_name][thr]["total"] += 1
                    gate_results[gate_name][thr]["total_depth"] += token_exit

            total += 1

        print0(f"\r\033[KBatch {batch_idx + 1}/{num_batches} | max_r acc: {max_r_correct}/{total} ({100 * max_r_correct / total:.1f}%)", end="")

    print0()

    # Reduce across ranks if distributed
    if world_size > 1:
        # Pack scalar counters into a tensor for a single all_reduce
        counters = torch.tensor(
            [max_r_correct, r1_correct, train_r_correct, total],
            dtype=torch.long, device=device,
        )
        dist.all_reduce(counters, op=dist.ReduceOp.SUM)
        max_r_correct, r1_correct, train_r_correct, total = counters.tolist()

        # Reduce gate results
        for gate_name, thresholds_dict in gate_results.items():
            gate_counters = []
            gate_keys = sorted(thresholds_dict.keys())
            for thr in gate_keys:
                s = thresholds_dict[thr]
                gate_counters.extend([s["correct"], s["total"], s["total_depth"]])
            gate_tensor = torch.tensor(gate_counters, dtype=torch.long, device=device)
            dist.all_reduce(gate_tensor, op=dist.ReduceOp.SUM)
            gate_list = gate_tensor.tolist()
            for i, thr in enumerate(gate_keys):
                thresholds_dict[thr]["correct"] = gate_list[i * 3]
                thresholds_dict[thr]["total"] = gate_list[i * 3 + 1]
                thresholds_dict[thr]["total_depth"] = gate_list[i * 3 + 2]

    return {
        "gate_results": gate_results,
        "max_r_correct": max_r_correct,
        "r1_correct": r1_correct,
        "train_r_correct": train_r_correct,
        "total": total,
        "num_recur": num_recur,
        "train_recur": train_recur,
    }


# ---------------------------------------------------------------------------
# FLOPs fraction computation


def compute_flops_fraction(model, exit_depth: float, num_recur: int) -> float:
    """Compute FLOPs fraction: (fixed + r_exit * recur) / (fixed + max_r * recur)."""
    params = model.num_scaling_params()
    fixed = params["prelude"] + params["coda"] + params["wte"] + params["lm_head"] + params["scalars"]
    recur = params["recur_block"] + params["inject"]
    full_cost = fixed + num_recur * recur
    exit_cost = fixed + exit_depth * recur
    return exit_cost / full_cost


# ---------------------------------------------------------------------------
# Plotting


def plot_pareto_curves(
    all_model_results: dict[str, dict],
    task_name: str,
    output_dir: Path,
):
    """
    Plot FLOPs fraction vs accuracy Pareto curves.

    One plot per task. Each gate is a curve (connected scatter). Baselines
    shown as horizontal/vertical lines. Faceted or overlaid across models.
    """
    num_models = len(all_model_results)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 6), squeeze=False)

    # Colorblind-safe palette (Wong)
    gate_colors = {
        "acceleration": "#332288",
        "relative_l2": "#E69F00",
        "kl_divergence": "#009E73",
    }
    gate_markers = {
        "acceleration": "o",
        "relative_l2": "s",
        "kl_divergence": "D",
    }

    for col_idx, (model_tag, result) in enumerate(all_model_results.items()):
        ax = axes[0, col_idx]
        model = result["model_ref"]
        num_recur = result["eval_result"]["num_recur"]
        train_recur = result["eval_result"]["train_recur"]
        total = result["eval_result"]["total"]
        max_r_acc = result["eval_result"]["max_r_correct"] / total
        r1_acc = result["eval_result"]["r1_correct"] / total
        train_r_acc = result["eval_result"]["train_r_correct"] / total

        # Baselines
        r1_flops = compute_flops_fraction(model, 1, num_recur)
        train_r_flops = compute_flops_fraction(model, train_recur, num_recur)
        ax.axhline(y=max_r_acc, color="black", linestyle="--", alpha=0.5, label=f"max_r={num_recur} ({max_r_acc:.3f})")
        ax.axhline(y=train_r_acc, color="#CC79A7", linestyle="-.", alpha=0.5, label=f"train_r={train_recur} ({train_r_acc:.3f})")
        ax.axvline(x=train_r_flops, color="#CC79A7", linestyle="-.", alpha=0.3)
        ax.axhline(y=r1_acc, color="gray", linestyle=":", alpha=0.5, label=f"r=1 ({r1_acc:.3f})")
        ax.axvline(x=r1_flops, color="gray", linestyle=":", alpha=0.3)

        # Gate curves
        for gate_name, thresholds_dict in result["eval_result"]["gate_results"].items():
            flops_list = []
            acc_list = []
            for thr, stats in sorted(thresholds_dict.items()):
                if stats["total"] == 0:
                    continue
                acc = stats["correct"] / stats["total"]
                mean_depth = stats["total_depth"] / stats["total"]
                flops_frac = compute_flops_fraction(model, mean_depth, num_recur)
                flops_list.append(flops_frac)
                acc_list.append(acc)

            if flops_list:
                ax.plot(
                    flops_list, acc_list,
                    color=gate_colors[gate_name],
                    marker=gate_markers[gate_name],
                    markersize=5,
                    label=gate_name,
                    linewidth=1.5,
                    alpha=0.8,
                )

        ax.set_xlabel("FLOPs Fraction", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{task_name} — {model_tag} (r={num_recur})", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"gating_pareto_{task_name.lower().replace('-', '_')}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print0(f"Plot saved to: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CSV output


def save_results_csv(
    all_model_results: dict[str, dict],
    task_name: str,
    output_dir: Path,
):
    """Save raw results as CSV for downstream analysis."""
    csv_path = output_dir / f"gating_results_{task_name.lower().replace('-', '_')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_tag", "task", "gate", "threshold",
            "accuracy", "mean_exit_depth", "flops_fraction",
            "correct", "total",
        ])

        for model_tag, result in all_model_results.items():
            model = result["model_ref"]
            num_recur = result["eval_result"]["num_recur"]
            total = result["eval_result"]["total"]

            # Baselines
            train_recur = result["eval_result"]["train_recur"]

            max_r_acc = result["eval_result"]["max_r_correct"] / total
            writer.writerow([model_tag, task_name, "max_r", num_recur, f"{max_r_acc:.6f}", num_recur, "1.000000", result["eval_result"]["max_r_correct"], total])

            train_r_acc = result["eval_result"]["train_r_correct"] / total
            train_r_flops = compute_flops_fraction(model, train_recur, num_recur)
            writer.writerow([model_tag, task_name, "train_r", train_recur, f"{train_r_acc:.6f}", train_recur, f"{train_r_flops:.6f}", result["eval_result"]["train_r_correct"], total])

            r1_acc = result["eval_result"]["r1_correct"] / total
            r1_flops = compute_flops_fraction(model, 1, num_recur)
            writer.writerow([model_tag, task_name, "r1", 1, f"{r1_acc:.6f}", 1, f"{r1_flops:.6f}", result["eval_result"]["r1_correct"], total])

            # Gate results
            for gate_name, thresholds_dict in result["eval_result"]["gate_results"].items():
                for thr, stats in sorted(thresholds_dict.items()):
                    if stats["total"] == 0:
                        continue
                    acc = stats["correct"] / stats["total"]
                    mean_depth = stats["total_depth"] / stats["total"]
                    flops_frac = compute_flops_fraction(model, mean_depth, num_recur)
                    writer.writerow([model_tag, task_name, gate_name, thr, f"{acc:.6f}", f"{mean_depth:.2f}", f"{flops_frac:.6f}", stats["correct"], stats["total"]])

    print0(f"CSV saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Main


def main():
    parser = argparse.ArgumentParser(description="Early-exit gating analysis for looped transformers")
    parser.add_argument("-i", "--source", type=str, required=True, help="Model source: base|sft|rl")
    parser.add_argument("-a", "--task-name", type=str, default="ARC-Easy", help="Pipe-separated task names (e.g., ARC-Easy|MMLU). Default: ARC-Easy.")
    parser.add_argument("-g", "--model-tags", type=str, default=None, help="Pipe-separated model tags (e.g., d12|d20). Default: largest.")
    parser.add_argument("-r", "--num-recur", type=int, default=None, help="Max recurrences (default: model default)")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size for categorical eval")
    parser.add_argument("-x", "--max-problems", type=int, default=None, help="Max problems to evaluate")
    parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps", ""])
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _ddp_rank, _ddp_local_rank, _ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # Parse model tags and task names
    model_tags = args.model_tags.split("|") if args.model_tags else [None]
    task_names = args.task_name.split("|")
    for tn in task_names:
        assert tn in TASK_MODULES, f"Unknown task: {tn}. Available: {list(TASK_MODULES.keys())}"

    # Output directory
    output_dir = Path(get_base_dir()) / "plots"
    output_dir.mkdir(exist_ok=True)

    # Load models once, reuse across tasks
    models = {}
    for model_tag in model_tags:
        tag_label = model_tag or "default"
        print0(f"\nLoading model: {tag_label} (source={args.source})")
        model, tokenizer, _meta = load_model(args.source, device, phase="eval", model_tag=model_tag)
        num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
        print0(f"  num_recur={num_recur}, params={model.num_scaling_params()}")
        models[tag_label] = (model, tokenizer, num_recur)

    # Evaluate each task
    for task_name in task_names:
        print0(f"\n{'#' * 60}")
        print0(f"Task: {task_name}")
        print0(f"{'#' * 60}")
        task_object = TASK_MODULES[task_name]()

        all_model_results = {}
        for tag_label, (model, tokenizer, num_recur) in models.items():
            print0(f"\n{'=' * 60}")
            print0(f"Model: {tag_label} (r={num_recur})")
            print0(f"{'=' * 60}")

            with autocast_ctx:
                eval_result = evaluate_with_gates(
                    model, tokenizer, task_object,
                    batch_size=args.batch_size,
                    num_recur=num_recur,
                    max_problems=args.max_problems,
                )

            total = eval_result["total"]
            print0(f"Results for {tag_label}:")
            print0(f"  max_r accuracy:   {eval_result['max_r_correct']}/{total} ({100 * eval_result['max_r_correct'] / total:.2f}%)")
            print0(f"  train_r accuracy: {eval_result['train_r_correct']}/{total} ({100 * eval_result['train_r_correct'] / total:.2f}%) [train_r={eval_result['train_recur']}]")
            print0(f"  r=1 accuracy:     {eval_result['r1_correct']}/{total} ({100 * eval_result['r1_correct'] / total:.2f}%)")

            all_model_results[tag_label] = {
                "eval_result": eval_result,
                "model_ref": model,
            }

            # Print gate summary
            for gate_name, thresholds_dict in eval_result["gate_results"].items():
                print0(f"\n  {gate_name}:")
                for thr, stats in sorted(thresholds_dict.items()):
                    if stats["total"] == 0:
                        continue
                    acc = stats["correct"] / stats["total"]
                    mean_depth = stats["total_depth"] / stats["total"]
                    flops_frac = compute_flops_fraction(model, mean_depth, num_recur)
                    print0(f"    thr={thr:<8} acc={acc:.4f}  mean_depth={mean_depth:.2f}  flops={flops_frac:.4f}")

        # Plot and save
        plot_pareto_curves(all_model_results, task_name, output_dir)
        save_results_csv(all_model_results, task_name, output_dir)

    compute_cleanup()


if __name__ == "__main__":
    main()
