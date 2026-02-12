"""
t-SNE visualization of latent states across multiple evaluation samples.

This script:
1. Loads multiple test cases from a chosen evaluation dataset
2. Generates responses using the looped transformer
3. Captures the recurrent state after each loop iteration for all tokens (input + output)
4. Projects the high-dimensional latent states into 2D via t-SNE
5. Plots the 2D embeddings colored by loop step and shaped by token type (input vs output)

Example:
    uv run python dev/analysis/visualize_latent_tsne.py -i sft --num-recur 16 --num-samples 5
    uv run python dev/analysis/visualize_latent_tsne.py -i sft --num-recur 16 --task-name MMLU --num-samples 5
"""

import argparse
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

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


def collect_latent_records(
    engine: Engine,
    tokenizer,
    task_object,
    num_samples: int,
    sample_idx: int,
    num_recur: int,
    max_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    kv_budget: int,
    use_warm_start: bool,
    autocast_ctx,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect latent state vectors across multiple evaluation samples.

    Returns:
        vectors: (N, hidden_dim) array of latent state vectors
        loop_steps: (N,) array of loop step indices
        token_types: (N,) array of 0 (input) or 1 (output)
        sample_ids: (N,) array of sample indices
    """
    all_vectors = []
    all_loop_steps = []
    all_token_types = []
    all_sample_ids = []

    for sample_offset in range(num_samples):
        idx = sample_idx + sample_offset
        print(f"\nProcessing sample {idx} ({sample_offset + 1}/{num_samples})")

        conversation = task_object.get_example(idx)
        question = conversation["messages"][0]["content"]
        prompt_tokens = tokenizer.render_for_completion(conversation)

        print(f"  Question: {question[:80]}...")
        print(f"  Prompt length: {len(prompt_tokens)} tokens")

        with autocast_ctx:
            generated_tokens, input_latent_states, output_latent_states, _ = generate_with_latent_tracking(
                engine=engine,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                num_recur=num_recur,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed + sample_offset,
                kv_budget=kv_budget,
                use_warm_start=use_warm_start,
            )

        latent_states_per_token = input_latent_states + output_latent_states
        num_input_tokens = len(input_latent_states)
        num_output_tokens = len(generated_tokens)
        print(f"  Generated {num_output_tokens} tokens")
        print(f"  Captured states for {len(latent_states_per_token)} tokens ({num_input_tokens} input + {len(output_latent_states)} output)")

        # Flatten: for each token, for each loop step, extract the vector
        for token_idx, states in enumerate(latent_states_per_token):
            is_output = 1 if token_idx >= num_input_tokens else 0
            for loop_step, state_tensor in enumerate(states):
                vec = state_tensor.flatten().float().numpy()
                all_vectors.append(vec)
                all_loop_steps.append(loop_step)
                all_token_types.append(is_output)
                all_sample_ids.append(sample_offset)

    vectors = np.stack(all_vectors)
    loop_steps = np.array(all_loop_steps)
    token_types = np.array(all_token_types)
    sample_ids = np.array(all_sample_ids)

    print(f"\nTotal latent state records: {len(vectors)}")
    print(f"  Input token records: {(token_types == 0).sum()}")
    print(f"  Output token records: {(token_types == 1).sum()}")

    return vectors, loop_steps, token_types, sample_ids


def stratified_subsample(
    vectors: np.ndarray,
    loop_steps: np.ndarray,
    token_types: np.ndarray,
    sample_ids: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Subsample points while preserving the ratio of (loop_step, token_type) strata."""
    n = len(vectors)
    if n <= max_points:
        return vectors, loop_steps, token_types, sample_ids

    print(f"Subsampling from {n} to {max_points} points (stratified)...")

    # Build stratum keys
    strata = loop_steps * 2 + token_types  # unique key per (loop_step, token_type)
    unique_strata = np.unique(strata)

    # Allocate budget proportionally per stratum
    stratum_indices = {s: np.where(strata == s)[0] for s in unique_strata}
    stratum_sizes = {s: len(idx) for s, idx in stratum_indices.items()}
    total = sum(stratum_sizes.values())

    selected = []
    for s in unique_strata:
        budget = max(1, int(round(stratum_sizes[s] / total * max_points)))
        budget = min(budget, stratum_sizes[s])
        chosen = rng.choice(stratum_indices[s], size=budget, replace=False)
        selected.append(chosen)

    selected = np.concatenate(selected)
    # Trim if over budget due to rounding
    if len(selected) > max_points:
        selected = rng.choice(selected, size=max_points, replace=False)

    return vectors[selected], loop_steps[selected], token_types[selected], sample_ids[selected]


def plot_tsne(
    embedding: np.ndarray,
    loop_steps: np.ndarray,
    token_types: np.ndarray,
    num_recur: int,
    num_samples: int,
    source: str,
    task_name: str,
    point_size: float,
    alpha: float,
    kv_budget: int,
    use_warm_start: bool,
    perplexity: float,
):
    """Plot t-SNE embedding colored by loop step and shaped by token type."""
    fig, ax = plt.subplots(figsize=(12, 10))

    cmap = plt.get_cmap("viridis", num_recur)
    input_mask = token_types == 0
    output_mask = token_types == 1

    markers = {0: "o", 1: "^"}  # input=circle, output=triangle
    labels = {0: "Input tokens", 1: "Output tokens"}

    # Plot each loop step in order so later steps render on top
    for step in range(num_recur):
        step_mask = loop_steps == step
        for token_type, marker in markers.items():
            type_mask = input_mask if token_type == 0 else output_mask
            mask = step_mask & type_mask
            if not mask.any():
                continue
            sc = ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[cmap(step / (num_recur - 1))] * mask.sum(),
                marker=marker,
                s=point_size,
                alpha=alpha,
                edgecolors="none",
            )

    # Colorbar via ScalarMappable (independent of plot order)
    norm = mcolors.Normalize(vmin=0, vmax=num_recur - 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    cbar = fig.colorbar(sm, ax=ax, label="Loop step", ticks=np.arange(0, num_recur, max(1, num_recur // 8)))
    cbar.ax.tick_params(labelsize=9)

    # Legend for marker shapes only (not colors)
    legend_handles = []
    for token_type, marker in markers.items():
        mask = input_mask if token_type == 0 else output_mask
        if mask.any():
            legend_handles.append(
                ax.scatter([], [], marker=marker, c="gray", s=40, label=labels[token_type], edgecolors="none")
            )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    warmstart_text = ", warm-start" if use_warm_start else ""
    ax.set_title(
        f"t-SNE of Latent States ({source} model)\n"
        f"num_recur={num_recur}, {num_samples} samples, kv_budget={kv_budget}, perplexity={perplexity}{warmstart_text}",
        fontsize=13,
    )
    ax.set_xlabel("t-SNE dim 1", fontsize=11)
    ax.set_ylabel("t-SNE dim 2", fontsize=11)
    ax.grid(True, alpha=0.2)

    # Save
    plots_dir = Path(get_base_dir()) / "plots"
    plots_dir.mkdir(exist_ok=True)
    warmstart_suffix = "_warmstart" if use_warm_start else ""
    task_slug = task_name.lower().replace("-", "_")
    output_path = plots_dir / f"{task_slug}_latent_tsne_recur{num_recur}_samples{num_samples}{warmstart_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nt-SNE plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization of latent states across evaluation samples")
    parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: base|sft|rl (default: sft)")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., d12). If not specified, uses largest model.")
    parser.add_argument("-a", "--task-name", type=str, default="GSM8K", choices=list(TASK_MODULES.keys()), help="Evaluation task to load (default: GSM8K)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index to start from (default: 0)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples (default: 5)")
    parser.add_argument("--num-recur", type=int, default=16, help="Number of recurrences (default: 16)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default: 0.0)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--kv-budget", type=int, default=1, help="Fixed KV-cache budget for recurrences (default: 1)")
    parser.add_argument("--use-rec-warm-start", action="store_true", help="Use recurrent warm-start")
    parser.add_argument("--max-points", type=int, default=5000, help="Max points for t-SNE (subsample if exceeded, default: 5000)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (default: 30)")
    parser.add_argument("--point-size", type=float, default=15.0, help="Scatter point size (default: 15)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Point transparency (default: 0.6)")
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

    num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
    print(f"Using num_recur={num_recur}")

    # Load dataset
    print(f"Loading {args.task_name} dataset...")
    task_object = TASK_MODULES[args.task_name]()

    # Collect latent states
    vectors, loop_steps, token_types, sample_ids = collect_latent_records(
        engine=engine,
        tokenizer=tokenizer,
        task_object=task_object,
        num_samples=args.num_samples,
        sample_idx=args.sample_idx,
        num_recur=num_recur,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        kv_budget=args.kv_budget,
        use_warm_start=args.use_rec_warm_start,
        autocast_ctx=autocast_ctx,
    )

    # Subsample if needed
    rng = np.random.default_rng(args.seed)
    vectors, loop_steps, token_types, sample_ids = stratified_subsample(
        vectors, loop_steps, token_types, sample_ids, args.max_points, rng
    )

    # Run t-SNE
    print(f"\nRunning t-SNE on {len(vectors)} points (perplexity={args.perplexity})...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(vectors)
    print("t-SNE complete.")

    # Plot
    plot_tsne(
        embedding=embedding,
        loop_steps=loop_steps,
        token_types=token_types,
        num_recur=num_recur,
        num_samples=args.num_samples,
        source=args.source,
        task_name=args.task_name,
        point_size=args.point_size,
        alpha=args.alpha,
        kv_budget=args.kv_budget,
        use_warm_start=args.use_rec_warm_start,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
