"""Arithmetic evaluation tasks for base models.

Procedurally generates graduated-difficulty arithmetic/algorithmic problems
to test whether looped (depth-recurrent) models benefit from additional
computation depth on tasks that require sequential reasoning.

Tasks (graduated difficulty):
  Tier 1 - Parity: is the sum of a digit sequence even or odd?
  Tier 3 - Addition: multi-digit addition with space-separated digit output

Evaluation is teacher-forced: the full prompt including the answer is fed
in a single forward pass, and we check if the model's argmax predictions
at continuation positions match the ground truth. Examples are batched
for throughput on small models with short sequences.

Example rendered prompts (5-shot parity, difficulty=3):

    Parity: 7 3 2 -> even
    Parity: 1 8 5 -> even
    Parity: 9 4 1 -> even
    Parity: 6 3 7 -> even
    Parity: 2 5 9 -> even
    Parity: 4 1 6 -> odd

The model must predict "odd" after the final " -> ".

Example rendered prompts (4-shot addition, difficulty=2):

    Add: 4 7 + 2 8 = 7 5
    Add: 9 1 + 3 6 = 1 2 7
    Add: 5 3 + 6 9 = 1 2 2
    Add: 8 4 + 1 5 = 9 9
    Add: 3 7 + 5 8 = 9 5

The model must predict "9 5" after the final " = ".
"""

import random
import time
from functools import partial

import torch
import torch.distributed as dist

from nanochat.common import print0
from nanochat.core_eval import forward_model, render_prompts_lm


# ---------------------------------------------------------------------------
# Data generators
#
# Each returns (data, task_meta) where:
#   data: list[dict] with "context" and "continuation" keys
#   task_meta: dict with "task_type", "num_fewshot", "continuation_delimiter"


def generate_parity_data(
    difficulty: int,
    num_examples: int = 500,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    Parity: determine if the sum of a digit sequence is even or odd.

    difficulty: sequence length (1-8 single digits)
    Random baseline: 50% (binary classification)
    """
    rng = random.Random(seed)
    data = []
    for _ in range(num_examples):
        digits = [rng.randint(1, 9) for _ in range(difficulty)]
        context = "Parity: " + " ".join(str(d) for d in digits)
        answer = "even" if sum(digits) % 2 == 0 else "odd"
        data.append({"context": context, "continuation": answer})
    task_meta = {
        "task_type": "language_modeling",
        "num_fewshot": 5,
        "continuation_delimiter": " -> ",
    }
    return data, task_meta


def generate_addition_data(
    difficulty: int,
    num_examples: int = 500,
    seed: int = 42,
    reverse: bool = False,
) -> tuple[list[dict], dict]:
    """
    Multi-digit addition with space-separated digit output.

    difficulty: number of digits per operand (2-5)
    reverse: if True, present digits in reverse order (LSB first),
             which is algorithmically easier for autoregressive models
             since carry propagation matches left-to-right generation.
    Random baseline: ~0% (exact match of all output digits)
    """
    rng = random.Random(seed)
    data = []
    for _ in range(num_examples):
        a = rng.randint(10 ** (difficulty - 1), 10**difficulty - 1)
        b = rng.randint(10 ** (difficulty - 1), 10**difficulty - 1)
        result = a + b

        a_digits = list(str(a))
        b_digits = list(str(b))
        r_digits = list(str(result))

        if reverse:
            a_digits = a_digits[::-1]
            b_digits = b_digits[::-1]
            r_digits = r_digits[::-1]

        context = f"Add: {' '.join(a_digits)} + {' '.join(b_digits)}"
        continuation = " ".join(r_digits)
        data.append({"context": context, "continuation": continuation})

    task_meta = {
        "task_type": "language_modeling",
        "num_fewshot": 4,
        "continuation_delimiter": " = ",
    }
    return data, task_meta


# ---------------------------------------------------------------------------
# Batched teacher-forced evaluation


@torch.no_grad()
def evaluate_task_batched(
    model,
    tokenizer,
    data: list[dict],
    device,
    task_meta: dict,
    batch_size: int = 64,
    num_recur: int | None = None,
    debug: int = 0,
) -> float:
    """
    Batched teacher-forced evaluation for language_modeling tasks.

    Packs multiple examples into a single forward pass instead of the
    one-at-a-time loop in core_eval.evaluate_task. Much faster for short
    sequences on small models.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]
    pad_token_id = tokenizer.get_bos_token_id()
    bos = tokenizer.get_bos_token_id()

    # Indices assigned to this rank
    rank_indices = list(range(rank, len(data), world_size))

    # Render and tokenize all examples for this rank
    all_tokens: list[list[int]] = []
    all_si: list[int] = []
    all_ei: list[int] = []

    for idx in rank_indices:
        item = data[idx]

        # Sample few-shot examples (same RNG as core_eval for consistency)
        rng = random.Random(1234 + idx)
        available = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

        # Render and tokenize
        prompt_without, prompt_with = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens_without = tokenizer(prompt_without, prepend=bos)
        tokens_with = tokenizer(prompt_with, prepend=bos)
        si, ei = len(tokens_without), len(tokens_with)
        assert si < ei, "continuation must be non-empty"
        assert tokens_without == tokens_with[:si], "prompt_without must be a prefix of prompt_with"

        # Truncate if model has max_seq_len
        if hasattr(model, "max_seq_len") and model.max_seq_len and ei > model.max_seq_len:
            crop = ei - model.max_seq_len
            tokens_with = tokens_with[crop:]
            si -= crop
            ei -= crop
            assert si >= 0

        all_tokens.append(tokens_with)
        all_si.append(si)
        all_ei.append(ei)

    # Process in mini-batches
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)

    for batch_start in range(0, len(rank_indices), batch_size):
        batch_end = min(batch_start + batch_size, len(rank_indices))
        batch_tokens = all_tokens[batch_start:batch_end]
        batch_si = all_si[batch_start:batch_end]
        batch_ei = all_ei[batch_start:batch_end]

        # Pad and stack into (B, T) tensor
        max_len = max(len(t) for t in batch_tokens)
        bsz = len(batch_tokens)
        input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long, device=device)
        for i, t in enumerate(batch_tokens):
            input_ids[i, : len(t)] = torch.tensor(t, dtype=torch.long)

        # Single forward pass for the whole batch
        _losses, predictions = forward_model(model, input_ids, num_recur=num_recur)

        # Check each example's continuation
        for i in range(bsz):
            si, ei = batch_si[i], batch_ei[i]
            predicted = predictions[i, si - 1 : ei - 1]
            actual = input_ids[i, si:ei]
            is_correct = torch.all(predicted == actual).item()
            global_idx = rank_indices[batch_start + i]
            correct[global_idx] = float(is_correct)

            if debug > 0 and rank == 0 and global_idx < debug:
                prompt_tokens = input_ids[i, :si].tolist()
                pred_tokens = predicted.tolist()
                actual_tokens = actual.tolist()
                prompt_str = tokenizer.decode(prompt_tokens)
                pred_str = tokenizer.decode(pred_tokens)
                actual_str = tokenizer.decode(actual_tokens)
                mark = "OK" if is_correct else "WRONG"
                print0(f"\n  [{mark}] Example {global_idx}")
                print0(f"  Prompt (last 80 chars): ...{prompt_str[-80:]}")
                print0(f"  Expected: {actual_str!r}  ({actual_tokens})")
                print0(f"  Got:      {pred_str!r}  ({pred_tokens})")

    # Sync across ranks
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    return correct.mean().item()


# ---------------------------------------------------------------------------
# Task registry
#
# Each entry: (label, generator_fn, difficulty_levels, random_baseline)

ARITHMETIC_TASKS = [
    ("parity", generate_parity_data, list(range(1, 9)), 0.5),
    ("addition", generate_addition_data, list(range(2, 6)), 0.0),
    ("addition_rev", partial(generate_addition_data, reverse=True), list(range(2, 6)), 0.0),
]


# ---------------------------------------------------------------------------
# Main evaluation entry point


def evaluate_arithmetic(
    model,
    tokenizer,
    device,
    num_examples: int = 500,
    batch_size: int = 64,
    num_recur: int | None = None,
    debug: int = 0,
) -> dict[str, float]:
    """
    Run all arithmetic evaluation tasks across all difficulty levels.

    Returns dict mapping "task_name/d{difficulty}" -> accuracy.
    debug: if > 0, print this many example prompts + predictions per task.
    """
    results = {}
    for task_name, generator, difficulties, baseline in ARITHMETIC_TASKS:
        for diff in difficulties:
            label = f"{task_name}/d{diff}"
            data, task_meta = generator(difficulty=diff, num_examples=num_examples, seed=42 + diff)
            print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, n={len(data)})... ", end="")
            start = time.time()
            accuracy = evaluate_task_batched(
                model, tokenizer, data, device, task_meta, batch_size=batch_size, num_recur=num_recur,
                debug=debug,
            )
            elapsed = time.time() - start
            results[label] = accuracy
            print0(f"accuracy: {accuracy:.4f} (baseline: {baseline:.2f}) | time: {elapsed:.2f}s")
    return results
