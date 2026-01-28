"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -i mid -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i mid -a ARC-Easy
"""

import argparse
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist

from nanochat.checkpoint_manager import load_model
from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_dist_info,
    print0,
)
from nanochat.engine import Engine
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.spellingbee import SpellingBee

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)


def run_generative_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
    num_recur=None,
    use_warm_start=False,
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            num_recur=num_recur,
            use_warm_start=use_warm_start,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(
            f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100 * num_passed / total:.2f}%)",
            end="",
            flush=True,
        )

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100 * num_passed / total:.2f}%)")

    # Return the accuracy
    return num_passed / total


# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.


def run_categorical_eval(
    task_object, tokenizer, model, batch_size, max_problems=None, num_recur=None
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()  # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {}  # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [
            tokenizer.render_for_completion(conversation) for conversation in conversations
        ]  # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [
            len(ids) - 1 for ids in prompt_ids
        ]  # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits, _ = model(prompt_ids, num_recur=num_recur)  # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation["letters"]
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed / total
    print0(f"Final: {num_passed}/{total} ({100 * average:.2f}%)")
    return average


# -----------------------------------------------------------------------------


def run_chat_eval(
    task_name,
    model,
    tokenizer,
    engine,
    batch_size=1,
    num_samples=1,
    max_new_tokens=512,
    temperature=0.0,
    top_k=50,
    max_problems=None,
    num_recur=None,
    use_warm_start=False,
):
    # Create the evaluation object
    task_module = {
        "HumanEval": HumanEval,
        "MMLU": partial(MMLU, subset="all", split="test"),
        "ARC-Easy": partial(ARC, subset="ARC-Easy", split="test"),
        "ARC-Challenge": partial(ARC, subset="ARC-Challenge", split="test"),
        "GSM8K": partial(GSM8K, subset="main", split="test"),
        "SpellingBee": partial(SpellingBee, size=256, split="test"),
    }[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == "generative":
        acc = run_generative_eval(
            task_object,
            tokenizer,
            model,
            engine,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            max_problems=max_problems,
            num_recur=num_recur,
            use_warm_start=use_warm_start,
        )
    elif task_object.eval_type == "categorical":
        acc = run_categorical_eval(
            task_object,
            tokenizer,
            model,
            batch_size,
            max_problems=max_problems,
            num_recur=num_recur,
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return acc


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--source", type=str, required=True, help="Source of the model: sft|mid|rl"
    )
    parser.add_argument(
        "-a",
        "--task-name",
        type=str,
        default=None,
        help="Task name. Default = all tasks. Use | to split multiple tasks.",
    )
    parser.add_argument(
        "-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"]
    )
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8, help="Batch size for categorical evaluation"
    )
    parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
    parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
    parser.add_argument(
        "-x", "--max-problems", type=int, default=None, help="Max problems to evaluate"
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="",
        choices=["cuda", "cpu", "mps"],
        help="Device type for evaluation: cuda|cpu|mps. empty => autodetect",
    )
    parser.add_argument(
        "--num-recur",
        type=str,
        default=None,
        help='Number of recurrences for recursive transformer. Can be single value (e.g. "4") or comma-separated list (e.g. "2,4,8,16") to evaluate multiple.',
    )
    parser.add_argument(
        "-rws",
        "--use-rec-warm-start",
        action="store_true",
        help="Use recurrent warm-start (carry recurrent state when decoding tokens)",
    )
    args = parser.parse_args()

    # Parse num_recur argument - can be single value or comma-separated list
    recur_values = [None]  # default: use model's default
    if args.num_recur is not None:
        recur_values = [int(x.strip()) for x in args.num_recur.split(",")]

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda"
        else nullcontext()
    )

    model, tokenizer, meta = load_model(
        args.source, device, phase="eval", model_tag=args.model_tag, step=args.step
    )
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
    baseline_accuracies = {
        "ARC-Easy": 0.25,  # multiple choice 1 of 4 => 25%
        "ARC-Challenge": 0.25,  # multiple choice 1 of 4 => 25%
        "MMLU": 0.25,  # multiple choice 1 of 4 => 25%
        "GSM8K": 0.0,  # open-ended => 0%
        "HumanEval": 0.0,  # open-ended => 0%
        "SpellingBee": 0.0,  # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split("|")

    # Run all the task evaluations for each num_recur value
    from nanochat.report import get_report

    all_results = {}
    for num_recur in recur_values:
        recur_label = f"r{num_recur}" if num_recur is not None else "default"
        print0(f"\n{'=' * 80}")
        print0(f"Evaluating with num_recur={num_recur if num_recur is not None else 'default'}")
        print0(f"{'=' * 80}")

        results = {}
        for task_name in task_names:
            with autocast_ctx:
                acc = run_chat_eval(
                    task_name,
                    model,
                    tokenizer,
                    engine,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    max_problems=args.max_problems,
                    num_recur=num_recur,
                    use_warm_start=args.use_rec_warm_start,
                )
                results[task_name] = acc
                print0(f"{task_name} accuracy: {100 * acc:.2f}%")

        all_results[recur_label] = results

        # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
        # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
        all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
        chatcore_metric_dict = {}
        if all_tasks_were_evaluated:
            centered_mean = 0
            for task_name, acc in results.items():
                baseline_acc = baseline_accuracies.get(task_name, 0.0)
                centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
                centered_mean += centered_acc
            chatcore_metric = centered_mean / len(results)
            chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
            print0(f"ChatCORE metric ({recur_label}): {chatcore_metric:.4f}")

        # Log to report for this recur value
        section_name = f"Chat evaluation {args.source}" + (
            f" (r={num_recur})" if num_recur is not None else ""
        )
        get_report().log(
            section=section_name,
            data=[
                vars(args),
                {"num_recur_here": num_recur},
                results,
                chatcore_metric_dict,
            ],
        )

    compute_cleanup()
