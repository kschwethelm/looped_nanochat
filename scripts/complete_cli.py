"""
Text completion CLI for base (non-instruction-tuned) models.
Feeds a raw text prompt and streams the model's continuation.

Usage:
    uv run python -m scripts.complete_cli -p "The quick brown fox"
    uv run python -m scripts.complete_cli  # interactive mode
"""

import argparse
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

parser = argparse.ArgumentParser(description="Text completion with a base model")
parser.add_argument("-g", "--model-tag", type=str, default=None, help="Model tag to load")
parser.add_argument("-s", "--step", type=int, default=None, help="Step to load")
parser.add_argument("-p", "--prompt", type=str, default="", help="Text prompt to complete (if empty, enters interactive mode)")
parser.add_argument("-m", "--max-tokens", type=int, default=256, help="Maximum tokens to generate")
parser.add_argument("-t", "--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-k sampling parameter")
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"], help="Device type (empty => autodetect)")
parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("-r", "--num-recur", type=int, default=None, help="Number of recurrences (optional, uses model default if not specified)")
parser.add_argument("-rws", "--use-rec-warm-start", action="store_true", help="Carry recurrent state across decoded tokens")
parser.add_argument("-kb", "--kv-budget", type=int, default=1, help="Fixed KV-cache budget for recurrences (default=1)")
args = parser.parse_args()

# Init model and tokenizer
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)

engine = Engine(model, tokenizer)

print("\nBase Model Completion Mode")
print("-" * 50)
print("Type 'quit' or 'exit' to stop")
print("-" * 50)


def complete(prompt_text: str):
    """Tokenize prompt, generate, and stream output."""
    bos = tokenizer.get_bos_token_id()
    tokens = [bos] + tokenizer.encode(prompt_text)

    print(f"\n--- Prompt ---\n{prompt_text}", end="", flush=True)
    print("\n--- Completion ---")

    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "use_warm_start": args.use_rec_warm_start,
        "kv_budget": args.kv_budget,
    }
    with autocast_ctx:
        for token_column, _token_masks in engine.generate(tokens, num_recur=args.num_recur, **generate_kwargs):
            token = token_column[0]
            # Stop on BOS (base models may emit it as a separator)
            if token == bos:
                break
            print(tokenizer.decode([token]), end="", flush=True)
    print("\n")


if args.prompt:
    complete(args.prompt)
else:
    while True:
        try:
            user_input = input("\nPrompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not user_input:
            continue
        complete(user_input)
