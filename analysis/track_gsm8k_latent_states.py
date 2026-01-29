"""
Track latent states during GSM8K response generation.

This script:
1. Loads GSM8K test cases
2. Generates responses using the looped transformer
3. Captures the recurrent state (u) after each loop iteration
4. Saves: question, response, latent_states_per_loop, tokens to a pickle file

The latent state tracking is done via PyTorch hooks on the final recurrent block.
Only the final token's latent state is saved for each generation.
"""

import argparse
import pickle
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K


@dataclass
class GenerationResult:
    """Container for a single GSM8K generation result."""

    question: str
    response: str
    tokens: list[int]
    latent_states: list[torch.Tensor]  # One state per recurrence iteration
    num_recur: int


class LatentStateHook:
    """Hook to capture latent states during recurrent block execution."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset state tracking for a new forward pass."""
        self.states = []
        self.forward_count = 0

    def __call__(self, module, input, output):  # noqa: ARG002
        """Hook called after each recurrent block execution."""
        # output is the activations after the last recurrent block: shape (B, T, hidden_dim)
        # We only want the final token's state
        final_token_state = output[:, -1, :].detach().cpu().clone()
        self.states.append(final_token_state)
        self.forward_count += 1


def generate_with_latent_tracking(
    engine,
    tokenizer,
    prompt_tokens,
    num_recur,
    max_tokens=512,
    temperature=0.7,
    top_k=50,
    seed=42,
):
    """
    Generate response while tracking latent states.

    Returns:
        tokens: list of generated token ids
        latent_states: list of tensors, one per recurrence iteration (for final token only)
    """
    # Create hook to capture latent states
    hook = LatentStateHook()

    # Register hook on the last recurrent block
    # This will fire num_recur times per forward pass during generation
    last_recur_block = engine.model.transformer.recur[-1]
    handle = last_recur_block.register_forward_hook(hook)

    try:
        # Generate tokens
        generated_tokens = []
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        bos = tokenizer.get_bos_token_id()

        # Track the latent states from the most recent token generation
        # After the loop completes, this will contain states from the final token
        latent_states_final = []
        states_before_token = len(hook.states)

        for token_column, _token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            num_recur=num_recur,
        ):
            token = token_column[0]  # batch size is 1

            # Stop if we hit a terminal token (check BEFORE appending)
            if token in (assistant_end, bos):
                break

            generated_tokens.append(token)

            # After this token is generated, capture its latent states
            # The new states are at the end of hook.states
            states_after_token = len(hook.states)
            new_states = hook.states[states_before_token:states_after_token]

            # Should have num_recur new states (one per loop iteration)
            if len(new_states) >= num_recur:
                # Keep updating latent_states_final with each token
                # After the loop, it will contain states from the final token
                latent_states_final = new_states[-num_recur:]

            states_before_token = states_after_token

    finally:
        # Clean up hook
        handle.remove()

    return generated_tokens, latent_states_final


def main():
    parser = argparse.ArgumentParser(description="Track latent states during GSM8K generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="base",
        help="Checkpoint to load (default: base)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of GSM8K test samples to process (default: 100)",
    )
    parser.add_argument(
        "--num-recur",
        type=int,
        default=None,
        help="Number of recurrences (default: use model's train_recur_mean)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per response (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gsm8k_latent_states.pkl",
        help="Output pickle file (default: gsm8k_latent_states.pkl)",
    )
    args = parser.parse_args()

    # Initialize compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    if ddp:
        print("Warning: Running with DDP, but this script is designed for single GPU")

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, tokenizer, meta = load_model(args.checkpoint, device, phase="eval")
    engine = Engine(model, tokenizer)

    # Determine num_recur
    num_recur = args.num_recur if args.num_recur is not None else int(model.config.train_recur_mean)
    print(f"Using num_recur={num_recur}")

    # Load GSM8K dataset
    print("Loading GSM8K dataset...")
    gsm8k = GSM8K(subset="main", split="test")
    num_samples = min(args.num_samples, gsm8k.num_examples())
    print(f"Processing {num_samples} samples")

    # Generate and track states
    results = []
    with autocast_ctx:
        for idx in range(num_samples):
            print(f"\nProcessing example {idx + 1}/{num_samples}")

            # Get the conversation
            conversation = gsm8k.get_example(idx)
            question = conversation["messages"][0]["content"]

            # Render prompt for completion (removes the assistant's response)
            prompt_tokens = tokenizer.render_for_completion(conversation)

            print(f"Question: {question[:100]}...")
            print(f"Prompt length: {len(prompt_tokens)} tokens")

            # Generate with latent state tracking
            generated_tokens, latent_states = generate_with_latent_tracking(
                engine=engine,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                num_recur=num_recur,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=args.seed + idx,  # Different seed per example
            )

            # Decode response
            response = tokenizer.decode(generated_tokens)
            print(f"Generated {len(generated_tokens)} tokens")
            print(f"Captured {len(latent_states)} latent states (one per loop iteration)")
            print(f"Response: {response[:100]}...")

            # Store result
            result = GenerationResult(
                question=question,
                response=response,
                tokens=generated_tokens,
                latent_states=latent_states,
                num_recur=num_recur,
            )
            results.append(result)

    # Save results
    output_path = args.output
    print(f"\nSaving results to {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"Done! Saved {len(results)} results")

    # Print summary statistics
    print("\nSummary:")
    print(f"  Total examples: {len(results)}")
    print(f"  Avg tokens generated: {sum(len(r.tokens) for r in results) / len(results):.1f}")
    print(
        f"  Avg latent state shape: {results[0].latent_states[0].shape if results and results[0].latent_states else 'N/A'}"
    )


if __name__ == "__main__":
    main()
