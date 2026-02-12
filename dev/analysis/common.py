"""Shared utilities for latent state analysis scripts."""

import torch

from nanochat.engine import Engine


class LatentStateHook:
    """Hook to capture latent states during recurrent block execution."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset state tracking for a new forward pass."""
        self.states = []
        self.is_prefill = True
        self.prefill_states = []

    def __call__(self, module, input, output):  # noqa: ARG002
        """Hook called after each recurrent block execution."""
        # output is the activations after the last recurrent block: shape (B, T, hidden_dim)
        if self.is_prefill:
            # During prefill, capture all token positions: (B, T, hidden_dim)
            self.prefill_states.append(output.detach().cpu().clone())
        else:
            # During decode, capture only the final position
            final_token_state = output[:, -1, :].detach().cpu().clone()
            self.states.append(final_token_state)


def generate_with_latent_tracking(
    engine: Engine,
    tokenizer,
    prompt_tokens: list[int],
    num_recur: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: int = 42,
    kv_budget: int = 1,
    use_warm_start: bool = False,
    return_intermediate_logits: bool = False,
) -> tuple[list[int], list[list[torch.Tensor]], list[list[torch.Tensor]], list[torch.Tensor] | None]:
    """
    Generate response while tracking latent states for ALL tokens (input + output).

    Delegates generation to engine.generate(), using a hook on recur[-1] to capture
    latent states. When return_intermediate_logits=True, engine.generate() also captures
    intermediate logits (per recurrence step) on the engine instance.

    The generator protocol enables correct hook state transitions: prefill runs before
    the first yield (is_prefill=True), and we flip to is_prefill=False before the
    generator resumes with the first decode forward.

    Args:
        return_intermediate_logits: When True, also returns logits at each
            recurrence step for every token (input + output). The hook on recur[-1]
            is unaffected because _predict runs coda + lm_head (not recur blocks).

    Returns:
        generated_tokens: list of generated token ids
        input_latent_states: list of lists, each containing num_recur tensors for input tokens
        output_latent_states: list of lists, each containing num_recur tensors for output tokens
        intermediate_logits: list of num_recur tensors, each (total_tokens, vocab_size) on CPU,
            or None if return_intermediate_logits is False
    """
    hook = LatentStateHook()

    # Register hook on the last recurrent block
    last_recur_block = engine.model.transformer.recur[-1]
    handle = last_recur_block.register_forward_hook(hook)

    try:
        num_input_tokens = len(prompt_tokens)
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        bos = tokenizer.get_bos_token_id()

        hook.is_prefill = True
        generated_tokens = []
        output_latent_states = []
        states_before_token = 0

        gen = engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            num_recur=num_recur,
            use_warm_start=use_warm_start,
            kv_budget=kv_budget,
            return_intermediate_logits=return_intermediate_logits,
        )

        for is_first, (token_column, _token_masks) in _enumerate_first(gen):
            if is_first:
                # Prefill forward has completed (runs before first yield).
                # Switch hook to decode mode before generator resumes with decode forward.
                hook.is_prefill = False

            # Capture latent states from the previous decode forward.
            # On the first yield no decode forward has run yet, so skip.
            if not is_first:
                states_after_token = len(hook.states)
                new_states = hook.states[states_before_token:states_after_token]
                if len(new_states) >= num_recur:
                    output_latent_states.append(new_states[-num_recur:])
                states_before_token = states_after_token

            token = token_column[0]  # num_samples=1
            if token in (assistant_end, bos):
                break
            generated_tokens.append(token)

        # After the for loop, capture states from the last decode forward.
        # In the max_tokens case the generator ran the last decode forward before
        # its while-loop break (StopIteration). In the assistant_end case we broke
        # before the generator resumed, so no new states exist here.
        states_after_token = len(hook.states)
        new_states = hook.states[states_before_token:states_after_token]
        if len(new_states) >= num_recur:
            output_latent_states.append(new_states[-num_recur:])

        # Extract per-token latent states from prefill
        input_latent_states = []
        if len(hook.prefill_states) >= num_recur:
            prefill_recur_states = hook.prefill_states[-num_recur:]
            for token_idx in range(num_input_tokens):
                token_states = [state[:, token_idx, :] for state in prefill_recur_states]
                input_latent_states.append(token_states)

        # Assemble intermediate logits from engine
        if return_intermediate_logits:
            prefill_intermediate = engine._prefill_intermediate
            # Build per-step chunks starting with prefill
            per_step_chunks: list[list[torch.Tensor]] = [
                [il[0].cpu()] for il in prefill_intermediate  # (T_input, vocab_size) each
            ]
            # Append decode chunks
            for decode_inter in engine._decode_intermediates:
                for step_idx, il in enumerate(decode_inter):
                    per_step_chunks[step_idx].append(il[0].cpu())  # (1, vocab_size)
            intermediate_logits = [torch.cat(chunks, dim=0) for chunks in per_step_chunks]
        else:
            intermediate_logits = None

    finally:
        handle.remove()

    return generated_tokens, input_latent_states, output_latent_states, intermediate_logits


def _enumerate_first(iterable):
    """Yield (is_first, item) pairs, where is_first is True only for the first item."""
    first = True
    for item in iterable:
        yield first, item
        first = False
