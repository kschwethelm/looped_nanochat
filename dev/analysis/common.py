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
    prompt_tokens,
    num_recur,
    max_tokens=512,
    temperature=0.7,
    top_k=50,
    seed=42,
    kv_budget=1,
    use_warm_start=False,
) -> tuple[list[int], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """
    Generate response while tracking latent states for ALL tokens (input + output).

    Returns:
        generated_tokens: list of generated token ids
        input_latent_states: list of lists, each containing num_recur tensors for input tokens
        output_latent_states: list of lists, each containing num_recur tensors for output tokens
    """
    hook = LatentStateHook()

    # Register hook on the last recurrent block
    last_recur_block = engine.model.transformer.recur[-1]
    handle = last_recur_block.register_forward_hook(hook)

    try:
        device = engine.model.get_device()
        num_input_tokens = len(prompt_tokens)

        # Step 1: Run prefill to capture input token states
        m = engine.model.config
        cache_num_recur = num_recur if num_recur is not None else int(m.train_recur_mean)
        effective_num_layers = m.n_prelude + (m.n_recur_block * kv_budget) + m.n_coda
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        from nanochat.engine import KVCache

        kv_model_kwargs = {
            "num_heads": m.n_kv_head,
            "head_dim": m.n_embd // m.n_head,
            "num_layers": effective_num_layers,
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(prompt_tokens),
            device=device,
            dtype=dtype,
            num_recur=cache_num_recur,
            kv_budget=kv_budget,
            **kv_model_kwargs,
        )

        # Run prefill with hook in prefill mode
        hook.is_prefill = True
        ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        logits, warm_start_state = engine.model.forward(ids, kv_cache=kv_cache_prefill, num_recur=num_recur)

        # Extract per-token states from prefill
        input_latent_states = []
        if len(hook.prefill_states) >= num_recur:
            # Take the last num_recur states (in case of extra calls)
            prefill_recur_states = hook.prefill_states[-num_recur:]
            # For each token position, collect its states across all recurrences
            for token_idx in range(num_input_tokens):
                token_states = [state[:, token_idx, :] for state in prefill_recur_states]
                input_latent_states.append(token_states)

        # Step 2: Switch to decode mode and run generation
        hook.is_prefill = False
        hook.states = []  # Reset decode states

        generated_tokens = []
        output_latent_states = []
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        bos = tokenizer.get_bos_token_id()

        states_before_token = 0

        for token_column, _token_masks in engine.generate(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            num_recur=num_recur,
            kv_budget=kv_budget,
            use_warm_start=use_warm_start,
        ):
            token = token_column[0]  # batch size is 1

            # Stop if we hit a terminal token
            if token in (assistant_end, bos):
                break

            generated_tokens.append(token)

            # Capture latent states for this token
            states_after_token = len(hook.states)
            new_states = hook.states[states_before_token:states_after_token]

            # Store all num_recur states for this token
            if len(new_states) >= num_recur:
                output_latent_states.append(new_states[-num_recur:])

            states_before_token = states_after_token

    finally:
        handle.remove()

    return generated_tokens, input_latent_states, output_latent_states
