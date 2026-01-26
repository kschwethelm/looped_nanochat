"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration

Looped Transformer specifics:
- Three-stage architecture: prelude (n_prelude layers) → recur (n_recur_block layers, run num_recur times) → coda (n_coda layers)
- KV cache persists across all recurrences, allowing each recurrence to attend to previous tokens
- Input injection at each recurrence: u = inject(concat(e, s)) where e=prelude output, s=recurrent state
  - inject layer initialized as identity-like [I|0] so inject(concat(e,s)) ≈ e initially
- Warm-start inference: final recurrent state s from token t-1 initializes state for token t (disabled for training)
- Truncated BPTT: gradients detached for all but last bptt_k recurrences to limit memory/compute
- Variable recurrence depth: num_recur sampled from distribution during training (mean=train_recur_mean, max=train_recur_max)
- Recurrent state flow: s ← e (init) → s ← recur(inject(e,s)) (loop num_recur times) → coda(s) → output
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.adamw import DistAdamW
from nanochat.common import get_dist_info, print0

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn
from nanochat.muon import DistMuon, Muon


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 65536
    n_layer: int = 20  # Here only used for scaling, not the actual number of layers
    n_head: int = 10  # number of query heads
    n_kv_head: int = 10  # number of key/value heads (GQA)
    n_embd: int = 1280
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"

    # Looped Transformer config options
    n_prelude: int = 2
    n_recur_block: int = 4
    n_coda: int = 2

    # Looped Transformer traning options
    train_recur_mean: float = 4.0  # mean recurrences during training (samples from distribution)
    train_recur_max: int = 16  # max recurrences sampled during training
    recur_warm_start: bool = True  # warm-start recurrence from previous token's final state
    bptt_k: int = (
        4  # truncate backprop to last k recurrences (you have this in code but not config)
    )


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Validate looped Transformer config
        assert config.n_layer >= (config.n_prelude + config.n_recur_block + config.n_coda), (
            "n_layer must be bigger or equal than n_prelude + n_recur_block + n_coda"
        )

        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "prelude": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_prelude)]
                ),
                "recur": nn.ModuleList(
                    [
                        Block(config, config.n_prelude + layer_idx)
                        for layer_idx in range(config.n_recur_block)
                    ]
                ),
                "coda": nn.ModuleList(
                    [
                        Block(config, config.n_prelude + config.n_recur_block + layer_idx)
                        for layer_idx in range(config.n_coda)
                    ]
                ),
            }
        )
        # Input injection adapter
        self.inject = nn.Linear(2 * config.n_embd, config.n_embd, bias=False)
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = (
            3**0.5 * n_embd**-0.5
        )  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in (
            list(self.transformer.prelude)
            + list(self.transformer.recur)
            + list(self.transformer.coda)
        ):
            torch.nn.init.uniform_(
                block.attn.c_q.weight, -s, s
            )  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Initialize inject layer as identity-like: output = e (first half of concat(e, s))
        # This ensures gradients flow on the first forward pass
        # Weight shape is (n_embd, 2*n_embd), we want [I | 0] so inject(concat(e,s)) ≈ e
        n_embd = self.config.n_embd
        with torch.no_grad():
            self.inject.weight.zero_()
            self.inject.weight[:, :n_embd].copy_(torch.eye(n_embd))

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config: GPTConfig):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_prelude + config.n_recur_block + config.n_coda):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self, num_recur=None):
        """
        Return the estimated FLOPs per token for the looped model (forward + backward).

        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)

        Args:
            num_recur: Number of recurrences to assume. If None, uses train_recur_mean.
        """
        if num_recur is None:
            num_recur = int(self.config.train_recur_mean)

        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )

        # 1. Count parameters by section
        prelude_params = sum(p.numel() for p in self.transformer.prelude.parameters())
        recur_params = sum(p.numel() for p in self.transformer.recur.parameters())
        coda_params = sum(p.numel() for p in self.transformer.coda.parameters())
        inject_params = sum(p.numel() for p in self.inject.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())

        # 2. Matmul FLOPs weighted by usage
        matmul_flops = 6 * (
            prelude_params  # prelude runs 1x
            + recur_params * num_recur  # recur runs num_recur times
            + coda_params  # coda runs 1x
            + inject_params * num_recur  # inject runs num_recur times
            + lm_head_params  # lm_head runs 1x
        )

        # 3. Attention FLOPs weighted by usage
        attn_flops = 0
        num_layers = self.config.n_prelude + self.config.n_recur_block + self.config.n_coda
        for layer_idx in range(num_layers):
            window = self.window_sizes[layer_idx][0]
            effective_seq = t if window < 0 else min(window, t)

            # Determine how many times this layer runs
            if layer_idx < self.config.n_prelude:
                multiplier = 1  # prelude
            elif layer_idx < self.config.n_prelude + self.config.n_recur_block:
                multiplier = num_recur  # recur
            else:
                multiplier = 1  # coda

            attn_flops += 12 * h * q * effective_seq * multiplier

        num_flops_per_token = matmul_flops + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        matrix_params = (
            list(self.transformer.prelude.parameters())
            + list(self.transformer.recur.parameters())
            + list(self.transformer.coda.parameters())
            + list(self.inject.parameters())
        )
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(
            lm_head_params
        )
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(
            betas=adam_betas, eps=1e-10, weight_decay=0.0
        )  # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        num_recur=None,
        warm_start_state=None,
    ):
        B, T = idx.size()
        if num_recur is None:
            num_recur = int(self.config.train_recur_mean)

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert self.cos.size(1) >= T, (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        # 1. Embedding + norm
        x = self.transformer.wte(idx)
        x = norm(x)

        # 2. Prelude blocks (run once)
        # For inference with KV cache, prelude uses cache_write=True
        for i, block in enumerate(self.transformer.prelude):
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
        e = x  # prelude output, used for injection into each recurrence

        # 3. Initialize recurrent state
        # If warm_start_state provided and config allows, use it; otherwise start from e
        if warm_start_state is not None and self.config.recur_warm_start:
            # warm_start_state may be (B, 1, h) from last token - broadcast to match e's shape (B, T, h)
            if warm_start_state.size(1) != T:
                s = warm_start_state.expand(-1, T, -1)
            else:
                s = warm_start_state
        else:
            s = e

        # 4. Recurrent block (run num_recur times)
        for i in range(num_recur):
            # Input injection: u = inject(concat(e, s))
            u = self.inject(torch.cat([e, s], dim=-1))
            # Run recur blocks with KV cache (all recurrences can attend to previous tokens)
            for j, block in enumerate(self.transformer.recur):
                u = block(u, cos_sin, self.window_sizes[self.config.n_prelude + j], kv_cache)
            # TODO: No normalization? u = norm(u)?
            s = u  # update recurrent state
            # Truncated BPTT: detach gradients for recurrences before the last bptt_k
            # This limits gradient flow depth to bptt_k * n_recur_block layers through recurrence
            if self.config.bptt_k is not None and i < num_recur - self.config.bptt_k:
                s = s.detach()

        # 5. Coda blocks (run once)
        x = s
        for i, block in enumerate(self.transformer.coda):
            x = block(
                x,
                cos_sin,
                self.window_sizes[self.config.n_prelude + self.config.n_recur_block + i],
                kv_cache,
            )
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(
            x
        )  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            return logits, s  # Return logits and final state

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42, num_recur=None):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        warm_start_state = None
        for _ in range(max_tokens):
            logits, warm_start_state = self.forward(
                ids, num_recur=num_recur, warm_start_state=warm_start_state
            )  # (B, T, vocab_size)
            # Only keep last position's state for warm-start (shape B,1,h)
            warm_start_state = warm_start_state[:, -1:, :]
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
