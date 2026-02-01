# Looped Transformer — Experiment Log

Running log of experiments on the looped (depth-recurrent) transformer, forked from [nanochat](https://github.com/karpathy/nanochat). Started ~Feb 1 2026.

**Goal:** Train a solid looped LLM baseline under limited compute. Stability and predictable training take priority over chasing small accuracy gains.

---

## 2026-02-01: Initial Scaling Laws

> **Note:** all scaling-law runs here use a fixed recursion depth for the FLOPs budget. In practice *r* is sampled from a Poisson log-normal with mean 4 during training, so the per-step FLOPs fluctuate around this value. Using the mean as a single point estimate is ok for now.

### FLOPs Estimation

**TLDR.** For recurrent block, multiply FLOPs by number of recursions

The standard "6N" rule (6 FLOPs per parameter per token, covering forward + backward) doesn't directly apply to a looped model because different sections of the network execute a different number of times per token. The estimation in `GPT.estimate_flops()` handles this by splitting into two components and weighting each by its execution count.

**Matmul FLOPs.** Parameters are counted per section and each section's contribution is scaled by how many times it runs:

| Section | Runs per token | Notes |
|---------|---------------|-------|
| prelude | 1× | |
| recur block | r× | the loop |
| inject | r× | re-injects prelude output at each recurrence |
| coda | 1× | |
| lm_head | 1× | |

The total is `6 × (prelude + r·recur + r·inject + coda + lm_head)`, where each term is a raw parameter count. The 6× factor is the standard multiply-accumulate in forward (2) plus 2× that in backward (4).

**Attention FLOPs.** Each layer again gets a multiplier of 1 or r depending on whether it sits in the prelude/coda or the recur block.

**What's excluded.** The token embedding is a table lookup, not a matmul, so it contributes zero FLOPs. The softmax exp/sum/divide is also omitted. These are the same two omissions relative to Chinchilla's formula, and together they account for roughly 1% error.

### Architectural changes

Since we are starting from scratch, we introduce some architectural changes:
- Sandwich norm + norm at end of recurrent block
- Trainable RMSNorm

### Experiment Setup

In standard nanochat, `depth` elegantly controls both width AND layers together. In a looped transformer, recursion count decouples effective depth from physical layer count. This means the layer structure (prelude, recur, coda) and num_recur are architectural choices, not scaling knobs — changing them between runs means comparing different architectures, not the same architecture at different sizes. Power laws won't fit cleanly through that.

The clean solution: fix the architecture at (2 prelude, 4 recur, 2 coda, r=4), giving a constant effective depth of 2 + 4×4 + 2 = 20 layers across the entire sweep. Width becomes the single scaling dial — both parameter count and FLOPs scale smoothly as ~O(width²), and every model in the sweep is structurally identical. Recursion sampling is disabled (--no-sample-recur) for clean FLOPs measurements.

| SIZE | WIDTH | HEADS | EFF. LAYERS |   PARAMS    | FLOPS / TOK  |
|------|-------|-------|-------------|-------------|--------------|
| 8    | 512   | 4     | 20          | 92,816,896  | 7.678034e+08 |
| 10   | 640   | 5     | 20          | 124,049,280 | 1.081651e+09 |
| 12   | 768   | 6     | 20          | 158,492,928 | 1.444258e+09 |
| 14   | 896   | 7     | 20          | 196,147,840 | 1.855623e+09 |
| 16   | 1024  | 8     | 20          | 237,014,016 | 2.315747e+09 |
| 18   | 1152  | 9     | 20          | 281,091,456 | 2.824630e+09 |
| 20   | 1280  | 10    | 20          | 328,380,160 | 3.382272e+09 |
