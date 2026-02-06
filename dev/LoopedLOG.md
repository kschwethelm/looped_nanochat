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

### Results

Given we did not make any training parameter changes, the scaling law results are surprisingly clean:

![scaling laws](scaling_laws_feb1.png)

**Initial observation:** Optimal configurations showed tokens/params ratio rising with scale (8.9 -> 11.4), initially suggesting parameter starvation. However, this didn't account for the recurrent block's parameter reuse.

| FLOPs | Eff Params | Tokens | Ratio | Val BPB |
|-------|------------|--------|-------|---------|
| 1e+18 | 116,180,734 | 1,009,724,988 | 8.9 | 0.9580 |
| 2e+18 | 159,155,139 | 1,482,003,055 | 9.3 | 0.9147 |
| 5e+18 | 214,639,515 | 2,263,974,151 | 10.7 | 0.8784 |
| 1e+19 | 295,598,854 | 3,351,493,312 | 11.4 | 0.8448 | 

**Accounting for parameter reuse:** Since the recurrent block's 4 layers execute 4× per forward pass while prelude/coda layers run once, the effective parameter count must weight by usage:

```
effective_params = 2.5 × params_transformer
```

(2 prelude layers × 1 use) + (2 coda layers × 1 use) + (4 recur layers × 4 uses) = 20 layer-uses / 8 total layers = 2.5× multiplier

**Corrected optimal configurations:**

| FLOPs | Eff Params | Tokens | Ratio | Val BPB |
|-------|------------|--------|-------|---------|
| 1e+18 | 171,176,667 | 1,006,769,967 | 6.1 | 0.9580 |
| 2e+18 | 245,888,809 | 1,484,904,920 | 6.0 | 0.9147 |
| 5e+18 | 348,014,355 | 2,268,826,257 | 6.6 | 0.8784 |
| 1e+19 | 506,477,134 | 3,339,016,342 | 6.7 | 0.8448 |


**Key finding:** When accounting for parameter reuse, the tokens/effective-params ratio stabilizes at ~6-7 across all scales (compared to Chinchilla's ~20 for standard transformers). This indicates the looped architecture efficiently utilizes parameters through recurrence.


