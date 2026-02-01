# Looped Transformer — Experiment Log

Running log of experiments on the looped (depth-recurrent) transformer, forked from [nanochat](https://github.com/karpathy/nanochat). Started ~Feb 1 2026.

**Goal:** Train a solid looped LLM baseline under limited compute. Stability and predictable training take priority over chasing small accuracy gains.

---

## Architecture

Follows Huginn architecture (Geiping et al., 2025). Some experiments were already done by [Trelis Research](https://github.com/TrelisResearch/nanochat/tree/recursive) but I would like to redo them.

Three-stage looped transformer: **prelude** (2 layers) → **recur** (4 layers, iterated *r* times) → **coda** (2 layers).

At each recurrence the prelude output is re-injected into the recurrent state via a learned `inject` layer (identity-initialized). The recurrence count *r* is sampled from a Poisson log-normal distribution (mean=4, max=16) during training to encourage generalization across iteration depths. Gradients are truncated to the last `bptt_k=4` recurrences.

Depth is the single scaling dial: `model_dim = depth × 64`, nudged up to the nearest multiple of `head_dim=128`. At the defaults (d20, r=4) the model has 8 unique layers but 20 effective layers (2 + 4×4 + 2).

### Changes compared to Trelis
- Sandwich norm (see Huginn)
- Sliding window attention in recursive block (planned)
- Value embeddings in recursive block (from current nanochat architecture; planned)
- Start with random normal instead of duplicate prelude output in input injection (see Huginn; planned)

## Training Pipeline

| Stage | Script | Description |
|-------|--------|-------------|
| 1 | `base_train` | Next-token prediction on web text. Training budget set by param-data ratio (default 4×). CORE evaluated periodically. |
| 2 | `chat_sft` | SFT on ~856K-row mixture: SmolTalk 460K, MMLU 100K, GSM8K 16K (×2ep), identity 2K (×2ep), SimpleSpelling 200K, SpellingBee 80K. Single epoch, bestfit-pad packing. |
| 3 | `chat_rl` | Simplified GRPO on GSM8K. On-policy, no KL/trust region. Advantage = reward − mean(reward), DAPO-style token-level normalization. |

## Primary Metric: CORE

22-task composite benchmark from the DCLM paper, spanning world knowledge, language understanding, commonsense reasoning, symbolic problem solving, and reading comprehension. Raw task accuracies are centered against random-guessing baselines and averaged into a single score.

**Target: beat nanochat at CORE = 0.2565.**

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

### Experiment Setup

In standard nanochat, `depth` elegantly controls both width AND layers together. For looped transformers, we've lost that simplicity—we have 4 knobs instead of 1.
The pragmatic solution: define discrete "model size" configs that span a reasonable FLOPs range, each with a sensible (prelude, recur, coda, num_recur, width) combination. We also disable recursion sampling for clean measurements.

| DEPTHS | N_PRELUDES | N_RECUR_BLOCKS | N_CODAS | N_RECURS |
|--------|------------|----------------|---------|----------|
| 6      | 1          | 2              | 1       | 2        |
| 8      | 1          | 2              | 1       | 3        |
| 10     | 1          | 2              | 1       | 4        |
| 12     | 2          | 2              | 2       | 4        |
| 14     | 1          | 3              | 1       | 4        |
| 16     | 2          | 3              | 2       | 4        |
