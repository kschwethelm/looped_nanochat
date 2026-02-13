# Looped Transformer — Experiment Log

Running log of experiments on the looped (depth-recurrent) transformer, forked from [nanochat](https://github.com/karpathy/nanochat). Started ~Feb 1 2026.

**Goal:** Exploratory work on understanding looped LLM behavior from the ground up. Small model sizes, limited compute budget, single runs for quick iteration. The end goal is a model with true dynamic compute and meaningful depth scaling at reasonable cost.

---

## 2026-02-12: Recursion Depth Sweep at Scale — Some Positive Signal

Repeated the 2026-02-11 depth sweep at S20 (1280 width, 328M params, 1.35e19 FLOPs) — a scale where models produce coherent text. Same setup: num_recur ∈ {1, 2, 4, 6}, fixed recurrence, default hyperparameters, 2× A100-SXM4-80GB. r=1 is the non-looped baseline (same architecture, recurrent block executes once).

### Base Model

| Model | r | FLOPs/tok | Tokens | Val BPB | Steps | Time |
|-------|---|-----------|--------|---------|-------|------|
| r1_1.35e19_s20 | 1 | 1.67e9 | 8.08B | 0.8330 | 15406 | 9:56h |
| r2_1.35e19_s20 | 2 | 2.24e9 | 6.02B | **0.8266** | 11486 | 9:43h |
| r4_1.35e19_s20 | 4 | 3.38e9 | 3.99B | 0.8333 | 7613 | 9:56h |
| r6_1.35e19_s20 | 6 | 4.52e9 | 2.98B | 0.8437 | 5693 | 8:18h |

r2 beats the non-looped r1 baseline despite seeing 2B fewer tokens. Beyond r2, val BPB degrades with r as before. Note that only r4 is at the compute-optimal token/param ratio for 1.35e19 FLOPs (from the scaling laws); other models are over- or under-trained, so base BPB differences partly reflect training efficiency rather than architectural capacity.

### SFT + ChatCORE

All base models SFT'd for 1 epoch (816 steps) with fixed recurrence at their respective r. Categorical tasks (ARC, MMLU) are identical under full vs minimal KV cache. Generative tasks differ — shown as kv=r / kv=1 below.

| Model | SFT Val BPB | ARC-E | ARC-C | MMLU | SpellingBee (r/1) | GSM8K (r/1) | HumanEval (r/1) | ChatCORE (r/1) |
|-------|-------------|-------|-------|------|-------------------|-------------|-----------------|----------------|
| r1 | 0.4137 | 0.416 | 0.300 | 0.324 | 0.965 | 0.017 | 0.018 | 0.2311 |
| r2 | 0.4027 | **0.453** | **0.361** | 0.337 | 0.957 / 0.922 | 0.021 / 0.024 | 0.006 / 0.000 | 0.2531 / 0.2468 |
| r4 | **0.4017** | 0.468 | 0.355 | **0.338** | **0.973** / 0.961 | 0.031 / **0.041** | 0.055 / **0.079** | **0.2677** / **0.2715** |
| r6 | 0.4078 | 0.444 | 0.325 | 0.338 | 0.973 / **0.969** | 0.031 / 0.036 | **0.061** / 0.049 | 0.2567 / 0.2549 |

SFT compresses the looped models within ~0.006 BPB, but r1 lags noticeably (0.4137 vs 0.4017–0.4078). ChatCORE spreads more than at S12 (0.2311–0.2677 vs 0.2248–0.2289), with r1 clearly weakest and r4 ahead overall. Notable trend: generative benchmarks now favor higher r. HumanEval climbs with depth (0.018 → 0.006 → 0.055 → 0.061), GSM8K jumps from r1/r2 to r4/r6. Categorical tasks don't show this. Absolute scores are still low, but this is a possible early signal that depth helps multi-step generation more than pattern matching.

Minimal KV cache barely changes performance — r4 even improves marginally (0.2677 → 0.2715), surprising given models were trained with full cache.

### Latent State Dynamics

Visualized latent convergence (L2 distance, cosine similarity, KL divergence between consecutive loop steps) on GSM8K & SpellingBee prompts at r=16 under full cache (kv=16) and minimal cache (kv=1).

**Convergence speed = training r.** Each model reaches its fixed point in approximately the number of loops it was trained with — r2 in ~2, r4 in ~4, r6 in ~6 — then near-identity. Same spread-computation pattern as S12.

**kv=1 smooths convergence.** With full cache, r4 and r6 show non-monotonic spikes during convergence. With kv=1, both converge faster and more smoothly. kv=1 only caches the last loop's KV entries from the previous token (the converged state), so attention sees a consistent "preview" of the fixed point across all loops, guiding early loops toward convergence faster. Full cache exposes unconverged intermediate states that can disrupt the trajectory.

**Fixed-step training causes convergence spikes.** The non-monotonic spikes in r4/r6 are absent in models trained with recurrence sampling (2026-02-09 variants). Fixed-step training learns a specific iteration trajectory that can overshoot before settling; sampled training learns a smoother monotonic path. Another argument for recurrence sampling — not for final performance (equivalent, see 2026-02-09), but for cleaner convergence dynamics, especially relevant for early-exit gating.

### Interpretation

The spread-computation pattern from S12 persists: more recursions don't clearly improve the fixed point, and r4 wins ChatCORE despite r6 having 50% more effective depth. But the picture is less one-sided than at S12. The base BPB gap narrowed, and generative benchmarks now favor higher r while categorical tasks remain flat. This could be early evidence that depth becomes more useful once models have enough capacity for multi-step generation, even if the signal is weak.

**Note: gradient truncation.** All models use bptt_k=4, which covers all iterations for r=2 and r=4 but truncates gradients for the first 2 iterations of r=6. r=6's weaker base BPB may partly reflect this rather than a fundamental depth limitation.

### Next Steps
- Train S20 r=4 & r=6 with sampled recursions
- Evaluation on synthetic arithmetic tasks
- Training-free early exiting

---

## 2026-02-11: Naive Recursion Depth Sweep — Diminishing Returns

Swept num_recur ∈ {2, 4, 6} at fixed IsoFLOPs (2.15e18) on S12 (768 width, 158M physical params). Architecture identical across runs (2 prelude, 4 recur block layers, 2 coda) — only the loop count changes. Fixed recurrence (no sampling), default hyperparameters.

Under a fixed FLOPs budget, more recursions means more FLOPs per token and proportionally fewer tokens seen.

### Base Model

| Model | r | FLOPs/tok | Tokens | Val BPB | Steps | Time |
|-------|---|-----------|--------|---------|-------|------|
| r2_2.15e18_s12 | 2 | 9.96e8 | 2.16B | **0.9060** | 4118 | 1:52h |
| r4_2.15e18_s12 | 4 | 1.44e9 | 1.49B | 0.9144 | 2839 | 1:50h |
| r6_2.15e18_s12 | 6 | 1.89e9 | 1.14B | 0.9281 | 2167 | 1:32h |

Val BPB degrades monotonically with r. The extra depth doesn't compensate for seeing fewer tokens — r2 wins by seeing nearly 2× the data of r6.

### SFT + ChatCORE

All base models SFT'd for 1 epoch (816 steps) with fixed recurrence at their respective r, same hyperparameters.

| Model | SFT Val BPB | ARC-E | ARC-C | MMLU | SpellingBee | GSM8K | HumanEval | ChatCORE |
|-------|-------------|-------|-------|------|-------------|-------|-----------|----------|
| r2 | 0.4561 | 0.366 | 0.282 | 0.321 | 0.969 | 0.007 | 0.079 | 0.2248 |
| r4 | **0.4555** | 0.367 | 0.290 | 0.313 | **0.992** | 0.008 | 0.079 | **0.2289** |
| r6 | 0.4607 | **0.383** | **0.304** | 0.300 | 0.961 | 0.009 | 0.079 | 0.2274 |

SFT closes the base model gap — all three land within ~0.005 BPB and ~0.004 ChatCORE. The base training differences wash out. r6 edges out on ARC (slightly harder reasoning) but the margins are within noise at this scale.

### Interpretation

Latent state visualization shows models trained with more recursions converge slower to a fixed point — they spread the same computation across more iterations. This matters because evaluating a single model (e.g. r6) at r=2 vs r=6 looks like test-time scaling works. But cross-model comparison reveals the illusion: r2 at r=2 matches or beats r6 at r=6. The model just needs more steps to reach the same fixed point, not a better one.

Consistent with 2026-02-09: recurrence sampling shifts the convergence point to the distribution's mean but doesn't raise the ceiling. At ~158M params, the recurrent block learns a near-identity mapping after convergence. Whether this changes with scale is open.

### Limitations

- **No coherent generation.** S12 models can't produce meaningful text. Generative benchmarks (HumanEval, GSM8K) score ~0 and are excluded from CORE.
- **Shallow evals.** >95% SpellingBee likely reflects pattern matching rather than depth-dependent reasoning — too shallow to probe recursion scaling. ARC/MMLU differences are within noise.
- **Single runs.** No error bars. Differences below ~0.01 BPB or ~0.005 ChatCORE should be treated as noise.

### Next steps

- **Scale up.** S18+ models trained at >1e19 FLOPs produce coherent text (e.g. correctly complete "The capital of France is"). Base training on 2× 94GB H100 takes ~4h. Revisit depth scaling there.
- **Depth-sensitive evals.** Design tasks tractable at small scale that reward iterative computation: multi-digit arithmetic, parity checks, string reversal/counting, simple logical deduction, synthetic multi-hop lookup in context. These isolate depth benefit from knowledge capacity.
- **Token-level gating.** Explore learned per-token loop exit to compress unnecessary iterations — the missing training signal linking extra compute to harder tokens.

### Speculation

Geiping's Huginn trains with many recursions but with little pressure to compress capability across loops. Self-distillation from a high-r teacher to a low-r student could force the model to be more loop-efficient, rather than just spreading the same computation across more steps. This also suggests that proposed "specialized" post-training on lower recursion counts may primarily compress loops rather than improve depth scaling itself. Worth investigating, but caution warranted until the compression-vs-scaling distinction is better understood.

---

## 2026-02-09: Recurrence Sampling & Random State Init — Negative Result (6c5710a)

Tested whether training with (a) sampled recurrence steps and (b) random initial recurrent state enables adaptive test-time compute. S12 model (768 width, ~158M params, r=4 mean), 2.15e18 token budget. SFT with recurrence sampling applied to all variants regardless of base training.

**Motivation.** Two training-time interventions aimed at enabling adaptive compute at inference:

1. **Recurrence step sampling** (`sample`): Sample num_recur from a Poisson lognormal distribution (1–16, mean 4) instead of fixed r=4. Goal: teach the model to benefit from variable compute budgets, enabling "think more on harder tokens." -> Source: Huginn (Geiping et al., 2025)
2. **Random initial state** (`init_random`): Initialize the recurrent latent state with noise instead of the prelude's output. Goal: force convergence from diverse starting points, making the state representation more robust. -> Source: Huginn (Geiping et al., 2025)

Training cost: Step sampling requires `torch.compile(dynamic=True)`, increasing base training time by ~30% (2:56h vs 2:14h on 2x A100 80GB).

**Four variants tested:** `r4` (baseline, fixed steps, prelude init), `r4_init_random`, `r4_sample`, `r4_sample_init_random`.

### Base Model — Validation BPB vs Recurrence Depth

| Model | r=2 | r=4 | r=8 | r=16 | r=32 |
|-------|-----|-----|-----|------|------|
| r4 (baseline) | 0.9943 | **0.9144** | 0.9215 | 0.9247 | 0.9247 |
| r4_init_random | 1.0197 | 0.9258 | 0.9263 | 0.9263 | 0.9264 |
| r4_sample | 0.9263 | 0.9174 | 0.9177 | 0.9177 | 0.9177 |
| r4_sample_init_random | 0.9277 | 0.9178 | 0.9184 | 0.9184 | 0.9184 |

Key observations from base eval:

- **No model benefits from more recurrences beyond r=4.** All variants plateau. Notably, the baseline `r4` trained at fixed r=4 shows only mild degradation at much higher depths (0.9144 → 0.9247 at r=32) — more stable than expected when evaluated far outside its training distribution. Sampling variants stay flat but don't improve either.
- **Sampling variants are more robust at low r.** At r=2, `r4_sample` (0.9263) is dramatically better than `r4` (0.9943). The model learned to produce useful output even with fewer iterations. However, this robustness comes at the cost of slightly worse performance at the training mean (0.9174 vs 0.9144).
- **Random init is redundant or slightly harmful.** `r4_init_random` is the weakest variant, and combining it with sampling (`r4_sample_init_random` ≈ `r4_sample`) shows no added benefit. The differences are small, but random init never helps — at best it's neutral when sampling already provides robustness.

### SFT Model — ChatCORE vs Recurrence Depth

All base models were SFT'd with recurrence sampling enabled (including variants not trained with it). Evaluated with KV budget=1, warm start.

| Model | r=2 | r=4 | r=8 | r=16 | r=32 |
|-------|-----|-----|-----|------|------|
| r4 (baseline) | 0.2094 | 0.2187 | 0.2218 | 0.2197 | 0.2197 |
| r4_init_random | 0.2054 | 0.2186 | 0.2208 | 0.2214 | 0.2208 |
| r4_sample | 0.2010 | 0.2196 | 0.2200 | 0.2205 | 0.2199 |
| r4_sample_init_random | 0.2054 | 0.2186 | 0.2208 | 0.2214 | 0.2208 |

Key observations after SFT:

- **SFT largely equalizes all variants.** All variants converge to similar ChatCORE across all recurrence depths. The base training differences wash out. This means SFT alone is sufficient — the 30% training overhead from `dynamic=True` in base training buys nothing in the final chat model.
- **Flat curves beyond r=4 everywhere.** The central finding holds post-SFT: extra recurrences don't improve downstream performance.

### Interpretation

The recurrent block seems to converge to a near-fixed point after ~4 iterations — additional iterations are close to identity. One possible explanation is that nothing in training incentivizes using extra steps: the loss treats all step counts equally, so a model that converges at r=4 and idles beyond that is never penalized. Sampling exposes the model to variable depths but doesn't teach it *when* to use them.

Random init doesn't help either. The initial recurrent state is formed by concatenating a second signal with the prelude output and projecting through a linear adapter. Without random init, the prelude output is simply duplicated for this — replacing the duplicate with noise adds no useful diversity.

**Caveat:** These are small models (~158M params) with a low loop count (r=4). Whether this generalizes to more powerful models is an open question. The s20 model (~328M params, trained longer without recurrence sampling) shows the same flat behavior at test time, which is at least consistent. Note that r=4 is also the default in other looped transformer work (Ouro, Mixture-of-Recursions), so we're not obviously under-looping — but the interaction between model capacity, loop count, and adaptive compute could look very different at larger scale.

### Decisions (for now)

- **Skip recurrence sampling in base training.** The 30% compute overhead from `dynamic=True` yields no benefit after SFT. SFT with sampling is enough for robustness.
- **Skip random initial state.** Redundant at best, slightly harmful at worst.
- **Fixed r=4 base training remains the default.** Simple, fast, and SFT recovers any robustness gap.

### Next steps

The models are robust (don't diverge with more recurrences) but flat (don't improve). Ideas to explore for actual test-time compute scaling:

1. **Token-level gating.** A learned gate that decides per-token whether to iterate more — the missing training signal linking extra compute to harder tokens.
2. **Introduce gating only in SFT.** Since SFT recovers base train differences, gating might not need to be in base training at all.
3. **Loss weighting by depth.** Weight the loss to reward predictions that improve with more recurrences.

---

## 2026-02-08: Hyperparameter Tuning (d79bc25)

Swept hyperparameters at S12 (768 width, ~158M params, ratio=4, fixed r=4) to check whether the looped architecture needs different settings from the non-looped defaults.

**Stage 1: Muon LR**
| matrix_lr | val_bpb |
|-----------|---------|
| 0.005     | 0.9163  |
| **0.01**  | **0.9108** |
| 0.02      | 0.9143  |
| 0.04      | 0.9751  |
| 0.08      | 2.261 (diverged) |

Small win for 0.01 over default 0.02 (~0.0035 bpb). Consistent with gradient accumulation across `bptt_k=4` recursions effectively doubling gradient magnitude, pushing optimal LR down ~2×. However, the gain is marginal — not enough to justify retraining existing models or deviating from the Muon standard 0.02.

**Stage 2: Weight Decay**
| weight_decay | val_bpb |
|-------------|---------|
| 0.05        | 0.9152  |
| 0.1         | 0.9134  |
| 0.2         | 0.9108  |
| **0.4**     | **0.9093** |
| 0.8         | 0.9273  |

0.4 marginally best but only 0.0015 over default 0.2. Not worth changing — 0.2 has an established `WD ∝ 1/width²` scaling law across model sizes.

**Stage 2b: Separate Weight Decay for Recurrent Block**

Recurrent block has ~2× the L1 param/gradient norms of prelude/coda (expected from gradient accumulation across recursions). Tested whether reducing WD on the recurrent block (via `recur_wd_scale=0.5`) helps.

| config | val_bpb |
|--------|---------|
| WD=0.2, recur_scale=0.5 | 0.9130 |
| WD=0.4, recur_scale=0.5 | 0.9111 |
| WD=0.2, uniform (baseline) | 0.9108 |

No benefit from differential weight decay. Uniform WD is better in both cases. The recurrent block doesn't need special treatment despite its larger norms.

**Stage 3: Warmdown Ratio**
| warmdown_ratio | val_bpb |
|----------------|---------|
| 0.2            | 0.9146  |
| 0.3            | 0.9127  |
| **0.4**        | **0.9108** |
| 0.5            | 0.9113  |
| 0.6            | 0.9115  |

All within noise. Default 0.4 is fine.

**Decision:** Keep all defaults as-is (matrix_lr=0.02, weight_decay=0.2, warmdown=0.4). The looped architecture is not sensitive to hyperparameters beyond what's already tuned for non-looped. The only notable signal was matrix_lr=0.01, which we keep in mind for future reference but don't adopt — existing models and scaling law results remain valid at 0.02.

**Implication for scaling law experiments:** The IsoFLOP sweeps run at default hyperparameters are not confounded by HP mismatch. The looped architecture's different scaling behavior (10:1 vs Chinchilla's 20:1) is a genuine architectural effect, not a tuning artifact.

**Future work:** Try Muon LR = 0.01 on s20 model, tune embedding and unembedding lr (expected little impact).

## 2026-02-01: Initial Scaling Laws (5ea018d)

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

**Initial observation (physical params):** The optimal tokens/params ratio sits around 9 to 11, already revealing the looped architecture is much more data-efficient than standard transformers (Chinchilla's ~20). However, the ratio rises with scale (8.9 → 11.4), making it harder to use for compute-optimal planning.

| FLOPs | Params | Tokens | Ratio | Val BPB |
|-------|--------|--------|-------|---------|
| 1e+18 | 116,180,734 | 1,009,724,988 | 8.9 | 0.9580 |
| 2e+18 | 159,155,139 | 1,482,003,055 | 9.3 | 0.9147 |
| 5e+18 | 214,639,515 | 2,263,974,151 | 10.7 | 0.8784 |
| 1e+19 | 295,598,854 | 3,351,493,312 | 11.4 | 0.8448 | 

**Refitting with usage-weighted params:** Since the recurrent block's 4 layers execute 4× per forward pass while prelude/coda layers run once, we can define an effective parameter count that weights by usage:
```
effective_params = once_params + num_recur × reused_params
```

(2 prelude layers × 1 use) + (2 coda layers × 1 use) + (4 recur layers × 4 uses) = 20 layer-uses / 8 total layers = 2.5× multiplier on average.

Refitting the scaling law with this definition as the independent variable gives a much more stable optimal ratio:

| FLOPs | Eff Params | Tokens | Ratio | Val BPB |
|-------|------------|--------|-------|---------|
| 1e+18 | 172,302,833 | 1,006,703,553 | 6.0 | 0.9580 |
| 2e+18 | 247,645,826 | 1,485,003,445 | 6.0 | 0.9147 |
| 5e+18 | 350,724,715 | 2,268,943,864 | 6.6 | 0.8784 |
| 1e+19 | 510,828,415 | 3,338,573,361 | 6.6 | 0.8448 |  

**Key finding:** Defining N as usage-weighted parameters yields a stable optimal tokens/params ratio of ~6–7 across scales, compared to the drifting 8.9–11.4 under physical parameter count. We use this definition for compute-optimal planning going forward. Note that this ratio isn't directly comparable to Chinchilla's ~20, since tied weights aren't equivalent to independent weights—but the physical-params ratio of ~10 vs ~20 already shows a clear data efficiency gain.

**Future work:** It would be interesting to fit curves with larger FLOPs budgets to see if the physical-params ratio converges (maybe to Chinchillas ~20) or continues rising, though this would require significant compute.