# Entropy-regularized exit gate (Ouro) on recurrent-depth sampling (Geiping)

This log documents the implementation bridge between:

- Geiping et al. recurrent-depth training (Poisson log-normal depth sampling + TBPTT) summarized in `../knowledge/summary_recurrent_depth_2502_05171.md`.
- Ouro / LoopLM learned depth allocation with entropy-regularized objective + Stage II gate tuning summarized in `../knowledge/summary_ouro_entropy_2510_25741.md`.

The goal is to enable token-adaptive loop depth while preserving this repo's training substrate.

## 1) Paper bridge: variable Tmax via sampled num_recur

Geiping training samples a single recurrence count `num_recur` per micro-batch and applies TBPTT. Ouro assumes you can compute losses at *all* steps t=1..Tmax to form an expected loss under a learned exit distribution. We bridge this by:

- Treating the sampled `num_recur` as the *per-batch* Tmax for the Ouro objective.
- Computing per-step losses for t=1..num_recur within the recurrence loop.
- Computing the exit distribution and entropy regularizer using those per-step losses.
- Keeping TBPTT as-is, but detaching after each step's own loss + gate computation.

This is compatible with heavy-tailed recurrence sampling and maintains the gate exploration behavior that the entropy regularizer is meant to enforce.

## 2) Equations used

### 2.1 Exit distribution from instantaneous exit probabilities

At each recurrence step t, the gate predicts a per-token exit probability:

```
lambda_t(x) = sigmoid(W h_t(x))
```

Survival probability:

```
S_t(x) = prod_{j=1}^t (1 - lambda_j(x)), S_0 = 1
```

Exit-step distribution (forced exit at Tmax):

```
p_exit(t) = lambda_t * S_{t-1} for t < Tmax
p_exit(Tmax) = S_{Tmax-1}
```

### 2.2 Stage I entropy-regularized expected loss (Ouro)

```
L = sum_{t=1..Tmax} p_exit(t) * L^{(t)} - beta * H(p_exit)
H(p_exit) = -sum_t p_exit(t) * log(p_exit(t))
```

This is equivalent to a KL-to-uniform prior (ELBO view) up to a constant.

### 2.3 Stage II focused gate loss (Ouro)

Compute per-token improvement:

```
I_i^{(t)} = max(0, L_{i,stop}^{(t-1)} - L_{i,stop}^{(t)})
```

Convert to soft "should continue" label:

```
w_i^{(t)} = sigmoid(k * (I_i^{(t)} - gamma))
```

BCE on continuation probability:

```
L_adapt^{(t)} = BCE(1 - lambda_i^{(t)}, w_i^{(t)})
```

Average over t = 2..Tmax and valid tokens.

### 2.4 Recurrence sampling + TBPTT (Geiping)

Recurrence count:

```
tau ~ N(log(r_bar) - 0.5*sigma^2, sigma)
r ~ Poisson(exp(tau)) + 1
```

Truncated backprop in depth:

- Detach recurrent state for all recurrences earlier than the last `bptt_k`.

## 3) Repo-specific deviations and design choices

1) **Prelude / recur / coda architecture**:
   - The repo has `prelude -> recur^t -> coda`.
   - Per-step prediction is defined as: stop after t recurrences, then run coda once.
   - This is faithful to how adaptive inference runs in this repo.

2) **Final-step distribution**:
   - We force the final step probability mass to `S_{Tmax-1}` (ignore lambda_Tmax).
   - This matches the Ouro exit distribution and prevents undefined halting.

3) **Batch-level adaptive exit**:
   - Q-exit currently halts when all rows in a batch meet `cdf >= q`.
   - This keeps KV cache updates consistent across the batch.
   - Per-row early exit is a future optimization.

4) **Loss reduction behavior**:
   - When `loss_reduction="none"`, we return the expected loss *without* entropy regularization.
   - This keeps BPB evaluation semantics intact.

## 4) Implementation mapping

### 4.1 Core model changes

Files:
- `nanochat/gpt.py`: exit gate module, exit distribution helper, Stage I loss, adaptive inference, stats helper.
- `nanochat/checkpoint_manager.py`: patch missing config keys.

Key points:
- `ExitGate`: linear head producing lambda_t for each token.
- Stage I loss computes per-step logits by running coda on `s_t` at each recurrence.
- Entropy is computed from `p_exit` (not lambda directly).
- Detach (TBPTT) happens after per-step loss + gate computation.
- `compute_exit_stats` logs entropy, expected depth, and p_last.

### 4.2 Training and eval plumbing

Files:
- `scripts/base_train.py`: new flags, config wiring, periodic gate stats.
- `scripts/base_eval.py`: new adaptive-exit flags for sample mode.
- `nanochat/engine.py`: Q-exit inference via `forward_adaptive_exit`.

### 4.3 Stage II script

File:
- `scripts/exit_gate_train.py`: loads a checkpoint, freezes LM, trains gate only.

Implementation summary:
- Computes per-step loss via coda at each recurrence.
- Improvements are computed from detached losses.
- BCE targets gate continuation probabilities.

## 5) Implementation notes and shapes

- `lambda_t`: shape (B, T), computed from `s_t`.
- `p_exit_t`: shape (B, T), sum over t = 1.
- Per-step loss: `loss_t` shape (B, T) using `ignore_index=-1`.
- Masked tokens (targets == -1) are excluded from all averages.
- For stability, use `log(p_exit_t.clamp_min(1e-12))` in entropy.

## 6) Experiment recipes

### 6.1 Stage I training

```
uv run python -m scripts.base_train \
  --run=exitgate_stage1 \
  --use-exit-gate \
  --exit-beta=0.05 \
  --exit-min-recur=1
```

Optional: disable recurrence sampling for bring-up

```
--no-sample-recur
```

### 6.2 Stage II gate tuning

```
uv run python -m scripts.exit_gate_train \
  --model-tag s12 \
  --run=exitgate_stage2 \
  --steps=1000 \
  --k=50.0 \
  --gamma=0.005
```

### 6.3 Adaptive inference (Q-exit)

```
uv run python -m scripts.base_eval \
  --model-tag s12 \
  --eval sample \
  --adaptive-exit-q 0.7 \
  --adaptive-exit-max-recur 16
```

## 7) Diagnostics and expected behavior

Metrics to track (logged from `base_train.py` when enabled):

- `gate/entropy`: should not collapse early (avoid near-zero).
- `gate/expected_t`: should spread across steps early, then specialize.
- `gate/p_last`: should be < 1.0 if gate is actually learning to exit early.

Failure modes:

- **Entropy collapse** (p_last ~ 1 quickly):
  - Increase `exit_beta`.
  - Ensure per-step losses are computed for all steps.
  - Verify entropy is computed on `p_exit`, not raw lambda.

- **Instability / divergence**:
  - Reduce LR.
  - Disable recurrence sampling temporarily (`--no-sample-recur`).

## 8) Open follow-ups

- Per-row adaptive exit to reduce wasted compute in batched generation.
- Optional early-exit head to avoid re-running coda at every step.
