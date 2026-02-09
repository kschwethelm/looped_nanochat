"""
Tests for exit gate utilities and Stage I training path.
"""

import torch

import nanochat.flash_attention as flash_attention
from nanochat.gpt import GPT, GPTConfig, compute_exit_distribution


def test_exit_distribution_sums_to_one():
    torch.manual_seed(0)
    tmax, batch, seq = 5, 3, 4
    lambdas = torch.rand(tmax, batch, seq) * 0.9  # keep away from 1.0
    p_exit = compute_exit_distribution(lambdas, min_recur=1)
    total = p_exit.sum(dim=0)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), "Exit distribution must sum to 1"


def test_q_exit_monotonicity():
    torch.manual_seed(0)
    tmax = 6
    lambdas = torch.rand(tmax, 1, 1) * 0.8
    p_exit = compute_exit_distribution(lambdas, min_recur=1)
    cdf = p_exit.cumsum(dim=0).view(tmax)
    q1, q2 = 0.2, 0.7
    step1 = int((cdf >= q1).nonzero(as_tuple=False)[0].item()) + 1
    step2 = int((cdf >= q2).nonzero(as_tuple=False)[0].item()) + 1
    assert step1 <= step2, "Higher q should not exit earlier"


def test_stage1_backward_has_gate_grads():
    prev_override = flash_attention._override_impl
    flash_attention._override_impl = "sdpa"
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        size=1,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        n_prelude=1,
        n_recur_block=1,
        n_coda=1,
        train_recur_mean=2.0,
        train_recur_max=3,
        bptt_k=2,
        use_exit_gate=True,
        exit_beta=0.05,
        exit_min_recur=1,
    )
    try:
        model = GPT(config)
        model.init_weights()
        model.train()
        # Nudge exit gate off 0.5 to ensure non-zero entropy gradient
        with torch.no_grad():
            model.exit_gate.proj.bias.fill_(0.2)

        idx = torch.randint(0, config.vocab_size, (2, 6))
        targets = torch.randint(0, config.vocab_size, (2, 6))
        loss = model(idx, targets=targets, loss_reduction="mean", num_recur=2)
        loss.backward()

        gate_grads = [p.grad for p in model.exit_gate.parameters() if p.grad is not None]
        assert gate_grads, "Exit gate should receive gradients"
        total_grad = sum(g.abs().sum().item() for g in gate_grads)
        assert total_grad > 0, "Exit gate gradients should be non-zero"
    finally:
        flash_attention._override_impl = prev_override
