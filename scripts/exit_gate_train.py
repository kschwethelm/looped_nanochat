"""
Stage II: focused exit gate training (Ouro-style).

This script freezes the base LM and trains only the exit gate using
per-step marginal loss improvements.

Supports training after either base pretraining or SFT:
    python -m scripts.exit_gate_train --source base --model-tag s12
    python -m scripts.exit_gate_train --source sft --model-tag s12
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import (
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
    sample_poisson_lognormal_recurrence,
)
from nanochat.dataloader import sft_data_loader, tokenizing_distributed_data_loader_with_state_bos_bestfit


def compute_gate_loss(model, idx, targets, num_recur, k, gamma):
    B, T = idx.size()
    if num_recur is None:
        num_recur = int(model.config.train_recur_mean)

    assert model.cos.size(1) >= T, f"Sequence length grew beyond rotary cache: {T} > {model.cos.size(1)}"
    assert idx.device == model.cos.device, f"Rotary embeddings and idx on different devices: {idx.device} != {model.cos.device}"
    cos_sin = (
        model.cos[:, :T],
        model.sin[:, :T],
    )

    x = model.transformer.wte(idx)
    x = model.norm_emb(x)
    for i, block in enumerate(model.transformer.prelude):
        x = block(x, cos_sin, model.window_sizes[i], kv_cache=None, layer_idx=None)
    e = x

    s = None
    prev_loss = None
    total_bce = torch.tensor(0.0, device=idx.device)
    total_count = torch.tensor(0.0, device=idx.device)
    for i in range(num_recur):
        u = model._state_transfer(e, s=s, warm_start_state=None)
        for j, block in enumerate(model.transformer.recur):
            u = block(u, cos_sin, model.window_sizes[model.config.n_prelude + j], kv_cache=None, layer_idx=None)
        s = model.norm_recur(u)

        logits_t = model._compute_logits_from_state(s, cos_sin, kv_cache=None, kv_budget=None)
        loss_t = F.cross_entropy(
            logits_t.view(-1, logits_t.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction="none",
        ).view(B, T)
        loss_t = loss_t.detach()

        lambda_t = model.exit_gate(s).float()
        if prev_loss is not None and i + 1 >= max(2, model.config.exit_min_recur):
            improvement = (prev_loss - loss_t).clamp_min(0.0)
            w_t = torch.sigmoid(k * (improvement - gamma))
            cont_prob = (1.0 - lambda_t).clamp(min=1e-6, max=1.0 - 1e-6)
            bce = F.binary_cross_entropy(cont_prob, w_t, reduction="none")
            mask = targets != -1
            total_bce += (bce * mask).sum()
            total_count += mask.sum()

        prev_loss = loss_t

        if model.config.bptt_k is not None and i < num_recur - model.config.bptt_k:
            s = s.detach()

    return total_bce / total_count.clamp(min=1.0)


def main():
    parser = argparse.ArgumentParser(description="Stage II exit gate training")
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    parser.add_argument("--source", type=str, choices=["base", "sft"], default="base", help="checkpoint source: 'base' (pretraining) or 'sft' (chat fine-tuned)")
    parser.add_argument("--model-tag", type=str, required=True, help="model tag to load from")
    parser.add_argument("--step", type=int, default=None, help="model step to load (default = last)")
    parser.add_argument("--output-tag", type=str, default=None, help="model tag to save to (default = '<model-tag>_gate')")
    parser.add_argument("--steps", type=int, default=1000, help="number of optimization steps")
    parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
    parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
    parser.add_argument("--max-seq-len", type=int, default=-1, help="max context length (-1 = use model config)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for exit gate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for exit gate optimizer")
    parser.add_argument("--adam-beta1", type=float, default=0.8, help="Adam beta1 for exit gate optimizer")
    parser.add_argument("--adam-beta2", type=float, default=0.95, help="Adam beta2 for exit gate optimizer")
    parser.add_argument("--k", type=float, default=50.0, help="sigmoid sharpness for improvement labels")
    parser.add_argument("--gamma", type=float, default=0.005, help="improvement threshold for labels")
    parser.add_argument("--no-sample-recur", action="store_true", help="disable sampling num_recur; use train_recur_mean")
    parser.add_argument("--log-every", type=int, default=100, help="log to wandb every N steps")
    parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")
    args = parser.parse_args()
    user_config = vars(args).copy()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

    model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.step)
    if not model.config.use_exit_gate or model.exit_gate is None:
        raise ValueError("Stage II requires a checkpoint trained with use_exit_gate=True")

    # Freeze all params except exit gate
    for p in model.parameters():
        p.requires_grad = False
    for p in model.exit_gate.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.exit_gate.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else meta["model_config"]["sequence_len"]
    if max_seq_len > model.config.sequence_len:
        raise ValueError(f"max_seq_len ({max_seq_len}) exceeds model config sequence_len ({model.config.sequence_len})")
    tokens_per_fwdbwd = args.device_batch_size * max_seq_len
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
    assert args.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
    print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
    print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

    if args.source == "sft":
        from tasks.common import TaskMixture
        from tasks.customjson import CustomJSON
        from tasks.gsm8k import GSM8K
        from tasks.mmlu import MMLU
        from tasks.smoltalk import SmolTalk
        from tasks.spellingbee import SimpleSpelling, SpellingBee

        base_dir = get_base_dir()
        identity_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
        train_dataset = TaskMixture(
            [
                SmolTalk(split="train"),
                MMLU(subset="auxiliary_train", split="train"),
                GSM8K(subset="main", split="train"),
                GSM8K(subset="main", split="train"),
                CustomJSON(filepath=identity_filepath),
                CustomJSON(filepath=identity_filepath),
                SimpleSpelling(size=200000, split="train"),
                SpellingBee(size=80000, split="train"),
            ]
        )
        train_loader = sft_data_loader(
            tokenizer,
            dataset=train_dataset,
            B=args.device_batch_size,
            T=max_seq_len,
            device=device,
            device_type=device_type,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
    else:
        train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            tokenizer,
            B=args.device_batch_size,
            T=max_seq_len,
            split="train",
            device=device,
        )

    output_tag = args.output_tag or f"{args.model_tag}_gate"
    checkpoint_subdir = "chatsft_checkpoints" if args.source == "sft" else "base_checkpoints"
    output_dir = os.path.join(get_base_dir(), checkpoint_subdir, output_tag)

    step = 0
    while step < args.steps:
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for _micro_step in range(grad_accum_steps):
            if args.no_sample_recur:
                num_recur = None
            else:
                num_recur = sample_poisson_lognormal_recurrence(
                    mean_recur=model.config.train_recur_mean,
                    sigma=0.5,
                    max_recur=model.config.train_recur_max,
                )
            x, y, _state = next(train_loader)
            with autocast_ctx:
                loss = compute_gate_loss(model, x, y, num_recur=num_recur, k=args.k, gamma=args.gamma)
            (loss / grad_accum_steps).backward()
        optimizer.step()

        loss_f = loss.item()
        dt = time.time() - t0
        if step % args.log_every == 0:
            log_data = {
                "step": step,
                "train/loss": loss_f,
                "train/dt": dt,
                "train/num_recur": num_recur if num_recur is not None else int(model.config.train_recur_mean),
            }
            wandb_run.log(log_data)
            print0(f"step {step:05d} | loss: {loss_f:.6f} | dt: {dt * 1000:.2f}ms")

        if (step + 1 == args.steps) or (args.save_every > 0 and step > 0 and step % args.save_every == 0):
            save_checkpoint(
                output_dir,
                step,
                model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "model_config": meta["model_config"],
                    "user_config": user_config,
                    "max_seq_len": max_seq_len,
                },
                rank=ddp_rank,
            )

        step += 1

    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    main()
