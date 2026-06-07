"""Train a nanogo network on katagoarchive .npz data.

Example (depth-6 net):
    uv run python scripts/train.py --data data --blocks 6 --channels 96 \
        --batch-size 256 --max-steps 20000 --out checkpoints/depth6.pt
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.net import data
from nanogo.common import get_device, set_seed, setup_logging
from nanogo.net.losses import LossWeights, compute_losses
from nanogo.net.model import ARCHS, Model, arch_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="dir with training .npz files")
    p.add_argument("--out", default="checkpoints/model.pt")
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--arch", default="b6c96-gpool", choices=list(ARCHS),
                   help="model architecture from the registry")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--val-files", type=int, default=1, help="# npz files held out for eval")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", default=None)
    return p.parse_args()


def lr_at(step, base_lr, warmup, max_steps):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, max_steps - warmup)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def infinite_batches(files, args, device, seed):
    epoch = 0
    while True:
        order = list(files)
        # cheap reshuffle by rotation + seed-derived permutation
        rng = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(order), generator=rng).tolist()
        order = [order[i] for i in perm]
        yield from data.read_batches(
            order, args.batch_size, args.pos_len, device,
            randomize_symmetries=True, seed=seed + epoch,
        )
        epoch += 1


@torch.no_grad()
def evaluate(model, files, args, device, weights, max_batches):
    model.eval()
    agg = {}
    n = 0
    for batch in data.read_batches(files, args.batch_size, args.pos_len, device,
                                   randomize_symmetries=False, seed=0, drop_last=False):
        outputs = model(batch["spatial"], batch["glob"])
        _, parts = compute_losses(outputs, batch, batch["spatial"], weights)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        if n >= max_batches:
            break
    model.train()
    return {k: v / max(1, n) for k, v in agg.items()}


def main():
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"device={device}")

    files = data.list_npz(args.data)
    if not files:
        raise SystemExit(f"No .npz files under {args.data!r}. Run scripts/download_data.py first.")
    val_files = files[: args.val_files] if len(files) > args.val_files else files[:1]
    train_files = files[args.val_files:] if len(files) > args.val_files else files
    print(f"{len(train_files)} train files, {len(val_files)} val files")

    import dataclasses
    config = dataclasses.replace(arch_config(args.arch), pos_len=args.pos_len)
    model = Model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {config}  params={n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = LossWeights()
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    batches = infinite_batches(train_files, args, device, args.seed)
    model.train()
    t0 = time.time()
    running = {}
    for step in range(start_step, args.max_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.max_steps)
        batch = next(batches)
        outputs = model(batch["spatial"], batch["glob"])
        loss, parts = compute_losses(outputs, batch, batch["spatial"], weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        for k, v in parts.items():
            running[k] = 0.9 * running.get(k, v) + 0.1 * v

        if step % 50 == 0:
            dt = time.time() - t0
            sps = (step - start_step + 1) / dt
            msg = " ".join(f"{k}={running[k]:.3f}" for k in ["total", "policy", "value", "score", "ownership"])
            print(f"step {step} lr={opt.param_groups[0]['lr']:.2e} {msg} ({sps:.1f} it/s)")

        if step > 0 and step % args.eval_interval == 0:
            ev = evaluate(model, val_files, args, device, weights, args.eval_batches)
            print(f"  [eval step {step}] " + " ".join(f"{k}={v:.3f}" for k, v in ev.items()))

        if step > 0 and step % args.save_interval == 0:
            save(model, opt, config, step, args.out)

    save(model, opt, config, args.max_steps, args.out)
    print(f"done -> {args.out}")


def save(model, opt, config, step, out):
    from nanogo.go import features as F
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "model_config": asdict(config),
        "step": step,
        "spatial_subset": F.SPATIAL_SUBSET,
        "global_subset": F.GLOBAL_SUBSET,
    }, out)


if __name__ == "__main__":
    main()
