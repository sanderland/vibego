"""Evaluate a checkpoint's loss on one or more .npz dirs/files — the comparison half of the
off-policy diagnostic: the SAME net's loss on archive (g170-trajectory) positions vs its own
match-game (student-trajectory) positions, both labeled by the same b18 teacher at the same
visits, isolates the distribution term. A big on-policy excess = off-policy gap confirmed.

    uv run python scripts/eval_loss.py --ckpt runs/s_nbt.pt \
        --data archive=/workspace/distill/distilled-b18-val onpolicy=/workspace/distill/onpolicy
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.common import get_device
from vibego.net import data
from vibego.net.losses import LossWeights, compute_losses
from vibego.net.model import Model, ModelConfig


@torch.no_grad()
def eval_dir(model, files, pos_len, batch_size, device, weights, max_batches):
    agg, n = {}, 0
    for batch in data.read_batches(files, batch_size, pos_len, device,
                                   randomize_symmetries=False, seed=0, drop_last=False,
                                   prefetch_ahead=4):
        outputs = model(batch["spatial"], batch["glob"])
        _, parts = compute_losses(outputs, batch, batch["spatial"], weights)
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
        if n >= max_batches:
            break
    return {k: v / max(1, n) for k, v in agg.items()}, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, action="append",
                   help="checkpoint .pt (repeatable: compare several nets on the same sets)")
    p.add_argument("--data", required=True, nargs="+", metavar="NAME=PATH",
                   help="named npz dirs/files to evaluate on")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-batches", type=int, default=100)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    weights = LossWeights()
    sets = []
    for spec in args.data:
        name, path = spec.split("=", 1)
        files = data.list_npz(path) if os.path.isdir(path) else [path]
        sets.append((name, files))

    keys = None
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)
        model = Model(ModelConfig(**ckpt["model_config"])).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for name, files in sets:
            parts, n = eval_dir(model, files, ckpt["model_config"].get("pos_len", 19),
                                args.batch_size, device, weights, args.max_batches)
            if keys is None:
                keys = sorted(parts)
                print(f"{'ckpt':12s} {'set':10s} {'batches':>7s} " +
                      "".join(f"{k:>10s}" for k in keys))
            print(f"{os.path.basename(ckpt_path):12s} {name:10s} {n:7d} " +
                  "".join(f"{parts[k]:10.4f}" for k in keys), flush=True)


if __name__ == "__main__":
    main()
