"""Inference-speed benchmark: measured wall-clock nnevals/s per architecture, so we can compare
nets at equal *inference cost* (speed-matched) instead of only equal params (size-matched).

FLOPs alone undersell nbt's real cost: its deeper, narrower, bottlenecked trunk has more
sequential layers and memory round-trips (lower arithmetic intensity), so wall-clock can be worse
than the FLOP count suggests — exactly the "framework timing vs real backend" gap the KataGo devs
flag. Speed is weight-independent, so this benches untrained models built from the registry.

    uv run python scripts/bench_net.py --device mps --batch-sizes 1,16 --iters 60
    uv run python scripts/bench_net.py --device cpu --arch b6c96-gpool --arch b6c112nbt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.common import get_device
from nanogo.go.features import NUM_GLOBAL, NUM_SPATIAL
from nanogo.net.model import Model, arch_config

# The bake-off + depth-width nets, in the order we report them.
DEFAULT = [
    ("old6b", "b6c96-gpool"), ("nbt6b", "b6c96nbt"), ("nbt6b_matched", "b6c112nbt"),
    ("old10b", "b10c128-gpool"), ("nbt10b", "b10c128nbt"), ("nbt10b_matched", "b10c152nbt"),
    ("dw7", "b7c106nbt"), ("dw8", "b8c102nbt"), ("dw9", "b9c92nbt"), ("dw10", "b10c88nbt"),
    # global-mixing study (every-3rd slot = gpool vs linattn vs rwkv; cf. experiments/IDEAS.md)
    ("linat6b", "b6c96-linat"), ("rwkv6b", "b6c96-rwkv"),
    ("linat10b", "b10c128-linat"), ("rwkv10b", "b10c128-rwkv"),
]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def bench(cfg, device, batch, iters, warmup):
    model = Model(cfg).to(device).eval()
    sp = torch.randn(batch, NUM_SPATIAL, cfg.pos_len, cfg.pos_len, device=device)
    gl = torch.randn(batch, NUM_GLOBAL, device=device)
    for _ in range(warmup):
        model(sp, gl)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(sp, gl)
    sync(device)
    dt = time.perf_counter() - t0
    ms_per_batch = 1000.0 * dt / iters
    evals_per_s = batch * iters / dt
    return ms_per_batch, evals_per_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--arch", action="append", help="arch name (repeatable); default = bake-off set")
    p.add_argument("--batch-sizes", default="1,16", help="comma-separated")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--warmup", type=int, default=15)
    args = p.parse_args()

    device = get_device(args.device)
    batches = [int(b) for b in args.batch_sizes.split(",")]
    nets = [(a, a) for a in args.arch] if args.arch else DEFAULT
    print(f"device={device}  iters={args.iters}  warmup={args.warmup}")

    # header
    cols = "".join(f"{'b%d ms' % b:>9s}{'b%d ev/s' % b:>10s}" for b in batches)
    print(f"{'name':15s} {'arch':15s} {'params':>8s}{cols}")
    base = {}  # batch -> baseline evals/s (first net) for a relative column
    for name, arch in nets:
        cfg = arch_config(arch)
        nparams = sum(x.numel() for x in Model(cfg).parameters()) / 1e6
        cells = ""
        for b in batches:
            ms, eps = bench(cfg, device, b, args.iters, args.warmup)
            cells += f"{ms:9.2f}{eps:10.0f}"
        print(f"{name:15s} {arch:15s} {nparams:7.3f}M{cells}", flush=True)


if __name__ == "__main__":
    main()
