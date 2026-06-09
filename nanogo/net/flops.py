"""Inference cost of a nanogo net: parameter count and FLOPs/eval (batch=1).

We count FLOPs by **dispatch** (torch FlopCounterMode), not forward hooks: the
counter tallies every FLOP-heavy aten op — conv *and* the matmul/einsum/bmm in
attention blocks — so any block type (regular / gpool / nbt / linattn / rwkv)
is handled without hardcoding the trunk. A hook-on-Conv2d/Linear counter would
silently miss linattn's attention einsums (undercount of ~7-18 MFLOP at our
sizes) — exactly the honesty check experiments/IDEAS.md demands. Elementwise
ops (BatchNorm, relu, the rwkv WKV softmax-pool) are not counted: they're tiny
next to the matmuls and BN fuses into the preceding conv at inference anyway.
get_total_flops() already returns 2 * MACs (one multiply + one add per MAC),
the usual convention; we derive macs = flops // 2 for the report.

CLI:  python -m nanogo.net.flops [arch ...]   (defaults to every arch in ARCHS)
"""
from __future__ import annotations

import argparse
import dataclasses

import torch
from torch.utils.flop_counter import FlopCounterMode

from .model import ARCHS, Model, ModelConfig, arch_config


def count_flops_params(cfg: ModelConfig, board: int = 19) -> dict:
    """Build Model(cfg with pos_len=board), run one batch=1 CPU forward under a
    FlopCounterMode (and no_grad), and report params / macs / flops."""
    cfg = dataclasses.replace(cfg, pos_len=board)
    model = Model(cfg).eval()
    spatial = torch.zeros(1, cfg.num_spatial, board, board)
    glob = torch.zeros(1, cfg.num_global)
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        model(spatial, glob)
    flops = int(fc.get_total_flops())
    params = sum(p.numel() for p in model.parameters())
    return {"params": int(params), "macs": flops // 2, "flops": flops}


def arch_flops(name: str, board: int = 19) -> dict:
    """Cost of a named architecture from the ARCHS registry."""
    return count_flops_params(arch_config(name), board=board)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archs", nargs="*", default=list(ARCHS),
                        help="arch names to measure (default: all in ARCHS)")
    parser.add_argument("--board", type=int, default=19, help="board size")
    args = parser.parse_args()
    names = args.archs or list(ARCHS)

    print(f"{'name':<16}{'params(M)':>12}{'MFLOP/eval':>14}")
    for name in names:
        r = arch_flops(name, board=args.board)
        print(f"{name:<16}{r['params'] / 1e6:>12.3f}{r['flops'] / 1e6:>14.1f}")


if __name__ == "__main__":
    main()
