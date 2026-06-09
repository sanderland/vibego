"""Inference cost of a nanogo net: parameter count and FLOPs/eval (batch=1).

We count MACs with forward hooks on the only FLOP-heavy layers — nn.Conv2d and
nn.Linear — so any block type (regular / gpool / nbt) is handled without
hardcoding the trunk shape. BatchNorm2d is skipped: it's an affine
per-element op (a few elementwise mul/adds), tiny next to the convs and fused
into the preceding conv at inference time anyway. FLOPs = 2 * MACs (one
multiply + one add per MAC), the usual convention.

CLI:  python -m nanogo.net.flops [arch ...]   (defaults to every arch in ARCHS)
"""
from __future__ import annotations

import argparse
import dataclasses

import torch
import torch.nn as nn

from .model import ARCHS, Model, ModelConfig, arch_config


def _conv_macs(module: nn.Conv2d, out: torch.Tensor) -> int:
    """MACs for one conv = out_elements * (in_channels/groups) * kh * kw."""
    out_elems = out.numel()  # B * out_channels * H_out * W_out
    kh, kw = module.kernel_size
    in_per_group = module.in_channels // module.groups
    return out_elems * in_per_group * kh * kw


def _linear_macs(module: nn.Linear, out: torch.Tensor) -> int:
    """MACs for a linear = out_elements * in_features (per output element we
    do a length-in_features dot product). out.numel() already folds in batch."""
    return out.numel() * module.in_features


def count_flops_params(cfg: ModelConfig, board: int = 19) -> dict:
    """Build Model(cfg with pos_len=board), run one batch=1 CPU forward under
    no_grad while hooks tally MACs, and report params / macs / flops."""
    cfg = dataclasses.replace(cfg, pos_len=board)
    model = Model(cfg).eval()

    macs = 0

    def hook(module, inputs, output):
        nonlocal macs
        if isinstance(module, nn.Conv2d):
            macs += _conv_macs(module, output)
        elif isinstance(module, nn.Linear):
            macs += _linear_macs(module, output)

    handles = [
        m.register_forward_hook(hook)
        for m in model.modules()
        if isinstance(m, (nn.Conv2d, nn.Linear))
    ]
    try:
        spatial = torch.zeros(1, cfg.num_spatial, board, board)
        glob = torch.zeros(1, cfg.num_global)
        with torch.no_grad():
            model(spatial, glob)
    finally:
        for h in handles:
            h.remove()

    params = sum(p.numel() for p in model.parameters())
    return {"params": int(params), "macs": int(macs), "flops": int(2 * macs)}


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
