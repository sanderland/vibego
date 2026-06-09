"""Guards the FLOP counter in scripts/bench_net.py — the north-star x-axis must stay correct
across torch versions, and the attention-block einsums must actually be counted (a hook-based
counter would silently miss them)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import bench_net  # noqa: E402

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from vibego.net.model import LinAttnResBlock, arch_config  # noqa: E402


def test_count_flops_matches_hand_computed_conv_convention():
    # A 6-block plain b6c96 trunk: each ResBlock = two 3x3 convs at 96ch over 19x19.
    # One such conv = 19*19*96*96*9 MACs; the counter must report MACs*2 (our convention),
    # so the trunk convs alone exceed 6*2 of them. We assert the per-conv FLOP unit precisely
    # via a single-conv-equivalent: the whole-net MFLOP must reproduce the documented b6c96-gpool.
    mflop = bench_net.count_flops(arch_config("b6c96-gpool"))
    assert abs(mflop - 774) < 5  # documented value (experiments/2026-06-08-flops-elo-frontier.md)


def test_attention_einsums_are_counted():
    # linattn's global mixing is einsums, not modules — a dispatch counter must see them. The
    # block's total FLOPs must strictly exceed its 1x1-conv-only FLOPs (qkv+proj+ffn), i.e. the
    # attention einsums contribute a non-zero, counted amount on top.
    c, n = 96, 9 * 9
    block = LinAttnResBlock(c).eval()
    x = torch.zeros(1, c, 9, 9)
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        block(x)
    total = fc.get_total_flops()
    # 1x1 convs only: qkv(3C*C) + proj(C*C) + ffn1(2C*C) + ffn2(C*2C) MACs per token, *2 for FLOPs
    conv_only = 2 * n * c * c * (3 + 1 + 2 + 2)
    assert total > conv_only > 0
