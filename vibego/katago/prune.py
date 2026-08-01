"""Structural pruning that stays *inside* KataGo's model format.

The point of the constraint: a net pruned this way is still an ordinary `.bin.gz`, so it loads in
the stock engine and can be measured by the existing harness (`policy_eval.py`, `move_eval.py`,
`match.py`, `arena.py`) at real backend speed. Nothing here needs a torch reimplementation, and
nothing here is a simulation -- the FLOPs actually go away.

Three levers, in increasing order of how local (and therefore how fine-grained) they are:

  drop_blocks       remove whole trunk blocks. Coarsest -- one block of a b10 net is 10% of depth.
  drop_inner_pairs  remove attention+FFN pairs from inside an `nbt` block's bottleneck stack.
  prune_heads       remove attention heads. Per-block, uncoupled from the trunk.
  narrow_ffn        shrink the FFN hidden dim. Per-block, uncoupled, and the finest-grained knob.

All four are exact structural edits: attention heads are concatenated and independent, FFN hidden
units are independent, and blocks are residual, so the smaller net is a well-formed net rather
than an approximation of one. What is *approximate* is the choice of what to remove.

Selection criterion (weight-only). With no activations to look at, importance is a magnitude
proxy over the two weight matrices a unit sits between -- for a head, ||W_v[:, h]|| * ||W_out[h, :]||;
for an FFN unit, ||W1[:, j]|| * ||W_gate[:, j]|| * ||W2[j, :]||. This is the cheap zeroth-order
criterion, in the spirit of Wanda without the activation term. It ignores how much signal actually
flows through a unit on real positions, so an activation-aware criterion (calibration positions
through a torch forward) should beat it; treat these numbers as the floor of what pruning can do,
not the ceiling.

The trunk residual stream is deliberately *not* prunable here: it is shared across every block and
both heads, so narrowing it needs one global mask, which is a different (and coupled) problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .binmodel import (
    KataModel,
    NestedBottleneckBlock,
    TransformerAttentionBlock,
    TransformerFFNBlock,
)


class PruneError(Exception):
    pass


@dataclass
class PruneRecord:
    """What a pruning pass actually did, for the notebook entry."""

    op: str
    detail: str


def sanitize_name(name: str, suffix: str) -> str:
    """KataGo requires model names to be <=96 chars of [A-Za-z0-9_-] (they end up in cache
    filenames), so a generated name has to be scrubbed."""
    combined = re.sub(r"[^A-Za-z0-9_-]", "-", f"{name}-{suffix}")
    return combined[:96]


# --------------------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------------------


def drop_blocks(model: KataModel, indices: list[int]) -> PruneRecord:
    """Remove whole trunk blocks. Valid because every trunk block is residual with matching
    in/out channels, so the trunk still type-checks with fewer of them."""
    blocks = model.trunk.blocks
    bad = [i for i in indices if not 0 <= i < len(blocks)]
    if bad:
        raise PruneError(f"block index out of range 0..{len(blocks) - 1}: {bad}")
    if len(set(indices)) == len(blocks):
        raise PruneError("refusing to drop every trunk block")
    keep = [b for i, b in enumerate(blocks) if i not in set(indices)]
    model.trunk.blocks = keep
    model.trunk.num_blocks = len(keep)
    return PruneRecord("drop_blocks", f"dropped {sorted(set(indices))}, {len(keep)} blocks remain")


def drop_inner_pairs(model: KataModel, block_index: int, count: int) -> PruneRecord:
    """Remove `count` attention+FFN pairs from one `nbt` block's internal stack.

    Finer-grained than dropping a whole trunk block: an `nbt` block of internal_length 2 holds
    four sub-blocks, so this removes ~5% of trunk depth instead of ~10%. Pairs are removed from
    the end of the stack, where the residual updates are usually smallest.
    """
    block = model.trunk.blocks[block_index]
    if not isinstance(block, NestedBottleneckBlock):
        raise PruneError(f"block {block_index} is {block.kind}, not a nested_bottleneck_block")
    kinds = [b.kind for b in block.blocks]
    pairs = len(block.blocks) // 2
    expected = ["transformer_attention_block", "transformer_ffn_block"] * pairs
    if kinds != expected:
        raise PruneError(
            f"block {block_index} inner stack is {kinds}, not alternating attention/FFN pairs"
        )
    if count >= pairs:
        raise PruneError(f"block {block_index} has {pairs} pairs; refusing to drop {count}")
    block.blocks = block.blocks[: 2 * (pairs - count)]
    block.num_blocks = len(block.blocks)
    return PruneRecord("drop_inner_pairs",
                       f"block {block_index}: {pairs} -> {pairs - count} attention/FFN pairs")


# --------------------------------------------------------------------------------------
# Width
# --------------------------------------------------------------------------------------


def head_importance(block: TransformerAttentionBlock) -> np.ndarray:
    """Per-head weight-only importance: the product of the norms of the two matrices the head's
    activations pass through (V projection in, output projection out)."""
    h, vd = block.num_heads, block.v_head_dim
    v = block.v_proj.weight.reshape(-1, h, vd)  # (c_in, heads, v_head_dim)
    o = block.out_proj.weight.reshape(h, vd, -1)  # (heads, v_head_dim, c_out)
    return np.linalg.norm(v, axis=(0, 2)) * np.linalg.norm(o, axis=(1, 2))


def prune_heads(block: TransformerAttentionBlock, keep_heads: int,
                importance: np.ndarray | None = None) -> PruneRecord:
    """Keep the `keep_heads` highest-importance attention heads, dropping the rest exactly.

    Heads are concatenated along the projection outputs and never mix until `out_proj`, so
    slicing them out of q/k/v/out (and the learnable RoPE frequency table) leaves a net that
    computes exactly what the full net would have computed with those heads zeroed -- minus
    their FLOPs.
    """
    h = block.num_heads
    if block.num_kv_heads != h:
        raise PruneError(
            f"{block.name}: grouped-query attention (num_kv_heads={block.num_kv_heads} != "
            f"num_heads={h}) is not supported -- q heads share kv heads, so they cannot be "
            "dropped independently"
        )
    if not 0 < keep_heads < h:
        raise PruneError(f"{block.name}: keep_heads must be in 1..{h - 1}, got {keep_heads}")

    scores = head_importance(block) if importance is None else np.asarray(importance)
    order = np.argsort(scores)[::-1][:keep_heads]
    keep = np.sort(order)
    qd, vd = block.q_head_dim, block.v_head_dim

    def slice_out(weight: np.ndarray, head_dim: int) -> np.ndarray:
        # (c_in, heads * head_dim) -> keep a subset of heads
        return weight.reshape(weight.shape[0], -1, head_dim)[:, keep, :].reshape(
            weight.shape[0], keep_heads * head_dim)

    block.q_proj.weight = slice_out(block.q_proj.weight, qd)
    block.q_proj.out_channels = keep_heads * qd
    block.k_proj.weight = slice_out(block.k_proj.weight, qd)
    block.k_proj.out_channels = keep_heads * qd
    block.v_proj.weight = slice_out(block.v_proj.weight, vd)
    block.v_proj.out_channels = keep_heads * vd

    out = block.out_proj.weight.reshape(-1, vd, block.out_proj.out_channels)[keep]
    block.out_proj.weight = out.reshape(keep_heads * vd, block.out_proj.out_channels)
    block.out_proj.in_channels = keep_heads * vd

    if block.rope_freqs is not None:
        block.rope_freqs = block.rope_freqs[keep]
    block.num_heads = keep_heads
    block.num_kv_heads = keep_heads
    return PruneRecord("prune_heads", f"{block.name}: {h} -> {keep_heads} heads "
                                      f"(kept {[int(i) for i in keep]})")


def ffn_importance(block: TransformerFFNBlock) -> np.ndarray:
    """Per-hidden-unit weight-only importance across the FFN's in/gate/out matrices."""
    score = np.linalg.norm(block.linear1.weight, axis=0) * np.linalg.norm(block.linear2.weight,
                                                                         axis=1)
    if block.linear_gate is not None:
        score = score * np.linalg.norm(block.linear_gate.weight, axis=0)
    return score


def narrow_ffn(block: TransformerFFNBlock, keep_units: int,
               importance: np.ndarray | None = None) -> PruneRecord:
    """Keep the `keep_units` highest-importance FFN hidden units.

    The finest-grained lever available: hidden units are independent (each contributes one
    rank-1 term to the output), so this is exact and can be applied a few percent at a time
    rather than in whole-block chunks.
    """
    n = block.ffn_channels
    if not 0 < keep_units < n:
        raise PruneError(f"{block.name}: keep_units must be in 1..{n - 1}, got {keep_units}")
    scores = ffn_importance(block) if importance is None else np.asarray(importance)
    keep = np.sort(np.argsort(scores)[::-1][:keep_units])

    block.linear1.weight = block.linear1.weight[:, keep]
    block.linear1.out_channels = keep_units
    if block.linear_gate is not None:
        block.linear_gate.weight = block.linear_gate.weight[:, keep]
        block.linear_gate.out_channels = keep_units
    block.linear2.weight = block.linear2.weight[keep, :]
    block.linear2.in_channels = keep_units
    block.ffn_channels = keep_units
    return PruneRecord("narrow_ffn", f"{block.name}: ffn {n} -> {keep_units}")


# --------------------------------------------------------------------------------------
# Whole-model passes
# --------------------------------------------------------------------------------------


def iter_attention_blocks(model: KataModel):
    for i, block in enumerate(model.trunk.blocks):
        if isinstance(block, TransformerAttentionBlock):
            yield i, block
        elif isinstance(block, NestedBottleneckBlock):
            for sub in block.blocks:
                if isinstance(sub, TransformerAttentionBlock):
                    yield i, sub


def iter_ffn_blocks(model: KataModel):
    for i, block in enumerate(model.trunk.blocks):
        if isinstance(block, TransformerFFNBlock):
            yield i, block
        elif isinstance(block, NestedBottleneckBlock):
            for sub in block.blocks:
                if isinstance(sub, TransformerFFNBlock):
                    yield i, sub


def prune_heads_everywhere(model: KataModel, keep_fraction: float,
                           importance: dict | None = None) -> list[PruneRecord]:
    """Uniform head pruning across every attention block. Uniform is the *baseline*, not the
    right answer -- per-block head budgets should come from a sensitivity sweep.

    `importance` maps block name -> per-head scores (see `calibrate.weighted_head_importance`);
    without it the weight-only proxy is used.
    """
    records = []
    for _, block in iter_attention_blocks(model):
        keep = max(1, int(round(block.num_heads * keep_fraction)))
        if keep < block.num_heads:
            scores = None if importance is None else importance.get(block.name)
            records.append(prune_heads(block, keep, scores))
    if not records:
        raise PruneError("no attention blocks with prunable heads found")
    return records


def narrow_ffn_everywhere(model: KataModel, keep_fraction: float,
                          importance: dict | None = None) -> list[PruneRecord]:
    records = []
    for _, block in iter_ffn_blocks(model):
        keep = max(1, int(round(block.ffn_channels * keep_fraction)))
        if keep < block.ffn_channels:
            scores = None if importance is None else importance.get(block.name)
            records.append(narrow_ffn(block, keep, scores))
    if not records:
        raise PruneError("no FFN blocks found")
    return records
