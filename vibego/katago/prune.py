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


# --------------------------------------------------------------------------------------
# Low-rank: the value path, which the file format already stores in factored form
# --------------------------------------------------------------------------------------


def _inverse_sqrt_and_sqrt(second_moment: np.ndarray, ridge: float = 1e-8):
    """Symmetric square root and inverse square root of a PSD second-moment matrix."""
    eig, vecs = np.linalg.eigh(second_moment.astype(np.float64))
    floor = max(float(eig.max()), 1e-30) * ridge
    eig = np.clip(eig, floor, None)
    root = vecs @ np.diag(np.sqrt(eig)) @ vecs.T
    inv_root = vecs @ np.diag(1.0 / np.sqrt(eig)) @ vecs.T
    return root, inv_root


def low_rank_value_path(block: TransformerAttentionBlock, keep_dim: int,
                        second_moment: np.ndarray | None = None) -> PruneRecord:
    """Shrink `v_head_dim` by low-rank approximation of each head's value->output map.

    This is the lever the phase-1 format table missed. A low-rank factorization normally needs an
    extra layer, which model format v17 cannot express -- but the value path is *already stored
    factored*: `v_proj` maps c_in -> heads*v_head_dim and `out_proj` maps heads*v_head_dim ->
    c_out, with `v_head_dim` a plain header field. Shrinking the dimension between them IS the
    low-rank approximation, in-format, no new layer type, and the engine loads the result.

    Per head the composed map is M = W_v @ W_out (c_in x c_out), of rank at most v_head_dim. We
    replace it with its best rank-`keep_dim` approximation *under the input distribution*: with
    C = E[x x^T] at the block's input, minimizing ||C^(1/2)(M - M')||_F rather than ||M - M'||_F,
    because directions the data never visits are not worth spending rank on. Pass
    `second_moment=None` to fall back to the plain (data-blind) SVD.

    Unlike head pruning, nothing is discarded outright -- every head keeps a (smaller) subspace,
    which matters here because the diagnostics found no head to be idle.

    Caveat: attention's row mixing is a left factor (out = A @ X @ M), so optimizing over the rows
    of X ignores that A contracts them. It is the standard approximation and it is conservative --
    the true error is no larger.
    """
    if block.num_kv_heads != block.num_heads:
        raise PruneError(f"{block.name}: grouped-query attention is not supported")
    h, vd = block.num_heads, block.v_head_dim
    if not 0 < keep_dim < vd:
        raise PruneError(f"{block.name}: keep_dim must be in 1..{vd - 1}, got {keep_dim}")

    c_in = block.v_proj.in_channels
    c_out = block.out_proj.out_channels
    w_v = block.v_proj.weight.astype(np.float64).reshape(c_in, h, vd)
    w_out = block.out_proj.weight.astype(np.float64).reshape(h, vd, c_out)

    if second_moment is None:
        root = inv_root = np.eye(c_in)
    else:
        root, inv_root = _inverse_sqrt_and_sqrt(second_moment)

    new_v = np.empty((c_in, h, keep_dim))
    new_out = np.empty((h, keep_dim, c_out))
    kept_energy = []
    for i in range(h):
        m = w_v[:, i, :] @ w_out[i]                      # (c_in, c_out), rank <= vd
        u, s, vt = np.linalg.svd(root @ m, full_matrices=False)
        kept_energy.append(float((s[:keep_dim] ** 2).sum() / max((s ** 2).sum(), 1e-30)))
        new_v[:, i, :] = inv_root @ (u[:, :keep_dim] * s[:keep_dim])
        new_out[i] = vt[:keep_dim]

    block.v_proj.weight = new_v.reshape(c_in, h * keep_dim).astype(np.float32)
    block.v_proj.out_channels = h * keep_dim
    block.out_proj.weight = new_out.reshape(h * keep_dim, c_out).astype(np.float32)
    block.out_proj.in_channels = h * keep_dim
    block.v_head_dim = keep_dim
    return PruneRecord("low_rank_value", f"{block.name}: v_head_dim {vd} -> {keep_dim} "
                                         f"(kept {100 * np.mean(kept_energy):.1f}% of weighted "
                                         f"spectral energy)")


def low_rank_value_everywhere(model: KataModel, keep_fraction: float,
                              moments: dict | None = None) -> list[PruneRecord]:
    records = []
    for _, block in iter_attention_blocks(model):
        keep = max(1, int(round(block.v_head_dim * keep_fraction)))
        if keep < block.v_head_dim:
            moment = None if moments is None else moments.get(block.name)
            records.append(low_rank_value_path(block, keep, moment))
    if not records:
        raise PruneError("no attention blocks with a reducible value path found")
    return records


def rope_pair_energy(block: TransformerAttentionBlock,
                     second_moment: np.ndarray | None = None) -> np.ndarray:
    """Per-(head, RoPE pair) importance: the product of the q and k energies that pair carries.

    Each RoPE pair is a separable frequency channel of the attention logit -- the pair's
    contribution to l_ij depends on the positions only through omega . (r_i - r_j) -- so its scale
    is set by E[|q_p|^2] * E[|k_p|^2]. With C = E[x x^T] at the block input those are
    trace(W^T C W) over the pair's two columns; without C it degrades to plain weight energy.
    """
    h, qd = block.num_heads, block.q_head_dim
    pairs = qd // 2
    c_in = block.q_proj.in_channels
    w_q = block.q_proj.weight.astype(np.float64).reshape(c_in, h, pairs, 2)
    w_k = block.k_proj.weight.astype(np.float64).reshape(c_in, h, pairs, 2)
    c = np.eye(c_in) if second_moment is None else second_moment.astype(np.float64)
    energy_q = np.einsum("chpi,cd,dhpi->hp", w_q, c, w_q)
    energy_k = np.einsum("chpi,cd,dhpi->hp", w_k, c, w_k)
    return energy_q * energy_k


def drop_rope_pairs(block: TransformerAttentionBlock, keep_pairs: int,
                    second_moment: np.ndarray | None = None) -> PruneRecord:
    """Shrink `q_head_dim` by keeping only the highest-energy RoPE frequency pairs.

    The query path cannot be factorized as freely as the value path: RoPE rotates the interleaved
    channel pairs (2p, 2p+1) by position-dependent angles, so an arbitrary change of basis would
    break the correspondence between a dimension and its learned frequency. What *is* free is
    choosing which frequencies to keep -- each pair is an independent additive term in the logit --
    and dropping a pair takes its two columns of `q_proj`/`k_proj` and its row of `rope_freqs` with
    it. `q_head_dim` is a header field, so the result stays in-format.

    Note this changes the attention softmax scale (1/sqrt(q_head_dim)), which the engine derives
    from the header -- the surviving logits are therefore rescaled, not merely a subset.
    """
    if block.num_kv_heads != block.num_heads:
        raise PruneError(f"{block.name}: grouped-query attention is not supported")
    if not block.use_rope or not block.learnable_rope:
        raise PruneError(f"{block.name}: only learnable-RoPE attention has pair structure to drop")
    h, qd = block.num_heads, block.q_head_dim
    pairs = qd // 2
    if not 0 < keep_pairs < pairs:
        raise PruneError(f"{block.name}: keep_pairs must be in 1..{pairs - 1}, got {keep_pairs}")

    c_in = block.q_proj.in_channels
    scores = rope_pair_energy(block, second_moment)
    keep = np.sort(np.argsort(scores, axis=1)[:, ::-1][:, :keep_pairs], axis=1)  # (h, keep_pairs)

    def slice_pairs(weight: np.ndarray) -> np.ndarray:
        w = weight.reshape(c_in, h, pairs, 2)
        out = np.stack([w[:, i, keep[i], :] for i in range(h)], axis=1)
        return out.reshape(c_in, h * keep_pairs * 2)

    block.q_proj.weight = slice_pairs(block.q_proj.weight)
    block.q_proj.out_channels = h * keep_pairs * 2
    block.k_proj.weight = slice_pairs(block.k_proj.weight)
    block.k_proj.out_channels = h * keep_pairs * 2
    block.rope_freqs = np.stack([block.rope_freqs[i, keep[i], :] for i in range(h)], axis=0)
    block.q_head_dim = keep_pairs * 2
    kept = float(np.mean([scores[i, keep[i]].sum() / max(scores[i].sum(), 1e-30) for i in range(h)]))
    return PruneRecord("drop_rope_pairs",
                       f"{block.name}: q_head_dim {qd} -> {keep_pairs * 2} "
                       f"(kept {100 * kept:.1f}% of q/k energy)")


def drop_rope_pairs_everywhere(model: KataModel, keep_fraction: float,
                               moments: dict | None = None) -> list[PruneRecord]:
    records = []
    for _, block in iter_attention_blocks(model):
        pairs = block.q_head_dim // 2
        keep = max(1, int(round(pairs * keep_fraction)))
        if keep < pairs:
            moment = None if moments is None else moments.get(block.name)
            records.append(drop_rope_pairs(block, keep, moment))
    if not records:
        raise PruneError("no learnable-RoPE attention blocks found")
    return records


# --------------------------------------------------------------------------------------
# Not compression at all: flushing subnormal weights
# --------------------------------------------------------------------------------------


def flush_subnormal_weights(model: KataModel, threshold: float = 1.17549435e-38) -> PruneRecord:
    """Zero every weight smaller in magnitude than the smallest normal float32.

    Numerically this is nothing -- subnormals below ~1.2e-38 cannot affect a net whose
    activations are order 1. Computationally it can be everything: x86 handles subnormal operands
    in microcode, at roughly two orders of magnitude the cost of normal arithmetic, and KataGo's
    C++ never sets the SSE flush-to-zero / denormals-are-zero flags, so its CPU backend pays that
    cost on every one it meets.

    This is the cheapest possible edit -- it removes no capacity, changes no shape, and leaves a
    file the stock engine loads. Measure before assuming it matters: nets differ enormously in how
    many subnormals they carry.
    """
    flushed = total = 0
    for _, layer in model.iter_layers():
        weight = getattr(layer, "weight", None)
        if weight is None or not isinstance(weight, np.ndarray):
            continue
        small = (np.abs(weight) < threshold) & (weight != 0)
        flushed += int(small.sum())
        total += int(weight.size)
        weight[small] = 0.0
    return PruneRecord("flush_subnormal",
                       f"zeroed {flushed:,} subnormal weights of {total:,} "
                       f"({100 * flushed / max(total, 1):.4f}%)")
