"""Params and FLOPs/eval for a KataGo `.bin.gz`, broken down per trunk block.

The FLOPs axis here is the same one the rest of the lab plots against (`vibego/net/flops.py`,
`scripts/bench_net.py`): **FLOPs = MACs x 2**, at a fixed board size. Counting it straight off the
weight file means a released net can be placed on the FLOPs<->Elo plane without a torch
reimplementation, and -- more to the point for compression work -- the per-block breakdown says
where the FLOPs actually *are*, which is what decides what is worth pruning.

Counted: convs, matmuls, and attention's two batched matmuls (QK^T and AV). Not counted: norms,
activations, softmax, RoPE rotation, pooling. Those are memory-bound elementwise work, a low
single-digit fraction of MACs, and they do not shift the comparison between two nets of the same
family -- but they are one reason measured ms/eval is not proportional to this number.

A matmul is charged per-position (x N) when it runs on every board point (all transformer
projections and FFNs), and once when it runs on a pooled/global vector (gpool->bias, the pass
logit, the value head).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .binmodel import (
    Conv,
    GPoolBlock,
    KataModel,
    MatMul,
    NestedBottleneckBlock,
    OrdinaryBlock,
    TransformerAttentionBlock,
    TransformerFFNBlock,
)


@dataclass
class Cost:
    params: int = 0
    flops: int = 0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(self.params + other.params, self.flops + other.flops)


@dataclass
class BlockCost:
    index: int
    kind: str
    detail: str  # short human-readable shape summary
    cost: Cost


@dataclass
class ModelCost:
    board: int
    total: Cost
    stem: Cost
    trunk: Cost
    policy_head: Cost
    value_head: Cost
    blocks: list[BlockCost] = field(default_factory=list)
    attention_flops: int = 0  # of trunk flops, the part that is QK^T/AV (scales as N^2)


def _conv(layer: Conv, n: int) -> Cost:
    macs = n * layer.in_channels * layer.out_channels * layer.diam_y * layer.diam_x
    return Cost(layer.num_parameters(), 2 * macs)


def _matmul(layer: MatMul, n: int) -> Cost:
    """`n` = number of positions it runs on: the board area, or 1 for pooled/global vectors."""
    return Cost(layer.num_parameters(), 2 * n * layer.in_channels * layer.out_channels)


def _params_only(*layers) -> Cost:
    return Cost(sum(layer.num_parameters() for layer in layers), 0)


def _block_cost(block, n: int) -> tuple[Cost, int, str]:
    """Returns (cost, attention_flops, detail)."""
    if isinstance(block, OrdinaryBlock):
        c = (_conv(block.regular_conv, n) + _conv(block.final_conv, n)
             + _params_only(block.pre_norm, block.mid_norm))
        return c, 0, f"c{block.regular_conv.out_channels}"

    if isinstance(block, GPoolBlock):
        c = (_conv(block.regular_conv, n) + _conv(block.gpool_conv, n)
             + _matmul(block.gpool_to_bias_mul, 1) + _conv(block.final_conv, n)
             + _params_only(block.pre_norm, block.gpool_norm, block.mid_norm))
        return c, 0, f"c{block.regular_conv.out_channels}+g{block.gpool_conv.out_channels}"

    if isinstance(block, TransformerAttentionBlock):
        c = (_matmul(block.q_proj, n) + _matmul(block.k_proj, n) + _matmul(block.v_proj, n)
             + _matmul(block.out_proj, n) + _params_only(block.pre_norm))
        # QK^T over all key positions, then the value-weighted sum. Both are N^2 in board area.
        attn = 2 * n * n * block.num_heads * block.q_head_dim
        attn += 2 * n * n * block.num_heads * block.v_head_dim
        c.flops += attn
        if block.rope_freqs is not None:
            c.params += int(block.rope_freqs.size)
        kv = "" if block.num_kv_heads == block.num_heads else f"kv{block.num_kv_heads}"
        detail = f"h{block.num_heads}{kv} q{block.q_head_dim} v{block.v_head_dim}"
        return c, attn, detail

    if isinstance(block, TransformerFFNBlock):
        c = _matmul(block.linear1, n) + _matmul(block.linear2, n) + _params_only(block.pre_norm)
        if block.linear_gate is not None:
            c = c + _matmul(block.linear_gate, n)
        glu = "swiglu" if block.use_swiglu else "ffn"
        return c, 0, f"{glu}{block.ffn_channels}"

    if isinstance(block, NestedBottleneckBlock):
        c = (_conv(block.pre_conv, n) + _conv(block.post_conv, n)
             + _params_only(block.pre_norm, block.post_norm))
        attn = 0
        inner_details = []
        for sub in block.blocks:
            sub_cost, sub_attn, sub_detail = _block_cost(sub, n)
            c = c + sub_cost
            attn += sub_attn
            inner_details.append(sub_detail)
        kinds = [b.kind for b in block.blocks]
        inner = f"{len(block.blocks)}x[{inner_details[0] if inner_details else ''}"
        if len(set(kinds)) > 1 and len(inner_details) > 1:
            inner += f" {inner_details[1]}"
        inner += "]"
        return c, attn, f"bottleneck c{block.bottleneck_channels} {inner}"

    raise TypeError(f"unknown block type {type(block)}")


def model_cost(model: KataModel, board: int = 19) -> ModelCost:
    n = board * board

    stem = _conv(model.trunk.initial_conv, n) + _matmul(model.trunk.initial_matmul, 1)
    if model.trunk.metadata_encoder is not None:
        meta = model.trunk.metadata_encoder
        stem = stem + _matmul(meta.mul1, 1) + _matmul(meta.mul2, 1) + _matmul(meta.mul3, 1)
        stem = stem + _params_only(meta.bias1, meta.bias2)

    trunk = Cost()
    blocks: list[BlockCost] = []
    attention_flops = 0
    for i, block in enumerate(model.trunk.blocks):
        cost, attn, detail = _block_cost(block, n)
        trunk = trunk + cost
        attention_flops += attn
        blocks.append(BlockCost(i, block.kind, detail, cost))
    trunk = trunk + _params_only(model.trunk.tip_norm)

    ph = model.policy_head
    policy = (_conv(ph.p1_conv, n) + _conv(ph.g1_conv, n) + _matmul(ph.gpool_to_bias_mul, 1)
              + _conv(ph.p2_conv, n) + _matmul(ph.gpool_to_pass_mul, 1)
              + _params_only(ph.g1_norm, ph.p1_norm))
    if ph.gpool_to_pass_mul2 is not None:
        policy = policy + _matmul(ph.gpool_to_pass_mul2, 1) + _params_only(ph.gpool_to_pass_bias)

    vh = model.value_head
    value = (_conv(vh.v1_conv, n) + _matmul(vh.v2_mul, 1) + _matmul(vh.v3_mul, 1)
             + _matmul(vh.sv3_mul, 1) + _conv(vh.ownership_conv, n)
             + _params_only(vh.v1_norm, vh.v2_bias, vh.v3_bias, vh.sv3_bias))

    total = stem + trunk + policy + value
    return ModelCost(board, total, stem, trunk, policy, value, blocks, attention_flops)


def arch_summary(model: KataModel) -> str:
    """A short KataGo-style name for what is actually in the file, e.g. `b10c384h6nbt-tf`."""
    trunk = model.trunk
    kinds = model.block_kinds()
    parts = [f"b{trunk.num_blocks}", f"c{trunk.trunk_channels}"]

    heads = set()
    nbt_internal = set()
    has_nbt = has_attn = has_ffn = has_gpool = False
    learnable_rope = fixed_rope = False

    def scan(block):
        nonlocal has_nbt, has_attn, has_ffn, has_gpool, learnable_rope, fixed_rope
        if isinstance(block, NestedBottleneckBlock):
            has_nbt = True
            # KataGo names these by internal_length: a transformer-inner nbt block holds
            # 2*internal_length sub-blocks (attention/FFN pairs), and `nbt` with no number
            # means internal_length 2.
            inner_attn = sum(1 for b in block.blocks
                             if isinstance(b, TransformerAttentionBlock))
            nbt_internal.add(inner_attn if inner_attn else len(block.blocks))
            for sub in block.blocks:
                scan(sub)
        elif isinstance(block, TransformerAttentionBlock):
            has_attn = True
            heads.add(block.num_heads)
            if block.use_rope:
                if block.learnable_rope:
                    learnable_rope = True
                else:
                    fixed_rope = True
        elif isinstance(block, TransformerFFNBlock):
            has_ffn = True

    for block in trunk.blocks:
        scan(block)
    # Only a *top-level* gpool block earns the suffix -- nbt blocks routinely carry gpool
    # sub-blocks, and KataGo does not name those (kata1-b18c384nbt, not ...nbtgpool).
    has_gpool = any(isinstance(b, GPoolBlock) for b in trunk.blocks)

    if heads:
        parts.append("h" + "/".join(str(h) for h in sorted(heads)))
    if has_nbt:
        lengths = sorted(nbt_internal)
        suffix = "" if lengths == [2] else "/".join(str(x) for x in lengths)
        parts.append(f"nbt{suffix}")
    if has_attn and has_ffn:
        parts.append("tf")
    if learnable_rope:
        parts.append("lrs")
    elif fixed_rope:
        parts.append("rs")
    if has_gpool and not has_attn:
        parts.append("gpool")
    name = "".join(parts)
    return f"{name} ({len(kinds)} top-level blocks: " + \
        ", ".join(f"{kinds.count(k)}x{k}" for k in dict.fromkeys(kinds)) + ")"
