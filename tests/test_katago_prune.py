"""Pruning has to produce a file the *stock engine* accepts, so the invariants that matter are
KataGo's own load-time checks: projection shapes must agree with the head counts, FFN matrices
with the hidden width, and the block stack with the declared block count. These tests assert them
after each edit, plus the exactness claim -- pruning a head must equal zeroing it.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibego.katago.binmodel import (
    NestedBottleneckBlock,
    TransformerAttentionBlock,
    TransformerFFNBlock,
    model_bytes,
    read_model_bytes,
)
from vibego.katago.cost import model_cost
from vibego.katago.prune import (
    PruneError,
    drop_blocks,
    drop_inner_pairs,
    narrow_ffn,
    narrow_ffn_everywhere,
    prune_heads,
    prune_heads_everywhere,
    sanitize_name,
)
from tests.test_katago_binmodel import make_model


def _attn(model) -> TransformerAttentionBlock:
    return model.trunk.blocks[2].blocks[0]


def _ffn(model) -> TransformerFFNBlock:
    return model.trunk.blocks[2].blocks[1]


def _check_engine_invariants(model) -> None:
    """The shape agreements KataGo's desc.cpp throws on at load time."""
    trunk = model.trunk
    assert trunk.num_blocks == len(trunk.blocks)
    for block in trunk.blocks:
        if isinstance(block, NestedBottleneckBlock):
            assert block.num_blocks == len(block.blocks)
            c_mid = block.pre_conv.out_channels
            assert block.post_conv.out_channels == trunk.trunk_channels
            for sub in block.blocks:
                _check_sub(sub, c_mid)


def _check_sub(sub, c_mid: int) -> None:
    if isinstance(sub, TransformerAttentionBlock):
        assert sub.num_heads % sub.num_kv_heads == 0
        assert sub.q_proj.in_channels == c_mid
        assert sub.q_proj.out_channels == sub.num_heads * sub.q_head_dim
        assert sub.k_proj.out_channels == sub.num_kv_heads * sub.q_head_dim
        assert sub.v_proj.out_channels == sub.num_kv_heads * sub.v_head_dim
        assert sub.out_proj.in_channels == sub.num_heads * sub.v_head_dim
        assert sub.out_proj.out_channels == c_mid
        assert sub.q_head_dim % 2 == 0  # RoPE rotates interleaved pairs
        if sub.rope_freqs is not None:
            assert sub.rope_freqs.shape == (sub.num_kv_heads, sub.q_head_dim // 2, 2)
    elif isinstance(sub, TransformerFFNBlock):
        assert sub.num_channels == c_mid
        assert sub.linear1.in_channels == c_mid
        assert sub.linear1.out_channels == sub.ffn_channels
        assert sub.linear2.in_channels == sub.ffn_channels
        assert sub.linear2.out_channels == c_mid
        if sub.linear_gate is not None:
            assert sub.linear_gate.out_channels == sub.ffn_channels


def test_head_prune_keeps_the_file_loadable():
    model = make_model()
    prune_heads(_attn(model), 1)
    _check_engine_invariants(model)
    reloaded = read_model_bytes(model_bytes(model))
    _check_engine_invariants(reloaded)
    assert _attn(reloaded).num_heads == 1


def test_head_prune_equals_zeroing_that_head():
    """A pruned head must be *exactly* the head removed, not an approximation: the surviving
    q/k/v/out slices have to be the same numbers, in the same order."""
    model = make_model()
    attn = _attn(model)
    before_v = attn.v_proj.weight.copy()
    before_out = attn.out_proj.weight.copy()
    vd = attn.v_head_dim
    from vibego.katago.prune import head_importance

    keep = int(np.argmax(head_importance(attn)))
    prune_heads(attn, 1)
    assert np.array_equal(attn.v_proj.weight, before_v[:, keep * vd:(keep + 1) * vd])
    assert np.array_equal(attn.out_proj.weight, before_out[keep * vd:(keep + 1) * vd, :])


def test_ffn_narrow_keeps_the_file_loadable():
    model = make_model()
    narrow_ffn(_ffn(model), 4)
    _check_engine_invariants(model)
    reloaded = read_model_bytes(model_bytes(model))
    assert _ffn(reloaded).ffn_channels == 4
    assert _ffn(reloaded).linear2.in_channels == 4


def test_drop_blocks_updates_the_declared_count():
    model = make_model()
    drop_blocks(model, [0])
    assert model.trunk.num_blocks == 2
    reloaded = read_model_bytes(model_bytes(model))
    assert reloaded.block_kinds() == ["gpool_block", "nested_bottleneck_block"]
    _check_engine_invariants(reloaded)


def test_drop_inner_pairs_updates_the_declared_count():
    model = make_model()
    drop_inner_pairs(model, 2, 1)
    assert len(model.trunk.blocks[2].blocks) == 2
    reloaded = read_model_bytes(model_bytes(model))
    _check_engine_invariants(reloaded)


def test_pruning_actually_removes_flops():
    model = make_model()
    before = model_cost(model)
    narrow_ffn_everywhere(model, 0.5)
    prune_heads_everywhere(model, 0.5)
    after = model_cost(model)
    assert after.total.flops < before.total.flops
    assert after.total.params < before.total.params


def test_refuses_impossible_requests():
    model = make_model()
    with pytest.raises(PruneError):
        drop_blocks(model, [0, 1, 2])  # everything
    with pytest.raises(PruneError):
        drop_blocks(model, [99])
    with pytest.raises(PruneError):
        prune_heads(_attn(model), 0)
    with pytest.raises(PruneError):
        narrow_ffn(_ffn(model), _ffn(model).ffn_channels)  # not actually narrower
    with pytest.raises(PruneError):
        drop_inner_pairs(model, 0, 1)  # not an nbt block
    with pytest.raises(PruneError):
        drop_inner_pairs(model, 2, 2)  # would empty the stack


def test_grouped_query_attention_is_refused_rather_than_corrupted():
    model = make_model()
    attn = _attn(model)
    attn.num_kv_heads = 1  # pretend GQA; q heads would share this kv head
    with pytest.raises(PruneError, match="grouped-query"):
        prune_heads(attn, 1)


def test_model_names_stay_engine_legal():
    name = sanitize_name("kata1-b18c384nbt-s999/d444", "ffn0.75")
    assert all(c.isalnum() or c in "_-" for c in name)
    assert len(name) <= 96


def test_flushing_subnormals_changes_nothing_a_net_can_notice():
    """Subnormals are below 1.2e-38; zeroing them must be numerically inert while removing the
    operands that make x86 fall into microcode."""
    import numpy as np

    from vibego.katago.prune import flush_subnormal_weights

    model = make_model()
    ffn = model.trunk.blocks[2].blocks[1]
    ffn.linear1.weight[0, 0] = 1e-40      # subnormal
    ffn.linear1.weight[0, 1] = 1e-30      # normal, must survive
    record = flush_subnormal_weights(model)
    assert ffn.linear1.weight[0, 0] == 0.0
    assert ffn.linear1.weight[0, 1] == np.float32(1e-30)
    assert "1 subnormal" in record.detail or "zeroed" in record.detail


def test_flushing_subnormals_leaves_shapes_and_the_file_untouched():
    from vibego.katago.binmodel import model_bytes, read_model_bytes
    from vibego.katago.cost import model_cost
    from vibego.katago.prune import flush_subnormal_weights

    model = make_model()
    before = model_cost(model)
    flush_subnormal_weights(model)
    after = model_cost(model)
    assert (after.total.params, after.total.flops) == (before.total.params, before.total.flops)
    _check_engine_invariants(read_model_bytes(model_bytes(model)))
