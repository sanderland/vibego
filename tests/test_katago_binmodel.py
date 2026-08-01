"""The contract for `vibego/katago/binmodel.py` is byte-exactness: a net we edit has to remain a
file the stock KataGo engine loads, so anything we *didn't* touch must come back out identical.

The synthetic model below exercises every layer kind the reader knows, including the v17
transformer path (attention + SwiGLU FFN + learnable RoPE nested inside an `nbt` block), so the
round-trip is tested without needing a 40MB download. Point `VIBEGO_KATAGO_MODEL` at a real
`.bin.gz` to additionally assert exactness against a file KataGo itself wrote.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pytest

from vibego.katago.binmodel import (
    Activation,
    BatchNorm,
    Conv,
    GPoolBlock,
    KataModel,
    MatBias,
    MatMul,
    ModelFormatError,
    NestedBottleneckBlock,
    OrdinaryBlock,
    PolicyHead,
    RMSNorm,
    Trunk,
    TransformerAttentionBlock,
    TransformerFFNBlock,
    TransformerRMSNorm,
    ValueHead,
    model_bytes,
    read_model,
    read_model_bytes,
    write_model,
)
from vibego.katago.cost import arch_summary, model_cost

EPS = "1e-20"


def _rand(*shape) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(shape)) % (2**32))
    return rng.standard_normal(shape).astype("<f4")


def _conv(name, dy, dx, ic, oc) -> Conv:
    return Conv(name, dy, dx, ic, oc, 1, 1, _rand(dy, dx, ic, oc))


def _matmul(name, ic, oc) -> MatMul:
    return MatMul(name, ic, oc, _rand(ic, oc))


def _bn(name, c, has_scale=1) -> BatchNorm:
    return BatchNorm(name, c, EPS, has_scale, 1, _rand(c), np.abs(_rand(c)) + 1.0,
                     _rand(c) if has_scale else None, _rand(c))


def _act(name, kind="ACTIVATION_SILU") -> Activation:
    return Activation(name, kind)


def _ordinary(name, c) -> OrdinaryBlock:
    return OrdinaryBlock(name, _bn(f"{name}.n1", c), _act(f"{name}.a1"),
                         _conv(f"{name}.c1", 3, 3, c, c), _bn(f"{name}.n2", c),
                         _act(f"{name}.a2"), _conv(f"{name}.c2", 3, 3, c, c))


def _gpool(name, c, g) -> GPoolBlock:
    return GPoolBlock(name, _bn(f"{name}.n1", c), _act(f"{name}.a1"),
                      _conv(f"{name}.cr", 3, 3, c, c), _conv(f"{name}.cg", 3, 3, c, g),
                      _bn(f"{name}.ng", g), _act(f"{name}.ag"), _matmul(f"{name}.lg", 3 * g, c),
                      _bn(f"{name}.n2", c), _act(f"{name}.a2"), _conv(f"{name}.c2", 3, 3, c, c))


def _attention(name, c, heads, qd, vd, learnable_rope=True) -> TransformerAttentionBlock:
    blk = TransformerAttentionBlock(
        name, heads, heads, qd, vd, 1, 1 if learnable_rope else 0,
        TransformerRMSNorm(f"{name}.norm1", c, "1e-05", _rand(c)),
        _matmul(f"{name}.q_proj", c, heads * qd), _matmul(f"{name}.k_proj", c, heads * qd),
        _matmul(f"{name}.v_proj", c, heads * vd), _matmul(f"{name}.out_proj", heads * vd, c),
    )
    blk.rope_name = f"{name}.rope_freqs" if learnable_rope else f"{name}.rope_theta"
    if learnable_rope:
        blk.rope_freqs = _rand(heads, qd // 2, 2)
    else:
        blk.rope_theta = "10000.0"
    return blk


def _ffn(name, c, ffn) -> TransformerFFNBlock:
    return TransformerFFNBlock(name, c, ffn, 1,
                               TransformerRMSNorm(f"{name}.norm", c, "1e-05", _rand(c)),
                               _matmul(f"{name}.l1", c, ffn), _matmul(f"{name}.lg", c, ffn),
                               _matmul(f"{name}.l2", ffn, c))


def _nbt_transformer(name, c_trunk, c_mid, heads=2, qd=4, vd=4, ffn=16) -> NestedBottleneckBlock:
    inner = [_attention(f"{name}.blockstack.0", c_mid, heads, qd, vd),
             _ffn(f"{name}.blockstack.1", c_mid, ffn),
             _attention(f"{name}.blockstack.2", c_mid, heads, qd, vd),
             _ffn(f"{name}.blockstack.3", c_mid, ffn)]
    return NestedBottleneckBlock(
        name, len(inner), _bn(f"{name}.np", c_trunk), _act(f"{name}.ap"),
        _conv(f"{name}.cp", 1, 1, c_trunk, c_mid), inner,
        _bn(f"{name}.nq", c_mid), _act(f"{name}.aq"), _conv(f"{name}.cq", 1, 1, c_mid, c_trunk),
    )


def make_model(version: int = 17, c: int = 8, c_mid: int = 4) -> KataModel:
    """A miniature but structurally complete net covering every block kind."""
    blocks = [_ordinary("model.blocks.0", c), _gpool("model.blocks.1", c, 2),
              _nbt_transformer("model.blocks.2", c, c_mid)]
    trunk = Trunk(
        "trunk", len(blocks), c, c, c - 2, 2, 2,
        norm_kind=1, unused=[0] * 5,
        initial_conv=_conv("model.conv_spatial", 3, 3, 22, c),
        initial_matmul=_matmul("model.linear_global", 19, c),
        metadata_encoder=None, blocks=blocks,
        tip_norm=RMSNorm("model.norm_trunkfinal", c, "1e-05", 0, 0, _rand(c), _rand(c)),
        tip_act=_act("model.act_trunkfinal"),
    )
    policy = PolicyHead(
        "model.policy_head", 2, [0, 0, 0],
        _conv("p.conv1p", 1, 1, c, 4), _conv("p.conv1g", 1, 1, c, 4), _bn("p.biasg", 4, has_scale=0),
        _act("p.actg"), _matmul("p.linear_g", 12, 4), _bn("p.bias2", 4, has_scale=0), _act("p.act2"),
        _conv("p.conv2p", 1, 1, 4, 2), _matmul("p.linear_pass", 12, 4),
        MatBias("p.linear_pass_bias", 4, _rand(4)), _act("p.act_pass", "ACTIVATION_IDENTITY"),
        _matmul("p.linear_pass2", 4, 2),
    )
    value = ValueHead(
        "model.value_head", [0, 0, 0], _conv("v.conv1", 1, 1, c, 4), _bn("v.bias1", 4, has_scale=0),
        _act("v.act1"), _matmul("v.linear2", 12, 8), MatBias("v.bias2", 8, _rand(8)), _act("v.act2"),
        _matmul("v.linear_valuehead", 8, 3), MatBias("v.bias_valuehead", 3, _rand(3)),
        _matmul("v.linear_miscvaluehead", 8, 6), MatBias("v.bias_miscvaluehead", 6, _rand(6)),
        _conv("v.conv_ownership", 1, 1, c, 1),
    )
    return KataModel("test-net", version, 22, 19,
                     post_process=["20.0", "20.0", "20.0", "20.0", "40.0", "0.25", "30.0"],
                     meta_encoder_version=0, prefer_pass_alive=0, unused=[0] * 6,
                     trunk=trunk, policy_head=policy, value_head=value)


def test_roundtrip_is_byte_exact():
    model = make_model()
    raw = model_bytes(model)
    assert model_bytes(read_model_bytes(raw)) == raw


def test_roundtrip_preserves_structure():
    model = read_model_bytes(model_bytes(make_model()))
    assert model.version == 17
    assert model.block_kinds() == ["ordinary_block", "gpool_block", "nested_bottleneck_block"]
    nbt = model.trunk.blocks[2]
    assert [b.kind for b in nbt.blocks] == [
        "transformer_attention_block", "transformer_ffn_block",
        "transformer_attention_block", "transformer_ffn_block",
    ]
    attn = nbt.blocks[0]
    assert attn.learnable_rope and attn.rope_freqs.shape == (2, 2, 2)
    assert nbt.blocks[1].use_swiglu and nbt.blocks[1].linear_gate is not None


def test_edited_weights_survive_the_roundtrip():
    model = make_model()
    nbt = model.trunk.blocks[2]
    nbt.blocks[1].linear2.weight[:] = 0.0
    reloaded = read_model_bytes(model_bytes(model))
    assert np.all(reloaded.trunk.blocks[2].blocks[1].linear2.weight == 0.0)


def test_older_version_omits_v15_and_v17_fields():
    """A v14 file has no metadata/trunk-norm/policy-out fields and no second pass matmul --
    the reader must not consume tokens that aren't there. It does still carry the v13
    post-process multipliers."""
    model = make_model(version=14)
    model.trunk.norm_kind = 0
    model.trunk.tip_norm = _bn("model.norm_trunkfinal", 8)
    raw = model_bytes(model)
    reloaded = read_model_bytes(raw)
    assert reloaded.version == 14
    assert reloaded.post_process == model.post_process
    assert reloaded.policy_head.gpool_to_pass_bias is None
    assert model_bytes(reloaded) == raw
    # Policy out channels are implied by version, not stored, below v17.
    assert reloaded.policy_out_channels() == 2
    assert make_model(version=11).policy_out_channels() == 1


def test_pre_v11_activations_carry_no_kind_token():
    model = make_model(version=8)
    model.trunk.norm_kind = 0
    model.trunk.tip_norm = _bn("model.norm_trunkfinal", 8)
    for act in [model.trunk.tip_act, model.trunk.blocks[0].pre_act]:
        act.kind = None
    # v8 has no transformer blocks in the wild, but the activation encoding is what differs.
    model.trunk.blocks = model.trunk.blocks[:2]
    model.trunk.num_blocks = 2
    for block in model.trunk.blocks:
        for layer in block.layers():
            if isinstance(layer, Activation):
                layer.kind = None
    for layer in model.policy_head.layers() + model.value_head.layers():
        if isinstance(layer, Activation):
            layer.kind = None
    raw = model_bytes(model)
    assert b"ACTIVATION_" not in raw
    assert model_bytes(read_model_bytes(raw)) == raw


def test_truncated_file_is_rejected():
    raw = model_bytes(make_model())
    with pytest.raises(ModelFormatError):
        read_model_bytes(raw[: len(raw) // 2])


def test_write_model_gzip_roundtrip(tmp_path):
    model = make_model()
    path = str(tmp_path / "m.bin.gz")
    write_model(model, path)
    with gzip.open(path, "rb") as f:
        assert f.read() == model_bytes(model)
    assert model_bytes(read_model(path)) == model_bytes(model)


def test_param_count_matches_sum_of_layers():
    model = make_model()
    cost = model_cost(model, board=19)
    assert cost.total.params == model.num_parameters()
    # RoPE frequencies are real parameters and must not be silently dropped.
    rope = sum(b.rope_freqs.size for b in model.trunk.blocks[2].blocks
               if isinstance(b, TransformerAttentionBlock))
    assert rope == 2 * 2 * 2 * 2
    assert model.num_parameters() > rope


def test_attention_flops_scale_with_board_area_squared():
    model = make_model()
    small = model_cost(model, board=5)
    big = model_cost(model, board=10)
    # 4x the positions -> 16x the N^2 attention matmuls.
    assert big.attention_flops == 16 * small.attention_flops


def test_arch_summary_names_the_shape():
    summary = arch_summary(make_model())
    assert summary.startswith("b3c8h2nbt")


REAL_MODEL = os.environ.get("VIBEGO_KATAGO_MODEL")


@pytest.mark.skipif(not REAL_MODEL, reason="set VIBEGO_KATAGO_MODEL to a real KataGo .bin.gz")
def test_real_model_roundtrips_byte_exactly():
    """The one test that proves we agree with KataGo's own exporter rather than just ourselves."""
    with gzip.open(REAL_MODEL, "rb") as f:
        original = f.read()
    assert model_bytes(read_model(REAL_MODEL)) == original
