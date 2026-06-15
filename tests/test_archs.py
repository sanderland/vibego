"""Tests for the model-architecture registry and checkpoint-config migration."""
import dataclasses

import pytest
import torch

from vibego.go.features import NUM_GLOBAL, NUM_SPATIAL
from vibego.net.model import (
    ARCHS, GlobalModBlock, GPoolResBlock, LinAttnResBlock, Model, ModelConfig, NBTResBlock,
    PatternEmbed, ResBlock, RWKVResBlock, _build_3x3_canon, _q_shift, arch_config,
)


def _block_types(config):
    return [type(b).__name__ for b in Model(config).blocks]


def test_registry_archs_build():
    for name in ARCHS:
        cfg = arch_config(name)
        model = Model(cfg)
        s = torch.zeros(1, NUM_SPATIAL, 19, 19)
        g = torch.zeros(1, NUM_GLOBAL)
        out = model(s, g)
        assert out[0].shape == (1, 19 * 19 + 1)


def test_gpool_arch_places_gpool_blocks():
    plain = _block_types(arch_config("b6c96"))
    gpool = _block_types(arch_config("b6c96-gpool"))
    assert plain == ["ResBlock"] * 6
    assert gpool == ["ResBlock", "ResBlock", "GPoolResBlock",
                     "ResBlock", "ResBlock", "GPoolResBlock"]


def test_nbt_arch_places_nbt_and_gpool_blocks():
    nbt = _block_types(arch_config("b6c96nbt"))
    assert nbt == ["NBTResBlock", "NBTResBlock", "GPoolResBlock",
                   "NBTResBlock", "NBTResBlock", "GPoolResBlock"]


def test_nbt_block_bottlenecks_and_is_residual():
    # The 3x3 convs run at the bottleneck width (c//2), and a zero-input passes through the
    # outer residual unchanged (a sanity check that it really is x + f(x)).
    block = NBTResBlock(96).eval()
    assert block.conv_in.out_channels == 48
    assert block.n1_conv1.in_channels == 48 and block.n1_conv1.out_channels == 48
    assert block.conv_out.out_channels == 96
    x = torch.zeros(1, 96, 9, 9)
    assert torch.allclose(block(x), x)


def test_nbt_is_smaller_per_block_than_regular():
    # The whole point of nbt: more conv depth per parameter -> fewer params at equal b/c label.
    nparams = lambda name: sum(p.numel() for p in Model(arch_config(name)).parameters())
    assert nparams("b6c96nbt") < nparams("b6c96-gpool")


def test_global_mixing_archs_place_their_block():
    # The global-mixing study swaps only the every-3rd slot; conv backbone stays regular.
    assert _block_types(arch_config("b6c96-linat")) == [
        "ResBlock", "ResBlock", "LinAttnResBlock",
        "ResBlock", "ResBlock", "LinAttnResBlock"]
    assert _block_types(arch_config("b6c96-rwkv")) == [
        "ResBlock", "ResBlock", "RWKVResBlock",
        "ResBlock", "ResBlock", "RWKVResBlock"]


@pytest.mark.parametrize("block", [LinAttnResBlock(64), RWKVResBlock(64)])
def test_global_mixing_blocks_are_residual_and_shape_preserving(block):
    block = block.eval()
    x = torch.randn(2, 64, 9, 9)
    out = block(x)
    assert out.shape == x.shape
    # Both blocks are wrapped in outer residuals built from convs with no bias and zeroed maps
    # only at init? They aren't zero-init, so just assert finiteness + that a zero input passes
    # (degenerate softmax over zeros is uniform, FFN of zero is zero) close to zero.
    assert torch.isfinite(out).all()
    z = torch.zeros(2, 64, 9, 9)
    assert torch.allclose(block(z), z, atol=1e-5)


def test_linattn_global_mixing_moves_information():
    # A single hot pixel must influence *other* positions (global mixing, not a local conv).
    torch.manual_seed(0)
    block = LinAttnResBlock(32).eval()
    x = torch.zeros(1, 32, 7, 7)
    x[0, :, 3, 3] = 5.0
    out = block(x) - x
    far = out[0, :, 0, 0].abs().sum().item()
    assert far > 0  # corner changed despite the spike being at the centre


def test_q_shift_moves_each_quarter_one_pixel():
    c = 8
    x = torch.arange(7 * 7, dtype=torch.float32).reshape(1, 1, 7, 7).repeat(1, c, 1, 1)
    sh = _q_shift(x)
    q = c // 4
    # group 0 takes the right neighbour: sh[...,j] == x[...,j+1]
    assert torch.equal(sh[0, 0, :, :-1], x[0, 0, :, 1:])
    # group 2 takes the lower neighbour: sh[...,i,:] == x[...,i+1,:]
    assert torch.equal(sh[0, 2 * q, :-1, :], x[0, 2 * q, 1:, :])


def test_linattn_requires_divisible_heads():
    with pytest.raises(ValueError):
        LinAttnResBlock(30, heads=4)


def test_globmod_arch_places_block_and_mixes_globally():
    assert _block_types(arch_config("b6c96-globmod")) == [
        "ResBlock", "ResBlock", "GlobalModBlock",
        "ResBlock", "ResBlock", "GlobalModBlock"]
    block = GlobalModBlock(32).eval()
    x = torch.zeros(1, 32, 7, 7)
    x[0, :, 3, 3] = 5.0
    out = block(x)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert (out - x)[0, :, 0, 0].abs().sum() > 0  # global summary modulates a far corner


def test_pattern_canon_is_dihedral_canonical():
    canon, k = _build_3x3_canon()
    assert k == 2862                             # exact D4-orbit count of base-3 3x3 (Burnside)
    # a single own stone at the four 3x3 corners is one symmetry class (cells 0,2,6,8)
    assert len({int(canon[3 ** i]) for i in (0, 2, 6, 8)}) == 1
    assert canon[3 ** 0] != canon[2 * 3 ** 0]    # own corner vs opp corner: no colour symmetry
    assert canon[3 ** 4] != canon[3 ** 0]        # centre-own is a distinct class from corner-own


def test_pattern_embed_zero_init_then_local_sensitivity():
    pe = PatternEmbed(8)
    # zero-init -> the lookup contributes nothing until trained (clean ablation), for any input
    assert torch.count_nonzero(pe(torch.zeros(1, NUM_SPATIAL, 5, 5))) == 0
    torch.nn.init.normal_(pe.embed.weight)
    sp = torch.zeros(1, NUM_SPATIAL, 5, 5)
    base = pe(sp)
    sp2 = sp.clone(); sp2[0, 1, 2, 2] = 1.0      # own stone at centre (channel 1 = own)
    diff = (pe(sp2) - base).abs().sum(1)[0]      # (5,5)
    assert diff[2, 2] > 0 and diff[1, 1] > 0 and diff[3, 3] > 0  # whole 3x3 neighbourhood shifts
    assert diff[0, 0] == 0                        # a cell whose window excludes (2,2) is unchanged


def test_pattern_embed_arch_builds_and_config_roundtrips():
    cfg = arch_config("b6c96-gpool-pat")
    assert cfg.pattern_embed is True
    model = Model(cfg)
    assert model.pattern_embed is not None
    out = model(torch.zeros(1, NUM_SPATIAL, 19, 19), torch.zeros(1, NUM_GLOBAL))
    assert out[0].shape == (1, 19 * 19 + 1)
    restored = ModelConfig(**dataclasses.asdict(cfg))
    assert restored == cfg and restored.pattern_embed is True


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        arch_config("does-not-exist")


def test_unknown_block_kind_raises():
    with pytest.raises(ValueError):
        Model(ModelConfig(block_kinds=("regular", "frobnicate")))


def test_config_roundtrips_through_dict():
    # How checkpoints persist/restore configs: asdict() -> ModelConfig(**dict).
    cfg = arch_config("b10c128-gpool")
    restored = ModelConfig(**dataclasses.asdict(cfg))
    assert restored == cfg
