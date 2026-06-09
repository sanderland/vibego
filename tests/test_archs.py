"""Tests for the model-architecture registry and checkpoint-config migration."""
import dataclasses

import pytest
import torch

from nanogo.go.features import NUM_GLOBAL, NUM_SPATIAL
from nanogo.net.model import ARCHS, GPoolResBlock, Model, ModelConfig, NBTResBlock, ResBlock, arch_config


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
