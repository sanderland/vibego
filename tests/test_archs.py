"""Tests for the model-architecture registry and checkpoint-config migration."""
import dataclasses

import pytest
import torch

from nanogo.go.features import NUM_GLOBAL, NUM_SPATIAL
from nanogo.net.model import ARCHS, GPoolResBlock, Model, ModelConfig, ResBlock, arch_config


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
