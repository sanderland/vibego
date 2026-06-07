"""The network: a small pre-activation ResNet trunk with four heads
(policy, value, score, ownership). Kept in one file, nanoGPT-style.

Input: spatial (B, num_spatial, P, P) + global (B, num_global).
Outputs:
  policy:    (B, P*P + 1)         move logits (last entry = pass)
  value:     (B, 3)              win / loss / no-result logits
  score:     (B,)               predicted score lead (points, from side-to-move POV)
  ownership: (B, 1, P, P)        tanh in [-1, 1] (side-to-move owns positive)
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from ..go.features import NUM_GLOBAL, NUM_SPATIAL


@dataclass
class ModelConfig:
    """A concrete architecture. The trunk is described by `block_kinds` (one entry per
    residual block) rather than ad-hoc boolean flags, so new block types can be added without
    growing the config surface. Pick named architectures via the ARCHS registry / arch_config."""
    pos_len: int = 19
    num_spatial: int = NUM_SPATIAL
    num_global: int = NUM_GLOBAL
    channels: int = 96
    head_channels: int = 32
    gpool_channels: int = 32
    block_kinds: tuple[str, ...] = ("regular",) * 6

    @property
    def num_blocks(self) -> int:
        return len(self.block_kinds)

    @classmethod
    def plain(cls, channels: int, num_blocks: int, **kw) -> "ModelConfig":
        """An all-regular trunk (handy for tiny test models)."""
        return cls(channels=channels, block_kinds=("regular",) * num_blocks, **kw)


def global_pool(x: torch.Tensor) -> torch.Tensor:
    """Mean + max over spatial dims -> (B, 2C)."""
    mean = x.mean(dim=(2, 3))
    mx = x.amax(dim=(2, 3))
    return torch.cat([mean, mx], dim=1)


class ResBlock(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, c: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.conv1(Fnn.relu(self.bn1(x)))
        h = self.conv2(Fnn.relu(self.bn2(h)))
        return x + h


class GPoolResBlock(nn.Module):
    """Residual block with a global-pooling bias path (KataGo's key ingredient): part of the
    conv output is globally pooled and broadcast back as a per-channel bias, so the block can
    use whole-board context (score, who's ahead, large-group status)."""

    def __init__(self, c: int, cg: int = 32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.conv_reg = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.conv_pool = nn.Conv2d(c, cg, 3, padding=1, bias=False)
        self.bn_pool = nn.BatchNorm2d(cg)
        self.fc_pool = nn.Linear(2 * cg, c)
        self.bn2 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        h = Fnn.relu(self.bn1(x))
        reg = self.conv_reg(h)
        pool = Fnn.relu(self.bn_pool(self.conv_pool(h)))
        reg = reg + self.fc_pool(global_pool(pool)).unsqueeze(-1).unsqueeze(-1)
        out = self.conv2(Fnn.relu(self.bn2(reg)))
        return x + out


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config.channels
        self.stem = nn.Conv2d(config.num_spatial, c, 3, padding=1, bias=False)
        self.global_fc = nn.Linear(config.num_global, c)
        blocks = []
        for kind in config.block_kinds:
            if kind == "regular":
                blocks.append(ResBlock(c))
            elif kind == "gpool":
                blocks.append(GPoolResBlock(c, config.gpool_channels))
            else:
                raise ValueError(f"unknown block kind {kind!r}")
        self.blocks = nn.ModuleList(blocks)
        self.trunk_bn = nn.BatchNorm2d(c)

        hc = config.head_channels
        # Policy head
        self.p_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(hc)
        self.p_out = nn.Conv2d(hc, 1, 1)
        self.p_pass = nn.Linear(2 * hc, 1)
        # Value + score head (from global pooling)
        self.v_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(hc)
        self.v_fc1 = nn.Linear(2 * hc, hc)
        self.v_value = nn.Linear(hc, 3)
        self.v_score = nn.Linear(hc, 1)
        # Ownership head
        self.o_conv = nn.Conv2d(c, hc, 1, bias=False)
        self.o_bn = nn.BatchNorm2d(hc)
        self.o_out = nn.Conv2d(hc, 1, 1)

    def forward(self, spatial: torch.Tensor, glob: torch.Tensor):
        B = spatial.shape[0]
        x = self.stem(spatial) + self.global_fc(glob).view(B, -1, 1, 1)
        for block in self.blocks:
            x = block(x)
        x = Fnn.relu(self.trunk_bn(x))

        # Policy
        p = Fnn.relu(self.p_bn(self.p_conv(x)))
        p_spatial = self.p_out(p).flatten(1)  # (B, P*P)
        p_pass = self.p_pass(global_pool(p))  # (B, 1)
        policy = torch.cat([p_spatial, p_pass], dim=1)  # (B, P*P+1)

        # Value + score
        v = Fnn.relu(self.v_bn(self.v_conv(x)))
        v = Fnn.relu(self.v_fc1(global_pool(v)))
        value = self.v_value(v)  # (B, 3)
        score = self.v_score(v).squeeze(1) * 20.0  # scale back to points

        # Ownership
        o = Fnn.relu(self.o_bn(self.o_conv(x)))
        ownership = torch.tanh(self.o_out(o))  # (B, 1, P, P)

        return policy, value, score, ownership


# ---- architecture registry ----
# Add new named architectures here; train.py / arena.py select them by name. Global pooling
# (and any future block type) lives entirely inside an arch's block_kinds, not as a global flag.

def _kinds(n: int, gpool: bool = False) -> tuple[str, ...]:
    return tuple("gpool" if (gpool and (i + 1) % 3 == 0) else "regular" for i in range(n))


ARCHS: dict[str, ModelConfig] = {
    "b6c96":         ModelConfig(channels=96,  block_kinds=_kinds(6)),
    "b6c96-gpool":   ModelConfig(channels=96,  block_kinds=_kinds(6, gpool=True)),
    "b10c128":       ModelConfig(channels=128, block_kinds=_kinds(10)),
    "b10c128-gpool": ModelConfig(channels=128, block_kinds=_kinds(10, gpool=True)),
}


def arch_config(name: str) -> ModelConfig:
    if name not in ARCHS:
        raise ValueError(f"unknown arch {name!r}; known: {', '.join(ARCHS)}")
    return dataclasses.replace(ARCHS[name])  # return a copy so callers can override fields
