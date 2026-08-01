"""Read/write KataGo's exported model format (`.bin.gz`), model versions 11-17.

Why: everything on the compression branch needs the weights of a *released* net, and the released
artifact is a `.bin.gz`, not a training checkpoint. This module parses one into a structured,
editable description and writes it back out **byte-for-byte identically** when nothing was changed
-- which is the whole point: a net we prune here still loads in the stock KataGo engine, so it can
be measured with the existing harness (`scripts/policy_eval.py`, `match.py`, ...) at real backend
speed, with no new inference code.

The format is a hybrid: whitespace-separated ASCII tokens describing shapes, with weight blobs
inlined as `@BIN@` followed by exactly the number of little-endian float32s the preceding shape
fields imply. So parsing is *structure-aware* -- you cannot skip a blob without knowing the layer
kind. This mirrors KataGo's own reader (`cpp/neuralnet/desc.cpp`) and writer
(`python/export_model_pytorch.py`) as of v1.17.1; the layer order below follows them exactly.

Scalars that KataGo writes via Python's `str(float)` (epsilons, the post-process multipliers, rope
theta) are kept as their original **token text** rather than reparsed floats, so re-emitting is
exact regardless of repr differences.

Weight arrays keep the file's own layout, not torch's:
  - conv:   (diam_y, diam_x, in_channels, out_channels)   [torch is (oc, ic, y, x)]
  - matmul: (in_channels, out_channels)                   [torch is (oc, ic)]
`Conv.torch_weight` / `MatMul.torch_weight` convert.

    from vibego.katago import read_model, write_model
    m = read_model("models/b10c384h6nbttflrs.bin.gz")
    print(m.version, m.trunk.num_blocks, m.trunk.trunk_channels, m.num_parameters())
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from typing import BinaryIO, Iterator

import numpy as np

# Block kind tokens, as written by export_model_pytorch.write_block.
ORDINARY_BLOCK = "ordinary_block"
GPOOL_BLOCK = "gpool_block"
NESTED_BOTTLENECK_BLOCK = "nested_bottleneck_block"
TRANSFORMER_ATTENTION_BLOCK = "transformer_attention_block"
TRANSFORMER_FFN_BLOCK = "transformer_ffn_block"

TRUNK_NORM_STANDARD = 0
TRUNK_NORM_RMSNORM = 1

_WHITESPACE = b" \t\r\n\v\f"


class ModelFormatError(Exception):
    pass


# --------------------------------------------------------------------------------------
# Token/blob stream
# --------------------------------------------------------------------------------------


class _Reader:
    """Tokenizer over the raw model bytes, with `@BIN@` float-blob support."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0
        self.version = 0  # set from the header before any layer is parsed

    def token(self) -> str:
        n = len(self.buf)
        i = self.pos
        while i < n and self.buf[i] in _WHITESPACE:
            i += 1
        start = i
        while i < n and self.buf[i] not in _WHITESPACE:
            i += 1
        if start == i:
            raise ModelFormatError(f"unexpected end of model file at byte {self.pos}")
        self.pos = i
        return self.buf[start:i].decode("ascii", errors="backslashreplace")

    def int(self) -> int:
        tok = self.token()
        try:
            return int(tok)
        except ValueError as e:
            raise ModelFormatError(f"expected int, got {tok!r}") from e

    def floats(self, count: int) -> np.ndarray:
        # KataGo's reader skips arbitrary bytes until '@' (with a sanity cap), then expects "BIN@".
        i = self.pos
        skipped = 0
        while i < len(self.buf) and self.buf[i] != 0x40:  # '@'
            i += 1
            skipped += 1
            if skipped > 100:
                raise ModelFormatError(
                    f"no @BIN@ block within 100 bytes of offset {self.pos} "
                    "(is this a .txt.gz model rather than a .bin.gz?)"
                )
        if self.buf[i : i + 5] != b"@BIN@":
            raise ModelFormatError(f"expected @BIN@ header at byte {i}")
        start = i + 5
        end = start + 4 * count
        if end > len(self.buf):
            raise ModelFormatError(f"truncated float block at byte {start} (wanted {count} floats)")
        self.pos = end
        # Copy rather than view: the point of parsing is to edit the weights, and a view onto
        # the file buffer is read-only.
        return np.frombuffer(self.buf, dtype="<f4", count=count, offset=start).copy()

    def expect_eof(self) -> None:
        rest = self.buf[self.pos :].strip(_WHITESPACE)
        if rest:
            raise ModelFormatError(f"{len(rest)} trailing bytes after end of model")


class _Writer:
    def __init__(self, version: int) -> None:
        self.parts: list[bytes] = []
        self.version = version

    def ln(self, value) -> None:
        self.parts.append((str(value) + "\n").encode("ascii", errors="backslashreplace"))

    def floats(self, arr: np.ndarray) -> None:
        flat = np.ascontiguousarray(np.asarray(arr, dtype="<f4").reshape(-1))
        self.parts.append(b"@BIN@" + flat.tobytes() + b"\n")

    def getvalue(self) -> bytes:
        return b"".join(self.parts)


# --------------------------------------------------------------------------------------
# Leaf layers
# --------------------------------------------------------------------------------------


@dataclass
class Conv:
    name: str
    diam_y: int
    diam_x: int
    in_channels: int
    out_channels: int
    dilation_y: int
    dilation_x: int
    weight: np.ndarray  # (diam_y, diam_x, in_channels, out_channels)

    @classmethod
    def read(cls, r: _Reader) -> "Conv":
        name = r.token()
        diam_y, diam_x = r.int(), r.int()
        in_channels, out_channels = r.int(), r.int()
        dil_y, dil_x = r.int(), r.int()
        w = r.floats(diam_y * diam_x * in_channels * out_channels)
        return cls(name, diam_y, diam_x, in_channels, out_channels, dil_y, dil_x,
                   w.reshape(diam_y, diam_x, in_channels, out_channels))

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        for v in (self.diam_y, self.diam_x, self.in_channels, self.out_channels,
                  self.dilation_y, self.dilation_x):
            w.ln(v)
        w.floats(self.weight)

    @property
    def torch_weight(self) -> np.ndarray:
        """(out_channels, in_channels, diam_y, diam_x) -- torch's conv layout."""
        return np.transpose(self.weight, (3, 2, 0, 1))

    def num_parameters(self) -> int:
        return int(self.weight.size)


@dataclass
class MatMul:
    name: str
    in_channels: int
    out_channels: int
    weight: np.ndarray  # (in_channels, out_channels)

    @classmethod
    def read(cls, r: _Reader) -> "MatMul":
        name = r.token()
        in_channels, out_channels = r.int(), r.int()
        w = r.floats(in_channels * out_channels)
        return cls(name, in_channels, out_channels, w.reshape(in_channels, out_channels))

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        w.ln(self.in_channels)
        w.ln(self.out_channels)
        w.floats(self.weight)

    @property
    def torch_weight(self) -> np.ndarray:
        """(out_channels, in_channels) -- torch's linear layout."""
        return np.transpose(self.weight, (1, 0))

    def num_parameters(self) -> int:
        return int(self.weight.size)


@dataclass
class MatBias:
    name: str
    num_channels: int
    weight: np.ndarray

    @classmethod
    def read(cls, r: _Reader) -> "MatBias":
        name = r.token()
        num_channels = r.int()
        return cls(name, num_channels, r.floats(num_channels))

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        w.ln(self.num_channels)
        w.floats(self.weight)

    def num_parameters(self) -> int:
        return int(self.weight.size)


@dataclass
class BatchNorm:
    """Covers both real batchnorm and the "bias mask" degenerate case -- KataGo writes them
    with the same fields (mean/variance are zeros/ones when there is no batchnorm)."""

    name: str
    num_channels: int
    epsilon: str  # raw token text, preserved for exact round-trip
    has_scale: int
    has_bias: int
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray | None
    bias: np.ndarray | None

    @classmethod
    def read(cls, r: _Reader) -> "BatchNorm":
        name = r.token()
        num_channels = r.int()
        epsilon = r.token()
        has_scale, has_bias = r.int(), r.int()
        mean = r.floats(num_channels)
        variance = r.floats(num_channels)
        scale = r.floats(num_channels) if has_scale else None
        bias = r.floats(num_channels) if has_bias else None
        return cls(name, num_channels, epsilon, has_scale, has_bias, mean, variance, scale, bias)

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        w.ln(self.num_channels)
        w.ln(self.epsilon)
        w.ln(self.has_scale)
        w.ln(self.has_bias)
        w.floats(self.mean)
        w.floats(self.variance)
        if self.has_scale:
            w.floats(self.scale)
        if self.has_bias:
            w.floats(self.bias)

    def num_parameters(self) -> int:
        n = self.mean.size + self.variance.size
        n += self.scale.size if self.scale is not None else 0
        n += self.bias.size if self.bias is not None else 0
        return int(n)


@dataclass
class Activation:
    """Pre-v11 files have no activation-kind token at all; the engine hardcodes ReLU there,
    so `kind` is None and nothing is emitted on write."""

    name: str
    kind: str | None  # ACTIVATION_RELU / _MISH / _SILU / _IDENTITY, or None for version < 11

    @classmethod
    def read(cls, r: _Reader) -> "Activation":
        name = r.token()
        return cls(name, r.token() if r.version >= 11 else None)

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        if w.version >= 11:
            w.ln(self.kind)

    def num_parameters(self) -> int:
        return 0


@dataclass
class RMSNorm:
    """Trunk-tip RMSNorm (`RMSNormMask`): has both gamma and beta, plus spatial/group flags."""

    name: str
    num_channels: int
    epsilon: str
    spatial: int
    cgroup_size: int
    gamma: np.ndarray
    beta: np.ndarray

    @classmethod
    def read(cls, r: _Reader) -> "RMSNorm":
        name = r.token()
        num_channels = r.int()
        epsilon = r.token()
        spatial, cgroup = r.int(), r.int()
        return cls(name, num_channels, epsilon, spatial, cgroup,
                   r.floats(num_channels), r.floats(num_channels))

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        w.ln(self.num_channels)
        w.ln(self.epsilon)
        w.ln(self.spatial)
        w.ln(self.cgroup_size)
        w.floats(self.gamma)
        w.floats(self.beta)

    def num_parameters(self) -> int:
        return int(self.gamma.size + self.beta.size)


@dataclass
class TransformerRMSNorm:
    """Inline pre-norm inside transformer blocks: a weight vector only, no bias."""

    name: str
    num_channels: int
    epsilon: str
    weight: np.ndarray

    @classmethod
    def read(cls, r: _Reader) -> "TransformerRMSNorm":
        name = r.token()
        num_channels = r.int()
        epsilon = r.token()
        return cls(name, num_channels, epsilon, r.floats(num_channels))

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        w.ln(self.num_channels)
        w.ln(self.epsilon)
        w.floats(self.weight)

    def num_parameters(self) -> int:
        return int(self.weight.size)


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


@dataclass
class OrdinaryBlock:
    kind = ORDINARY_BLOCK
    name: str
    pre_norm: BatchNorm
    pre_act: Activation
    regular_conv: Conv
    mid_norm: BatchNorm
    mid_act: Activation
    final_conv: Conv

    @classmethod
    def read(cls, r: _Reader) -> "OrdinaryBlock":
        return cls(r.token(), BatchNorm.read(r), Activation.read(r), Conv.read(r),
                   BatchNorm.read(r), Activation.read(r), Conv.read(r))

    def write(self, w: _Writer) -> None:
        w.ln(self.kind)
        w.ln(self.name)
        for layer in self.layers():
            layer.write(w)

    def layers(self) -> list:
        return [self.pre_norm, self.pre_act, self.regular_conv,
                self.mid_norm, self.mid_act, self.final_conv]


@dataclass
class GPoolBlock:
    kind = GPOOL_BLOCK
    name: str
    pre_norm: BatchNorm
    pre_act: Activation
    regular_conv: Conv
    gpool_conv: Conv
    gpool_norm: BatchNorm
    gpool_act: Activation
    gpool_to_bias_mul: MatMul
    mid_norm: BatchNorm
    mid_act: Activation
    final_conv: Conv

    @classmethod
    def read(cls, r: _Reader) -> "GPoolBlock":
        return cls(r.token(), BatchNorm.read(r), Activation.read(r), Conv.read(r), Conv.read(r),
                   BatchNorm.read(r), Activation.read(r), MatMul.read(r),
                   BatchNorm.read(r), Activation.read(r), Conv.read(r))

    def write(self, w: _Writer) -> None:
        w.ln(self.kind)
        w.ln(self.name)
        for layer in self.layers():
            layer.write(w)

    def layers(self) -> list:
        return [self.pre_norm, self.pre_act, self.regular_conv, self.gpool_conv,
                self.gpool_norm, self.gpool_act, self.gpool_to_bias_mul,
                self.mid_norm, self.mid_act, self.final_conv]


@dataclass
class NestedBottleneckBlock:
    """`nbt`. Also the carrier for `nbttflrs`: the inner stack is then attention/FFN blocks
    running at the *bottleneck* width (`pre_conv.out_channels`), not the trunk width."""

    kind = NESTED_BOTTLENECK_BLOCK
    name: str
    num_blocks: int
    pre_norm: BatchNorm
    pre_act: Activation
    pre_conv: Conv
    blocks: list
    post_norm: BatchNorm
    post_act: Activation
    post_conv: Conv

    @classmethod
    def read(cls, r: _Reader) -> "NestedBottleneckBlock":
        name = r.token()
        num_blocks = r.int()
        pre_norm, pre_act, pre_conv = BatchNorm.read(r), Activation.read(r), Conv.read(r)
        blocks = [read_block(r) for _ in range(num_blocks)]
        return cls(name, num_blocks, pre_norm, pre_act, pre_conv, blocks,
                   BatchNorm.read(r), Activation.read(r), Conv.read(r))

    def write(self, w: _Writer) -> None:
        w.ln(self.kind)
        w.ln(self.name)
        w.ln(self.num_blocks)
        self.pre_norm.write(w)
        self.pre_act.write(w)
        self.pre_conv.write(w)
        for b in self.blocks:
            b.write(w)
        self.post_norm.write(w)
        self.post_act.write(w)
        self.post_conv.write(w)

    @property
    def bottleneck_channels(self) -> int:
        return self.pre_conv.out_channels

    def layers(self) -> list:
        return [self.pre_norm, self.pre_act, self.pre_conv, self.post_norm,
                self.post_act, self.post_conv]


@dataclass
class TransformerAttentionBlock:
    kind = TRANSFORMER_ATTENTION_BLOCK
    name: str
    num_heads: int
    num_kv_heads: int
    q_head_dim: int
    v_head_dim: int
    use_rope: int
    learnable_rope: int
    pre_norm: TransformerRMSNorm
    q_proj: MatMul
    k_proj: MatMul
    v_proj: MatMul
    out_proj: MatMul
    rope_name: str | None = None
    rope_freqs: np.ndarray | None = None  # (num_kv_heads, q_head_dim // 2, 2)
    rope_theta: str | None = None  # raw token text, non-learnable rope only

    @classmethod
    def read(cls, r: _Reader) -> "TransformerAttentionBlock":
        name = r.token()
        num_heads, num_kv_heads = r.int(), r.int()
        q_head_dim, v_head_dim = r.int(), r.int()
        use_rope, learnable_rope = r.int(), r.int()
        blk = cls(name, num_heads, num_kv_heads, q_head_dim, v_head_dim, use_rope, learnable_rope,
                  TransformerRMSNorm.read(r), MatMul.read(r), MatMul.read(r),
                  MatMul.read(r), MatMul.read(r))
        if use_rope:
            blk.rope_name = r.token()
            if learnable_rope:
                n_kv, n_pairs, dim2 = r.int(), r.int(), r.int()
                if dim2 != 2:
                    raise ModelFormatError(f"{name}: rope freq dim2 must be 2, got {dim2}")
                blk.rope_freqs = r.floats(n_kv * n_pairs * 2).reshape(n_kv, n_pairs, 2)
            else:
                blk.rope_theta = r.token()
        return blk

    def write(self, w: _Writer) -> None:
        w.ln(self.kind)
        w.ln(self.name)
        for v in (self.num_heads, self.num_kv_heads, self.q_head_dim, self.v_head_dim,
                  self.use_rope, self.learnable_rope):
            w.ln(v)
        self.pre_norm.write(w)
        self.q_proj.write(w)
        self.k_proj.write(w)
        self.v_proj.write(w)
        self.out_proj.write(w)
        if self.use_rope:
            w.ln(self.rope_name)
            if self.learnable_rope:
                w.ln(self.rope_freqs.shape[0])
                w.ln(self.rope_freqs.shape[1])
                w.ln(self.rope_freqs.shape[2])
                w.floats(self.rope_freqs)
            else:
                w.ln(self.rope_theta)

    def layers(self) -> list:
        return [self.pre_norm, self.q_proj, self.k_proj, self.v_proj, self.out_proj]

    def num_parameters(self) -> int:
        n = sum(layer.num_parameters() for layer in self.layers())
        return n + (int(self.rope_freqs.size) if self.rope_freqs is not None else 0)


@dataclass
class TransformerFFNBlock:
    kind = TRANSFORMER_FFN_BLOCK
    name: str
    num_channels: int
    ffn_channels: int
    use_swiglu: int
    pre_norm: TransformerRMSNorm
    linear1: MatMul
    linear_gate: MatMul | None
    linear2: MatMul

    @classmethod
    def read(cls, r: _Reader) -> "TransformerFFNBlock":
        name = r.token()
        num_channels, ffn_channels, use_swiglu = r.int(), r.int(), r.int()
        pre_norm = TransformerRMSNorm.read(r)
        linear1 = MatMul.read(r)
        gate = MatMul.read(r) if use_swiglu else None
        return cls(name, num_channels, ffn_channels, use_swiglu, pre_norm, linear1, gate,
                   MatMul.read(r))

    def write(self, w: _Writer) -> None:
        w.ln(self.kind)
        w.ln(self.name)
        w.ln(self.num_channels)
        w.ln(self.ffn_channels)
        w.ln(self.use_swiglu)
        self.pre_norm.write(w)
        self.linear1.write(w)
        if self.use_swiglu:
            self.linear_gate.write(w)
        self.linear2.write(w)

    def layers(self) -> list:
        out = [self.pre_norm, self.linear1]
        if self.linear_gate is not None:
            out.append(self.linear_gate)
        out.append(self.linear2)
        return out


_BLOCK_READERS = {
    ORDINARY_BLOCK: OrdinaryBlock,
    GPOOL_BLOCK: GPoolBlock,
    NESTED_BOTTLENECK_BLOCK: NestedBottleneckBlock,
    TRANSFORMER_ATTENTION_BLOCK: TransformerAttentionBlock,
    TRANSFORMER_FFN_BLOCK: TransformerFFNBlock,
}


def read_block(r: _Reader):
    kind = r.token()
    reader = _BLOCK_READERS.get(kind)
    if reader is None:
        raise ModelFormatError(f"unknown block kind: {kind}")
    return reader.read(r)


# --------------------------------------------------------------------------------------
# Trunk / heads / model
# --------------------------------------------------------------------------------------


@dataclass
class SGFMetadataEncoder:
    name: str
    num_input_meta_channels: int
    mul1: MatMul
    bias1: MatBias
    act1: Activation
    mul2: MatMul
    bias2: MatBias
    act2: Activation
    mul3: MatMul

    @classmethod
    def read(cls, r: _Reader) -> "SGFMetadataEncoder":
        return cls(r.token(), r.int(), MatMul.read(r), MatBias.read(r), Activation.read(r),
                   MatMul.read(r), MatBias.read(r), Activation.read(r), MatMul.read(r))

    def write(self, w: _Writer) -> None:
        w.ln(self.name)
        w.ln(self.num_input_meta_channels)
        for layer in self.layers():
            layer.write(w)

    def layers(self) -> list:
        return [self.mul1, self.bias1, self.act1, self.mul2, self.bias2, self.act2, self.mul3]


@dataclass
class Trunk:
    name: str
    num_blocks: int
    trunk_channels: int
    mid_channels: int
    regular_channels: int
    dilated_channels: int  # unused by the engine, preserved verbatim
    gpool_channels: int
    norm_kind: int
    unused: list[int]
    initial_conv: Conv
    initial_matmul: MatMul
    metadata_encoder: SGFMetadataEncoder | None
    blocks: list
    tip_norm: BatchNorm | RMSNorm
    tip_act: Activation

    @classmethod
    def read(cls, r: _Reader, version: int, meta_encoder_version: int) -> "Trunk":
        name = r.token()
        num_blocks = r.int()
        trunk_channels, mid_channels = r.int(), r.int()
        regular_channels, dilated_channels, gpool_channels = r.int(), r.int(), r.int()
        norm_kind, unused = TRUNK_NORM_STANDARD, []
        if version >= 15:
            norm_kind = r.int()
            unused = [r.int() for _ in range(5)]
        initial_conv = Conv.read(r)
        initial_matmul = MatMul.read(r)
        meta = SGFMetadataEncoder.read(r) if meta_encoder_version > 0 else None
        blocks = [read_block(r) for _ in range(num_blocks)]
        tip_norm = RMSNorm.read(r) if norm_kind == TRUNK_NORM_RMSNORM else BatchNorm.read(r)
        return cls(name, num_blocks, trunk_channels, mid_channels, regular_channels,
                   dilated_channels, gpool_channels, norm_kind, unused, initial_conv,
                   initial_matmul, meta, blocks, tip_norm, Activation.read(r))

    def write(self, w: _Writer, version: int) -> None:
        w.ln(self.name)
        w.ln(self.num_blocks)
        w.ln(self.trunk_channels)
        w.ln(self.mid_channels)
        w.ln(self.regular_channels)
        w.ln(self.dilated_channels)
        w.ln(self.gpool_channels)
        if version >= 15:
            w.ln(self.norm_kind)
            for v in self.unused:
                w.ln(v)
        self.initial_conv.write(w)
        self.initial_matmul.write(w)
        if self.metadata_encoder is not None:
            self.metadata_encoder.write(w)
        for b in self.blocks:
            b.write(w)
        self.tip_norm.write(w)
        self.tip_act.write(w)


@dataclass
class PolicyHead:
    name: str
    out_channels: int  # 2 (regular + optimistic) or 4 (+ q winloss/score); v17 only
    unused: list[int]
    p1_conv: Conv
    g1_conv: Conv
    g1_norm: BatchNorm
    g1_act: Activation
    gpool_to_bias_mul: MatMul
    p1_norm: BatchNorm
    p1_act: Activation
    p2_conv: Conv
    gpool_to_pass_mul: MatMul
    gpool_to_pass_bias: MatBias | None
    pass_act: Activation | None
    gpool_to_pass_mul2: MatMul | None

    @classmethod
    def read(cls, r: _Reader, version: int) -> "PolicyHead":
        name = r.token()
        out_channels, unused = 0, []
        if version >= 17:
            out_channels = r.int()
            unused = [r.int() for _ in range(3)]
        p1_conv, g1_conv = Conv.read(r), Conv.read(r)
        g1_norm, g1_act = BatchNorm.read(r), Activation.read(r)
        gpool_to_bias_mul = MatMul.read(r)
        p1_norm, p1_act = BatchNorm.read(r), Activation.read(r)
        p2_conv = Conv.read(r)
        gpool_to_pass_mul = MatMul.read(r)
        pass_bias = pass_act = pass_mul2 = None
        if version >= 15:
            pass_bias, pass_act, pass_mul2 = MatBias.read(r), Activation.read(r), MatMul.read(r)
        return cls(name, out_channels, unused, p1_conv, g1_conv, g1_norm, g1_act,
                   gpool_to_bias_mul, p1_norm, p1_act, p2_conv, gpool_to_pass_mul,
                   pass_bias, pass_act, pass_mul2)

    def write(self, w: _Writer, version: int) -> None:
        w.ln(self.name)
        if version >= 17:
            w.ln(self.out_channels)
            for v in self.unused:
                w.ln(v)
        for layer in [self.p1_conv, self.g1_conv, self.g1_norm, self.g1_act,
                      self.gpool_to_bias_mul, self.p1_norm, self.p1_act, self.p2_conv,
                      self.gpool_to_pass_mul]:
            layer.write(w)
        if version >= 15:
            self.gpool_to_pass_bias.write(w)
            self.pass_act.write(w)
            self.gpool_to_pass_mul2.write(w)

    def layers(self) -> list:
        out = [self.p1_conv, self.g1_conv, self.g1_norm, self.g1_act, self.gpool_to_bias_mul,
               self.p1_norm, self.p1_act, self.p2_conv, self.gpool_to_pass_mul]
        out += [x for x in (self.gpool_to_pass_bias, self.pass_act, self.gpool_to_pass_mul2)
                if x is not None]
        return out


@dataclass
class ValueHead:
    name: str
    unused: list[int]
    v1_conv: Conv
    v1_norm: BatchNorm
    v1_act: Activation
    v2_mul: MatMul
    v2_bias: MatBias
    v2_act: Activation
    v3_mul: MatMul
    v3_bias: MatBias
    sv3_mul: MatMul
    sv3_bias: MatBias
    ownership_conv: Conv

    @classmethod
    def read(cls, r: _Reader, version: int) -> "ValueHead":
        name = r.token()
        unused = [r.int() for _ in range(3)] if version >= 17 else []
        return cls(name, unused, Conv.read(r), BatchNorm.read(r), Activation.read(r),
                   MatMul.read(r), MatBias.read(r), Activation.read(r), MatMul.read(r),
                   MatBias.read(r), MatMul.read(r), MatBias.read(r), Conv.read(r))

    def write(self, w: _Writer, version: int) -> None:
        w.ln(self.name)
        if version >= 17:
            for v in self.unused:
                w.ln(v)
        for layer in self.layers():
            layer.write(w)

    def layers(self) -> list:
        return [self.v1_conv, self.v1_norm, self.v1_act, self.v2_mul, self.v2_bias,
                self.v2_act, self.v3_mul, self.v3_bias, self.sv3_mul, self.sv3_bias,
                self.ownership_conv]


@dataclass
class KataModel:
    name: str
    version: int
    num_bin_features: int
    num_global_features: int
    post_process: list[str] = field(default_factory=list)  # 7 raw tokens, v>=13
    meta_encoder_version: int = 0
    prefer_pass_alive: int = 0
    unused: list[int] = field(default_factory=list)  # 6 zeros, v>=15
    trunk: Trunk = None
    policy_head: PolicyHead = None
    value_head: ValueHead = None

    # -- convenience -------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(layer.num_parameters() for _, layer in self.iter_layers())

    def iter_layers(self, include_rope: bool = True) -> Iterator[tuple[str, object]]:
        """Yield (path, leaf layer) over the whole model, in file order."""

        def walk_block(path: str, block):
            if isinstance(block, NestedBottleneckBlock):
                for i, name in enumerate(("pre_norm", "pre_act", "pre_conv")):
                    yield f"{path}.{name}", getattr(block, name)
                for i, sub in enumerate(block.blocks):
                    yield from walk_block(f"{path}.inner{i}", sub)
                for name in ("post_norm", "post_act", "post_conv"):
                    yield f"{path}.{name}", getattr(block, name)
            else:
                for layer in block.layers():
                    yield f"{path}.{layer.name.rsplit('.', 1)[-1]}", layer
                if include_rope and isinstance(block, TransformerAttentionBlock) \
                        and block.rope_freqs is not None:
                    yield f"{path}.rope_freqs", _RopeParams(block.rope_freqs)

        yield "trunk.initial_conv", self.trunk.initial_conv
        yield "trunk.initial_matmul", self.trunk.initial_matmul
        if self.trunk.metadata_encoder is not None:
            for layer in self.trunk.metadata_encoder.layers():
                yield f"trunk.meta.{layer.name.rsplit('.', 1)[-1]}", layer
        for i, block in enumerate(self.trunk.blocks):
            yield from walk_block(f"trunk.block{i}", block)
        yield "trunk.tip_norm", self.trunk.tip_norm
        yield "trunk.tip_act", self.trunk.tip_act
        for layer in self.policy_head.layers():
            yield f"policy.{layer.name.rsplit('.', 1)[-1]}", layer
        for layer in self.value_head.layers():
            yield f"value.{layer.name.rsplit('.', 1)[-1]}", layer

    def policy_out_channels(self) -> int:
        """Number of policy output channels the engine will use. Only v17 stores it explicitly;
        older versions imply it from the model version (see PolicyHeadDesc in desc.cpp)."""
        if self.version >= 17:
            return self.policy_head.out_channels
        if self.version == 16:
            return 4
        return 2 if self.version >= 12 else 1

    def block_kinds(self) -> list[str]:
        """Top-level trunk block kinds, e.g. ['nested_bottleneck_block'] * 10."""
        return [b.kind for b in self.trunk.blocks]


@dataclass
class _RopeParams:
    """Wrapper so learnable-RoPE frequencies show up in iter_layers/param counts."""

    weight: np.ndarray

    def num_parameters(self) -> int:
        return int(self.weight.size)


# --------------------------------------------------------------------------------------
# Top-level read/write
# --------------------------------------------------------------------------------------


def _read_model(r: _Reader) -> KataModel:
    name = r.token()
    version = r.int()
    if version < 8:
        raise ModelFormatError(
            f"model version {version} is older than this reader supports (>= 8); "
            "such nets are long unsupported by the engine too"
        )
    r.version = version
    num_bin, num_global = r.int(), r.int()
    post_process = [r.token() for _ in range(7)] if version >= 13 else []
    meta_encoder_version = prefer_pass_alive = 0
    unused: list[int] = []
    if version >= 15:
        meta_encoder_version = r.int()
        prefer_pass_alive = r.int()
        unused = [r.int() for _ in range(6)]
    trunk = Trunk.read(r, version, meta_encoder_version)
    policy_head = PolicyHead.read(r, version)
    value_head = ValueHead.read(r, version)
    r.expect_eof()
    return KataModel(name, version, num_bin, num_global, post_process, meta_encoder_version,
                     prefer_pass_alive, unused, trunk, policy_head, value_head)


def read_model(path: str) -> KataModel:
    """Parse a KataGo `.bin.gz` (or uncompressed `.bin`) model file."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        buf = f.read()
    return _read_model(_Reader(buf))


def read_model_bytes(buf: bytes) -> KataModel:
    return _read_model(_Reader(buf))


def model_bytes(model: KataModel) -> bytes:
    """Serialize to the exact bytes KataGo's exporter would have written."""
    w = _Writer(model.version)
    w.ln(model.name)
    w.ln(model.version)
    w.ln(model.num_bin_features)
    w.ln(model.num_global_features)
    if model.version >= 13:
        for tok in model.post_process:
            w.ln(tok)
    if model.version >= 15:
        w.ln(model.meta_encoder_version)
        w.ln(model.prefer_pass_alive)
        for v in model.unused:
            w.ln(v)
    model.trunk.write(w, model.version)
    model.policy_head.write(w, model.version)
    model.value_head.write(w, model.version)
    return w.getvalue()


def write_model(model: KataModel, path: str, compresslevel: int = 6) -> None:
    """Write a model to `path`. Gzips when the path ends in `.gz`.

    Note: gzip output is not byte-identical to KataGo's own (compression settings and the
    mtime header differ) -- it is the *decompressed* stream that round-trips exactly, which is
    what the engine reads. Use `model_bytes` to compare.
    """
    data = model_bytes(model)
    if path.endswith(".gz"):
        with gzip.GzipFile(path, "wb", compresslevel=compresslevel, mtime=0) as f:
            f.write(data)
    else:
        with open(path, "wb") as f:
            f.write(data)
