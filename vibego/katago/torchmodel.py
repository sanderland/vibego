"""A torch forward pass for a parsed KataGo `.bin.gz`.

The loader (`binmodel.py`) gets us the weights; this runs them. Not for playing -- the stock engine
is faster and already correct -- but for the things the engine cannot do: look at activations.
Activation-aware pruning criteria, effective-rank of the residual stream, per-block angular
distance, head masking -- all of them need the intermediate tensors, and none of them need the
forward to be fast.

Layer semantics follow KataGo's own reference implementation
(`python/katago/train/model_pytorch.py`, v1.17.1), which the exported file format is a flattening
of. Where the export folds things together (batchnorm into scale/bias, `conv1x1` into the centre of
a larger kernel), this reads the folded form directly.

Known limitations, all deliberate:
  - No SGF metadata encoder, no GAB/TAB attention bias, no inline registers, no grouped-query
    attention. None appear in the released nets; the loader raises on the metadata encoder and
    this module raises on the rest.
  - The score-belief subhead is not exported at all, so score *distribution* outputs are absent.
    Value, score mean/lead/stdev and ownership are all present.

    from vibego.katago import read_model
    from vibego.katago.torchmodel import KataTorchModel
    net = KataTorchModel(read_model("models/b10c384h6nbttflrs.bin.gz"))
    out = net(spatial, glob)          # spatial (N,22,19,19), glob (N,19)
    out["policy_logits"]              # (N, num_policy_outputs, 19*19+1), channel 0 is the policy
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .binmodel import (
    Activation,
    BatchNorm,
    Conv,
    GPoolBlock,
    KataModel,
    MatBias,
    MatMul,
    NestedBottleneckBlock,
    OrdinaryBlock,
    RMSNorm,
    TransformerAttentionBlock,
    TransformerFFNBlock,
    TransformerRMSNorm,
)

_ACTIVATIONS = {
    "ACTIVATION_RELU": F.relu,
    "ACTIVATION_SILU": F.silu,
    "ACTIVATION_MISH": F.mish,
    "ACTIVATION_IDENTITY": lambda x: x,
    None: F.relu,  # pre-v11 files carry no kind token; the engine hardcodes ReLU
}


def _t(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


class _Norm(nn.Module):
    """Exported batchnorm / bias-mask, folded to a per-channel affine.

    The file stores mean/variance/scale/bias with `variance` already defined so that
    sqrt(variance + epsilon) is the running std, so y = (x - mean) / std * scale + bias collapses
    to a single multiply-add. Bias-mask layers are the same thing with mean 0 and variance 1.
    """

    def __init__(self, desc: BatchNorm):
        super().__init__()
        eps = float(desc.epsilon)
        std = np.sqrt(desc.variance.astype(np.float64) + eps)
        scale = desc.scale if desc.scale is not None else np.ones_like(desc.mean)
        bias = desc.bias if desc.bias is not None else np.zeros_like(desc.mean)
        a = scale / std
        b = bias - desc.mean * a
        self.register_buffer("a", _t(a).view(1, -1, 1, 1))
        self.register_buffer("b", _t(b).view(1, -1, 1, 1))

    def forward(self, x, mask):
        return (x * self.a + self.b) * mask


class _RMSNormMask(nn.Module):
    """Trunk-tip RMSNorm, then gamma/beta.

    Two modes, both in the wild: per-position (RMS across channels, one divisor per board point)
    and `spatial` (RMS across channels *and* masked board points, one divisor per position in the
    batch). Grouped spatial RMSNorm exists in the training code but the exporter refuses it.
    """

    def __init__(self, desc: RMSNorm):
        super().__init__()
        if desc.cgroup_size:
            raise NotImplementedError("grouped spatial RMSNorm is not exportable, so not supported")
        self.spatial = bool(desc.spatial)
        self.c_in = desc.num_channels
        self.eps = float(desc.epsilon)
        self.register_buffer("gamma", _t(desc.gamma).view(1, -1, 1, 1))
        self.register_buffer("beta", _t(desc.beta).view(1, -1, 1, 1))

    def forward(self, x, mask):
        if self.spatial:
            mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)
            mean_sq = (torch.sum(x * x * mask, dim=(1, 2, 3), keepdim=True)
                       / (self.c_in * mask_sum_hw + self.eps))
        else:
            mean_sq = torch.mean(x * x, dim=1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return (x / rms * self.gamma + self.beta) * mask


class _SeqRMSNorm(nn.Module):
    """Inline pre-norm inside transformer blocks, on (N, S, C): weight only, no bias."""

    def __init__(self, desc: TransformerRMSNorm):
        super().__init__()
        self.eps = float(desc.epsilon)
        self.register_buffer("weight", _t(desc.weight))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class _Conv(nn.Module):
    def __init__(self, desc: Conv):
        super().__init__()
        if desc.dilation_y != 1 or desc.dilation_x != 1:
            raise NotImplementedError(f"{desc.name}: dilated convolutions are not implemented")
        self.padding = (desc.diam_y // 2, desc.diam_x // 2)
        self.register_buffer("weight", _t(desc.torch_weight))

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding)


class _Linear(nn.Module):
    """Exported matmul (+ optional separate bias layer)."""

    def __init__(self, desc: MatMul, bias: MatBias | None = None):
        super().__init__()
        self.register_buffer("weight", _t(desc.torch_weight))
        self.register_buffer("bias", _t(bias.weight) if bias is not None else None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


def _gpool(x, mask, mask_sum_hw):
    """KataGo's trunk/policy pooling: (mean, mean scaled by board size, max), masked."""
    offset = torch.sqrt(mask_sum_hw) - 14.0
    mean = torch.sum(x, dim=(2, 3), keepdim=True) / mask_sum_hw
    maxed = torch.max((x + (mask - 1.0)).flatten(2), dim=2).values.view(x.shape[0], -1, 1, 1)
    return torch.cat((mean, mean * (offset / 10.0), maxed), dim=1)


def _value_gpool(x, mask, mask_sum_hw):
    """The value head pools three *scalings of the mean* instead of taking a max."""
    offset = torch.sqrt(mask_sum_hw) - 14.0
    mean = torch.sum(x, dim=(2, 3), keepdim=True) / mask_sum_hw
    return torch.cat((mean, mean * (offset / 10.0),
                      mean * ((offset * offset) / 100.0 - 0.1)), dim=1)


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


class _OrdinaryBlock(nn.Module):
    def __init__(self, desc: OrdinaryBlock):
        super().__init__()
        self.name = desc.name
        self.norm1, self.conv1 = _Norm(desc.pre_norm), _Conv(desc.regular_conv)
        self.norm2, self.conv2 = _Norm(desc.mid_norm), _Conv(desc.final_conv)
        self.act1 = _ACTIVATIONS[desc.pre_act.kind]
        self.act2 = _ACTIVATIONS[desc.mid_act.kind]

    def forward(self, x, ctx):
        out = self.conv1(self.act1(self.norm1(x, ctx.mask)))
        return self.conv2(self.act2(self.norm2(out, ctx.mask)))


class _GPoolBlock(nn.Module):
    def __init__(self, desc: GPoolBlock):
        super().__init__()
        self.name = desc.name
        self.norm1 = _Norm(desc.pre_norm)
        self.conv_r, self.conv_g = _Conv(desc.regular_conv), _Conv(desc.gpool_conv)
        self.norm_g = _Norm(desc.gpool_norm)
        self.linear_g = _Linear(desc.gpool_to_bias_mul)
        self.norm2, self.conv2 = _Norm(desc.mid_norm), _Conv(desc.final_conv)
        self.act1 = _ACTIVATIONS[desc.pre_act.kind]
        self.act_g = _ACTIVATIONS[desc.gpool_act.kind]
        self.act2 = _ACTIVATIONS[desc.mid_act.kind]

    def forward(self, x, ctx):
        h = self.act1(self.norm1(x, ctx.mask))
        r, g = self.conv_r(h), self.conv_g(h)
        g = self.act_g(self.norm_g(g, ctx.mask))
        pooled = _gpool(g, ctx.mask, ctx.mask_sum_hw).flatten(1)
        r = r + self.linear_g(pooled).unsqueeze(-1).unsqueeze(-1)
        return self.conv2(self.act2(self.norm2(r, ctx.mask)))


class _AttentionBlock(nn.Module):
    def __init__(self, desc: TransformerAttentionBlock, pos_len: int):
        super().__init__()
        self.name = desc.name
        if desc.num_kv_heads != desc.num_heads:
            raise NotImplementedError(f"{desc.name}: grouped-query attention is not implemented")
        self.num_heads = desc.num_heads
        self.q_head_dim, self.v_head_dim = desc.q_head_dim, desc.v_head_dim
        self.scale = 1.0 / math.sqrt(desc.q_head_dim)
        self.norm = _SeqRMSNorm(desc.pre_norm)
        self.q_proj, self.k_proj = _Linear(desc.q_proj), _Linear(desc.k_proj)
        self.v_proj, self.out_proj = _Linear(desc.v_proj), _Linear(desc.out_proj)

        if not desc.use_rope:
            self.register_buffer("cos", None)
            self.register_buffer("sin", None)
            return
        if not desc.learnable_rope:
            raise NotImplementedError(f"{desc.name}: fixed-theta RoPE is not implemented")
        # angle(pos, head, pair) = x * omega_x + y * omega_y, over the row-major board grid.
        idx = np.arange(pos_len * pos_len)
        sx = (idx % pos_len).astype(np.float64)
        sy = (idx // pos_len).astype(np.float64)
        freqs = desc.rope_freqs.astype(np.float64)  # (heads, pairs, 2)
        angles = sx[:, None, None] * freqs[None, :, :, 0] + sy[:, None, None] * freqs[None, :, :, 1]
        self.register_buffer("cos", _t(np.cos(angles)))  # (S, heads, pairs)
        self.register_buffer("sin", _t(np.sin(angles)))

    def _rope(self, x):
        # x: (N, S, heads, dim) -> rotate interleaved channel pairs (2p, 2p+1)
        n, s, h, d = x.shape
        pairs = x.view(n, s, h, d // 2, 2)
        x0, x1 = pairs.unbind(dim=-1)
        cos, sin = self.cos.unsqueeze(0), self.sin.unsqueeze(0)
        return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).reshape(n, s, h, d)

    def forward(self, x, ctx):
        n, c, h, w = x.shape
        seq = x.view(n, c, -1).permute(0, 2, 1)
        xn = self.norm(seq)
        s = seq.shape[1]
        if ctx.capture_input_moment:
            # E[x x^T] over on-board positions, at the input to the q/k/v projections. This is
            # the metric a data-aware low-rank approximation of the projections has to use.
            flat = xn * ctx.mask.reshape(n, -1, 1)
            moment = torch.einsum("nsc,nsd->cd", flat, flat).double()
            prev = ctx.input_moment.get(self.name)
            ctx.input_moment[self.name] = moment if prev is None else prev + moment

        q = self.q_proj(xn).view(n, s, self.num_heads, self.q_head_dim)
        k = self.k_proj(xn).view(n, s, self.num_heads, self.q_head_dim)
        v = self.v_proj(xn).view(n, s, self.num_heads, self.v_head_dim)
        if self.cos is not None:
            q, k = self._rope(q), self._rope(k)

        q, k, v = (t.permute(0, 2, 1, 3) for t in (q, k, v))
        attn_mask = None
        if ctx.has_offboard:
            flat = ctx.mask.reshape(n, 1, 1, s)
            attn_mask = torch.zeros_like(flat)
            attn_mask.masked_fill_(flat == 0, float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        out = out.permute(0, 2, 1, 3).reshape(n, s, self.num_heads * self.v_head_dim)
        if ctx.capture_heads:
            ctx.head_out[self.name] = out.detach()
        out = self.out_proj(out)
        return out.permute(0, 2, 1).view(n, c, h, w)


class _FFNBlock(nn.Module):
    def __init__(self, desc: TransformerFFNBlock):
        super().__init__()
        self.name = desc.name
        self.norm = _SeqRMSNorm(desc.pre_norm)
        self.linear1 = _Linear(desc.linear1)
        self.gate = _Linear(desc.linear_gate) if desc.linear_gate is not None else None
        self.linear2 = _Linear(desc.linear2)
        self.use_swiglu = bool(desc.use_swiglu)

    def forward(self, x, ctx):
        n, c, h, w = x.shape
        seq = x.view(n, c, -1).permute(0, 2, 1)
        xn = self.norm(seq)
        hidden = F.silu(self.linear1(xn))
        if self.use_swiglu:
            hidden = hidden * self.gate(xn)
        if ctx.capture_hidden:
            ctx.hidden_out[self.name] = hidden.detach()
        out = self.linear2(hidden)
        return out.permute(0, 2, 1).view(n, c, h, w)


class _NestedBottleneckBlock(nn.Module):
    def __init__(self, desc: NestedBottleneckBlock, pos_len: int):
        super().__init__()
        self.name = desc.name
        self.norm_p, self.conv_p = _Norm(desc.pre_norm), _Conv(desc.pre_conv)
        self.norm_q, self.conv_q = _Norm(desc.post_norm), _Conv(desc.post_conv)
        self.act_p = _ACTIVATIONS[desc.pre_act.kind]
        self.act_q = _ACTIVATIONS[desc.post_act.kind]
        self.inner = nn.ModuleList([_build_block(b, pos_len) for b in desc.blocks])

    def forward(self, x, ctx):
        out = self.conv_p(self.act_p(self.norm_p(x, ctx.mask)))
        for block in self.inner:
            residual = block(out, ctx)
            if ctx.capture_residuals:
                ctx.residuals.append((block.name, out.detach(), residual.detach()))
            out = out + residual
        return self.conv_q(self.act_q(self.norm_q(out, ctx.mask)))


def _build_block(desc, pos_len: int) -> nn.Module:
    if isinstance(desc, OrdinaryBlock):
        return _OrdinaryBlock(desc)
    if isinstance(desc, GPoolBlock):
        return _GPoolBlock(desc)
    if isinstance(desc, NestedBottleneckBlock):
        return _NestedBottleneckBlock(desc, pos_len)
    if isinstance(desc, TransformerAttentionBlock):
        return _AttentionBlock(desc, pos_len)
    if isinstance(desc, TransformerFFNBlock):
        return _FFNBlock(desc)
    raise NotImplementedError(f"no torch implementation for {type(desc).__name__}")


class _Ctx:
    """Per-forward state: the board mask, plus whatever the caller asked to capture."""

    def __init__(self, mask, capture_residuals=False, capture_heads=False, capture_hidden=False,
                 capture_input_moment=False):
        self.mask = mask
        self.mask_sum_hw = torch.sum(mask, dim=(2, 3), keepdim=True)
        self.has_offboard = bool((mask == 0).any())
        self.capture_residuals = capture_residuals
        self.capture_heads = capture_heads
        self.capture_hidden = capture_hidden
        self.capture_input_moment = capture_input_moment
        self.input_moment: dict = {}
        self.residuals: list = []
        self.trunk_pre_norm = None
        self.head_out: dict = {}
        self.hidden_out: dict = {}


# --------------------------------------------------------------------------------------
# Whole model
# --------------------------------------------------------------------------------------


class KataTorchModel(nn.Module):
    def __init__(self, model: KataModel, pos_len: int = 19):
        super().__init__()
        if model.trunk.metadata_encoder is not None:
            raise NotImplementedError("SGF metadata encoder is not implemented")
        self.desc = model
        self.pos_len = pos_len
        self.num_policy_outputs = model.policy_out_channels()

        trunk = model.trunk
        self.conv_spatial = _Conv(trunk.initial_conv)
        self.linear_global = _Linear(trunk.initial_matmul)
        self.blocks = nn.ModuleList([_build_block(b, pos_len) for b in trunk.blocks])
        self.tip_norm = (_RMSNormMask(trunk.tip_norm) if isinstance(trunk.tip_norm, RMSNorm)
                         else _Norm(trunk.tip_norm))
        self.tip_act = _ACTIVATIONS[trunk.tip_act.kind]

        ph = model.policy_head
        self.p1_conv, self.g1_conv = _Conv(ph.p1_conv), _Conv(ph.g1_conv)
        self.g1_norm, self.p1_norm = _Norm(ph.g1_norm), _Norm(ph.p1_norm)
        self.g1_act = _ACTIVATIONS[ph.g1_act.kind]
        self.p1_act = _ACTIVATIONS[ph.p1_act.kind]
        self.gpool_to_bias = _Linear(ph.gpool_to_bias_mul)
        self.p2_conv = _Conv(ph.p2_conv)
        self.pass_linear = _Linear(ph.gpool_to_pass_mul, ph.gpool_to_pass_bias)
        self.pass_act = _ACTIVATIONS[ph.pass_act.kind] if ph.pass_act else None
        self.pass_linear2 = (_Linear(ph.gpool_to_pass_mul2)
                             if ph.gpool_to_pass_mul2 is not None else None)

        vh = model.value_head
        self.v1_conv, self.v1_norm = _Conv(vh.v1_conv), _Norm(vh.v1_norm)
        self.v1_act = _ACTIVATIONS[vh.v1_act.kind]
        self.v2_linear = _Linear(vh.v2_mul, vh.v2_bias)
        self.v2_act = _ACTIVATIONS[vh.v2_act.kind]
        self.v3_linear = _Linear(vh.v3_mul, vh.v3_bias)
        self.sv3_linear = _Linear(vh.sv3_mul, vh.sv3_bias)
        self.ownership_conv = _Conv(vh.ownership_conv)

        # Post-process multipliers, in the order export_model_pytorch writes them.
        p = model.post_process
        self.score_mean_mult = float(p[1]) if p else 20.0
        self.lead_mult = float(p[3]) if p else 20.0

    def trunk_forward(self, spatial, glob, ctx: _Ctx):
        out = self.conv_spatial(spatial) + self.linear_global(glob).unsqueeze(-1).unsqueeze(-1)
        for block in self.blocks:
            residual = block(out, ctx)
            if ctx.capture_residuals:
                ctx.residuals.append((block.name, out.detach(), residual.detach()))
            out = out + residual
        # The residual stream *before* the trunk-tip norm and activation. Per-channel statistics
        # have to be read here: the tip RMSNorm's gamma rescales every channel and SiLU squashes
        # negatives, so measuring after them reports the norm's parameters as much as the stream.
        ctx.trunk_pre_norm = out.detach()
        return self.tip_act(self.tip_norm(out, ctx.mask))

    def forward(self, spatial, glob, capture: dict | None = None):
        capture = capture or {}
        mask = spatial[:, 0:1].contiguous()
        ctx = _Ctx(mask, capture.get("residuals", False), capture.get("heads", False),
                   capture.get("hidden", False), capture.get("input_moment", False))
        trunk_out = self.trunk_forward(spatial, glob, ctx)

        p1 = self.p1_conv(trunk_out)
        g1 = self.g1_act(self.g1_norm(self.g1_conv(trunk_out), mask))
        pooled = _gpool(g1, mask, ctx.mask_sum_hw).flatten(1)
        pass_logit = self.pass_linear(pooled)
        if self.pass_linear2 is not None:
            pass_logit = self.pass_linear2(self.pass_act(pass_logit))
        p1 = p1 + self.gpool_to_bias(pooled).unsqueeze(-1).unsqueeze(-1)
        policy = self.p2_conv(self.p1_act(self.p1_norm(p1, mask)))
        policy = policy - (1.0 - mask) * 5000.0
        policy_logits = torch.cat((policy.flatten(2), pass_logit.unsqueeze(-1)), dim=2)

        v1 = self.v1_act(self.v1_norm(self.v1_conv(trunk_out), mask))
        v_pooled = _value_gpool(v1, mask, ctx.mask_sum_hw).flatten(1)
        v2 = self.v2_act(self.v2_linear(v_pooled))
        value = self.v3_linear(v2)
        miscvalue = self.sv3_linear(v2)
        ownership = torch.tanh(self.ownership_conv(v1)) * mask

        result = {
            "policy_logits": policy_logits,
            "value": value,
            "miscvalue": miscvalue,
            "ownership": ownership,
            "trunk_out": trunk_out,
            "trunk_pre_norm": ctx.trunk_pre_norm,
        }
        if ctx.capture_residuals:
            result["residuals"] = ctx.residuals
        if ctx.capture_heads:
            result["head_out"] = ctx.head_out
        if ctx.capture_hidden:
            result["hidden_out"] = ctx.hidden_out
        if ctx.capture_input_moment:
            result["input_moment"] = ctx.input_moment
        return result

    # -- readable outputs --------------------------------------------------------------

    def policy(self, out: dict, channel: int = 0) -> torch.Tensor:
        """Softmax policy over board points + pass, for one policy output channel."""
        return torch.softmax(out["policy_logits"][:, channel], dim=-1)

    def value_probs(self, out: dict, no_result_possible: bool = False) -> torch.Tensor:
        """(win, loss, no-result) probabilities.

        KataGo suppresses the no-result output entirely unless the ruleset can actually produce
        one -- i.e. simple ko or territory scoring (`nneval.cpp`). Under the usual superko + area
        rules it forces the probability to zero and renormalizes, so reproducing the engine's
        winrate means doing the same.
        """
        logits = out["value"]
        if not no_result_possible:
            logits = logits.clone()
            logits[:, 2] = -1e9
        return torch.softmax(logits, dim=-1)

    def winrate(self, out: dict, no_result_possible: bool = False) -> torch.Tensor:
        """Win probability, counting a no-result as half a win (the analysis engine's convention)."""
        probs = self.value_probs(out, no_result_possible)
        return probs[:, 0] + 0.5 * probs[:, 2]

    def score_lead(self, out: dict, no_result_possible: bool = False) -> torch.Tensor:
        """Expected score lead. Unconditional, so it is scaled down by the no-result probability
        exactly as the engine does.

        Caveat: measured against the engine this matches to ~0.06 points on v15/v17 nets but
        drifts by ~0.5-2 points on the old v8 g170 nets, whose misc-value head is only four
        channels wide and whose lead post-processing the engine handles differently. Policy and
        value agree closely on those nets regardless -- see scripts/kata_torch_check.py.
        """
        no_result = self.value_probs(out, no_result_possible)[:, 2]
        return out["miscvalue"][:, 2] * self.lead_mult * (1.0 - no_result)
