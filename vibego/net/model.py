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

import numpy as np
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
    pattern_embed: bool = False   # add a local-3x3-pattern lookup embedding to the stem (memory-for-FLOPs)

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


class NBTResBlock(nn.Module):
    """KataGo's nested bottleneck residual block (the `nbt` in b18c384nbt / b40c768nbt): a 1x1
    bottleneck from c to bc channels, then two *nested* pre-activation 3x3 residual blocks run at
    the reduced width bc, then a 1x1 projection back to c, all wrapped in an outer residual.
    Four 3x3 convs at half-width cost ~the same as two at full width, so an nbt block buys more
    conv depth (and the nesting) per parameter — that's why KataGo's strong nets are all nbt."""

    def __init__(self, c: int, bc: int | None = None):
        super().__init__()
        bc = bc if bc is not None else c // 2
        self.bn_in = nn.BatchNorm2d(c)
        self.conv_in = nn.Conv2d(c, bc, 1, bias=False)
        self.n1_bn1 = nn.BatchNorm2d(bc); self.n1_conv1 = nn.Conv2d(bc, bc, 3, padding=1, bias=False)
        self.n1_bn2 = nn.BatchNorm2d(bc); self.n1_conv2 = nn.Conv2d(bc, bc, 3, padding=1, bias=False)
        self.n2_bn1 = nn.BatchNorm2d(bc); self.n2_conv1 = nn.Conv2d(bc, bc, 3, padding=1, bias=False)
        self.n2_bn2 = nn.BatchNorm2d(bc); self.n2_conv2 = nn.Conv2d(bc, bc, 3, padding=1, bias=False)
        self.bn_out = nn.BatchNorm2d(bc)
        self.conv_out = nn.Conv2d(bc, c, 1, bias=False)

    def forward(self, x):
        h = self.conv_in(Fnn.relu(self.bn_in(x)))
        h = h + self.n1_conv2(Fnn.relu(self.n1_bn2(self.n1_conv1(Fnn.relu(self.n1_bn1(h))))))
        h = h + self.n2_conv2(Fnn.relu(self.n2_bn2(self.n2_conv1(Fnn.relu(self.n2_bn1(h))))))
        out = self.conv_out(Fnn.relu(self.bn_out(h)))
        return x + out


class LinAttnResBlock(nn.Module):
    """Linearized-attention block — global token-mixing at O(N·d) instead of softmax's O(N²·d),
    via a non-negative feature map φ(x)=elu(x)+1 (Katharopoulos et al. 2020, "Transformers are
    RNNs"). Each board point is a token (N=P·P), multi-head. Structured like a transformer block:
    a pre-norm linear-attention sublayer + a pre-norm squared-ReLU FFN, both residual.

    Honest caveats (see experiments/IDEAS.md): (1) at N=361 the FLOP win over softmax is *marginal*
    (softmax≈N²·d, linear≈N·d² are comparable when d~C/heads is not ≪ N) — this block exists to
    *measure* the trade-off, not to assume it. (2) Linear attention is permutation-equivariant and
    carries **no internal positional encoding**, so it is meant to be *interspersed with conv
    blocks* (the registry archs do this — convs supply locality/position, this supplies cheap
    global mixing, replacing gpool's role)."""

    def __init__(self, c: int, heads: int = 4, ffn_mult: int = 2):
        super().__init__()
        if c % heads != 0:
            raise ValueError(f"channels {c} not divisible by heads {heads}")
        self.heads = heads
        self.dh = c // heads
        self.bn_attn = nn.BatchNorm2d(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.bn_ffn = nn.BatchNorm2d(c)
        self.ffn1 = nn.Conv2d(c, ffn_mult * c, 1, bias=False)
        self.ffn2 = nn.Conv2d(ffn_mult * c, c, 1, bias=False)

    def _attn(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(self.bn_attn(x))  # BN as pre-norm (no relu: would zero half of q/k)
        q, k, v = qkv.view(B, 3, self.heads, self.dh, N).permute(1, 0, 2, 4, 3).unbind(0)
        # each is (B, heads, N, dh); feature map φ=elu+1 keeps the attention weights non-negative.
        fq = Fnn.elu(q) + 1.0
        fk = Fnn.elu(k) + 1.0
        kv = torch.einsum("bhnd,bhne->bhde", fk, v)        # (B,heads,dh,dh)  = Σ_j φk_j v_jᵀ
        z = fk.sum(dim=2)                                  # (B,heads,dh)     = Σ_j φk_j
        num = torch.einsum("bhnd,bhde->bhne", fq, kv)      # (B,heads,N,dh)
        den = torch.einsum("bhnd,bhd->bhn", fq, z).clamp_min(1e-6).unsqueeze(-1)
        out = (num / den).permute(0, 1, 3, 2).reshape(B, C, H, W)
        return self.proj(out)

    def _ffn(self, x):
        h = self.ffn1(self.bn_ffn(x))
        h = Fnn.relu(h) ** 2  # squared-ReLU (smoother, common in modern transformer FFNs)
        return self.ffn2(h)

    def forward(self, x):
        x = x + self._attn(x)
        x = x + self._ffn(x)
        return x


def _q_shift(x: torch.Tensor) -> torch.Tensor:
    """Vision-RWKV's omnidirectional token shift: split channels into four groups and shift each
    by one pixel (right/left/down/up neighbour), zero-padded at the border. Injects cheap local
    context before the (otherwise position-free) global WKV mix. Any channel remainder folds into
    the last (up-shift) group."""
    B, C, H, W = x.shape
    q = C // 4
    out = torch.zeros_like(x)
    out[:, 0 * q:1 * q, :, :-1] = x[:, 0 * q:1 * q, :, 1:]    # take right neighbour
    out[:, 1 * q:2 * q, :, 1:] = x[:, 1 * q:2 * q, :, :-1]    # take left neighbour
    out[:, 2 * q:3 * q, :-1, :] = x[:, 2 * q:3 * q, 1:, :]    # take lower neighbour
    out[:, 3 * q:, 1:, :] = x[:, 3 * q:, :-1, :]              # take upper neighbour (+remainder)
    return out


class RWKVResBlock(nn.Module):
    """A Vision-RWKV-style block (Duan et al. 2024) adapted to the non-causal Go board. RWKV's
    headline win — KV-cache-free *autoregressive* generation — does not transfer (we do a single
    pass over 361 fixed tokens), so we keep only the cheap global-mixing primitive:

      * **token shift** (`_q_shift`) + per-channel lerp — local context, mimicking RWKV's mix;
      * **WKV** as a *non-causal global* mix: per channel, a softmax-over-the-board weighting of v
        by exp(k), with a learned self-bonus `u`. We **drop RWKV's spatial decay** (a board has no
        canonical 1-D order, so distance-based decay is meaningless) — the honest 2-D adaptation;
      * a **receptance** sigmoid gate (r) on the mix, then output projection.

    Followed by RWKV's channel-mix FFN (token-shift → squared-ReLU gate → receptance). All maps are
    1×1 convs, so the whole block stays in (B,C,H,W). Like `linattn`, it's a low-FLOP global-mixing
    primitive meant to be interspersed with conv blocks — a candidate to beat gpool/softmax on the
    FLOPs↔Elo frontier (experiments/IDEAS.md)."""

    def __init__(self, c: int, ffn_mult: int = 2):
        super().__init__()
        self.bn_s = nn.BatchNorm2d(c)
        self.mu_sr = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
        self.mu_sk = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
        self.mu_sv = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
        self.s_r = nn.Conv2d(c, c, 1, bias=False)
        self.s_k = nn.Conv2d(c, c, 1, bias=False)
        self.s_v = nn.Conv2d(c, c, 1, bias=False)
        self.s_o = nn.Conv2d(c, c, 1, bias=False)
        self.bonus = nn.Parameter(torch.zeros(1, c, 1, 1))   # u: learned self-emphasis in WKV
        self.bn_c = nn.BatchNorm2d(c)
        self.mu_cr = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
        self.mu_ck = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
        self.c_r = nn.Conv2d(c, c, 1, bias=False)
        self.c_k = nn.Conv2d(c, ffn_mult * c, 1, bias=False)
        self.c_v = nn.Conv2d(ffn_mult * c, c, 1, bias=False)

    def _spatial_mix(self, x):
        h = self.bn_s(x)
        sh = _q_shift(h)
        r = torch.sigmoid(self.s_r(h * self.mu_sr + sh * (1 - self.mu_sr)))
        k = self.s_k(h * self.mu_sk + sh * (1 - self.mu_sk))
        v = self.s_v(h * self.mu_sv + sh * (1 - self.mu_sv))
        # Non-causal global WKV: softmax(k)-weighted board mean of v, with a self-bonus exp(u).
        ek = torch.exp(k - k.amax(dim=(2, 3), keepdim=True))
        extra = (torch.exp(self.bonus) - 1.0) * ek          # extra weight on the self token
        num = (ek * v).sum(dim=(2, 3), keepdim=True) + extra * v
        den = ek.sum(dim=(2, 3), keepdim=True) + extra + 1e-6
        return self.s_o(r * (num / den))

    def _channel_mix(self, x):
        h = self.bn_c(x)
        sh = _q_shift(h)
        r = torch.sigmoid(self.c_r(h * self.mu_cr + sh * (1 - self.mu_cr)))
        k = Fnn.relu(self.c_k(h * self.mu_ck + sh * (1 - self.mu_ck))) ** 2
        return r * self.c_v(k)

    def forward(self, x):
        x = x + self._spatial_mix(x)
        x = x + self._channel_mix(x)
        return x


class GlobalModBlock(nn.Module):
    """Smolgen-lite (Lc0-inspired) global-modulation block: a learned **global board summary**
    drives a **content-dependent, spatially-varying FiLM modulation** of a local conv path — richer
    than gpool's rank-0 broadcast *bias* (this is a multiplicative gain+shift that varies per cell),
    at O(N·C·d), with no dense N² attention. The per-cell gain/shift `(γ,β)` is the sum of a global
    term (from the pooled summary `g`) and a per-cell term (a 1×1 conv of the local features), so the
    modulation is conditioned on the whole board yet differs at every intersection."""

    def __init__(self, c: int, d: int = 64):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.fc_g = nn.Linear(2 * c, d)            # board summary from mean+max pool
        self.fc_film = nn.Linear(d, 2 * c)         # global -> per-channel (γ, β), broadcast over space
        self.conv_film = nn.Conv2d(c, 2 * c, 1)    # per-cell (γ, β) correction
        self.bn2 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)

    def forward(self, x):
        h = self.conv1(Fnn.relu(self.bn1(x)))
        g = Fnn.relu(self.fc_g(global_pool(x)))                       # (B, d) global summary
        film = self.fc_film(g).unsqueeze(-1).unsqueeze(-1) + self.conv_film(h)  # (B, 2C, H, W)
        gamma, beta = film[:, : x.shape[1]], film[:, x.shape[1]:]
        h = h * (1.0 + gamma) + beta                                  # global-conditioned, per-cell FiLM
        out = self.conv2(Fnn.relu(self.bn2(h)))
        return x + out


_CANON3X3_CACHE: dict = {}


def _build_3x3_canon():
    """Map every base-3 3×3 pattern id (cell ∈ {empty=0, own=1, opp=2}, row-major, 3^9=19683 ids) to
    a dense index over its **dihedral(D4)-canonical** representative (the min id over the 8 rotations/
    reflections). Returns (canon int64[19683], K) where K = 2862 (the exact D4-orbit count of base-3
    3×3 by Burnside; own/opp kept distinct, so ~2× the classical colour-symmetric ~1.4k). Cached."""
    if 3 in _CANON3X3_CACHE:
        return _CANON3X3_CACHE[3]
    transforms = [
        lambda r, c: (r, c),            lambda r, c: (c, 2 - r),       # identity, rot90
        lambda r, c: (2 - r, 2 - c),    lambda r, c: (2 - c, r),       # rot180, rot270
        lambda r, c: (r, 2 - c),        lambda r, c: (2 - r, c),       # mirror cols, mirror rows
        lambda r, c: (c, r),            lambda r, c: (2 - c, 2 - r),   # transpose, anti-transpose
    ]
    perms = []
    for t in transforms:
        p = [0] * 9
        for i in range(9):
            rr, cc = t(i // 3, i % 3)
            p[rr * 3 + cc] = i          # new cell (rr,cc) takes the value from original cell i
        perms.append(p)
    pow3 = (3 ** np.arange(9)).astype(np.int64)
    ids = np.arange(3 ** 9, dtype=np.int64)
    digits = (ids[:, None] // pow3[None, :]) % 3                       # (19683, 9)
    canon_raw = ids.copy()
    for p in perms:
        canon_raw = np.minimum(canon_raw, (digits[:, p] * pow3[None, :]).sum(1))
    uniq = np.unique(canon_raw)
    remap = np.zeros(canon_raw.max() + 1, dtype=np.int64)
    remap[uniq] = np.arange(len(uniq))
    canon = remap[canon_raw]
    _CANON3X3_CACHE[3] = (canon, len(uniq))
    return canon, len(uniq)


class PatternEmbed(nn.Module):
    """Local-pattern lookup embedding (params-as-memory, not FLOPs): map each cell's **dihedral-
    canonical 3×3 {empty/own/opp} neighbourhood** to a learned C-dim embedding (a table gather, ~0
    MACs) and scatter-add it into the stem — storing local Go shapes in a small ~Kx C table instead
    of spending conv FLOPs to recompute them. own/opp are the side-to-move-relative stone channels
    (model spatial channels 1 and 2). Init zero so it's a no-op until trained (clean ablation)."""

    OWN_CH, OPP_CH = 1, 2

    def __init__(self, c: int):
        super().__init__()
        canon, k = _build_3x3_canon()
        self.register_buffer("canon", torch.from_numpy(canon))                   # (19683,) long
        self.register_buffer("pow3", torch.tensor([3 ** i for i in range(9)]).view(1, 9, 1))
        self.embed = nn.Embedding(k, c)
        nn.init.zeros_(self.embed.weight)

    def forward(self, spatial):
        B, _, H, W = spatial.shape
        state = spatial[:, self.OWN_CH] + 2.0 * spatial[:, self.OPP_CH]          # (B,H,W) in {0,1,2}
        patches = Fnn.unfold(state.unsqueeze(1), kernel_size=3, padding=1)        # (B, 9, H*W), pad=empty
        raw = (patches.long() * self.pow3).sum(1)                                # (B, H*W) base-3 id
        emb = self.embed(self.canon[raw])                                        # (B, H*W, C)
        return emb.transpose(1, 2).reshape(B, -1, H, W)


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
            elif kind == "nbt":
                blocks.append(NBTResBlock(c))
            elif kind == "linattn":
                blocks.append(LinAttnResBlock(c))
            elif kind == "rwkv":
                blocks.append(RWKVResBlock(c))
            elif kind == "globmod":
                blocks.append(GlobalModBlock(c))
            else:
                raise ValueError(f"unknown block kind {kind!r}")
        self.blocks = nn.ModuleList(blocks)
        self.pattern_embed = PatternEmbed(c) if config.pattern_embed else None
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
        if self.pattern_embed is not None:
            x = x + self.pattern_embed(spatial)
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

def _kinds(n: int, gpool: bool = False, base: str = "regular", glob: str = "gpool") -> tuple[str, ...]:
    """n blocks of `base`, with every 3rd a global-mixing block (`glob`, default full-width gpool)
    when gpool=True. `glob` can be any global primitive — gpool / linattn / rwkv — so an arch can
    swap *which* cheap global op fills the every-3rd slot while holding the conv backbone fixed."""
    return tuple(glob if (gpool and (i + 1) % 3 == 0) else base for i in range(n))


ARCHS: dict[str, ModelConfig] = {
    # --- classic g170-style ResNet ladder (regular blocks + interspersed gpool) ---
    "b6c96":          ModelConfig(channels=96,  block_kinds=_kinds(6)),
    "b6c96-gpool":    ModelConfig(channels=96,  block_kinds=_kinds(6,  gpool=True)),
    "b10c128":        ModelConfig(channels=128, block_kinds=_kinds(10)),
    "b10c128-gpool":  ModelConfig(channels=128, block_kinds=_kinds(10, gpool=True)),
    "b15c192-gpool":  ModelConfig(channels=192, block_kinds=_kinds(15, gpool=True)),
    # --- nested-bottleneck ladder (KataGo's nbt; every 3rd block a full-width gpool) ---
    "b6c96nbt":       ModelConfig(channels=96,  block_kinds=_kinds(6,  gpool=True, base="nbt")),
    "b10c128nbt":     ModelConfig(channels=128, block_kinds=_kinds(10, gpool=True, base="nbt")),
    "b15c192nbt":     ModelConfig(channels=192, block_kinds=_kinds(15, gpool=True, base="nbt")),
    # nbt widened to ~match the param budget of the same-depth regular arch (bottleneck makes nbt
    # smaller at equal b/c, so to spend the *same* params we widen): b6c112nbt≈b6c96-gpool (1.07 vs
    # 1.09M), b10c152nbt≈b10c128-gpool (3.07 vs 3.12M). The fair "same budget, better block?" test.
    "b6c112nbt":      ModelConfig(channels=112, block_kinds=_kinds(6,  gpool=True, base="nbt")),
    "b10c152nbt":     ModelConfig(channels=152, block_kinds=_kinds(10, gpool=True, base="nbt")),
    # depth-vs-width study, all nbt, every arch ~param-matched to old6b's ~1.09M budget: as depth
    # grows the channel count shrinks to hold params fixed (b6c112 widest → b10c88 deepest).
    "b7c106nbt":      ModelConfig(channels=106, block_kinds=_kinds(7,  gpool=True, base="nbt")),
    "b8c102nbt":      ModelConfig(channels=102, block_kinds=_kinds(8,  gpool=True, base="nbt")),
    "b9c92nbt":       ModelConfig(channels=92,  block_kinds=_kinds(9,  gpool=True, base="nbt")),
    "b10c88nbt":      ModelConfig(channels=88,  block_kinds=_kinds(10, gpool=True, base="nbt")),
    # --- global-mixing study: regular conv backbone, every-3rd block is the cheap global op
    # (gpool vs linearized-attention vs RWKV-style mixing). Same conv locality, swapped global
    # primitive — a direct FLOPs↔Elo comparison of the candidate trunks in experiments/IDEAS.md.
    # (channels divisible by 4 for linattn's heads.) Speculative — measure on RunPod, don't assume.
    "b6c96-linat":    ModelConfig(channels=96,  block_kinds=_kinds(6,  gpool=True, glob="linattn")),
    "b6c96-rwkv":     ModelConfig(channels=96,  block_kinds=_kinds(6,  gpool=True, glob="rwkv")),
    "b10c128-linat":  ModelConfig(channels=128, block_kinds=_kinds(10, gpool=True, glob="linattn")),
    "b10c128-rwkv":   ModelConfig(channels=128, block_kinds=_kinds(10, gpool=True, glob="rwkv")),
    # smolgen-lite global modulation (Lc0-inspired): every-3rd slot = global-conditioned per-cell
    # FiLM instead of gpool — richer (multiplicative, spatially-varying) global mixing at low FLOPs.
    "b6c96-globmod":  ModelConfig(channels=96,  block_kinds=_kinds(6,  gpool=True, glob="globmod")),
    "b10c128-globmod": ModelConfig(channels=128, block_kinds=_kinds(10, gpool=True, glob="globmod")),
    # local-pattern lookup embedding (params-as-memory, ~0 FLOPs): a dihedral-canonical 3x3 table
    # added to the stem of an otherwise-standard arch — ablate the memory-for-FLOPs bet vs the base.
    "b6c96-gpool-pat": ModelConfig(channels=96, block_kinds=_kinds(6, gpool=True), pattern_embed=True),
    "b6c96nbt-pat":   ModelConfig(channels=96, block_kinds=_kinds(6, gpool=True, base="nbt"), pattern_embed=True),
}


def arch_config(name: str) -> ModelConfig:
    if name not in ARCHS:
        raise ValueError(f"unknown arch {name!r}; known: {', '.join(ARCHS)}")
    return dataclasses.replace(ARCHS[name])  # return a copy so callers can override fields
