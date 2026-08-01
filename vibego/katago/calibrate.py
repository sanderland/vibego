"""Activation-aware statistics over a calibration set: what the weights alone cannot tell you.

The weight-only importance in `prune.py` asks how big a unit's weights are. That is the floor of
what a pruning criterion can do, because it is blind to how much signal actually flows through the
unit on real positions -- a large weight applied to a channel that is always near zero costs
nothing to remove. The standard fix (Wanda, Sun et al. 2023) multiplies the weight norm by the
observed activation magnitude, which needs a forward pass over real inputs; that is what this
module does.

It also collects the structural diagnostics that decide whether *any* criterion can help:

  effective_rank      how many dimensions of the c384 residual stream actually carry variance.
                      If it is 384, no trunk-width pruning is possible at any quality.
  block_angles        how far each block rotates the residual stream. A block whose output is
                      nearly parallel to its input is doing little, and is the one to drop.
  head_importance     per-head activation-weighted importance, the upgrade over weight norms.
  ffn_importance      per-hidden-unit activation-weighted importance.

Calibration inputs come from KataGo training `.npz` rows, which carry the exact V7 features
KataGo itself encoded -- no feature re-derivation, so no reconstruction error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .binmodel import KataModel, NestedBottleneckBlock, TransformerAttentionBlock, TransformerFFNBlock
from .torchmodel import KataTorchModel


def load_calibration(npz_path: str, n: int, pos_len: int = 19, seed: int = 0):
    """Real positions with their exact V7 features, sampled from a training-data shard."""
    data = np.load(npz_path)
    spatial = np.unpackbits(data["binaryInputNCHWPacked"], axis=2)[:, :, : pos_len * pos_len]
    spatial = spatial.reshape(-1, 22, pos_len, pos_len).astype(np.float32)
    glob = data["globalInputNC"].astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(spatial))[:n]
    return torch.from_numpy(spatial[idx]), torch.from_numpy(glob[idx])


@dataclass
class Calibration:
    """Everything one pass over the calibration set produces."""

    head_importance: dict = field(default_factory=dict)   # attention block name -> (heads,)
    ffn_importance: dict = field(default_factory=dict)    # ffn block name -> (ffn_channels,)
    block_angles: dict = field(default_factory=dict)      # block name -> mean cosine(in, in+res)
    block_relative_norm: dict = field(default_factory=dict)  # block name -> ||res|| / ||in||
    trunk_singular_values: np.ndarray | None = None
    trunk_channel_variance: np.ndarray | None = None
    policy: np.ndarray | None = None                      # parent policy, for damage measurement
    score_lead: np.ndarray | None = None
    winrate: np.ndarray | None = None


def _accumulate(store: dict, key: str, value: np.ndarray) -> None:
    store[key] = value if key not in store else store[key] + value


def run_calibration(net: KataTorchModel, spatial: torch.Tensor, glob: torch.Tensor,
                    batch: int = 8, trunk_samples: int = 20000, seed: int = 0) -> Calibration:
    """One pass over the calibration set, collecting importances and diagnostics."""
    cal = Calibration()
    n = len(spatial)
    counts: dict = {}
    trunk_rows = []
    policies, leads, winrates = [], [], []
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for i in range(0, n, batch):
            sp, gl = spatial[i:i + batch], glob[i:i + batch]
            out = net(sp, gl, capture={"residuals": True, "heads": True, "hidden": True})
            policies.append(net.policy(out).numpy())
            leads.append(net.score_lead(out).numpy())
            winrates.append(net.winrate(out).numpy())

            mask = sp[:, 0:1]
            flat_mask = mask.reshape(mask.shape[0], -1, 1)  # (N, S, 1)

            for name, head_out in out["head_out"].items():
                # head_out: (N, S, heads*v_head_dim). Masked RMS over on-board positions only.
                block = _attention_by_name(net, name)
                h, vd = block.num_heads, block.v_head_dim
                per_head = head_out.view(head_out.shape[0], head_out.shape[1], h, vd)
                sq = (per_head.pow(2).sum(dim=-1) * flat_mask).sum(dim=(0, 1))  # (heads,)
                _accumulate(cal.head_importance, name, sq.numpy())
                counts[name] = counts.get(name, 0.0) + float(flat_mask.sum())

            for name, hidden in out["hidden_out"].items():
                sq = (hidden.pow(2) * flat_mask).sum(dim=(0, 1))  # (ffn_channels,)
                _accumulate(cal.ffn_importance, name, sq.numpy())
                counts[name] = counts.get(name, 0.0) + float(flat_mask.sum())

            for name, block_in, residual in out["residuals"]:
                a = block_in.flatten(1)
                b = (block_in + residual).flatten(1)
                cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
                _accumulate(cal.block_angles, name, cos.sum().numpy().reshape(()))
                rel = residual.flatten(1).norm(dim=1) / block_in.flatten(1).norm(dim=1).clamp_min(1e-9)
                _accumulate(cal.block_relative_norm, name, rel.sum().numpy().reshape(()))
                counts["__blocks__"] = counts.get("__blocks__", 0.0)

            # Trunk residual stream, sampled per board point, for the effective-rank estimate.
            # Pre-norm, so the statistics describe the stream rather than the tip norm's gamma.
            trunk = out["trunk_pre_norm"]
            vecs = trunk.permute(0, 2, 3, 1).reshape(-1, trunk.shape[1])
            on_board = mask.permute(0, 2, 3, 1).reshape(-1) > 0
            vecs = vecs[on_board].numpy()
            take = min(len(vecs), max(1, trunk_samples // max(1, (n + batch - 1) // batch)))
            trunk_rows.append(vecs[rng.permutation(len(vecs))[:take]])

    for name in cal.head_importance:
        cal.head_importance[name] = np.sqrt(cal.head_importance[name] / counts[name])
    for name in cal.ffn_importance:
        cal.ffn_importance[name] = np.sqrt(cal.ffn_importance[name] / counts[name])
    for name in cal.block_angles:
        cal.block_angles[name] = float(cal.block_angles[name]) / n
        cal.block_relative_norm[name] = float(cal.block_relative_norm[name]) / n

    trunk_matrix = np.concatenate(trunk_rows)
    centered = trunk_matrix - trunk_matrix.mean(axis=0, keepdims=True)
    cal.trunk_singular_values = np.linalg.svd(centered, compute_uv=False)
    # Per-*channel* variance is a different question from the PCA spectrum: channel pruning can
    # only delete axis-aligned coordinates, so a stream that is low-rank in some rotated basis is
    # not necessarily narrowable in the basis the weights actually live in.
    cal.trunk_channel_variance = centered.var(axis=0)
    cal.policy = np.concatenate(policies)
    cal.score_lead = np.concatenate(leads)
    cal.winrate = np.concatenate(winrates)
    return cal


def _attention_by_name(net: KataTorchModel, name: str):
    for module in net.modules():
        if getattr(module, "name", None) == name:
            return module
    raise KeyError(name)


def axis_aligned_concentration(channel_variance: np.ndarray) -> dict:
    """How concentrated the residual stream is in the *channel* basis -- the only basis that
    channel pruning can exploit. Compare with `effective_rank`, which is rotation-invariant: a
    big gap between them means the redundancy is real but not reachable without a change of
    basis (i.e. a low-rank factorization, which the model file format cannot express)."""
    var = np.sort(channel_variance.astype(np.float64))[::-1]
    cum = np.cumsum(var) / var.sum()
    return {
        "channels": int(len(var)),
        "participation_ratio": float(var.sum() ** 2 / (var ** 2).sum()),
        "channels_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "channels_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "frac_below_1pct_of_max": float(np.mean(var < 0.01 * var.max())),
    }


def effective_rank(singular_values: np.ndarray) -> dict:
    """Two standard summaries of "how many dimensions is this really using".

    `participation_ratio` = (sum s^2)^2 / sum s^4, the flat-spectrum-equivalent dimension.
    `dims_for_99pct` counts components needed for 99% of the variance.
    """
    var = singular_values.astype(np.float64) ** 2
    total = var.sum()
    cum = np.cumsum(var) / total
    return {
        "dims": int(len(var)),
        "participation_ratio": float(total ** 2 / (var ** 2).sum()),
        "dims_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
        "dims_for_99pct": int(np.searchsorted(cum, 0.99) + 1),
        "dims_for_999pct": int(np.searchsorted(cum, 0.999) + 1),
    }


def weighted_head_importance(model: KataModel, cal: Calibration) -> dict:
    """Activation-aware head importance: observed output magnitude x the norm of the matrix it
    feeds. The weight-only criterion is the second factor alone."""
    out = {}
    for block in _iter_inner(model, TransformerAttentionBlock):
        act = cal.head_importance.get(block.name)
        if act is None:
            continue
        h, vd = block.num_heads, block.v_head_dim
        w_out = np.linalg.norm(block.out_proj.weight.reshape(h, vd, -1), axis=(1, 2))
        out[block.name] = act * w_out
    return out


def weighted_ffn_importance(model: KataModel, cal: Calibration) -> dict:
    out = {}
    for block in _iter_inner(model, TransformerFFNBlock):
        act = cal.ffn_importance.get(block.name)
        if act is None:
            continue
        out[block.name] = act * np.linalg.norm(block.linear2.weight, axis=1)
    return out


def _iter_inner(model: KataModel, kind):
    for block in model.trunk.blocks:
        if isinstance(block, kind):
            yield block
        elif isinstance(block, NestedBottleneckBlock):
            for sub in block.blocks:
                if isinstance(sub, kind):
                    yield sub
