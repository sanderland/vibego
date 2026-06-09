"""Multi-head training loss. Target tensor indices follow KataGo's
metrics_pytorch.py (see katago/python/katago/train/metrics_pytorch.py:491-540):

  policy:    policyTargetsNCMove[:, 0, :]   visit counts -> normalized distribution
  value:     globalTargetsNC[:, 0:3]        win / loss / no-result probabilities
  score:     globalTargetsNC[:, 3]          score mean (points)
  ownership: valueTargetsNCHW[:, 0, :, :]   in [-1, 1], weighted by globalTargetsNC[:, 27]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    policy: float = 1.0
    value: float = 0.6
    score: float = 0.02
    ownership: float = 0.15


def compute_losses(outputs, batch, spatial, weights: LossWeights):
    policy, value, score, ownership = outputs

    # --- Policy: cross-entropy against the (normalized) visit-count distribution ---
    # Mask off-board cells (board < pos_len) so they don't enter the softmax denominator.
    pt = batch["policyTargetsNCMove"][:, 0, :].clamp(min=0)
    psum = pt.sum(dim=1, keepdim=True)
    valid = (psum.squeeze(1) > 0).float()
    target_p = pt / psum.clamp(min=1.0)
    b = spatial.shape[0]
    onboard_flat = spatial[:, 0].reshape(b, -1)  # (B, P*P)
    pass_col = torch.ones(b, 1, device=policy.device, dtype=onboard_flat.dtype)
    legal_mask = torch.cat([onboard_flat, pass_col], dim=1)  # (B, P*P+1), pass always legal
    logp = F.log_softmax(policy.masked_fill(legal_mask == 0, float("-inf")), dim=1)
    logp = logp.masked_fill(legal_mask == 0, 0.0)  # off-board logp is -inf; targets there are 0
    policy_loss = -(target_p * logp).sum(dim=1)
    policy_loss = (policy_loss * valid).sum() / valid.sum().clamp(min=1.0)

    # --- Value: cross-entropy over {win, loss, no-result} ---
    tv = batch["globalTargetsNC"][:, 0:3]
    value_loss = -(tv * F.log_softmax(value, dim=1)).sum(dim=1).mean()

    # --- Score: Huber on the score lead in points ---
    ts = batch["globalTargetsNC"][:, 3]
    score_loss = F.huber_loss(score, ts, delta=10.0)

    # --- Ownership: per-point cross-entropy in [-1, 1], masked to on-board, weighted ---
    to = batch["valueTargetsNCHW"][:, 0, :, :]
    ow = batch["globalTargetsNC"][:, 27]
    pred = ownership[:, 0, :, :].clamp(-1 + 1e-6, 1 - 1e-6)
    onboard = spatial[:, 0, :, :]
    t_pos = (1.0 + to) / 2.0
    ce = -(t_pos * torch.log((1.0 + pred) / 2.0) + (1.0 - t_pos) * torch.log((1.0 - pred) / 2.0))
    per_sample = (ce * onboard).sum(dim=(1, 2)) / onboard.sum(dim=(1, 2)).clamp(min=1.0)
    ownership_loss = (per_sample * ow).sum() / ow.sum().clamp(min=1.0)

    total = (
        weights.policy * policy_loss
        + weights.value * value_loss
        + weights.score * score_loss
        + weights.ownership * ownership_loss
    )
    return total, {
        "policy": policy_loss.item(),
        "value": value_loss.item(),
        "score": score_loss.item(),
        "ownership": ownership_loss.item(),
        "total": total.item(),
    }
