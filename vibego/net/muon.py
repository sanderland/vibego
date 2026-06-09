"""Muon optimizer (+ a small multi-optimizer wrapper and weight EMA) for the training study.

Muon (MomentUm Orthogonalized by Newton-schulz) is the single most-used lever on the Parameter
Golf leaderboard and the KataGo discord's "fast early training" pick. It updates each 2-D weight
matrix by the *orthogonalization* of its momentum buffer (via a few Newton-Schulz iterations),
which equalizes the singular values of the step — empirically a much better-conditioned update
than Adam for matrix params. It only applies to >=2-D params (conv/linear weights, conv reshaped
to (out, in*kh*kw)); 1-D params (norm scales, biases) should stay on AdamW. `make_muon_adamw`
builds exactly that split, and `MultiOpt` lets train.py drive both as one optimizer.

Refs: Keller Jordan's Muon (github.com/KellerJordan/Muon); momentum/LR knobs from ROADMAP #7.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Orthogonalize a 2-D matrix G via 5 quintic Newton-Schulz iterations (the coefficients are
    Keller Jordan's, tuned so the iteration pushes every singular value toward 1). Runs in bf16 for
    speed; the result has ~orthogonal rows/cols and the same shape as G."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transpose = G.size(0) > G.size(1)
    if transpose:  # iterate on the thin orientation
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon for >=2-D params. Each step: momentum-accumulate the grad (Nesterov), orthogonalize it
    (conv weights are reshaped to 2-D first), and take an RMS-matched step. `momentum` can be
    rescheduled per-step by the caller (set `group['momentum']`), as ROADMAP #7 warms it 0.92->0.99."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf
                shape = g.shape
                g2 = g.reshape(shape[0], -1) if g.ndim > 2 else g
                u = _zeropower_via_newtonschulz5(g2, group["ns_steps"])
                # RMS-match the update to the param's fan so a single LR works across shapes.
                scale = max(1.0, g2.size(0) / g2.size(1)) ** 0.5
                p.add_(u.reshape(shape), alpha=-group["lr"] * scale)


class MultiOpt:
    """Drives several torch optimizers as one. Exposes a flat `param_groups` (so train.py's lr
    scheduling / logging just works) and forwards step/zero_grad/state_dict/load_state_dict."""

    def __init__(self, opts: list):
        self.opts = opts

    @property
    def param_groups(self):
        return [g for o in self.opts for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.opts]}

    def load_state_dict(self, sd):
        for o, s in zip(self.opts, sd["opts"]):
            o.load_state_dict(s)


def make_muon_adamw(model, muon_lr: float, scalar_lr: float, momentum: float = 0.95,
                    weight_decay: float = 0.0) -> MultiOpt:
    """Muon for matrix params (ndim>=2), AdamW for everything 1-D (norms/biases). Each group gets a
    `base_lr` field so the caller can scale all groups by one schedule shape."""
    matrix = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    scalar = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    muon = Muon(matrix, lr=muon_lr, momentum=momentum)
    adamw = torch.optim.AdamW(scalar, lr=scalar_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    opt = MultiOpt([muon, adamw])
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt


class ModelEMA:
    """Exponential moving average of model weights (incl. buffers like BN running stats). EMA params
    are often a free strength bump at eval/export. `store/copy_to/restore` swap EMA weights into the
    live model for eval or saving without disturbing training."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1 - d)
            else:  # int buffers (e.g. BN num_batches_tracked): just track latest
                s.copy_(v)

    def store(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model):
        if self._backup is not None:
            model.load_state_dict(self._backup, strict=True)
            self._backup = None
