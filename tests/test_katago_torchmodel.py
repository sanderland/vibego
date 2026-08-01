"""The torch forward exists so we can look at activations; these tests pin the parts that would
silently corrupt what we see.

The headline check is that structural pruning is *exact*: removing an attention head must produce
bit-comparable outputs to leaving it in place with its output contribution zeroed. That one
assertion covers both the pruning slicing and this module's head layout -- if either transposed a
head, the two would diverge.

Agreement with the real engine is a separate, opt-in check: `scripts/kata_torch_check.py`.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from vibego.katago.calibrate import Calibration, effective_rank, run_calibration
from vibego.katago.prune import narrow_ffn, prune_heads
from vibego.katago.torchmodel import KataTorchModel
from tests.test_katago_binmodel import make_model

POS_LEN = 9


def _inputs(n: int = 2, pos_len: int = POS_LEN, off_board: int = 0):
    rng = np.random.default_rng(0)
    spatial = np.zeros((n, 22, pos_len, pos_len), dtype=np.float32)
    live = pos_len - off_board
    spatial[:, 0, :live, :live] = 1.0  # mask
    stones = rng.random((n, 2, live, live)) < 0.15
    spatial[:, 1, :live, :live] = stones[:, 0]
    spatial[:, 2, :live, :live] = stones[:, 1] * (1 - stones[:, 0])
    glob = rng.standard_normal((n, 19)).astype(np.float32) * 0.1
    return torch.from_numpy(spatial), torch.from_numpy(glob)


def _net(model=None):
    return KataTorchModel(model or make_model(), pos_len=POS_LEN).eval()


def test_forward_shapes():
    net = _net()
    spatial, glob = _inputs()
    with torch.no_grad():
        out = net(spatial, glob)
    assert out["policy_logits"].shape == (2, 2, POS_LEN * POS_LEN + 1)
    assert out["value"].shape == (2, 3)
    assert out["miscvalue"].shape == (2, 6)
    assert out["ownership"].shape == (2, 1, POS_LEN, POS_LEN)
    assert torch.allclose(net.policy(out).sum(dim=1), torch.ones(2), atol=1e-5)


def test_offboard_points_get_no_policy():
    net = _net()
    spatial, glob = _inputs(off_board=2)
    with torch.no_grad():
        policy = net.policy(net(spatial, glob))
    board = policy[:, :-1].reshape(2, POS_LEN, POS_LEN)
    assert float(board[:, POS_LEN - 2:, :].max()) < 1e-9
    assert float(board[:, :, POS_LEN - 2:].max()) < 1e-9
    assert float(board[:, : POS_LEN - 2, : POS_LEN - 2].sum(dim=(1, 2)).min()) > 0.1


def test_head_pruning_equals_zeroing_the_head():
    """Structural head removal must be exact, not approximate."""
    model = make_model()
    attn = model.trunk.blocks[2].blocks[0]
    from vibego.katago.prune import head_importance

    drop = int(np.argmin(head_importance(attn)))
    vd = attn.v_head_dim

    zeroed = copy.deepcopy(model)
    z_attn = zeroed.trunk.blocks[2].blocks[0]
    z_attn.out_proj.weight[drop * vd:(drop + 1) * vd, :] = 0.0

    pruned = copy.deepcopy(model)
    prune_heads(pruned.trunk.blocks[2].blocks[0], attn.num_heads - 1)

    spatial, glob = _inputs()
    with torch.no_grad():
        a = _net(zeroed)(spatial, glob)["policy_logits"]
        b = _net(pruned)(spatial, glob)["policy_logits"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_ffn_narrowing_equals_zeroing_those_units():
    model = make_model()
    ffn = model.trunk.blocks[2].blocks[1]
    from vibego.katago.prune import ffn_importance

    keep_units = ffn.ffn_channels // 2
    keep = np.sort(np.argsort(ffn_importance(ffn))[::-1][:keep_units])
    drop = np.setdiff1d(np.arange(ffn.ffn_channels), keep)

    zeroed = copy.deepcopy(model)
    zeroed.trunk.blocks[2].blocks[1].linear2.weight[drop, :] = 0.0

    pruned = copy.deepcopy(model)
    narrow_ffn(pruned.trunk.blocks[2].blocks[1], keep_units)

    spatial, glob = _inputs()
    with torch.no_grad():
        a = _net(zeroed)(spatial, glob)["policy_logits"]
        b = _net(pruned)(spatial, glob)["policy_logits"]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_block_dropping_changes_the_output():
    """A sanity floor: if dropping a block were a no-op, every diagnostic below would be too."""
    model = make_model()
    spatial, glob = _inputs()
    with torch.no_grad():
        before = _net(model)(spatial, glob)["policy_logits"]
    from vibego.katago.prune import drop_blocks

    dropped = copy.deepcopy(model)
    drop_blocks(dropped, [2])
    with torch.no_grad():
        after = _net(dropped)(spatial, glob)["policy_logits"]
    assert not torch.allclose(before, after, atol=1e-3)


def test_calibration_collects_every_block_and_unit():
    model = make_model()
    net = _net(model)
    spatial, glob = _inputs(n=4)
    cal = run_calibration(net, spatial, glob, batch=2, trunk_samples=200)

    attn = model.trunk.blocks[2].blocks[0]
    ffn = model.trunk.blocks[2].blocks[1]
    assert cal.head_importance[attn.name].shape == (attn.num_heads,)
    assert cal.ffn_importance[ffn.name].shape == (ffn.ffn_channels,)
    assert np.all(cal.head_importance[attn.name] > 0)
    # Every trunk block plus every inner sub-block gets an angle.
    assert len(cal.block_angles) == 3 + len(model.trunk.blocks[2].blocks)
    assert all(-1.0 <= v <= 1.0 for v in cal.block_angles.values())
    assert cal.policy.shape == (4, POS_LEN * POS_LEN + 1)


def test_effective_rank_bounds():
    flat = effective_rank(np.ones(64))
    assert flat["participation_ratio"] == pytest.approx(64.0)
    spiked = effective_rank(np.array([10.0] + [1e-6] * 63))
    assert spiked["participation_ratio"] < 1.01
    assert spiked["dims_for_99pct"] == 1


def test_calibration_dataclass_defaults_are_independent():
    a, b = Calibration(), Calibration()
    a.head_importance["x"] = np.zeros(1)
    assert "x" not in b.head_importance
