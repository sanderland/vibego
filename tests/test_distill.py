"""Distillation (logit forcing) data path: relabel a position with a teacher's outputs and
confirm the result is a valid, trainable vibego training npz."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import relabel  # noqa: E402

from vibego.net import data, losses  # noqa: E402
from vibego.net.model import Model, ModelConfig  # noqa: E402


class FakeTeacher:
    """Returns a uniform policy / neutral eval, and checks the query is well-formed.
    Mimics the pipelined send()/recv() interface relabel_file uses."""

    def __init__(self):
        self._pending = {}

    def send(self, q):
        assert q["initialPlayer"] == "B"
        assert q["includePolicy"] and q["includeOwnership"]
        n = q["boardXSize"] * q["boardYSize"]
        self._pending[q["id"]] = {"id": q["id"], "policy": [1.0 / (n + 1)] * (n + 1),
                                  "ownership": [0.0] * n,
                                  "rootInfo": {"winrate": 0.5, "scoreLead": 1.0}}

    def recv(self, qid):
        return self._pending.pop(qid)


def _synth_source(path, pos_len=19):
    # 9x9 board placed top-left, a couple of stones; only channels 0/1/2 + global komi matter.
    full = np.zeros((2, 22, pos_len, pos_len), dtype=np.uint8)
    full[:, 0, :9, :9] = 1          # on-board
    full[:, 1, 2, 2] = 1            # own (black) stone
    full[:, 2, 3, 3] = 1            # opp (white) stone
    packed = np.packbits(full.reshape(2, 22, pos_len * pos_len), axis=2)
    glob = np.zeros((2, 19), dtype=np.float32)
    glob[:, 5] = 7.5 / 20.0
    np.savez(path, binaryInputNCHWPacked=packed, globalInputNC=glob)


def test_relabel_produces_trainable_npz(tmp_path):
    src = str(tmp_path / "src.npz")
    out = str(tmp_path / "out.npz")
    _synth_source(src)
    relabel.relabel_file(src, out, FakeTeacher(), pos_len=19, visits=1)

    batches = list(data.read_batches([out], batch_size=2, pos_len=19, device="cpu",
                                     randomize_symmetries=False, shuffle_buffer=1))
    assert batches, "relabeled npz produced no batches"
    b = batches[0]
    assert b["policyTargetsNCMove"].shape == (2, 1, 19 * 19 + 1)
    assert b["valueTargetsNCHW"].shape[1] == 1

    model = Model(ModelConfig.plain(8, 2))
    out_t = model(b["spatial"], b["glob"])
    loss, parts = losses.compute_losses(out_t, b, b["spatial"], losses.LossWeights())
    assert np.isfinite(loss.item())
    assert parts["policy"] > 0  # learning signal present


def test_read_batches_drop_last(tmp_path):
    # A val set smaller than batch_size yields a partial batch with drop_last=False (so eval
    # isn't silently empty), and nothing with drop_last=True.
    src = str(tmp_path / "src.npz")
    out = str(tmp_path / "out.npz")
    _synth_source(src)  # 2 rows
    relabel.relabel_file(src, out, FakeTeacher(), pos_len=19, visits=1)
    none = list(data.read_batches([out], batch_size=8, pos_len=19, device="cpu",
                                  randomize_symmetries=False, shuffle_buffer=1, drop_last=True))
    partial = list(data.read_batches([out], batch_size=8, pos_len=19, device="cpu",
                                     randomize_symmetries=False, shuffle_buffer=1, drop_last=False))
    assert none == []
    assert len(partial) == 1 and partial[0]["spatial"].shape[0] == 2
