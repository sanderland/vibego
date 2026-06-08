"""Tests for the batching evaluator's failure handling."""
import math

import numpy as np
import pytest
import torch

from nanogo.engine.search import MCTS, NNEvaluator
from nanogo.go.board import BLACK, PASS, Board
from nanogo.go.features import NUM_GLOBAL, NUM_SPATIAL


class _BoomModel(torch.nn.Module):
    def forward(self, spatial, glob):
        raise RuntimeError("boom")


class _FakeEval:
    """Returns a fixed eval (uniform policy) so we can exercise search without a model."""

    def evaluate_boards(self, boards, komi, pos_len):
        out = []
        for b in boards:
            moves = b.legal_moves(b.to_move) + [PASS]
            out.append({"v": 0.0, "winrate": 0.5, "score": 6.0,
                        "policy": {mv: 1.0 / len(moves) for mv in moves},
                        "ownership": np.zeros((pos_len, pos_len), dtype=np.float32)})
        return out


def test_score_enters_search_utility():
    ev = {"v": 0.2, "score": 15.0}
    m = MCTS(None, komi=7.5, pos_len=19)  # KataGo defaults: static 0.1@2.0, dynamic 0.3@0.75
    m.sqrt_area = 19.0  # score_center defaults to 0 here
    expected = (2 / math.pi) * (0.1 * math.atan(15 / (2.0 * 19)) + 0.3 * math.atan(15 / (0.75 * 19)))
    assert abs(m._utility(ev, BLACK) - (0.2 + expected)) < 1e-9
    m0 = MCTS(None, komi=7.5, pos_len=19, static_score_factor=0.0, dynamic_score_factor=0.0)
    assert m0._utility(ev, BLACK) == 0.2  # score off -> win-loss only


def test_reporting_separates_winloss_and_score():
    root = MCTS(_FakeEval(), komi=7.5, pos_len=19).run(Board(7), visits=16, batch_size=4)
    assert root.N >= 16
    assert -1.0 <= root.winloss() <= 1.0
    assert abs(root.score()) <= 6.0 + 1e-6  # reported score lead is the win-loss-independent stat


def test_evaluator_surfaces_errors_instead_of_hanging():
    # If the model raises, infer() must raise (not block its caller forever).
    ev = NNEvaluator(_BoomModel(), "cpu")
    feats = [(np.zeros((NUM_SPATIAL, 19, 19), dtype=np.float32),
              np.zeros((NUM_GLOBAL,), dtype=np.float32))]
    with pytest.raises(RuntimeError, match="boom"):
        ev.infer(feats)
