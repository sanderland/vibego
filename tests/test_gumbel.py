"""Tests for the optional Gumbel-AlphaZero root mode (Danihelka et al. 2022)."""
import math

import numpy as np

from vibego.engine.search import (
    MCTS, gumbel_top_k, rank_children, sequential_halving_schedule,
)
from vibego.go.board import PASS, Board


class _FakeEval:
    """Deterministic eval without a model: fixed value/score, non-uniform policy (weights cycle
    1..7 over the legal-move list) so the search has real preferences and is reproducible."""

    def evaluate_boards(self, boards, komi, pos_len):
        out = []
        for b in boards:
            moves = b.legal_moves(b.to_move) + [PASS]
            ws = np.array([(i % 7) + 1 for i in range(len(moves))], dtype=np.float64)
            ws /= ws.sum()
            out.append({"v": 0.1, "winrate": 0.55, "score": 2.0,
                        "policy": {mv: float(w) for mv, w in zip(moves, ws)},
                        "ownership": np.zeros((pos_len, pos_len), dtype=np.float32)})
        return out


# ---- (a) Gumbel-top-k sampling without replacement ----

def test_gumbel_top_k_no_replacement_and_marginals():
    probs = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    logp = np.log(probs)
    rng = np.random.default_rng(0)
    counts = np.zeros(len(probs))
    trials = 20000
    for _ in range(trials):
        idx, g = gumbel_top_k(logp, 3, rng)
        assert len(idx) == 3 and len(set(int(i) for i in idx)) == 3  # without replacement
        assert len(g) == len(probs)
        counts[idx[0]] += 1
    # The top-1 of log p + Gumbel noise is an exact sample from softmax(log p) = probs:
    # empirical top-1 frequencies must match (binomial se < 0.0036, allow ~4 sigma).
    freqs = counts / trials
    assert np.all(np.abs(freqs - probs) < 0.015), freqs

    # Deterministic under a fixed seed.
    i1, g1 = gumbel_top_k(logp, 3, np.random.default_rng(42))
    i2, g2 = gumbel_top_k(logp, 3, np.random.default_rng(42))
    assert np.array_equal(i1, i2) and np.array_equal(g1, g2)


# ---- (b) sequential halving visit accounting ----

def test_sequential_halving_schedule_sums_to_budget():
    for m in (1, 2, 3, 4, 7, 16, 32):
        for budget in (0, 1, 5, 16, 32, 48, 64, 200):
            sched = sequential_halving_schedule(m, budget)
            assert sum(sum(a) for a in sched) == budget, (m, budget, sched)
            # phase sizes halve down to <=2 candidates, never grow
            sizes = [len(a) for a in sched]
            assert sizes[0] == max(1, m)
            for a, b in zip(sizes, sizes[1:]):
                assert b == max(1, a // 2)
            assert all(v >= 0 for a in sched for v in a)

    # the canonical paper-style case: m=16, n=64 -> 4 phases x 16 visits each
    sched = sequential_halving_schedule(16, 64)
    assert [len(a) for a in sched] == [16, 8, 4, 2]
    assert [sum(a) for a in sched] == [16, 16, 16, 16]


# ---- (c) 9x9 smoke game: legal move, budget respected, deterministic under seed ----

def test_gumbel_smoke_9x9():
    board = Board(9, 9)
    m = MCTS(_FakeEval(), komi=7.0, pos_len=9, gumbel_root=True, gumbel_m=8)
    root = m.run(board, visits=32, batch_size=4, seed=7)
    assert root.N == 32                       # halving schedule spends exactly the budget
    assert sum(ch.N for ch in root.children) == 32
    ranking = m.gumbel_ranking
    assert ranking, "gumbel search must produce a ranking"
    best = ranking[0]
    legal = board.legal_moves(board.to_move)
    assert best.move in legal or best.move is PASS
    assert best.N > 0
    # visits concentrated on at most m sampled candidates
    assert sum(1 for ch in root.children if ch.N > 0) <= 8

    # deterministic: same seed -> same visit distribution and same best move
    m2 = MCTS(_FakeEval(), komi=7.0, pos_len=9, gumbel_root=True, gumbel_m=8)
    root2 = m2.run(board, visits=32, batch_size=4, seed=7)
    assert [(ch.move, ch.N) for ch in root.children] == [(c.move, c.N) for c in root2.children]
    assert m2.gumbel_ranking[0].move == best.move


def test_gumbel_final_score_perspective():
    # sigma must be applied to Q in the ROOT mover's perspective: a candidate whose backed-up
    # value is great for the root mover (child.q() very negative) must rank above one whose
    # value is terrible, gumbel/prior being equal.
    m = MCTS(_FakeEval(), komi=7.0, pos_len=9, gumbel_root=True)
    root = m.run(Board(9, 9), visits=8, batch_size=1, seed=1)
    good, bad = root.children[0], root.children[1]
    # hand-craft stats: child stats are stored from the CHILD mover's view
    good.N, good.W = 4, -2.0      # child mean -0.5 -> +0.5 for the root mover
    bad.N, bad.W = 4, +2.0        # child mean +0.5 -> -0.5 for the root mover
    cands = [(bad, 0.0, math.log(0.5)), (good, 0.0, math.log(0.5))]
    ranked = m._gumbel_rank(root, cands)
    assert ranked[0][0] is good and ranked[1][0] is bad


# ---- (d) default-off equivalence: gumbel disabled == pre-change search, bit for bit ----

# Golden visit counts captured from the pre-change MCTS (commit before the gumbel feature):
# MCTS(_FakeEval(), komi=7.0, pos_len=9).run(Board(5, 5), visits=24, batch_size=4).
_GOLDEN_ROOT_Q = -0.17340232148388665
_GOLDEN_VISITS = {
    (0, 1): 1, (0, 2): 1, (0, 4): 1, (1, 1): 4, (1, 2): 1, (1, 3): 1,
    (2, 0): 1, (2, 2): 1, (2, 3): 1, (3, 0): 1, (3, 2): 4, (3, 3): 1,
    (3, 4): 1, (4, 0): 1, (4, 1): 1, (4, 3): 1, (4, 4): 1, PASS: 1,
}


def test_default_off_matches_pre_change_search():
    m = MCTS(_FakeEval(), komi=7.0, pos_len=9)   # gumbel_root defaults to False
    root = m.run(Board(5, 5), visits=24, batch_size=4)
    assert root.N == 24
    got = {ch.move: ch.N for ch in root.children if ch.N > 0}
    assert got == _GOLDEN_VISITS
    assert abs(root.q() - _GOLDEN_ROOT_Q) < 1e-15
    assert m.gumbel_ranking is None              # untouched by the default path
    assert rank_children(root.children)          # normal move selection still applies
