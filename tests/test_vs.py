"""Tests for the referee's randomized openings — the fix for deterministic self-play (two
deterministic engines otherwise replay one identical game per colour, making the arena degenerate)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import vs  # noqa: E402

from vibego.go.board import BLACK, WHITE  # noqa: E402


def test_random_opening_seed_deterministic_and_distinct():
    a = vs.random_opening(19, 8, seed=3)
    assert a == vs.random_opening(19, 8, seed=3)      # same seed -> identical opening
    assert a != vs.random_opening(19, 8, seed=4)      # different seed -> different
    assert len(a) == 8
    pts = [xy for _, xy in a]
    assert len(set(pts)) == 8                         # distinct points (all legal on empty board)
    assert [s for s, _ in a] == [BLACK, WHITE] * 4    # alternating colours, Black first
    assert all(min(x, 18 - x) >= 2 and min(y, 18 - y) >= 2 for x, y in pts)  # 3rd line inward


def test_random_opening_zero_plies_is_noop():
    assert vs.random_opening(19, 0, 0) == []
