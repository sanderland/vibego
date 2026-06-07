import numpy as np

from nanogo.eval.arena import Competitor, run_tournament
from nanogo.eval.elo import bayes_elo
from nanogo.eval.selfplay import area_score, play_game
from nanogo.engine.search import NNEvaluator
from nanogo.go.board import BLACK, WHITE, Board
from nanogo.net.model import Model, ModelConfig


# ---- area scoring ----
def test_area_score_empty_is_komi():
    assert area_score(Board(3), komi=7.5) == -7.5  # all dame -> only komi counts


def test_area_score_single_stone_controls_board():
    b = Board(3)
    b.grid[1, 1] = BLACK  # one black stone, rest empty bordered only by black
    assert area_score(b, komi=0.0) == 9.0  # 1 stone + 8 territory


def test_area_score_dame_between_colors():
    b = Board(3)
    b.grid[0, 0] = BLACK
    b.grid[2, 2] = WHITE
    # remaining empties border both colors -> neutral
    assert area_score(b, komi=0.0) == 0.0


# ---- bayesian elo ----
def test_bayes_elo_orders_players():
    # A beats B and C every time; B beats C every time.
    wins = [
        [0, 10, 10],
        [0, 0, 10],
        [0, 0, 0],
    ]
    elo, se = bayes_elo(wins)
    assert elo[0] > elo[1] > elo[2]
    assert np.isclose(elo.mean(), 0.0, atol=1e-6)
    assert np.all(np.isfinite(elo)) and np.all(se > 0)


def test_bayes_elo_undefeated_is_finite():
    # A never loses; prior keeps its rating finite.
    wins = [
        [0, 8, 8],
        [0, 0, 4],
        [0, 4, 0],
    ]
    elo, se = bayes_elo(wins)
    assert np.all(np.isfinite(elo))
    assert elo[0] == max(elo)


def test_bayes_elo_equal_players():
    wins = [[0, 5], [5, 0]]
    elo, se = bayes_elo(wins)
    assert abs(elo[0] - elo[1]) < 1e-6


# ---- self-play + tournament (tiny untrained nets, small board, fast) ----
def _tiny_competitor(name, device="cpu"):
    cfg = ModelConfig.plain(8, 2)
    model = Model(cfg)
    return Competitor(name, NNEvaluator(model, device), cfg.pos_len)


def test_play_game_returns_valid_result():
    a = _tiny_competitor("a")
    b = _tiny_competitor("b")
    res = play_game(a.evaluator, b.evaluator, a.pos_len, board_size=7, komi=2.5,
                    visits=4, opening_moves=0, max_moves=30, seed=1)
    assert res["winner"] in ("B", "W", "D")
    assert res["moves"] > 0


def test_tournament_runs_and_ranks():
    comps = [_tiny_competitor("a"), _tiny_competitor("b")]
    standings, wins = run_tournament(comps, games_per_pair=2, visits=4, board_size=7,
                                     komi=2.5, opening_moves=0, seed=0, log=None)
    assert len(standings) == 2
    assert {s["name"] for s in standings} == {"a", "b"}
    assert all(np.isfinite(s["elo"]) for s in standings)
    # total decided games (wins) sum to games played (2), draws split as 0.5/0.5
    assert abs(sum(s["wins"] for s in standings) - 2.0) < 1e-6
