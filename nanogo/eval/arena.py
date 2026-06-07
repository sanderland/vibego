"""Round-robin tournament between networks, scored with Bayesian Elo."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .elo import bayes_elo
from .selfplay import play_game


@dataclass
class Competitor:
    name: str
    evaluator: object   # an NNEvaluator
    pos_len: int


def run_tournament(competitors: list[Competitor], games_per_pair: int = 10, visits: int = 100,
                   board_size: int = 19, komi: float = 7.5, opening_moves: int = 8,
                   temperature: float = 1.0, seed: int = 0,
                   log: Callable[[str], None] | None = None):
    """Play every pair `games_per_pair` times (alternating colors); return standings + wins."""
    n = len(competitors)
    wins = [[0.0] * n for _ in range(n)]
    gid = 0
    for i in range(n):
        for j in range(i + 1, n):
            for g in range(games_per_pair):
                b, w = (i, j) if g % 2 == 0 else (j, i)
                res = play_game(
                    competitors[b].evaluator, competitors[w].evaluator, competitors[b].pos_len,
                    board_size=board_size, komi=komi, visits=visits,
                    opening_moves=opening_moves, temperature=temperature, seed=seed + gid,
                )
                gid += 1
                if res["winner"] == "B":
                    wins[b][w] += 1.0
                elif res["winner"] == "W":
                    wins[w][b] += 1.0
                else:
                    wins[b][w] += 0.5
                    wins[w][b] += 0.5
                if log:
                    log(f"game {gid}: {competitors[b].name}(B) vs {competitors[w].name}(W)"
                        f" -> {res['winner']} (score {res['score_black']:+.1f}, {res['moves']} mv)")

    elo, stderr = bayes_elo(wins)
    standings = []
    for k, c in enumerate(competitors):
        standings.append({
            "name": c.name,
            "elo": float(elo[k]),
            "stderr": float(stderr[k]),
            "wins": sum(wins[k]),
            "losses": sum(wins[r][k] for r in range(n)),
        })
    standings.sort(key=lambda s: -s["elo"])
    return standings, wins


def format_standings(standings) -> str:
    lines = [f"{'#':>2}  {'name':<24} {'Elo':>7}  {'95% CI':>9}   {'W-L':>11}"]
    for rank, s in enumerate(standings, 1):
        wl = f"{s['wins']:.1f}-{s['losses']:.1f}"
        lines.append(f"{rank:>2}  {s['name']:<24} {s['elo']:>7.0f}  ±{1.96 * s['stderr']:>7.0f}   {wl:>11}")
    return "\n".join(lines)
