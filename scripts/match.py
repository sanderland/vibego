"""Statistical head-to-head match between two KataGo-analysis-protocol engines.

Unlike vs.py (which plays games serially and reports a raw mean), this plays many games
**concurrently** (each game is independent, so wall-clock ≈ serial/​workers) and reports both
a continuous metric (mean judge scoreLead ± stderr) and a win/loss **Elo difference with a
95% CI** — enough to resolve the small per-visit search differences that single-digit-game
runs cannot (per-game sd is ~40 pts at 48 visits, so ~8 games can't see a 20-pt change).

    CFG=/opt/homebrew/Cellar/katago/1.16.4/share/katago/configs/analysis_example.cfg
    KATA="katago analysis -model models/g170-b6c96.bin.gz -config $CFG"
    JUDGE="katago analysis -model models/kata1-b18c384nbt.bin.gz -config $CFG"
    uv run python scripts/match.py --games 64 --visits 48 --workers 4 \
      --a "uv run python scripts/run_engine.py -pos-len 19 -proxy \"$KATA\"" --a-name our-search \
      --b "$KATA" --b-name katago \
      --judge "$JUDGE" --judge-visits 256

A is Black in even games, White in odd, so colors are balanced. All scores are reported from
A's perspective (positive = A ahead / A favored).
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vs import Engine, play_once  # noqa: E402  (reuse the referee)


class _LockedJudge:
    """One shared judge engine, serialized: concurrent games finish and score at once, but the
    final-position scoring is fast relative to a whole game, so one judge with a lock suffices."""

    def __init__(self, command: str):
        self._e = Engine(command)
        self._lock = threading.Lock()

    def score(self, *a):
        with self._lock:
            return self._e.score(*a)

    def close(self):
        self._e.close()


def winrate_to_elo(p: float) -> float:
    p = min(1.0 - 1e-9, max(1e-9, p))
    return -400.0 * math.log10(1.0 / p - 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="command for engine A")
    p.add_argument("--b", required=True, help="command for engine B")
    p.add_argument("--a-name", default="A")
    p.add_argument("--b-name", default="B")
    p.add_argument("--games", type=int, default=64)
    p.add_argument("--workers", type=int, default=4, help="games to run concurrently")
    p.add_argument("--visits", type=int, default=48)
    p.add_argument("--board", type=int, default=19)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--max-moves", type=int, default=200)
    p.add_argument("--judge", default=None, help="neutral judge (e.g. a strong KataGo b18)")
    p.add_argument("--judge-visits", type=int, default=256)
    p.add_argument("--opening-plies", type=int, default=8,
                   help="forced random opening stones for game diversity (0 = deterministic; "
                        "needed when both engines are deterministic, e.g. vibego-vs-vibego)")
    p.add_argument("--opening-seed", type=int, default=0)
    args = p.parse_args()

    judge = _LockedJudge(args.judge) if args.judge else None
    print_lock = threading.Lock()

    def play(g: int):
        a_black = (g % 2 == 0)
        bk, wh = (args.a, args.b) if a_black else (args.b, args.a)
        # color-reversed pair (g, g+1) shares an opening seed → balanced
        score_b, nmoves, finished = play_once(bk, wh, args, judge,
                                              opening_seed=args.opening_seed + g // 2)
        a = score_b if a_black else -score_b  # A's perspective
        with print_lock:
            tag = "fin" if finished else "cap"
            print(f"game {g + 1:>3}: {args.a_name}-as-{'B' if a_black else 'W'} "
                  f"-> {a:+.1f}  ({nmoves} mv, {tag})", flush=True)
        return a

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            a_scores = list(ex.map(play, range(args.games)))
    finally:
        if judge is not None:
            judge.close()

    n = len(a_scores)
    mean = sum(a_scores) / n
    se = statistics.pstdev(a_scores) / n ** 0.5 if n > 1 else float("nan")
    metric = "judge scoreLead" if args.judge else "area"

    # Win/loss Elo (draws = 0.5). Normal-approx CI on the win rate -> Elo.
    wins = sum(1.0 if a > 0 else 0.0 if a < 0 else 0.5 for a in a_scores)
    wr = wins / n
    se_wr = (wr * (1.0 - wr) / n) ** 0.5
    elo = winrate_to_elo(wr)
    elo_lo, elo_hi = winrate_to_elo(wr - 1.96 * se_wr), winrate_to_elo(wr + 1.96 * se_wr)

    # PAIRED scoreLead (the sensitive, low-variance estimator): games (2k, 2k+1) share an opening
    # with A as Black then White, so the pair-mean cancels the opening + color variance that
    # dominates per-game noise. CI over pair-means is much tighter than the per-game stderr above,
    # and scoreLead resolves small gaps that win-rate Elo can't (parity: -8.7±5.4 vs ±214 Elo).
    pairs = [(a_scores[i] + a_scores[i + 1]) / 2 for i in range(0, n - n % 2, 2)]
    pmean = sum(pairs) / len(pairs) if pairs else float("nan")
    pse = statistics.pstdev(pairs) / len(pairs) ** 0.5 if len(pairs) > 1 else float("nan")
    plo, phi = pmean - 1.96 * pse, pmean + 1.96 * pse

    print(f"\n{n} games @ {args.visits} visits ({args.a_name} vs {args.b_name}), "
          f"{args.a_name}'s perspective:")
    print(f"  {metric}: {mean:+.1f} ± {se:.1f} (stderr), per-game sd "
          f"{statistics.pstdev(a_scores):.1f}")
    print(f"  PAIRED {metric}: {pmean:+.2f} ± {pse:.2f}  [95% CI {plo:+.2f}, {phi:+.2f}]  "
          f"({len(pairs)} pairs)  decisive={'yes' if pairs and (plo > 0 or phi < 0) else 'no'}")
    print(f"  win rate: {wr * 100:.1f}% ({wins:.1f}/{n})  ->  "
          f"Elo {elo:+.0f}  [95% CI {elo_lo:+.0f}, {elo_hi:+.0f}]")


if __name__ == "__main__":
    main()
