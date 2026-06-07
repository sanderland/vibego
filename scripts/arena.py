"""Run a round-robin tournament between nanogo checkpoints and rank them by Bayesian Elo.

    uv run python scripts/arena.py --models checkpoints/depth6.pt checkpoints/depth10.pt \
        --games 20 --visits 100
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.common import get_device
from nanogo.eval.arena import Competitor, format_standings, run_tournament
from run_engine import load_evaluator  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="checkpoint paths")
    p.add_argument("--names", nargs="+", default=None, help="optional display names")
    p.add_argument("--games", type=int, default=20, help="games per pair")
    p.add_argument("--visits", type=int, default=100)
    p.add_argument("--board", type=int, default=19)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--opening-moves", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--quiet", action="store_true", help="don't log every game")
    args = p.parse_args()

    device = get_device(args.device)
    names = args.names or [os.path.splitext(os.path.basename(m))[0] for m in args.models]
    assert len(names) == len(args.models), "names must match models"

    competitors = []
    for name, path in zip(names, args.models):
        ev, config = load_evaluator(path, device)
        competitors.append(Competitor(name, ev, config.pos_len))
        print(f"loaded {name}: {path} ({config.num_blocks}b{config.channels}c) on {device}")

    standings, _wins = run_tournament(
        competitors, games_per_pair=args.games, visits=args.visits, board_size=args.board,
        komi=args.komi, opening_moves=args.opening_moves, temperature=args.temperature,
        seed=args.seed, log=(None if args.quiet else print),
    )
    print("\n=== Bayesian Elo standings ===")
    print(format_standings(standings))


if __name__ == "__main__":
    main()
