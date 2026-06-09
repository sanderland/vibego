"""Self-play a short game with a trained checkpoint to sanity-check search + board.

    uv run python scripts/play_demo.py -model checkpoints/depth6.pt --moves 40 --visits 50
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.go.board import BLACK, Board, PASS, xy_to_gtp
from vibego.common import get_device
from vibego.engine.search import MCTS

# reuse the checkpoint loader
from run_engine import load_evaluator  # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-model", required=True)
    p.add_argument("--size", type=int, default=19)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--moves", type=int, default=40)
    p.add_argument("--visits", type=int, default=50)
    p.add_argument("-device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    ev, config = load_evaluator(args.model, device)

    board = Board(args.size, args.size)
    passes = 0
    for i in range(args.moves):
        mcts = MCTS(ev, args.komi, config.pos_len)
        root = mcts.run(board, args.visits)
        visited = [c for c in root.children if c.N > 0]
        if not visited:
            break
        best = max(visited, key=lambda c: c.N)
        wr = (1 + root.winloss()) / 2
        print(f"move {i+1}: {'B' if board.to_move == BLACK else 'W'} plays "
              f"{xy_to_gtp(best.move, args.size):4s} visits={best.N} winrate={wr:.2f} "
              f"score={root.eval['score']:+.1f}")
        if best.move is PASS:
            passes += 1
            if passes >= 2:
                print("two passes -> game over")
                break
        else:
            passes = 0
        board.play(board.to_move, best.move)

    print("\nfinal position:")
    print(board)


if __name__ == "__main__":
    main()
