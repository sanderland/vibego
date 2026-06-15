"""Convert match.py --save-games JSONL records into source-format .npz position files.

Output matches the katagotraining source layout that relabel.py reads (full 22-channel packed
binary planes + 19-dim global; channels our encoder doesn't compute stay zero — nothing
downstream reads them), with zero targets: the point is to relabel these with the b18 teacher
and get STUDENT-TRAJECTORY training/eval positions, the off-policy diagnostic's missing half.

Positions are sampled before each move with ply >= --skip-plies (default: each game's forced
random opening, which is not the engines' play) at --stride to cut adjacent-ply correlation.

    uv run python scripts/game_positions.py --games runs/diag_games.jsonl \
        --out /workspace/distill/onpolicy --stride 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relabel import _write_npz  # noqa: E402  (same npz conventions, atomic write)
from vibego.go.board import BLACK, WHITE, Board, gtp_to_xy
from vibego.go.features import encode_board

FULL_SPATIAL = list(range(22))
FULL_GLOBAL = list(range(19))


def positions_from_game(rec: dict, pos_len: int, skip_plies: int, stride: int):
    """Yield (bin_full(22,P,P) uint8, glob_full(19) f32) for sampled positions of one game."""
    board = Board(rec["board"], rec["board"])
    names = {"B": BLACK, "W": WHITE}
    skip = rec.get("opening_plies", skip_plies) or skip_plies
    for ply, (side, gtp) in enumerate(rec["moves"]):
        if ply >= skip and (ply - skip) % stride == 0:
            sp, gl = encode_board(board, rec["komi"], pos_len, FULL_SPATIAL, FULL_GLOBAL)
            yield sp.astype(np.uint8), gl
        board.play(names[side], gtp_to_xy(gtp, rec["board"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="JSONL from match.py --save-games")
    p.add_argument("--out", required=True, help="output dir for .npz shards")
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--skip-plies", type=int, default=8, help="fallback when a record lacks opening_plies")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--shard-size", type=int, default=4096)
    args = p.parse_args()

    spatial, glob = [], []
    n_games = 0
    with open(args.games) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["board"] != args.pos_len:
                continue
            for sp, gl in positions_from_game(rec, args.pos_len, args.skip_plies, args.stride):
                spatial.append(sp)
                glob.append(gl)
            n_games += 1

    n = len(spatial)
    print(f"{n_games} games -> {n} positions (stride {args.stride})")
    pp1 = args.pos_len * args.pos_len + 1
    for s in range(0, n, args.shard_size):
        e = min(n, s + args.shard_size)
        sp = np.stack(spatial[s:e])                                    # (k, 22, P, P)
        packed = np.packbits(sp.reshape(sp.shape[0], 22, -1), axis=2)  # (k, 22, ceil(P*P/8))
        gl = np.stack(glob[s:e]).astype(np.float32)
        k = e - s
        _write_npz(os.path.join(args.out, f"game_pos_{s:08d}.npz"), packed, gl,
                   np.zeros((k, 1, pp1), np.float32), np.zeros((k, 64), np.float32),
                   np.zeros((k, 1, args.pos_len, args.pos_len), np.float32))
        print(f"  wrote game_pos_{s:08d}.npz ({k} rows)")


if __name__ == "__main__":
    main()
