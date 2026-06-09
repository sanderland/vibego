"""Low-noise per-move search diagnostic. On a set of fixed positions, ask our engine and a
reference engine each for their move at the same visits, then have a strong judge (b18) score
the position after each move. The mean "points conceded vs the reference per move" is a clean,
low-variance measure of search quality — far less noisy than full games.

Tests both to-move colors so a color asymmetry shows up.

    uv run python scripts/move_eval.py --src data --n 40 --visits 48 \
      --ours  'uv run python scripts/run_engine.py -pos-len 19 -proxy "katago analysis -model models/g170-b6c96.bin.gz -config CFG"' \
      --ref   "katago analysis -model models/g170-b6c96.bin.gz -config CFG" \
      --judge "katago analysis -model models/kata1-b18c384nbt.bin.gz -config CFG"
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import relabel  # noqa: E402  (TeacherEngine send/recv)
from vibego.go.board import BLACK, WHITE, xy_to_gtp  # noqa: E402
from vibego.net import data as ndata  # noqa: E402


def position(bin_row, glob_row, pos_len, tomove):
    """Build (stones, initialPlayer, komi, xs, ys) for a position with `tomove` to play."""
    onb = bin_row[0]
    xs, ys = int(onb[0].sum()), int(onb[:, 0].sum())
    own, opp = bin_row[1], bin_row[2]
    own_c = "B" if tomove == BLACK else "W"
    opp_c = "W" if tomove == BLACK else "B"
    stones = []
    for y in range(ys):
        for x in range(xs):
            if own[y, x]:
                stones.append([own_c, xy_to_gtp((x, y), ys)])
            elif opp[y, x]:
                stones.append([opp_c, xy_to_gtp((x, y), ys)])
    self_komi = float(glob_row[5]) * 20.0
    white_komi = self_komi if tomove == WHITE else -self_komi
    return stones, own_c, round(white_komi * 2) / 2, xs, ys


def genmove(eng, stones, player, komi, xs, ys, visits, qid):
    eng.send({"id": qid, "rules": "tromp-taylor", "komi": komi, "boardXSize": xs, "boardYSize": ys,
              "initialStones": stones, "initialPlayer": player, "moves": [],
              "analyzeTurns": [0], "maxVisits": visits})
    r = eng.recv(qid)
    infos = r.get("moveInfos", [])
    return infos[0]["move"] if infos else "pass"


def score_after(judge, stones, player, move, komi, xs, ys, visits, qid):
    """b18 scoreLead (Black perspective) after `player` plays `move`."""
    judge.send({"id": qid, "rules": "tromp-taylor", "komi": komi, "boardXSize": xs, "boardYSize": ys,
                "initialStones": stones, "initialPlayer": player, "moves": [[player, move]],
                "analyzeTurns": [1], "maxVisits": visits,
                "overrideSettings": {"reportAnalysisWinratesAs": "BLACK"}})
    return judge.recv(qid)["rootInfo"]["scoreLead"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--visits", type=int, default=48)
    p.add_argument("--judge-visits", type=int, default=256)
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--ours", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--judge", required=True)
    args = p.parse_args()

    rows = []
    for path in ndata.list_npz(args.src):
        if len(rows) >= args.n:
            break
        with np.load(path) as npz:
            b = np.unpackbits(npz["binaryInputNCHWPacked"], axis=2)[:, :, : args.pos_len ** 2]
            b = b.reshape(b.shape[0], 22, args.pos_len, args.pos_len)
            g = npz["globalInputNC"]
        for i in range(b.shape[0]):
            if len(rows) >= args.n:
                break
            rows.append((b[i], g[i]))
    print(f"{len(rows)} positions")

    ours = relabel.TeacherEngine(args.ours)
    ref = relabel.TeacherEngine(args.ref)
    judge = relabel.TeacherEngine(args.judge)
    try:
        for tomove, label in [(BLACK, "Black"), (WHITE, "White")]:
            sign = 1.0 if tomove == BLACK else -1.0
            losses = []
            for k, (bin_row, glob_row) in enumerate(rows):
                stones, player, komi, xs, ys = position(bin_row, glob_row, args.pos_len, tomove)
                om = genmove(ours, stones, player, komi, xs, ys, args.visits, f"o{k}")
                rm = genmove(ref, stones, player, komi, xs, ys, args.visits, f"r{k}")
                if om == rm:
                    losses.append(0.0)
                    continue
                oa = score_after(judge, stones, player, om, komi, xs, ys, args.judge_visits, f"oa{k}")
                ra = score_after(judge, stones, player, rm, komi, xs, ys, args.judge_visits, f"ra{k}")
                losses.append(sign * (ra - oa))  # >0 means our move was worse than ref's
            arr = np.array(losses)
            se = arr.std() / len(arr) ** 0.5
            print(f"{label}-to-move: our move loses {arr.mean():+.2f} ± {se:.2f} pts/move vs ref "
                  f"(differed on {int((arr != 0).sum())}/{len(arr)})")
    finally:
        ours.close(); ref.close(); judge.close()


if __name__ == "__main__":
    main()
