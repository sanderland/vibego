"""Referee: play two KataGo-analysis-protocol engines against each other.

Because nanogo's engine speaks the same JSON protocol as KataGo, this pits *any* two
engines: nanogo vs nanogo, or nanogo vs the real KataGo binary (e.g. its b6c96 net).
Each engine computes its own input features internally, so feature differences don't matter.

Examples:
    # nanogo vs nanogo
    uv run python scripts/vs.py \
        --black "uv run python scripts/run_engine.py -model checkpoints/depth6.pt" \
        --white "uv run python scripts/run_engine.py -model checkpoints/untrained.pt" \
        --visits 64 --board 19

    # nanogo toy vs real KataGo b6c96 (needs a katago binary + the b6c96 model)
    uv run python scripts/vs.py \
        --black "uv run python scripts/run_engine.py -model checkpoints/depth6.pt" \
        --white "katago analysis -model b6c96.bin.gz -config analysis.cfg" \
        --visits 64 --board 19
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.eval.selfplay import area_score
from nanogo.go.board import BLACK, WHITE, Board, gtp_to_xy


class Engine:
    def __init__(self, command: str):
        self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.n = 0

    def _query(self, query) -> dict:
        self.proc.stdin.write(json.dumps(query) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine closed unexpectedly")
            r = json.loads(line)
            if r.get("id") == query["id"] and not r.get("isDuringSearch"):
                return r

    def genmove(self, moves, komi, size, visits) -> str:
        self.n += 1
        r = self._query({"id": f"q{self.n}", "rules": "tromp-taylor", "komi": komi,
                         "boardXSize": size, "boardYSize": size, "moves": moves,
                         "analyzeTurns": [len(moves)], "maxVisits": visits})
        infos = r.get("moveInfos", [])
        return infos[0]["move"] if infos else "pass"

    def score(self, moves, komi, size, visits) -> float:
        """Engine's estimated final scoreLead from Black's perspective (KaTrain-style)."""
        self.n += 1
        r = self._query({"id": f"s{self.n}", "rules": "tromp-taylor", "komi": komi,
                         "boardXSize": size, "boardYSize": size, "moves": moves,
                         "analyzeTurns": [len(moves)], "maxVisits": visits,
                         "includeOwnership": True,
                         "overrideSettings": {"reportAnalysisWinratesAs": "BLACK"}})
        return r["rootInfo"]["scoreLead"]

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()


def play_once(black_cmd, white_cmd, args, judge):
    """Play one game; return (judge_or_area_score_BLACK_perspective, n_moves, finished)."""
    engines = {BLACK: Engine(black_cmd), WHITE: Engine(white_cmd)}
    names = {BLACK: "B", WHITE: "W"}
    board = Board(args.board, args.board)
    moves: list[list[str]] = []
    max_moves = args.max_moves or args.board * args.board * 2
    passes = 0
    try:
        for _ in range(max_moves):
            side = board.to_move
            mv = engines[side].genmove(moves, args.komi, args.board, args.visits)
            moves.append([names[side], mv])
            board.play(side, gtp_to_xy(mv, args.board))
            passes = passes + 1 if mv.lower() == "pass" else 0
            if passes >= 2:
                break
        if judge is not None:
            score = judge.score(moves, args.komi, args.board, args.judge_visits)
        else:
            score = area_score(board, args.komi)
    finally:
        engines[BLACK].close()
        engines[WHITE].close()
    return score, len(moves), passes >= 2


def main():
    import statistics
    p = argparse.ArgumentParser()
    p.add_argument("--black", required=True, help="command for engine A (Black in odd games)")
    p.add_argument("--white", required=True, help="command for engine B")
    p.add_argument("--visits", type=int, default=64)
    p.add_argument("--board", type=int, default=19)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--max-moves", type=int, default=None)
    p.add_argument("--judge", default=None, help="neutral judge engine (e.g. a strong KataGo b18)")
    p.add_argument("--judge-visits", type=int, default=256)
    p.add_argument("--games", type=int, default=1, help="play N games, alternating colors")
    args = p.parse_args()

    judge = Engine(args.judge) if args.judge else None
    try:
        a_scores = []  # from engine A's (=--black) perspective
        for g in range(args.games):
            a_black = (g % 2 == 0)
            bk, wh = (args.black, args.white) if a_black else (args.white, args.black)
            score_b, nmoves, finished = play_once(bk, wh, args, judge)
            a = score_b if a_black else -score_b
            a_scores.append(a)
            tag = "finished" if finished else "cap"
            print(f"game {g+1}: A-as-{'B' if a_black else 'W'} -> A {a:+.1f}  ({nmoves} mv, {tag})")
    finally:
        if judge is not None:
            judge.close()

    mean = sum(a_scores) / len(a_scores)
    metric = "judge scoreLead" if args.judge else "area"
    if len(a_scores) > 1:
        sd = statistics.pstdev(a_scores)
        se = sd / len(a_scores) ** 0.5
        print(f"\n{args.games} games @ {args.visits} visits — A ({metric}, B's perspective): "
              f"mean {mean:+.1f} ± {se:.1f} (stderr), per-game sd {sd:.1f}")
    else:
        print(f"\nresult: A {mean:+.1f} ({metric})")


if __name__ == "__main__":
    main()
