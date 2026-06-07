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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--black", required=True, help="command for the Black engine")
    p.add_argument("--white", required=True, help="command for the White engine")
    p.add_argument("--visits", type=int, default=64)
    p.add_argument("--board", type=int, default=19)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--max-moves", type=int, default=None)
    p.add_argument("--judge", default=None,
                   help="command for a neutral judge engine (e.g. a strong KataGo b18)")
    p.add_argument("--judge-visits", type=int, default=256)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    engines = {BLACK: Engine(args.black), WHITE: Engine(args.white)}
    names = {BLACK: "B", WHITE: "W"}
    board = Board(args.board, args.board)
    moves: list[list[str]] = []
    max_moves = args.max_moves or args.board * args.board * 2
    passes = 0
    try:
        for i in range(max_moves):
            side = board.to_move
            mv = engines[side].genmove(moves, args.komi, args.board, args.visits)
            moves.append([names[side], mv])
            board.play(side, gtp_to_xy(mv, args.board))
            if args.verbose:
                print(f"{i+1}: {names[side]} {mv}")
            passes = passes + 1 if mv.lower() == "pass" else 0
            if passes >= 2:
                break

        # Players' own estimates (cheap) and objective area count, for reference.
        finished = passes >= 2
        eb = engines[BLACK].score(moves, args.komi, args.board, args.visits)
        ew = engines[WHITE].score(moves, args.komi, args.board, args.visits)
        engine_score = (eb + ew) / 2.0  # Black perspective
        area = area_score(board, args.komi)
        # Authoritative verdict from a neutral judge engine, if given.
        judge_score = None
        if args.judge:
            judge = Engine(args.judge)
            try:
                judge_score = judge.score(moves, args.komi, args.board, args.judge_visits)
            finally:
                judge.close()
    finally:
        engines[BLACK].close()
        engines[WHITE].close()

    verdict = judge_score if judge_score is not None else engine_score
    winner = "Black" if verdict > 0 else ("White" if verdict < 0 else "Draw")
    print(f"\n{len(moves)} moves, {'finished' if finished else 'hit move cap'}")
    print(f"player estimates (B): black-net {eb:+.1f}, white-net {ew:+.1f}")
    print(f"rules-based area (B): {area:+.1f}")
    if judge_score is not None:
        print(f"judge scoreLead (B): {judge_score:+.1f}  [{args.judge_visits} visits]")
    print(f"result: {winner}+{abs(verdict):.1f}")


if __name__ == "__main__":
    main()
