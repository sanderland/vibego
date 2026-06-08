"""A drop-in evaluator that gets raw policy/value/ownership from an external KataGo-protocol
engine instead of a local model. This lets nanogo's MCTS run on a *known-strong* net (e.g.
KataGo's b6c96), so we can test our tree search in isolation: if our search + b6c96 evals plays
as well as KataGo's own engine at equal visits, the search is sound; if not, it has a bug.

Positions are sent as stones (no move history), so the teacher's ko/recent-move features are
approximate — fine for a search-correctness check. Uses maxVisits=1 for the raw net by default.
"""
from __future__ import annotations

import json
import shlex
import subprocess

import numpy as np

from ..go.board import BLACK, EMPTY, PASS, WHITE, xy_to_gtp


class KataGoEvaluator:
    def __init__(self, command: str, visits: int = 1):
        self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.visits = visits
        self._buf: dict = {}
        self._n = 0
        self.forwards = 0          # observability parity with NNEvaluator
        self.max_batch_seen = 0

    def _send(self, q):
        self.proc.stdin.write(json.dumps(q) + "\n")
        self.proc.stdin.flush()

    def _recv(self, qid):
        if qid in self._buf:
            return self._buf.pop(qid)
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("proxy engine closed")
            r = json.loads(line)
            if r.get("isDuringSearch"):
                continue
            if r.get("id") == qid:
                return r
            self._buf[r.get("id")] = r

    def _query(self, board, komi, qid):
        q = {"id": qid, "rules": "tromp-taylor", "komi": round(komi * 2) / 2,
             "boardXSize": board.x_size, "boardYSize": board.y_size,
             "maxVisits": self.visits, "includePolicy": True, "includeOwnership": True}
        if board.move_history:
            # Built by replaying from empty (our self-play): send the moves so the teacher
            # recomputes exact features (ko, last-N moves, ladder history).
            moves = [["B" if p == BLACK else "W", "pass" if mv is PASS else xy_to_gtp(mv, board.y_size)]
                     for p, mv in board.move_history]
            q.update(initialStones=[], moves=moves, analyzeTurns=[len(moves)],
                     initialPlayer="B" if board.move_history[0][0] == BLACK else "W")
        else:
            # No history (e.g. positions given as stones): send the current stones directly.
            stones = []
            for y in range(board.y_size):
                for x in range(board.x_size):
                    v = board.grid[y, x]
                    if v == BLACK:
                        stones.append(["B", xy_to_gtp((x, y), board.y_size)])
                    elif v == WHITE:
                        stones.append(["W", xy_to_gtp((x, y), board.y_size)])
            q.update(initialStones=stones, moves=[], analyzeTurns=[0],
                     initialPlayer="B" if board.to_move == BLACK else "W")
        return q

    def _parse(self, r, board, pos_len):
        xs = board.x_size
        pol = r["policy"]
        # Only keep moves legal on OUR board: KataGo under tromp-taylor allows suicide / has
        # different superko, which our board rejects — expanding such a move would crash search.
        legal = set(board.legal_moves(board.to_move))
        policy, s = {}, 0.0
        for i, p in enumerate(pol[:-1]):
            if p >= 0 and (i % xs, i // xs) in legal:
                policy[(i % xs, i // xs)] = p
                s += p
        pp = max(0.0, pol[-1])
        policy[PASS] = pp
        s += pp
        if s > 0:
            policy = {k: v / s for k, v in policy.items()}
        own = np.zeros((pos_len, pos_len), dtype=np.float32)
        for i, o in enumerate(r["ownership"]):
            own[i // xs, i % xs] = o
        wr = float(r["rootInfo"]["winrate"])
        return {"v": 2.0 * wr - 1.0, "winrate": wr,
                "score": float(r["rootInfo"]["scoreLead"]), "policy": policy, "ownership": own}

    def evaluate_boards(self, boards, komi, pos_len):
        ids = []
        for b in boards:
            self._n += 1
            qid = f"e{self._n}"
            ids.append(qid)
            self._send(self._query(b, komi, qid))
        self.forwards += 1
        self.max_batch_seen = max(self.max_batch_seen, len(boards))
        return [self._parse(self._recv(qid), b, pos_len) for qid, b in zip(ids, boards)]
