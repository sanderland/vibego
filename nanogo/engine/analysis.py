"""A minimal KataGo-compatible JSON analysis engine.

Speaks the subset of the analysis protocol that KaTrain uses: line-delimited JSON
queries on stdin, line-delimited JSON results on stdout. Winrates/scores/ownership are
reported from Black's perspective when overrideSettings.reportAnalysisWinratesAs == "BLACK"
(KaTrain forces this), otherwise from the side to move.

Concurrency: a reader thread parses stdin so that `terminate` / `query_version` are handled
immediately. A single scheduler thread runs all active searches round-robin (a chunk of
playouts per job per pass), so short queries (e.g. an AI move) finish promptly even while a
never-ending ponder (maxVisits=10000000) keeps running. Partial results (`isDuringSearch:
true`) stream every `reportDuringSearchEvery` seconds; terminated jobs stop and emit no final.

Reference: katago/docs/Analysis_Engine.md ; consumer: katrain/core/engine.py + game_node.py.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time

from ..go.board import BLACK, WHITE, Board, PASS, gtp_to_xy, opp, xy_to_gtp
from .search import MCTS, NNEvaluator, adaptive_batch, rank_children


def _player(s: str) -> int:
    return BLACK if s.upper().startswith("B") else WHITE


def _player_str(p: int) -> str:
    return "B" if p == BLACK else "W"


class AnalysisEngine:
    def __init__(self, evaluator: NNEvaluator, pos_len: int,
                 default_visits: int = 100, leaf_batch: int = 16, lcb_stdevs: float = 1.0):
        self.ev = evaluator
        self.pos_len = pos_len
        self.default_visits = default_visits
        self.leaf_batch = leaf_batch  # leaves collected per search step (virtual loss)
        self.lcb_stdevs = lcb_stdevs  # move selection LCB width (0 = pure mean-value)
        self._terminated: set[str] = set()
        self._term_epoch = 0  # incremented by terminate_all; a query started earlier is dead
        self._queue: queue.Queue = queue.Queue()
        self._out_lock = threading.Lock()

    # ---- output ----
    def _emit(self, obj: dict):
        with self._out_lock:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()

    # ---- board construction ----
    def _build_board(self, q: dict, upto: int) -> Board:
        xs = int(q.get("boardXSize", 19))
        ys = int(q.get("boardYSize", 19))
        b = Board(xs, ys)
        for color, coord in q.get("initialStones", []):
            mv = gtp_to_xy(coord, ys)
            if mv is not PASS:
                x, y = mv
                b.grid[y, x] = _player(color)
        moves = q.get("moves", [])
        if moves:
            b.to_move = _player(moves[0][0])
        elif q.get("initialPlayer"):
            b.to_move = _player(q["initialPlayer"])
        else:
            b.to_move = BLACK
        for color, coord in moves[:upto]:
            b.play(_player(color), gtp_to_xy(coord, ys))
        return b

    # ---- result building from a (partially) searched root ----
    def _build_result(self, q: dict, turn: int, root, during_search: bool) -> dict:
        ys = int(q.get("boardYSize", 19))
        xs = int(q.get("boardXSize", 19))
        report_as = q.get("overrideSettings", {}).get("reportAnalysisWinratesAs", "SIDETOMOVE")
        mover = root.board.to_move
        as_black = report_as.upper() == "BLACK"
        sign = 1.0 if (not as_black or mover == BLACK) else -1.0

        # Search-averaged win-loss and score (reporting uses these, not the search utility).
        root_wl = root.winloss() if root.N else root.eval["v"]
        root_score = root.score() if root.N else root.eval["score"]
        root_winrate = (1.0 + root_wl) / 2.0
        root_info = {
            "visits": root.N,
            "winrate": root_winrate if sign > 0 else 1.0 - root_winrate,
            "scoreLead": sign * root_score,
            "scoreSelfplay": sign * root_score,
            "scoreStdev": 0.0,
            "utility": (root.q() if sign > 0 else -root.q()),
            "currentPlayer": _player_str(mover),
        }

        # Order by LCB (robust value), not raw visit count — what gets played is moveInfos[0].
        visited = rank_children(root.children, self.lcb_stdevs)
        move_infos = []
        for order, ch in enumerate(visited):
            mv_wl = -ch.winloss()      # child stats are in the opponent's perspective
            mv_score = -ch.score()
            wr = (1.0 + mv_wl) / 2.0
            move_infos.append({
                "move": xy_to_gtp(ch.move, ys),
                "visits": ch.N,
                "winrate": wr if sign > 0 else 1.0 - wr,
                "scoreLead": sign * mv_score,
                "scoreSelfplay": sign * mv_score,
                "scoreStdev": 0.0,
                "prior": ch.P,
                "utility": (-ch.q()) if sign > 0 else ch.q(),
                "order": order,
                "pv": self._pv(ch, ys),
            })

        result = {
            "id": q.get("id"),
            "turnNumber": turn,
            "moveInfos": move_infos,
            "rootInfo": root_info,
            "isDuringSearch": during_search,
        }
        if q.get("includePolicy"):
            result["policy"] = self._policy_array(root, xs, ys)
        if q.get("includeOwnership"):
            result["ownership"] = self._ownership_array(root, xs, ys, sign)
        return result

    def _pv(self, node, ys) -> list[str]:
        pv, cur = [], node
        for _ in range(20):
            pv.append(xy_to_gtp(cur.move, ys))
            visited = [c for c in cur.children if c.N > 0]
            if not visited:
                break
            cur = max(visited, key=lambda c: c.N)
        return pv

    def _policy_array(self, root, xs, ys) -> list[float]:
        pol = root.eval["policy"]
        arr = [-1.0] * (xs * ys + 1)
        for mv, p in pol.items():
            if mv is PASS:
                arr[-1] = p
            else:
                x, y = mv
                arr[y * xs + x] = p
        return arr

    def _ownership_array(self, root, xs, ys, sign) -> list[float]:
        own = root.eval["ownership"]
        return [float(sign * own[y, x]) for y in range(ys) for x in range(xs)]

    # ---- running a search with streaming reports ----
    def _is_terminated(self, qid, start_epoch: int) -> bool:
        # Terminated if this id was terminated, or a terminate_all arrived after we started.
        return qid in self._terminated or self._term_epoch > start_epoch

    def _run_one(self, q: dict, turn: int):
        qid = q.get("id")
        start_epoch = self._term_epoch
        if self._is_terminated(qid, start_epoch):
            return
        komi = float(q.get("komi", 7.5))
        visits = int(q.get("maxVisits", self.default_visits))
        report_every = q.get("reportDuringSearchEvery")
        board = self._build_board(q, turn)
        mcts = MCTS(self.ev, komi, self.pos_len)
        root = mcts.prepare(board)
        last_report = time.monotonic()
        done = 0
        while done < visits:
            if self._is_terminated(qid, start_epoch):
                return  # discarded; KaTrain doesn't want stale results
            done += mcts.step(root, adaptive_batch(self.leaf_batch, done, visits))
            if report_every and (time.monotonic() - last_report) >= float(report_every):
                self._emit(self._build_result(q, turn, root, during_search=True))
                last_report = time.monotonic()
        if not self._is_terminated(qid, start_epoch):
            self._emit(self._build_result(q, turn, root, during_search=False))

    def _process_query(self, q: dict):
        turns = q.get("analyzeTurns")
        if turns is None:
            turns = [len(q.get("moves", []))]
        for turn in turns:
            self._run_one(q, turn)

    # ---- stdin reader (immediate handling of control actions) ----
    def _reader(self):
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = q.get("action")
            if action == "query_version":
                self._emit({"id": q.get("id"), "version": "1.0.0-nanogo"})
            elif action == "terminate":
                tid = q.get("terminateId")
                if tid is not None:
                    self._terminated.add(tid)
                self._emit({"id": q.get("id"), "terminateId": tid})
            elif action == "terminate_all":
                self._term_epoch += 1  # everything started before now is terminated
                self._emit({"id": q.get("id"), "action": "terminate_all"})
            elif action == "clear_cache":
                self._emit({"id": q.get("id"), "action": "clear_cache"})
            else:
                self._queue.put(q)
        self._queue.put(None)  # EOF sentinel

    def _worker(self, q: dict):
        try:
            self._process_query(q)
        except Exception as e:
            self._emit({"id": q.get("id"), "error": str(e)})

    def run(self):
        sys.stderr.write("nanogo analysis engine: started, ready to begin handling requests.\n")
        sys.stderr.flush()
        reader = threading.Thread(target=self._reader, daemon=True)
        reader.start()
        # One thread per query: concurrent searches share the batching evaluator, so a
        # ponder and other queries (AI move, analysis) run at once and batch together.
        threads: list[threading.Thread] = []
        while True:
            q = self._queue.get()
            if q is None:
                break
            t = threading.Thread(target=self._worker, args=(q,), daemon=True)
            t.start()
            threads.append(t)
            threads = [t for t in threads if t.is_alive()]
        for t in threads:
            t.join()
