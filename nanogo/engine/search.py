"""PUCT MCTS with search/inference decoupled, KataGo-style.

- `NNEvaluator` runs on its own thread: it collects feature tensors from *all* callers
  (every concurrent query, and every leaf in a single query's batch), stacks them into one
  batch, runs a single `model()` forward, and scatters raw outputs back. This is what keeps
  the device busy and lets many searches share batches.
- `MCTS` does *leaf-parallel* search: each `step()` selects a batch of distinct leaves using
  **virtual loss** (so the same leaf isn't picked repeatedly), encodes them, evaluates the
  whole batch in one evaluator call, then expands and backs them all up. Each query owns its
  own tree and is driven by a single thread, so there are no shared-tree races.

Value convention: every node stores stats from the perspective of the player to move at that
node; a child's mean value is the opponent's, so the parent negates it.
"""
from __future__ import annotations

import math
import queue
import threading

import numpy as np
import torch

from ..go.board import PASS, Board
from ..go.features import encode_board


class _Req:
    __slots__ = ("feats", "out", "error", "event")

    def __init__(self, feats):
        self.feats = feats          # list of (spatial, glob) numpy arrays
        self.out = None             # list of raw output tuples
        self.error = None           # exception from the eval thread, re-raised in infer()
        self.event = threading.Event()


class NNEvaluator:
    """Batches feature tensors from all callers into single forward passes."""

    def __init__(self, model, device, max_batch: int = 64, max_wait: float = 0.003,
                 spatial_subset=None, global_subset=None):
        from ..go import features as _F
        self.model = model
        self.device = device
        self.spatial_subset = spatial_subset if spatial_subset is not None else _F.SPATIAL_SUBSET
        self.global_subset = global_subset if global_subset is not None else _F.GLOBAL_SUBSET
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.max_batch_seen = 0   # observability: largest batch actually run
        self.forwards = 0         # number of forward passes
        self.model.eval()
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def infer(self, feats):
        """feats: list of (spatial, glob). Returns list of (policy_logits, value_probs,
        score, ownership) raw arrays, one per input. Blocks until the batch runs."""
        req = _Req(feats)
        self._q.put(req)
        req.event.wait()
        if req.error is not None:
            raise req.error
        return req.out

    def evaluate_boards(self, boards, komi: float, pos_len: int) -> list[dict]:
        """Board -> eval dict (encode our feature subset, batched forward, legal-masked)."""
        feats = [encode_board(b, komi, pos_len, self.spatial_subset, self.global_subset)
                 for b in boards]
        raws = self.infer(feats)
        return [raw_to_eval(b, raw, pos_len) for b, raw in zip(boards, raws)]

    def _loop(self):
        from time import monotonic
        while True:
            first = self._q.get()
            reqs = [first]
            total = len(first.feats)
            deadline = monotonic() + self.max_wait
            while total < self.max_batch:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    break
                try:
                    r = self._q.get(timeout=timeout)
                except queue.Empty:
                    break
                reqs.append(r)
                total += len(r.feats)
            self._run(reqs)

    @torch.no_grad()
    def _run(self, reqs):
        try:
            spatials = [s for r in reqs for (s, _g) in r.feats]
            globs = [g for r in reqs for (_s, g) in r.feats]
            S = torch.from_numpy(np.stack(spatials)).to(self.device)
            G = torch.from_numpy(np.stack(globs)).to(self.device)
            self.max_batch_seen = max(self.max_batch_seen, S.shape[0])
            self.forwards += 1
            policy, value, score, ownership = self.model(S, G)
            policy = policy.cpu().numpy()
            value = torch.softmax(value, dim=1).cpu().numpy()
            score = score.cpu().numpy()
            ownership = ownership[:, 0].cpu().numpy()
            i = 0
            for r in reqs:
                out = []
                for _ in range(len(r.feats)):
                    out.append((policy[i], value[i], float(score[i]), ownership[i]))
                    i += 1
                r.out = out
                r.event.set()
        except Exception as e:
            # Never leave callers blocked on req.event: surface the error in infer().
            for r in reqs:
                if not r.event.is_set():
                    r.error = e
                    r.event.set()


def raw_to_eval(board: Board, raw, pos_len: int) -> dict:
    """Turn a raw net output for `board` into an eval dict (legal-masked policy, value, ...)."""
    policy_logits, value, score, ownership = raw
    moves = board.legal_moves(board.to_move)
    idxs = [y * pos_len + x for (x, y) in moves]
    idxs.append(pos_len * pos_len)  # pass
    moves = moves + [PASS]
    logits = policy_logits[idxs].astype(np.float64)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    policy = {mv: float(p) for mv, p in zip(moves, probs)}
    v = float(value[0] - value[1])
    winrate = float(value[0] + 0.5 * value[2])
    return {"v": v, "winrate": winrate, "score": score, "policy": policy, "ownership": ownership}


class Node:
    # W = sum of search *utility* (winloss + score) used for selection;
    # Wwl / Wsc = sums of win-loss value and score lead, kept separately for reporting.
    __slots__ = ("move", "P", "N", "W", "Wwl", "Wsc", "vloss", "board", "eval",
                 "children", "expanded")

    def __init__(self, move, prior):
        self.move = move
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Wwl = 0.0          # win-loss value sum (reporting)
        self.Wsc = 0.0          # score-lead sum (reporting)
        self.vloss = 0          # outstanding virtual losses
        self.board: Board | None = None
        self.eval = None
        self.children: list[Node] = []
        self.expanded = False

    def q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def winloss(self):
        return self.Wwl / self.N if self.N > 0 else 0.0

    def score(self):
        return self.Wsc / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, evaluator: NNEvaluator, komi: float, pos_len: int,
                 c_puct: float = 1.0, fpu: float = 0.25, vloss_weight: float = 1.0,
                 score_weight: float = 0.5, score_scale: float = 30.0,
                 c_puct_log: float = 0.45, c_puct_base: float = 500.0):
        self.ev = evaluator
        self.komi = komi
        self.pos_len = pos_len
        self.c_puct = c_puct
        self.c_puct_log = c_puct_log    # cpuct grows ~log(visits), KataGo-style
        self.c_puct_base = c_puct_base
        self.fpu = fpu
        self.vloss_weight = vloss_weight
        # Search utility = win-loss + score_weight * tanh(scoreLead / score_scale), so the
        # search values the score margin (not win-rate only) — like KataGo's utility.
        self.score_weight = score_weight
        self.score_scale = score_scale

    def _utility(self, ev: dict) -> float:
        return ev["v"] + self.score_weight * math.tanh(ev["score"] / self.score_scale)

    def _eval_boards(self, boards: list[Board]) -> list[dict]:
        # The evaluator owns board -> eval, so alternative evaluators (e.g. a KataGo proxy that
        # asks an external engine) can plug in without touching the search.
        return self.ev.evaluate_boards(boards, self.komi, self.pos_len)

    def prepare(self, root_board: Board) -> Node:
        root = Node(None, 1.0)
        board = root_board.copy()
        root.eval = self._eval_boards([board])[0]
        root.board = board
        root.children = [Node(mv, p) for mv, p in root.eval["policy"].items()]
        root.expanded = True
        return root

    def _select(self, node: Node) -> Node:
        # Effective stats include virtual loss (N+vloss visits, each vloss counted as a loss).
        parent_neff = node.N + node.vloss
        sqrt_n = math.sqrt(parent_neff + 1)
        cpuct = self.c_puct + self.c_puct_log * math.log(
            (parent_neff + self.c_puct_base) / self.c_puct_base)
        parent_v = self._utility(node.eval)
        best, best_score = None, -1e18
        for ch in node.children:
            neff = ch.N + ch.vloss
            if neff > 0:
                # Virtual loss = a pretend loss for the *selecting* player, i.e. a pretend win
                # in the child's own (opponent's) perspective, so the child looks worse to us.
                q = -((ch.W + self.vloss_weight * ch.vloss) / neff)
            else:
                q = parent_v - self.fpu
            u = cpuct * ch.P * sqrt_n / (1 + neff)
            s = q + u
            if s > best_score:
                best_score, best = s, ch
        return best

    def step(self, root: Node, batch_size: int) -> int:
        """Collect up to batch_size leaves (virtual loss), evaluate as one batch, back up."""
        paths = []
        for _ in range(batch_size):
            path = [root]
            node = root
            while node.expanded and node.children:
                node = self._select(node)
                path.append(node)
                if not node.expanded:
                    break
            for n in path:
                n.vloss += 1
            paths.append(path)

        # Build boards for leaves that still need evaluation.
        to_eval, eval_boards = [], []
        for path in paths:
            leaf = path[-1]
            if not leaf.expanded and leaf.board is None:
                parent = path[-2]
                b = parent.board.copy()
                b.play(parent.board.to_move, leaf.move)
                leaf.board = b
                to_eval.append(leaf)
                eval_boards.append(b)
        if eval_boards:
            evals = self._eval_boards(eval_boards)
            for leaf, ev in zip(to_eval, evals):
                if not leaf.expanded:
                    leaf.eval = ev
                    leaf.children = [Node(mv, p) for mv, p in ev["policy"].items()]
                    leaf.expanded = True

        # Back up every path, undoing virtual loss. Accumulate utility (for selection) plus
        # win-loss and score separately (for reporting), with the per-node perspective sign.
        for path in paths:
            leaf = path[-1]
            util = self._utility(leaf.eval)
            wl = leaf.eval["v"]
            sc = leaf.eval["score"]
            mover = leaf.board.to_move
            for n in path:
                sign = 1.0 if n.board.to_move == mover else -1.0
                n.vloss -= 1
                n.N += 1
                n.W += sign * util
                n.Wwl += sign * wl
                n.Wsc += sign * sc
        return len(paths)

    def run(self, root_board: Board, visits: int, batch_size: int = 16) -> Node:
        root = self.prepare(root_board)
        done = 0
        while done < visits:
            done += self.step(root, adaptive_batch(batch_size, done, visits))
        return root


def adaptive_batch(batch_size: int, done: int, visits: int) -> int:
    """Don't collect more leaves than visits already done, so the early tree gets sequential
    value feedback instead of expanding a batch of leaves blind (which is weak at low visits)."""
    return max(1, min(batch_size, visits - done, max(1, done)))


def lcb(child: "Node", z: float = 1.0) -> float:
    """Lower-confidence-bound on the move's value from the parent's perspective (utility mean
    minus a visit-count uncertainty penalty). Robust move selection vs raw max-visits."""
    if child.N <= 0:
        return -1e18
    return -child.q() - z / math.sqrt(child.N)
