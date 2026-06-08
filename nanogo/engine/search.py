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

from ..go.board import BLACK, PASS, Board
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


TWO_OVER_PI = 2.0 / math.pi


SQRT2 = math.sqrt(2.0)


class Node:
    # W = sum of search *utility* (winloss + score) MC-accumulated; Wsq = sum of utility^2
    # (for LCB variance); Wwl / Wsc = win-loss value and score-lead sums (reporting). V is the
    # node's recomputed value in its own perspective, used for in-tree selection: it equals the
    # MC mean q() when value-weighting/subtree-bias are off, and incorporates them when on.
    __slots__ = ("move", "P", "N", "W", "Wsq", "Wwl", "Wsc", "V", "vloss", "board", "eval",
                 "children", "expanded")

    def __init__(self, move, prior):
        self.move = move
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Wsq = 0.0          # sum of utility^2 (for LCB)
        self.Wwl = 0.0          # win-loss value sum (reporting)
        self.Wsc = 0.0          # score-lead sum (reporting)
        self.V = 0.0            # recomputed node value (own perspective), for selection
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
    # Defaults stolen from KataGo (cpp/search/searchparams.cpp recommended preset).
    def __init__(self, evaluator: NNEvaluator, komi: float, pos_len: int,
                 c_puct: float = 1.0, fpu: float = 0.2, root_fpu: float = 0.1,
                 vloss_weight: float = 1.0, c_puct_log: float = 0.45, c_puct_base: float = 500.0,
                 winloss_factor: float = 1.0, static_score_factor: float = 0.1,
                 dynamic_score_factor: float = 0.3, static_score_scale: float = 2.0,
                 dynamic_score_scale: float = 0.75, cpuct_stdev_scale: float = 0.0,
                 cpuct_stdev_prior: float = 0.40, cpuct_stdev_prior_weight: float = 2.0,
                 fpu_parent_pow: float = 2.0, value_weight_exp: float = 0.25,
                 subtree_bias: float = 0.0):
        self.ev = evaluator
        self.komi = komi
        self.pos_len = pos_len
        self.c_puct = c_puct
        self.c_puct_log = c_puct_log    # cpuct grows ~log(visits), KataGo-style
        self.c_puct_base = c_puct_base
        self.fpu = fpu                  # FPU reduction = fpu * sqrt(visited policy mass)
        self.root_fpu = root_fpu        # smaller at root -> explore more candidate moves
        self.fpu_parent_pow = fpu_parent_pow  # blend parent-avg vs raw eval by mass^pow
        # KataGo recursive value recompute (searchupdatehelpers.cpp). value_weight_exp re-weights
        # children toward the above-average ones (z-score CDF^exp) when forming a node's value;
        # subtree_bias pulls the node's own nn-eval toward its children's average. Both 0 => the
        # recompute telescopes to the plain MC mean (q()), i.e. exactly the prior behaviour.
        self.value_weight_exp = value_weight_exp
        self.subtree_bias = subtree_bias
        self.vloss_weight = vloss_weight
        # Utility = winloss_factor*winloss + score utility, where the score utility is KataGo's
        # (2/pi)*atan(score/(scale*sqrtArea)) split into static + dynamic terms.
        self.winloss_factor = winloss_factor
        self.static_score_factor = static_score_factor
        self.dynamic_score_factor = dynamic_score_factor
        self.static_score_scale = static_score_scale
        self.dynamic_score_scale = dynamic_score_scale
        self.sqrt_area = float(pos_len)   # updated to the real board in prepare()
        self.score_center = 0.0           # KataGo's recentScoreCenter (Black perspective)
        # Scale cpuct by the node's utility uncertainty: explore more at uncertain nodes.
        self.cpuct_stdev_scale = cpuct_stdev_scale
        self.cpuct_stdev_prior = cpuct_stdev_prior
        self.cpuct_stdev_prior_weight = cpuct_stdev_prior_weight

    def _cpuct_stdev_factor(self, node) -> float:
        n = node.N
        prior = self.cpuct_stdev_prior
        if n <= 1:
            stdev = prior
        else:
            uavg = node.W / n
            usqavg = max(node.Wsq / n, uavg * uavg)
            var = (((uavg * uavg + prior * prior) * self.cpuct_stdev_prior_weight + usqavg * n)
                   / (self.cpuct_stdev_prior_weight + n - 1.0) - uavg * uavg)
            stdev = math.sqrt(max(0.0, var))
        return 1.0 + self.cpuct_stdev_scale * (stdev / prior - 1.0)

    def _recompute_value(self, node: "Node") -> None:
        """Set node.V = KataGo-style recomputed value (node's own perspective): a weight-1 prior
        of its own nn eval plus its visited children's values weighted by their visit counts,
        with optional value-weighting (upweight above-average children) and subtree-bias (pull
        the self term toward the children average). With both factors 0 this is the MC mean."""
        self_util = self._utility(node.eval, node.board.to_move)
        visited = [c for c in node.children if c.N > 0]
        if not visited:
            node.V = self_util
            return
        us = [-c.V for c in visited]            # child values in this node's perspective
        ws = [float(c.N) for c in visited]
        sw = sum(ws)
        if self.value_weight_exp > 0.0:
            mean = sum(w * u for w, u in zip(ws, us)) / sw
            rew = []
            for u, w in zip(us, ws):
                stdev = math.sqrt(1e-8 + 1.0 / (1.5 * math.sqrt(w)))
                z = (u - mean) / stdev
                p = 0.5 * (1.0 + math.erf(z / SQRT2)) + 1e-4
                rew.append(w * p ** self.value_weight_exp)
            ws = rew
            sw = sum(ws)
        children_avg = sum(w * u for w, u in zip(ws, us)) / sw
        self_corr = self_util + self.subtree_bias * (children_avg - self_util)
        node.V = (self_corr + sw * children_avg) / (1.0 + sw)

    def _utility(self, ev: dict, mover: int) -> float:
        return self.winloss_factor * ev["v"] + self._score_utility(ev["score"], mover)

    def _score_utility(self, score: float, mover: int) -> float:
        # score is from `mover`'s perspective; the dynamic term is centered on the current
        # expected score (recentScoreCenter), oriented into the mover's perspective.
        a = self.sqrt_area
        center = self.score_center if mover == BLACK else -self.score_center
        return TWO_OVER_PI * (
            self.static_score_factor * math.atan(score / (self.static_score_scale * a))
            + self.dynamic_score_factor * math.atan((score - center) / (self.dynamic_score_scale * a)))

    def _eval_boards(self, boards: list[Board]) -> list[dict]:
        # The evaluator owns board -> eval, so alternative evaluators (e.g. a KataGo proxy that
        # asks an external engine) can plug in without touching the search.
        return self.ev.evaluate_boards(boards, self.komi, self.pos_len)

    def prepare(self, root_board: Board) -> Node:
        self.sqrt_area = math.sqrt(root_board.x_size * root_board.y_size)
        root = Node(None, 1.0)
        board = root_board.copy()
        root.eval = self._eval_boards([board])[0]
        root.board = board
        # Center the dynamic score term on the current expected score (Black perspective).
        self.score_center = root.eval["score"] if board.to_move == BLACK else -root.eval["score"]
        root.children = [Node(mv, p) for mv, p in root.eval["policy"].items()]
        root.expanded = True
        self._recompute_value(root)
        return root

    def _select(self, node: Node, is_root: bool = False) -> Node:
        # Effective stats include virtual loss (N+vloss visits, each vloss counted as a loss).
        parent_neff = node.N + node.vloss
        sqrt_n = math.sqrt(parent_neff + 1)
        cpuct = self.c_puct + self.c_puct_log * math.log(
            (parent_neff + self.c_puct_base) / self.c_puct_base)
        cpuct *= self._cpuct_stdev_factor(node)
        # FPU value for unvisited children (KataGo searchexplorehelpers.cpp): the base blends
        # the node's recomputed value with its raw net eval, weighted by how much policy mass
        # has already been visited (mass^pow), then a mass-scaled reduction is applied so
        # unvisited moves look progressively worse as the good ones get explored.
        raw_v = self._utility(node.eval, node.board.to_move)
        parent_avg = node.V if node.N > 0 else raw_v
        visited_mass = sum(ch.P for ch in node.children if ch.N + ch.vloss > 0)
        avg_w = min(1.0, visited_mass ** self.fpu_parent_pow)
        parent_v = avg_w * parent_avg + (1.0 - avg_w) * raw_v
        fpu_max = self.root_fpu if is_root else self.fpu
        fpu = fpu_max * math.sqrt(max(0.0, visited_mass))
        best, best_score = None, -1e18
        for ch in node.children:
            neff = ch.N + ch.vloss
            if neff > 0:
                # Child value (recomputed) negated to our perspective, with virtual loss = a
                # pretend loss for the selecting player so the same leaf isn't picked twice.
                q = -((ch.V * ch.N + self.vloss_weight * ch.vloss) / neff)
            else:
                q = parent_v - fpu
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
                node = self._select(node, is_root=(node is root))
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
            util = self._utility(leaf.eval, leaf.board.to_move)
            wl = leaf.eval["v"]
            sc = leaf.eval["score"]
            mover = leaf.board.to_move
            for n in path:
                sign = 1.0 if n.board.to_move == mover else -1.0
                n.vloss -= 1
                n.N += 1
                n.W += sign * util
                n.Wsq += util * util       # perspective-independent (sign^2 = 1)
                n.Wwl += sign * wl
                n.Wsc += sign * sc

        # Recompute node values bottom-up (leaf -> root) so each parent sees its children's
        # updated values. Done after all MC updates so visit counts are final this step.
        for path in paths:
            for n in reversed(path):
                self._recompute_value(n)
        return len(paths)

    def run(self, root_board: Board, visits: int, batch_size: int = 16) -> Node:
        root = self.prepare(root_board)
        done = 0
        while done < visits:
            done += self.step(root, adaptive_batch(batch_size, done, visits))
        return root


def adaptive_batch(batch_size: int, done: int, visits: int) -> int:
    """How many leaves to collect per search step. Every leaf in a step is selected under
    virtual loss before any of them is evaluated, so a large step expands many leaves "blind".
    KataGo's threads re-select asynchronously as each eval lands, so its effective blindness is
    far lower than a synchronous batch of the same size; with a synchronous batch we keep it
    small *relative to total visits* (cap ~1/8) so low-visit search stays fine-grained — this
    measurably helps at 48 visits (per-move loss ~4.1 -> ~3.2 pts). At high visits the cap
    relaxes back to batch_size for throughput. Also never collect more than visits already done,
    so the opening of the search is effectively sequential."""
    cap = min(batch_size, max(1, visits // 8))
    return max(1, min(cap, visits - done, max(1, done)))


def lcb(child: "Node", lcb_stdevs: float = 5.0) -> float:
    """KataGo-style LCB: parent-perspective utility mean minus lcb_stdevs * standard error,
    where the std error uses the measured variance of the child's utility samples."""
    if child.N <= 0:
        return -1e18
    mean = -child.q()                                  # parent perspective
    var = max(0.0, child.Wsq / child.N - child.q() ** 2)
    return mean - lcb_stdevs * math.sqrt(var / child.N)


def rank_children(children, lcb_stdevs: float = 1.0, min_visit_prop: float = 0.15):
    """Order visited children for move selection (best first): among those with enough visits,
    rank by LCB; the rest fall below, ranked by raw visits — so a barely-visited move with a
    flukey LCB can't outrank a well-searched one. (KataGo's minVisitPropForLCB.)"""
    visited = [c for c in children if c.N > 0]
    if not visited:
        return []
    thresh = min_visit_prop * max(c.N for c in visited)

    def key(c):
        if c.move is PASS:
            return (-1, lcb(c, lcb_stdevs))  # never prefer PASS over a real move (area scoring:
            #                                  a harmless move never loses points), unless forced
        return (1, lcb(c, lcb_stdevs)) if c.N >= thresh else (0, c.N)

    return sorted(visited, key=key, reverse=True)
