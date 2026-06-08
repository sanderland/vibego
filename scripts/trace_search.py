"""Deterministic node-by-node trace of OUR MCTS at tiny scale, to diff against KataGo's own
search move-for-move. Runs batch_size=1 (no virtual loss, no concurrency -> fully sequential
and deterministic) on the SAME net (KataGo b6c96 via the proxy), so any divergence from
KataGo's own engine at equal visits is pure search algorithm.

Use an ASYMMETRIC position so KataGo's root symmetry pruning is inactive (else it shares visits
across symmetric moves and the comparison isn't apples-to-apples).

    CFG=config/analysis_1t.cfg
    KATA="katago analysis -model models/g170-b6c96.bin.gz -config $CFG"
    uv run python scripts/trace_search.py --proxy "$KATA" --size 9 --komi 7 --visits 10 \
        --moves B:E5 W:G5 B:C3
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.engine.proxy import KataGoEvaluator
from nanogo.engine.search import MCTS, Node
from nanogo.go.board import BLACK, WHITE, Board, PASS, gtp_to_xy, xy_to_gtp


def mv_str(mv, ys):
    return "pass" if mv is PASS else xy_to_gtp(mv, ys)


def puct_table(mcts: MCTS, node: Node, is_root: bool):
    """Recompute the exact selection score for every child (mirrors MCTS._select) and return
    rows (move, N, V, Q, U, score, prior, chosen) sorted by selection score."""
    parent_neff = node.N + node.vloss
    sqrt_n = math.sqrt(parent_neff + 1)
    cpuct = mcts.c_puct + mcts.c_puct_log * math.log((parent_neff + mcts.c_puct_base) / mcts.c_puct_base)
    cpuct *= mcts._cpuct_stdev_factor(node)
    raw_v = mcts._utility(node.eval, node.board.to_move)
    parent_avg = node.V if node.N > 0 else raw_v
    visited_mass = sum(ch.P for ch in node.children if ch.N + ch.vloss > 0)
    avg_w = min(1.0, visited_mass ** mcts.fpu_parent_pow)
    parent_v = avg_w * parent_avg + (1.0 - avg_w) * raw_v
    fpu_max = mcts.root_fpu if is_root else mcts.fpu
    fpu = fpu_max * math.sqrt(max(0.0, visited_mass))
    rows = []
    for ch in node.children:
        neff = ch.N + ch.vloss
        if neff > 0:
            q = -((ch.V * ch.N + mcts.vloss_weight * ch.vloss) / neff)
        else:
            q = parent_v - fpu
        u = cpuct * ch.P * sqrt_n / (1 + neff)
        rows.append([ch, q, u, q + u])
    rows.sort(key=lambda r: -r[3])
    return rows, cpuct, parent_v


def trace_visit(mcts: MCTS, root: Node, ys: int, vnum: int, topk: int):
    # ---- selection (sequential, no virtual loss) ----
    path = [root]
    node = root
    print(f"\n=== visit {vnum} ===")
    while node.expanded and node.children:
        rows, cpuct, parent_v = puct_table(mcts, node, is_root=(node is root))
        chosen = rows[0][0]
        depth = len(path) - 1
        if depth == 0:  # only print full table at the root to keep it readable
            print(f"  root select (cpuct={cpuct:.3f} fpuBase={parent_v:+.4f}):")
            print(f"    {'move':>5} {'N':>3} {'V':>8} {'Q':>8} {'U':>8} {'Q+U':>8} {'prior':>7}")
            for ch, q, u, s in rows[:topk]:
                mark = " <--" if ch is chosen else ""
                print(f"    {mv_str(ch.move, ys):>5} {ch.N:>3} {ch.V:>+8.4f} {q:>+8.4f} "
                      f"{u:>8.4f} {s:>+8.4f} {ch.P:>7.4f}{mark}")
        else:
            print(f"    depth {depth}: pick {mv_str(chosen.move, ys):>4} "
                  f"(N={chosen.N} V={chosen.V:+.4f} Q={rows[0][1]:+.4f} U={rows[0][2]:.4f})")
        node = chosen
        path.append(node)
        if not node.expanded:
            break

    # ---- expand leaf ----
    leaf = path[-1]
    if not leaf.expanded:
        parent = path[-2]
        b = parent.board.copy()
        b.play(parent.board.to_move, leaf.move)
        leaf.board = b
        ev = mcts._eval_boards([b])[0]
        leaf.eval = ev
        leaf.children = [Node(mv, p) for mv, p in ev["policy"].items()]
        leaf.expanded = True

    util = mcts._utility(leaf.eval, leaf.board.to_move)
    wl, sc = leaf.eval["v"], leaf.eval["score"]
    print(f"  leaf {mv_str(leaf.move, ys) if leaf.move else 'ROOT'} "
          f"(to_move={'B' if leaf.board.to_move == BLACK else 'W'}): "
          f"winrate={leaf.eval['winrate']:.4f} score={sc:+.2f} util={util:+.4f}")

    # ---- backup ----
    mover = leaf.board.to_move
    for n in path:
        sign = 1.0 if n.board.to_move == mover else -1.0
        n.N += 1
        n.W += sign * util
        n.Wsq += util * util
        n.Wwl += sign * wl
        n.Wsc += sign * sc
    for n in reversed(path):
        mcts._recompute_value(n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proxy", required=True, help="KataGo analysis command for the net")
    p.add_argument("--size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.0)
    p.add_argument("--visits", type=int, default=10)
    p.add_argument("--moves", nargs="*", default=[], help="setup moves like B:E5 W:G5")
    p.add_argument("--topk", type=int, default=8)
    args = p.parse_args()

    ev = KataGoEvaluator(args.proxy)
    board = Board(args.size, args.size)
    first = None
    for tok in args.moves:
        c, gtp = tok.split(":")
        pl = BLACK if c.upper().startswith("B") else WHITE
        if first is None:
            first = pl
            board.to_move = pl
        board.play(pl, gtp_to_xy(gtp, args.size))
    print(f"position: {args.size}x{args.size} komi={args.komi} to_move="
          f"{'B' if board.to_move == BLACK else 'W'} moves={args.moves}")

    mcts = MCTS(ev, args.komi, args.size)
    root = mcts.prepare(board)
    print(f"root eval: winrate={root.eval['winrate']:.4f} score={root.eval['score']:+.2f} "
          f"V={root.V:+.4f}  (#legal children={len(root.children)})")

    for v in range(1, args.visits + 1):
        trace_visit(mcts, root, args.size, v, args.topk)

    print("\n===== FINAL root children (sorted by visits) =====")
    print(f"{'move':>5} {'visits':>6} {'V':>8} {'winrate':>8} {'score':>7} {'prior':>7}")
    for ch in sorted(root.children, key=lambda c: -c.N):
        if ch.N == 0:
            continue
        wr = (1.0 - ch.winloss()) / 2.0 + 0.5  # child winrate in parent perspective approx
        wr = (1.0 + (-ch.winloss())) / 2.0
        print(f"{mv_str(ch.move, args.size):>5} {ch.N:>6} {ch.V:>+8.4f} {wr:>8.4f} "
              f"{-ch.score():>+7.2f} {ch.P:>7.4f}")
    print(f"\nroot: visits={root.N} winrate={(1 + root.winloss()) / 2:.4f} "
          f"score={root.score():+.2f} V={root.V:+.4f}")


if __name__ == "__main__":
    main()
