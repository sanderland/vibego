"""Replay relabeling: walk GAME RECORDS and query the teacher with the real move prefix, so its
search has true history/ko context — the fix for the e6/e7 finding that searched targets from
stones-only (history-less) relabeling are poisoned (search compounds the missing-context error;
h2h −38.7 sL). Inputs get real history channels too (encode_board on the replayed board).

Sources: g170 archive .sgfs files (one complete sgf per line) and/or match.py --save-games JSONL.
Only clean 19×19 games without setup stones (AB/AW) are used.

Targets are to-move-perspective via reportAnalysisWinratesAs=SIDETOMOVE (the stones-only path
instead re-posed every position as Black-to-move). Ownership perspective is verified by the
smoke test in tests/ — KataGo reports it from the configured perspective.

    uv run python scripts/replay_relabel.py --sgfs '/workspace/distill/g170raw/*/*/sgfs/*.sgfs' \
        --out /workspace/distill/replay32 --visits 32 --policy-temp 1.0 --stride 6
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relabel import TeacherEngine, _targets_from_response, _write_npz  # noqa: E402
from vibego.go.board import BLACK, WHITE, Board, opp, xy_to_gtp
from vibego.go.features import encode_board

FULL_SPATIAL = list(range(22))
FULL_GLOBAL = list(range(19))

SGF_MOVE_RE = re.compile(r";([BW])\[([a-t]{0,2})\]")
SGF_PROP_RE = re.compile(r"(SZ|KM|AB|AW)\[([^\]]*)\]")


def parse_sgf_line(line: str):
    """Minimal parser for KataGo selfplay sgfs. Returns (size, komi, moves[(color,(x,y)|None)])
    or None for games we skip (non-19×19, setup stones, no moves)."""
    props = dict()
    for k, v in SGF_PROP_RE.findall(line):
        props.setdefault(k, v)
    if props.get("SZ") != "19" or "AB" in props or "AW" in props:
        return None
    komi = float(props.get("KM", "7.5"))
    moves = []
    for color, mv in SGF_MOVE_RE.findall(line):
        if mv in ("", "tt"):
            moves.append((color, None))
        else:
            x = ord(mv[0]) - ord("a")
            y = ord(mv[1]) - ord("a")
            moves.append((color, (x, y)))
    return (19, komi, moves) if moves else None


def iter_games(sgfs_glob, jsonl_paths):
    if sgfs_glob:
        for path in sorted(globmod.glob(sgfs_glob, recursive=True)):
            with open(path) as fh:
                for line in fh:
                    g = parse_sgf_line(line)
                    if g:
                        yield g
    for path in jsonl_paths or []:
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                if rec.get("board") != 19:
                    continue
                moves = []
                for color, gtp in rec["moves"]:
                    if gtp.strip().lower() == "pass":
                        moves.append((color, None))
                    else:
                        col = gtp[0].upper()
                        x = ord(col) - ord("A") - (1 if col > "I" else 0)
                        y = 19 - int(gtp[1:])
                        moves.append((color, (x, y)))
                if moves:
                    yield (19, rec.get("komi", 7.5), moves)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sgfs", default=None, help="glob over .sgfs files (one sgf per line)")
    p.add_argument("--games", action="append", default=None, help="match.py --save-games JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--visits", type=int, default=32)
    p.add_argument("--policy-temp", type=float, default=1.0)
    p.add_argument("--stride", type=int, default=5,
                   help="sample every Nth ply; keep ODD so sampled to-move colors alternate "
                        "(an even stride samples a single perspective per game)")
    p.add_argument("--skip-plies", type=int, default=4)
    p.add_argument("--max-pos", type=int, default=None)
    p.add_argument("--shard-size", type=int, default=8192)
    p.add_argument("--inflight", type=int, default=64,
                   help="positions in flight (each costs --visits evals)")
    p.add_argument("--pos-len", type=int, default=19)
    args = p.parse_args()

    teacher = TeacherEngine(args.teacher)
    names = {"B": BLACK, "W": WHITE}
    pos_len = args.pos_len
    pp1 = pos_len * pos_len + 1

    pending: dict = {}      # qid -> (bin_full, glob_full)
    rows = []               # accumulated finished rows
    shard_idx = 0
    n_q = 0
    n_done = 0

    def flush_shard(force=False):
        nonlocal shard_idx, rows
        while len(rows) >= args.shard_size or (force and rows):
            chunk, rows = rows[: args.shard_size], rows[args.shard_size:]
            packed = np.packbits(np.stack([r[0] for r in chunk]).reshape(len(chunk), 22, -1), axis=2)
            gl = np.stack([r[1] for r in chunk]).astype(np.float32)
            pol = np.stack([r[2] for r in chunk])[:, None, :]
            gt = np.zeros((len(chunk), 64), np.float32)
            gt[:, 0:3] = np.stack([r[3] for r in chunk])
            gt[:, 3] = [r[4] for r in chunk]
            gt[:, 27] = 1.0
            own = np.stack([r[5] for r in chunk])[:, None]
            _write_npz(os.path.join(args.out, f"replay_{shard_idx:06d}.npz"),
                       packed, gl, pol, gt, own)
            print(f"wrote replay_{shard_idx:06d}.npz ({len(chunk)} rows)", flush=True)
            shard_idx += 1

    def drain(block_until=0):
        nonlocal n_done
        while len(pending) > block_until:
            r = teacher.recv_any()
            meta = pending.pop(r.get("id"), None)
            if meta is None:
                continue
            if "policy" not in r or "rootInfo" not in r:
                continue
            pol, val, sc, own = _targets_from_response(r, pos_len, pos_len, pos_len,
                                                       args.policy_temp)
            rows.append((meta[0], meta[1], pol, val, sc, own))
            n_done += 1
        flush_shard()

    os.makedirs(args.out, exist_ok=True)
    stop = False
    for size, komi, moves in iter_games(args.sgfs, args.games):
        if stop:
            break
        board = Board(size, size)
        gtp_moves = []
        for ply, (color, xy) in enumerate(moves):
            side = names[color]
            if ply >= args.skip_plies and (ply - args.skip_plies) % args.stride == 0 \
                    and board.to_move == side:
                sp, gl = encode_board(board, komi, pos_len, FULL_SPATIAL, FULL_GLOBAL)
                qid = f"g{n_q}"
                n_q += 1
                teacher.send({
                    "id": qid, "rules": "chinese", "komi": komi,
                    "boardXSize": size, "boardYSize": size,
                    "initialPlayer": "B", "moves": list(gtp_moves),
                    "maxVisits": args.visits,
                    "includePolicy": True, "includeOwnership": True,
                    "overrideSettings": {"reportAnalysisWinratesAs": "SIDETOMOVE"},
                })
                pending[qid] = (sp.astype(np.uint8), gl)
                if len(pending) >= args.inflight:
                    drain(block_until=args.inflight // 2)
                if args.max_pos and n_done + len(pending) >= args.max_pos:
                    stop = True
                    break
            gtp_moves.append([color, "pass" if xy is None else xy_to_gtp(xy, size)])
            try:
                board.play(side, xy)
            except ValueError:
                # rules gap (g170 plays situational-superko/button variants our Board doesn't
                # model) — truncate this game, keep the samples already emitted from it
                break
    drain(0)
    flush_shard(force=True)
    teacher.close()
    print(f"done: {n_done} positions, {shard_idx} shards -> {args.out}")


if __name__ == "__main__":
    main()
