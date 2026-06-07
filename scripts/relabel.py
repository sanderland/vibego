"""Distillation / "logit forcing": relabel positions with a stronger teacher's outputs.

Reads source .npz training files, reconstructs each position's stones from the input feature
planes, queries a teacher engine (any KataGo-analysis-protocol command, e.g. a strong b18)
for its policy / value / ownership / score, and writes new .npz files in nanogo's training
format with the TEACHER's outputs as targets. Training on these = distilling the teacher into
the student (the policy cross-entropy against the teacher's soft policy is the logit forcing).

The input feature planes are copied verbatim from the source (so ladders/history/ko stay
intact); only the targets are replaced. Note: the teacher is queried from the stone position
(no move history), so its ko/last-move context is approximate.

Example:
    uv run python scripts/relabel.py --src data --out distilled --n-files 200 --visits 1 \
        --teacher "katago analysis -model models/kata1-b18c384nbt.bin.gz -config katago/cpp/configs/analysis_example.cfg"
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.go.board import xy_to_gtp
from nanogo.net import data as ndata


class TeacherEngine:
    def __init__(self, command: str):
        self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._buf: dict = {}  # responses that arrived out of order, keyed by id

    def ask(self, query: dict) -> dict:
        self.send(query)
        return self.recv(query["id"])

    def send(self, query: dict):
        self.proc.stdin.write(json.dumps(query) + "\n")
        self.proc.stdin.flush()

    def recv(self, qid: str) -> dict:
        if qid in self._buf:
            return self._buf.pop(qid)
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("teacher engine closed")
            r = json.loads(line)
            if r.get("isDuringSearch"):
                continue
            if r.get("id") == qid:
                return r
            self._buf[r.get("id")] = r  # arrived out of order; stash for later

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.terminate()


def _row_query(bin_full, glob_full, pos_len, komi_idx, qid, visits):
    """Build a teacher query for one position from its unpacked feature planes."""
    onboard = bin_full[0]
    xs = int(onboard[0].sum())
    ys = int(onboard[:, 0].sum())
    own, opp_ = bin_full[1], bin_full[2]
    stones = []
    for y in range(ys):
        for x in range(xs):
            if own[y, x]:
                stones.append(["B", xy_to_gtp((x, y), ys)])
            elif opp_[y, x]:
                stones.append(["W", xy_to_gtp((x, y), ys)])
    self_komi = float(glob_full[komi_idx]) * 20.0  # global[5] = selfKomi / 20, to-move = Black
    komi = round(-self_komi * 2) / 2  # KataGo requires an integer/half-integer komi
    return {
        "id": qid, "rules": "chinese", "komi": komi,
        "boardXSize": xs, "boardYSize": ys, "initialStones": stones,
        "initialPlayer": "B", "moves": [], "maxVisits": visits,
        "includePolicy": True, "includeOwnership": True,
        "overrideSettings": {"reportAnalysisWinratesAs": "BLACK"},
    }, xs, ys


def _targets_from_response(r, xs, ys, pos_len):
    policy = np.zeros(pos_len * pos_len + 1, dtype=np.float32)
    tp = r["policy"]
    for i, p in enumerate(tp[:-1]):
        if p >= 0:
            x, y = i % xs, i // xs
            policy[y * pos_len + x] = p
    policy[pos_len * pos_len] = max(0.0, tp[-1])
    own = np.zeros((pos_len, pos_len), dtype=np.float32)
    for i, o in enumerate(r["ownership"]):
        own[i // xs, i % xs] = o
    wr = float(r["rootInfo"]["winrate"])
    value = np.array([wr, 1.0 - wr, 0.0], dtype=np.float32)
    score = float(r["rootInfo"]["scoreLead"])
    return policy, value, score, own


def relabel_file(path, out_path, teacher, pos_len, visits):
    with np.load(path) as npz:
        packed = npz["binaryInputNCHWPacked"]
        global_nc = npz["globalInputNC"]
    bin_nchw = np.unpackbits(packed, axis=2)[:, :, : pos_len * pos_len]
    bin_nchw = bin_nchw.reshape(bin_nchw.shape[0], bin_nchw.shape[1], pos_len, pos_len)
    n = bin_nchw.shape[0]
    pol = np.zeros((n, 1, pos_len * pos_len + 1), dtype=np.float32)
    gt = np.zeros((n, 64), dtype=np.float32)
    own = np.zeros((n, 1, pos_len, pos_len), dtype=np.float32)
    # Pipeline all queries for this file so the teacher engine batches them.
    sizes = {}
    for i in range(n):
        q, xs, ys = _row_query(bin_nchw[i], global_nc[i], pos_len, 5, f"r{i}", visits)
        sizes[f"r{i}"] = (xs, ys)
        teacher.send(q)
    for i in range(n):
        r = teacher.recv(f"r{i}")
        xs, ys = sizes[f"r{i}"]
        p, v, s, o = _targets_from_response(r, xs, ys, pos_len)
        pol[i, 0] = p
        gt[i, 0:3] = v
        gt[i, 3] = s
        gt[i, 27] = 1.0  # ownership weight
        own[i, 0] = o
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, binaryInputNCHWPacked=packed, globalInputNC=global_nc,
                        policyTargetsNCMove=pol, globalTargetsNC=gt, valueTargetsNCHW=own)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data")
    p.add_argument("--out", default="distilled")
    p.add_argument("--teacher", required=True, help="teacher engine command (KataGo protocol)")
    p.add_argument("--n-files", type=int, default=100)
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--visits", type=int, default=1, help="teacher visits (1 = raw policy)")
    args = p.parse_args()

    files = ndata.list_npz(args.src)[: args.n_files]
    teacher = TeacherEngine(args.teacher)
    try:
        for k, f in enumerate(files):
            out = os.path.join(args.out, f"distilled_{k:06d}.npz")
            relabel_file(f, out, teacher, args.pos_len, args.visits)
            if k % 50 == 0:
                print(f"relabeled {k+1}/{len(files)}")
    finally:
        teacher.close()
    print(f"done -> {args.out}/ ({len(ndata.list_npz(args.out))} files)")


if __name__ == "__main__":
    main()
