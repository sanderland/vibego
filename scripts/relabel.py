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
import queue
import shlex
import subprocess
import sys
import threading
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.go.board import xy_to_gtp
from nanogo.net import data as ndata


class TeacherEngine:
    """KataGo analysis-protocol engine. A background thread drains stdout into a queue so we
    can keep many queries in flight (relabel_stream) without the classic pipe deadlock — if we
    sent thousands of queries while never reading, KataGo would block writing responses and then
    stop reading our stdin. recv(qid) (specific id, used by relabel_file/policy_eval) and
    recv_any() (next-to-arrive, used by relabel_stream) both pull from that one queue."""

    def __init__(self, command: str):
        self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._q: "queue.Queue" = queue.Queue()
        self._buf: dict = {}  # responses pulled while waiting for a specific id (recv)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("isDuringSearch"):
                continue
            self._q.put(r)
        self._q.put(None)  # EOF sentinel

    def _get(self) -> dict:
        r = self._q.get()
        if r is None:
            raise RuntimeError("teacher engine closed")
        return r

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
            r = self._get()
            if r.get("id") == qid:
                return r
            self._buf[r.get("id")] = r  # arrived out of order; stash for later

    def recv_any(self) -> dict:
        """Return the next response to arrive (any id). Drains stashed ones first."""
        if self._buf:
            return self._buf.popitem()[1]
        return self._get()

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
    keep = []  # KataGo returns an error response (no "policy") for some reconstructed
    for i in range(n):  # positions (e.g. illegal initialStones / terminal board); drop those.
        r = teacher.recv(f"r{i}")  # must recv all n to stay in sync with the n sent queries
        if "policy" not in r or "rootInfo" not in r:
            continue
        xs, ys = sizes[f"r{i}"]
        p, v, s, o = _targets_from_response(r, xs, ys, pos_len)
        pol[i, 0] = p
        gt[i, 0:3] = v
        gt[i, 3] = s
        gt[i, 27] = 1.0  # ownership weight
        own[i, 0] = o
        keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    packed, global_nc = packed[keep], global_nc[keep]
    pol, gt, own = pol[keep], gt[keep], own[keep]
    return _write_npz(out_path, packed, global_nc, pol, gt, own)


def _write_npz(out_path, packed, global_nc, pol, gt, own):
    """Atomic write: temp-then-rename so a crash can't leave a truncated file the cache later
    skips as 'done'. Write via a file HANDLE — np.savez_compressed appends '.npz' to a path arg,
    which would break the rename."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, binaryInputNCHWPacked=packed, globalInputNC=global_nc,
                            policyTargetsNCMove=pol, globalTargetsNC=gt, valueTargetsNCHW=own)
    os.replace(tmp, out_path)
    return packed.shape[0]


def relabel_stream(items, teacher, pos_len, visits, max_inflight=2048):
    """Relabel many files with a large query window kept in flight, so the teacher's NN batch
    stays full — relabeling one file at a time leaves the GPU ~idle between files (the dominant
    cost at visits=1). Yields (out_path, n_kept) as each file's responses arrive and it's written.
    items: list of (src_path, out_path)."""
    loaded = {}   # idx -> per-file state (input planes + target arrays + pending count)
    qmap = {}     # qid -> (idx, pos_i, xs, ys)
    send_q = deque()
    qid = 0
    inflight = 0
    cursor = 0

    def load(idx):
        src, _out = items[idx]
        with np.load(src) as npz:
            packed = npz["binaryInputNCHWPacked"]
            global_nc = npz["globalInputNC"]
        bn = np.unpackbits(packed, axis=2)[:, :, : pos_len * pos_len]
        bn = bn.reshape(bn.shape[0], bn.shape[1], pos_len, pos_len)
        n = bn.shape[0]
        loaded[idx] = {
            "out": items[idx][1], "packed": packed, "global_nc": global_nc, "bin": bn, "n": n,
            "pol": np.zeros((n, 1, pos_len * pos_len + 1), dtype=np.float32),
            "gt": np.zeros((n, 64), dtype=np.float32),
            "own": np.zeros((n, 1, pos_len, pos_len), dtype=np.float32),
            "keep": [], "pending": n,
        }
        for i in range(n):
            send_q.append((idx, i))
        return n

    def finalize(idx):
        st = loaded.pop(idx)
        keep = np.array(st["keep"], dtype=np.int64)
        _write_npz(st["out"], st["packed"][keep], st["global_nc"][keep],
                   st["pol"][keep], st["gt"][keep], st["own"][keep])
        return st["out"], len(keep)

    while True:
        while inflight + len(send_q) < max_inflight and cursor < len(items):
            idx = cursor
            cursor += 1
            if load(idx) == 0:
                yield finalize(idx)  # empty source -> empty output, done immediately
        while inflight < max_inflight and send_q:
            idx, i = send_q.popleft()
            st = loaded[idx]
            q, xs, ys = _row_query(st["bin"][i], st["global_nc"][i], pos_len, 5, f"q{qid}", visits)
            qmap[f"q{qid}"] = (idx, i, xs, ys)
            qid += 1
            teacher.send(q)
            inflight += 1
        if inflight == 0 and not send_q and cursor >= len(items):
            return
        r = teacher.recv_any()
        key = r.get("id")
        if key not in qmap:
            continue  # stray/global message not tied to an in-flight query; ignore
        inflight -= 1
        idx, i, xs, ys = qmap.pop(key)
        st = loaded[idx]
        if "policy" in r and "rootInfo" in r:  # else KataGo errored on this position -> drop it
            p, v, s, o = _targets_from_response(r, xs, ys, pos_len)
            st["pol"][i, 0] = p
            st["gt"][i, 0:3] = v
            st["gt"][i, 3] = s
            st["gt"][i, 27] = 1.0
            st["own"][i, 0] = o
            st["keep"].append(i)
        st["pending"] -= 1
        if st["pending"] == 0:
            yield finalize(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data")
    p.add_argument("--out", default="distilled")
    p.add_argument("--teacher", required=True, help="teacher engine command (KataGo protocol)")
    p.add_argument("--n-files", type=int, default=None, help="cap on source files to consider")
    p.add_argument("--max-pos", type=int, default=None,
                   help="stop after ~this many newly-relabeled positions")
    p.add_argument("--max-gb", type=float, default=None,
                   help="stop when the output dir reaches ~this many GB (cached + new)")
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--visits", type=int, default=1, help="teacher visits (1 = raw policy)")
    p.add_argument("--inflight", type=int, default=2048,
                   help="teacher queries kept in flight; fills the NN batch (throughput knob)")
    p.add_argument("--overwrite", action="store_true", help="re-relabel even if a cached output exists")
    args = p.parse_args()

    # Resumable cache: each output is named after its source file (already a content hash), so the
    # (src -> output) mapping is stable across runs; existing outputs are reused, not recomputed.
    # NOTE: a teacher's targets are net-specific -> use a SEPARATE --out dir per teacher.
    files = ndata.list_npz(args.src)
    if args.n_files is not None:
        files = files[: args.n_files]
    os.makedirs(args.out, exist_ok=True)
    # Bytes already on disk (cached outputs) count toward --max-gb; cheap stat, no npz loads.
    cached_bytes = sum(e.stat().st_size for e in os.scandir(args.out) if e.name.endswith(".npz"))
    items = [(f, os.path.join(args.out, os.path.basename(f))) for f in files]
    items = [(f, o) for f, o in items if args.overwrite or not os.path.exists(o)]
    cached = len(files) - len(items)
    max_bytes = int(args.max_gb * 1e9) if args.max_gb is not None else None

    print(f"{len(files)} src files: {cached} cached ({cached_bytes/1e9:.2f} GB), {len(items)} to do"
          + (f"; target ~{args.max_gb} GB" if max_bytes else ""))
    teacher = TeacherEngine(args.teacher)
    done = pos = new_bytes = 0
    try:
        if max_bytes is not None and cached_bytes >= max_bytes:
            print(f"already at {cached_bytes/1e9:.2f} GB >= target; nothing to do")
        else:
            for out_path, nk in relabel_stream(items, teacher, args.pos_len, args.visits, args.inflight):
                done += 1
                pos += nk
                new_bytes += os.path.getsize(out_path)
                if done % 50 == 0:
                    print(f"{done} new + {cached} cached, ~{pos} pos, "
                          f"{(cached_bytes + new_bytes)/1e9:.2f} GB", flush=True)
                if max_bytes is not None and cached_bytes + new_bytes >= max_bytes:
                    print(f"reached ~{args.max_gb} GB target")
                    break
                if args.max_pos is not None and pos >= args.max_pos:
                    print(f"reached ~{args.max_pos} pos target")
                    break
    finally:
        teacher.close()
    print(f"done -> {args.out}/ : {done} new + {cached} cached = "
          f"{len(ndata.list_npz(args.out))} files, ~{pos} new pos, "
          f"{(cached_bytes + new_bytes)/1e9:.2f} GB total")


if __name__ == "__main__":
    main()
