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
        # Close stdin first: the engine sees EOF and shuts down cleanly. Then WAIT for it to
        # actually exit before returning — otherwise a dying engine is still freeing its GPU
        # memory while the next engine (e.g. the next net, or a heavy ref like b40) starts
        # allocating, and on MPS/Metal that race can OOM the new process at load (seen as a
        # BrokenPipe / "engine closed" when running many engines back to back).
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15)
        except Exception:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


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
    cost at visits=1). Yields (src_path, out_path, packed, global_nc, pol, gt, own) — the kept
    rows for one source file — as its responses arrive. The caller writes/accumulates them.
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
        return (items[idx][0], st["out"], st["packed"][keep], st["global_nc"][keep],
                st["pol"][keep], st["gt"][keep], st["own"][keep])

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


class ShardWriter:
    """Packs relabeled rows into large shard npz (shard_NNNNNN.npz) instead of one file per
    source — networked volumes often cap FILE COUNT (inodes), not bytes, and 60-row shards blow
    through it. Idempotent/resumable via an append-only manifest: a source basename is recorded
    ONLY after the shard containing its rows is durably written. So a crash loses at most the
    current in-memory buffer (re-relabeled next run) — never silently drops or double-counts a
    committed shard. With --delete-src, source files are removed once safely in a shard."""

    def __init__(self, out_dir, shard_size):
        self.out, self.S = out_dir, shard_size
        self.mpath = os.path.join(out_dir, "_manifest.txt")
        self.done = set()
        if os.path.exists(self.mpath):
            with open(self.mpath) as f:
                self.done = {ln.strip() for ln in f if ln.strip()}
        idxs = [int(n[6:12]) for n in os.listdir(out_dir) if n.startswith("shard_") and n.endswith(".npz")]
        self.idx = max(idxs, default=-1) + 1
        self.out_bytes = sum(e.stat().st_size for e in os.scandir(out_dir) if e.name.endswith(".npz"))
        self.shards = 0
        self._reset()

    def _reset(self):
        self._bp, self._bg, self._bpol, self._bgt, self._bown = [], [], [], [], []
        self._n = 0
        self._pending = []  # (basename, [paths_to_delete]) for rows now in the buffer

    def add(self, packed, global_nc, pol, gt, own, basename, del_paths):
        if packed.shape[0] > 0:
            self._bp.append(packed); self._bg.append(global_nc); self._bpol.append(pol)
            self._bgt.append(gt); self._bown.append(own); self._n += packed.shape[0]
        self._pending.append((basename, del_paths))
        if self._n >= self.S:
            self._flush()

    def _flush(self):
        if not self._pending:
            return
        if self._n > 0:  # write the shard FIRST (durable) so the manifest never over-claims
            shard = os.path.join(self.out, f"shard_{self.idx:06d}.npz")
            _write_npz(shard, np.concatenate(self._bp), np.concatenate(self._bg),
                       np.concatenate(self._bpol), np.concatenate(self._bgt), np.concatenate(self._bown))
            self.out_bytes += os.path.getsize(shard)
            self.idx += 1; self.shards += 1
        with open(self.mpath, "a") as mf:  # then commit the manifest (fsync), then free sources
            for base, _ in self._pending:
                mf.write(base + "\n")
            mf.flush(); os.fsync(mf.fileno())
        for base, paths in self._pending:
            self.done.add(base)
            for pth in paths:
                try:
                    sz = os.path.getsize(pth)
                    os.remove(pth)
                    if pth.startswith(self.out):
                        self.out_bytes -= sz
                except OSError:
                    pass
        self._reset()

    def close(self):
        self._flush()


def _load_targets(path):
    with np.load(path) as z:
        return (z["binaryInputNCHWPacked"], z["globalInputNC"], z["policyTargetsNCMove"],
                z["globalTargetsNC"], z["valueTargetsNCHW"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data")
    p.add_argument("--out", default="distilled")
    p.add_argument("--teacher", required=True, help="teacher engine command (KataGo protocol)")
    p.add_argument("--n-files", type=int, default=None, help="cap on source files to consider")
    p.add_argument("--max-pos", type=int, default=None,
                   help="stop after ~this many newly-relabeled positions")
    p.add_argument("--max-gb", type=float, default=None,
                   help="stop when the output dir reaches ~this many GB")
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--visits", type=int, default=1, help="teacher visits (1 = raw policy)")
    p.add_argument("--inflight", type=int, default=2048,
                   help="teacher queries kept in flight; fills the NN batch (throughput knob)")
    p.add_argument("--shard-size", type=int, default=None,
                   help="if set, pack rows into shards of ~this many positions (fewer files/inodes)")
    p.add_argument("--delete-src", action="store_true",
                   help="delete each source file once its rows are safely written to a shard")
    p.add_argument("--overwrite", action="store_true", help="re-relabel even if cached")
    args = p.parse_args()

    files = ndata.list_npz(args.src)
    if args.n_files is not None:
        files = files[: args.n_files]
    os.makedirs(args.out, exist_ok=True)
    max_bytes = int(args.max_gb * 1e9) if args.max_gb is not None else None

    if args.shard_size:
        _run_sharded(args, files, max_bytes)
    else:
        _run_per_file(args, files, max_bytes)


def _run_sharded(args, files, max_bytes):
    # NOTE: a teacher's targets are net-specific -> use a SEPARATE --out dir per teacher.
    sw = ShardWriter(args.out, args.shard_size)
    src_by_base = {os.path.basename(f): f for f in files}
    reached = lambda: (max_bytes is not None and sw.out_bytes >= max_bytes) \
        or (args.max_pos is not None and sw.pos_seen >= args.max_pos)
    sw.pos_seen = 0

    # On resume, source files already in a shard (manifest) were skipped, not re-relabeled, so
    # --delete-src never fired on them — sweep them now so leftover done-source can't leak inodes.
    if args.delete_src:
        swept = 0
        for f in files:
            if os.path.basename(f) in sw.done:
                try:
                    os.remove(f); swept += 1
                except OSError:
                    pass
        if swept:
            print(f"swept {swept} already-done source files", flush=True)

    # Phase 1: fold any pre-existing per-source tiny files (legacy layout) into shards, then drop
    # them — this both repacks old output and reclaims its inodes. Their name IS the source base.
    legacy = sorted(n for n in os.listdir(args.out)
                    if n.endswith(".npz") and not n.startswith("shard_") and n != "_manifest.txt")
    if legacy:
        print(f"repacking {len(legacy)} legacy per-file outputs into shards...", flush=True)
    for k, name in enumerate(legacy):
        path = os.path.join(args.out, name)
        if name in sw.done:  # already in a shard; just drop the stray tiny file
            try: os.remove(path)
            except OSError: pass
            continue
        try:
            packed, gnc, pol, gt, own = _load_targets(path)
        except Exception:
            continue
        dels = [path] + ([src_by_base[name]] if args.delete_src and name in src_by_base else [])
        sw.add(packed, gnc, pol, gt, own, name, dels)
        sw.pos_seen += packed.shape[0]
        if (k + 1) % 2000 == 0:
            print(f"  repacked {k+1}/{len(legacy)}, {sw.shards} shards, {sw.out_bytes/1e9:.1f} GB", flush=True)
    sw._flush()  # commit ALL repacked legacy to the manifest before computing todo, so phase 2
    #              never re-relabels a source still sitting in the (sub-shard-size) buffer.

    # Phase 2: relabel source files not yet done (skip via manifest = the idempotent cache).
    todo = [(f, os.path.join(args.out, os.path.basename(f)))
            for f in files if args.overwrite or os.path.basename(f) not in sw.done]
    print(f"sharded: {len(sw.done)} done, {len(todo)} to relabel; out={sw.out_bytes/1e9:.2f} GB"
          + (f", target ~{args.max_gb} GB" if max_bytes else ""), flush=True)
    teacher = TeacherEngine(args.teacher)
    n = 0
    try:
        if not reached():
            for src, _o, packed, gnc, pol, gt, own in relabel_stream(
                    todo, teacher, args.pos_len, args.visits, args.inflight):
                dels = [src] if args.delete_src else []
                sw.add(packed, gnc, pol, gt, own, os.path.basename(src), dels)
                sw.pos_seen += packed.shape[0]
                n += 1
                if n % 200 == 0:
                    print(f"{n} relabeled, {sw.shards} shards, {sw.out_bytes/1e9:.2f} GB", flush=True)
                if reached():
                    print("reached target"); break
    finally:
        sw.close()
        teacher.close()
    print(f"done -> {args.out}/ : {sw.shards} shards this run, "
          f"{len([1 for _ in os.scandir(args.out)])} files in dir, {sw.out_bytes/1e9:.2f} GB", flush=True)


def _run_per_file(args, files, max_bytes):
    # Legacy resumable cache: output named after its source file (already a content hash).
    cached_bytes = sum(e.stat().st_size for e in os.scandir(args.out) if e.name.endswith(".npz"))
    items = [(f, os.path.join(args.out, os.path.basename(f))) for f in files]
    items = [(f, o) for f, o in items if args.overwrite or not os.path.exists(o)]
    cached = len(files) - len(items)
    print(f"{len(files)} src files: {cached} cached ({cached_bytes/1e9:.2f} GB), {len(items)} to do"
          + (f"; target ~{args.max_gb} GB" if max_bytes else ""))
    teacher = TeacherEngine(args.teacher)
    done = pos = new_bytes = 0
    try:
        if max_bytes is not None and cached_bytes >= max_bytes:
            print(f"already at {cached_bytes/1e9:.2f} GB >= target; nothing to do")
        else:
            for src, out_path, packed, gnc, pol, gt, own in relabel_stream(
                    items, teacher, args.pos_len, args.visits, args.inflight):
                _write_npz(out_path, packed, gnc, pol, gt, own)
                done += 1; pos += packed.shape[0]; new_bytes += os.path.getsize(out_path)
                if done % 50 == 0:
                    print(f"{done} new + {cached} cached, ~{pos} pos, "
                          f"{(cached_bytes + new_bytes)/1e9:.2f} GB", flush=True)
                if max_bytes is not None and cached_bytes + new_bytes >= max_bytes:
                    print(f"reached ~{args.max_gb} GB target"); break
                if args.max_pos is not None and pos >= args.max_pos:
                    print(f"reached ~{args.max_pos} pos target"); break
    finally:
        teacher.close()
    print(f"done -> {args.out}/ : {done} new + {cached} cached = "
          f"{len(ndata.list_npz(args.out))} files, ~{pos} new pos, "
          f"{(cached_bytes + new_bytes)/1e9:.2f} GB total")


if __name__ == "__main__":
    main()
