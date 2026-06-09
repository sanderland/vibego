"""Raw-network diagnostic (no search, no games): on a fixed set of positions, compare each
engine's RAW policy (maxVisits=1) against a strong reference (b18) — top-1 / top-5 move
agreement and value MAE. This isolates *network* quality from search and from game variance.

If our net agrees with b18 far less than b6c96 does, the gap is in the net (features/data/
training); if it's comparable, the net is fine and the gap is in search.

    uv run python scripts/policy_eval.py --src data --n 400 \
      --ref  "katago analysis -model models/kata1-b18c384nbt.bin.gz -config katago/cpp/configs/analysis_example.cfg" \
      --engine "ours=uv run python scripts/run_engine.py -model checkpoints/depth6_distill_1m.pt" \
      --engine "b6c96=katago analysis -model models/g170-b6c96.bin.gz -config katago/cpp/configs/analysis_example.cfg"
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for sibling 'relabel'

import relabel  # noqa: E402  (reuse TeacherEngine + _row_query board reconstruction)
from nanogo.net import data as ndata  # noqa: E402


def collect(engine_cmd, queries, chunk=16):
    """Return per-position dicts: policy, winrate, score, ownership.

    Send/recv in small chunks: sending *all* queries before reading deadlocks for engines with
    large responses (ours: 361 policy + 361 ownership floats per position) — the engine's stdout
    pipe fills, it blocks on write and stops reading our stdin. Reading after each small chunk
    keeps the pipe drained."""
    eng = relabel.TeacherEngine(engine_cmd)
    out = []
    try:
        for i in range(0, len(queries), chunk):
            batch = queries[i:i + chunk]
            for q in batch:
                eng.send(q)
            for q in batch:
                r = eng.recv(q["id"])
                out.append({
                    "policy": np.asarray(r["policy"], dtype=np.float64),
                    "winrate": float(r["rootInfo"]["winrate"]),
                    "score": float(r["rootInfo"]["scoreLead"]),
                    "ownership": np.asarray(r.get("ownership", []), dtype=np.float64),
                })
    finally:
        eng.close()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data")
    p.add_argument("--n", type=int, default=400, help="number of positions")
    p.add_argument("--pos-len", type=int, default=19)
    p.add_argument("--ref", required=True, help="reference engine cmd (ground truth, e.g. b18)")
    p.add_argument("--engine", action="append", required=True, help="name=cmd (repeatable)")
    args = p.parse_args()

    # Build queries from real positions (reconstructed from npz feature planes).
    queries = []
    for path in ndata.list_npz(args.src):
        if len(queries) >= args.n:
            break
        with np.load(path) as npz:
            bin_nchw = np.unpackbits(npz["binaryInputNCHWPacked"], axis=2)[:, :, : args.pos_len ** 2]
            bin_nchw = bin_nchw.reshape(bin_nchw.shape[0], 22, args.pos_len, args.pos_len)
            glob = npz["globalInputNC"]
        for i in range(bin_nchw.shape[0]):
            if len(queries) >= args.n:
                break
            q, _xs, _ys = relabel._row_query(bin_nchw[i], glob[i], args.pos_len,
                                             5, f"q{len(queries)}", 1)
            queries.append(q)
    print(f"{len(queries)} positions")

    ref = collect(args.ref, queries)
    ref_top = [int(np.argmax(r["policy"])) for r in ref]

    print(f"\nvs reference (ground truth) — agreement / mean-abs-error per channel")
    print(f"{'engine':>14} {'pol top1%':>9} {'top5%':>7} {'winrate MAE':>12} "
          f"{'score MAE':>10} {'ownership MAE':>14}")
    for spec in args.engine:
        name, cmd = spec.split("=", 1)
        res = collect(cmd, queries)
        n = len(res)
        top1 = np.mean([int(np.argmax(res[i]["policy"]) == ref_top[i]) for i in range(n)])
        top5 = np.mean([int(ref_top[i] in np.argsort(res[i]["policy"])[-5:]) for i in range(n)])
        wr = np.mean([abs(res[i]["winrate"] - ref[i]["winrate"]) for i in range(n)])
        sc = np.mean([abs(res[i]["score"] - ref[i]["score"]) for i in range(n)])
        ow = np.mean([np.mean(np.abs(res[i]["ownership"] - ref[i]["ownership"]))
                      for i in range(n) if res[i]["ownership"].size == ref[i]["ownership"].size])
        print(f"{name:>14} {100*top1:>8.1f} {100*top5:>6.1f} {wr:>12.3f} {sc:>10.2f} {ow:>14.3f}")


if __name__ == "__main__":
    main()
