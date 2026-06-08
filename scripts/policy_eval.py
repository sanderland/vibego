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


def collect_policies(engine_cmd, queries):
    eng = relabel.TeacherEngine(engine_cmd)
    out = []
    try:
        for q in queries:
            eng.send(q)
        for q in queries:
            r = eng.recv(q["id"])
            out.append((np.asarray(r["policy"], dtype=np.float64),
                        float(r["rootInfo"]["winrate"])))
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

    ref = collect_policies(args.ref, queries)
    ref_top = [int(np.argmax(p)) for p, _ in ref]

    print(f"\n{'engine':>10} {'top1%':>7} {'top5%':>7} {'valueMAE':>9}")
    for spec in args.engine:
        name, cmd = spec.split("=", 1)
        res = collect_policies(cmd, queries)
        top1 = np.mean([int(np.argmax(p) == ref_top[i]) for i, (p, _) in enumerate(res)])
        top5 = np.mean([int(ref_top[i] in np.argsort(p)[-5:]) for i, (p, _) in enumerate(res)])
        vmae = np.mean([abs(res[i][1] - ref[i][1]) for i in range(len(res))])
        print(f"{name:>10} {100*top1:>6.1f} {100*top5:>6.1f} {vmae:>9.3f}")


if __name__ == "__main__":
    main()
