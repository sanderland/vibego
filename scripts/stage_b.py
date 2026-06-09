"""Stage-B validation: play each candidate checkpoint vs the fixed anchor gauntlet (g170-b6c96 at
fixed visits, b18 neutral judge) and record Elo±CI to the registry, so the REAL frontier (Elo vs
FLOPs / CPU-ms) can be drawn. Screening (val-loss) only ranks candidates; Elo is the verdict.

Each candidate is one `scripts/match.py` run (our engine vs anchor); we parse its Elo line and merge
elo/elo_lo/elo_hi onto that id's existing screening row (carrying arch/flops/val_loss). Candidates
run sequentially — each match.py already parallelizes its games across --workers.

    uv run python scripts/stage_b.py --candidates t_muon_lr04,m_dw10,m_rwkv6 --games 48 --visits 48
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KATAGO = os.path.join(REPO, "katago_bin", "katago.sh")
CFG = os.path.join(REPO, "katago_bin", "analysis.cfg")
ELO_RE = re.compile(r"Elo\s+([+-]?\d+)\s+\[95% CI\s+([+-]?\d+),\s*([+-]?\d+)\]")
WR_RE = re.compile(r"win rate:\s+([\d.]+)%")


def run_match(ckpt, anchor_model, judge_model, games, visits, judge_visits, workers, name):
    a = f"uv run python scripts/run_engine.py -model {ckpt} -early-stop"  # safe locked-winner stop
    b = f"{KATAGO} analysis -model {anchor_model} -config {CFG}"
    judge = f"{KATAGO} analysis -model {judge_model} -config {CFG}"
    cmd = ["uv", "run", "python", "scripts/match.py",
           "--a", a, "--a-name", name, "--b", b, "--b-name", "anchor",
           "--judge", judge, "--games", str(games), "--visits", str(visits),
           "--judge-visits", str(judge_visits), "--workers", str(workers)]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    log = out.stdout + "\n" + out.stderr
    m = ELO_RE.search(log)
    wr = WR_RE.search(log)
    res = {"rc": out.returncode}
    if m:
        res["elo"], res["elo_lo"], res["elo_hi"] = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if wr:
        res["winrate"] = float(wr.group(1))
    return res, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", required=True, help="comma-separated run ids (ckpt = runs/<id>.pt)")
    p.add_argument("--runs-dir", default="/workspace/distill/runs")
    p.add_argument("--anchor", default=os.path.join(REPO, "models/b6c96.bin.gz"))
    p.add_argument("--judge", default=os.path.join(REPO, "models/b18.bin.gz"))
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--visits", type=int, default=48)
    p.add_argument("--judge-visits", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    existing = {r["id"]: r for r in reg.read_rows()}  # carry arch/flops/val_loss onto the B row
    cands = [c for c in args.candidates.split(",") if c]
    print(f"Stage B: {len(cands)} candidates vs anchor (b6c96 @ {args.visits}v, b18 judge {args.judge_visits}v), {args.games} games each")
    for cid in cands:
        ckpt = os.path.join(args.runs_dir, f"{cid}.pt")
        if not os.path.exists(ckpt):
            print(f"  !! {cid}: no checkpoint at {ckpt}; skip"); continue
        print(f"  [{cid}] playing {args.games}...", flush=True)
        res, log = run_match(ckpt, args.anchor, args.judge, args.games, args.visits,
                             args.judge_visits, args.workers, cid)
        base = dict(existing.get(cid, {"id": cid, "axis": "arch", "arch": "?"}))
        base.update({"id": cid, "stage": "B",
                     "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "extra": {"games": args.games, "visits": args.visits, "anchor": "g170-b6c96"}})
        for k in ("elo", "elo_lo", "elo_hi", "winrate"):
            if k in res:
                base[k] = res[k]
        reg.append_row(base)
        print(f"  [{cid}] Elo={res.get('elo')} [{res.get('elo_lo')},{res.get('elo_hi')}] "
              f"wr={res.get('winrate')}% (rc={res['rc']})", flush=True)

    print("\n=== real frontier (FLOPs↓ vs Elo↑) ===")
    rows = reg.read_rows()
    front, _ = reg.pareto(rows, "flops", "elo", minimize_x=True, minimize_y=False)
    for r in sorted(front, key=lambda r: r.get("flops", 0)):
        print(f"  {r['id']:14s} {r.get('arch'):14s} {r.get('flops')} MFLOP  Elo={r.get('elo')}")


if __name__ == "__main__":
    main()
