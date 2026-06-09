"""Stage-A screening: train many small configs CONCURRENTLY (the GPU is far underutilized by a
single ~1M-param net at batch 256, so packing N runs is the throughput win), then record each
config's val-loss + params + FLOPs to the registry and print the screening Pareto (val-loss↓ vs
FLOPs↓). This is the cheap pre-filter; promising points graduate to Stage-B arena Elo.

Each config is one `train.py` subprocess (its own CUDA context). We cap concurrency, parse the
final eval line, and append one registry row per run (axis/arch/params/flops/val_loss, stage=A).

    uv run python scripts/screen.py --batch a0 --data /workspace/distill/screen \
        --steps 8000 --concurrency 8
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for registry.py

from nanogo.net.flops import arch_flops
import registry as reg


# A config = (id, arch, axis, [extra train.py flags]). One row per config in the registry.
# Batch a0: lead with the training axis (Muon/EMA vs AdamW on the champion arch dw7), plus a few
# arch baselines on the new 75G-subset data to seed the frontier. One lever per row.
def _train_axis(arch="b7c106nbt"):
    return [
        ("t_adamw", arch, "train", ["--optimizer", "adamw", "--lr", "2e-3"]),
        ("t_adamw_ema", arch, "train", ["--optimizer", "adamw", "--lr", "2e-3", "--ema", "0.999"]),
        ("t_muon", arch, "train", ["--optimizer", "muon", "--lr", "0.02"]),
        ("t_muon_ema", arch, "train", ["--optimizer", "muon", "--lr", "0.02", "--ema", "0.999"]),
        ("t_muon_lr01", arch, "train", ["--optimizer", "muon", "--lr", "0.01"]),
        ("t_muon_lr04", arch, "train", ["--optimizer", "muon", "--lr", "0.04"]),
    ]


def _arch_axis():  # fixed optimizer (AdamW) so arch is the only lever; comparable to the old frontier
    base = ["--optimizer", "adamw", "--lr", "2e-3"]
    # (dw7/b7c106nbt + AdamW is already covered by t_adamw in the train axis, so omit it here.)
    return [(f"a_{n}", a, "arch", base) for n, a in [
        ("old6b", "b6c96-gpool"), ("nbt6b", "b6c96nbt"), ("dw10", "b10c88nbt"),
    ]]


BATCHES = {"a0": _train_axis() + _arch_axis()}

EVAL_RE = re.compile(r"\[eval final\]\s+(.*)")
KV_RE = re.compile(r"(\w+)=([\d.eE+-]+)")


def parse_final_eval(log_path):
    """Return the dict from the last '[eval final]' line, or None."""
    last = None
    try:
        with open(log_path) as f:
            for line in f:
                m = EVAL_RE.search(line)
                if m:
                    last = m.group(1)
    except OSError:
        return None
    if not last:
        return None
    return {k: float(v) for k, v in KV_RE.findall(last)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", default="a0", choices=list(BATCHES))
    p.add_argument("--data", default="/workspace/distill/screen")
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-files", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--runs-dir", default="/workspace/distill/runs")
    p.add_argument("--tag", default="", help="suffix appended to each run id (e.g. a date)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    configs = BATCHES[args.batch]
    os.makedirs(args.runs_dir, exist_ok=True)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"batch {args.batch}: {len(configs)} configs, steps={args.steps}, concurrency={args.concurrency}")
    for cid, arch, axis, flags in configs:
        print(f"  {cid:16s} {arch:14s} [{axis}] {' '.join(flags)}")
    if args.dry_run:
        return

    def make_cmd(cid, arch, flags):
        out = os.path.join(args.runs_dir, f"{cid}{args.tag}.pt")
        log = os.path.join(args.runs_dir, f"{cid}{args.tag}.log")
        cmd = ["uv", "run", "python", "scripts/train.py",
               "--data", args.data, "--arch", arch, "--out", out,
               "--max-steps", str(args.steps), "--batch-size", str(args.batch_size),
               "--val-files", str(args.val_files), "--seed", str(args.seed),
               "--eval-interval", str(max(500, args.steps // 8)),
               "--save-interval", str(args.steps)] + flags
        return cmd, out, log

    # bounded-concurrency process pool
    pending = list(configs)
    running = []  # (cid, arch, axis, proc, log, out)
    results = []
    t0 = time.time()
    while pending or running:
        while pending and len(running) < args.concurrency:
            cid, arch, axis, flags = pending.pop(0)
            cmd, out, log = make_cmd(cid, arch, flags)
            lf = open(log, "w")
            proc = subprocess.Popen(cmd, cwd=repo, stdout=lf, stderr=subprocess.STDOUT)
            running.append([cid, arch, axis, proc, log, out, lf])
            print(f"[{time.time()-t0:5.0f}s] launched {cid} (pid {proc.pid}), {len(running)} running")
        # reap finished
        still = []
        for entry in running:
            cid, arch, axis, proc, log, out, lf = entry
            if proc.poll() is None:
                still.append(entry)
                continue
            lf.close()
            ev = parse_final_eval(log)
            fp = arch_flops(arch)
            row = {
                "id": f"{cid}{args.tag}", "axis": axis, "arch": arch,
                "params": round(fp["params"] / 1e6, 4), "flops": round(fp["flops"] / 1e6, 1),
                "stage": "A", "data": os.path.basename(args.data.rstrip("/")),
                "steps": args.steps, "seed": args.seed,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            if ev:
                row["val_loss"] = round(ev.get("total", float("nan")), 4)
                row["val_policy"] = round(ev.get("policy", float("nan")), 4)
            row["extra"] = {"rc": proc.returncode}
            reg.append_row(row)
            results.append((cid, arch, axis, row.get("val_loss"), proc.returncode))
            tag = "OK" if proc.returncode == 0 else f"rc={proc.returncode}"
            print(f"[{time.time()-t0:5.0f}s] done {cid} [{tag}] val_loss={row.get('val_loss')}")
        running = still
        if running:
            time.sleep(5)

    print(f"\n=== batch {args.batch} done in {(time.time()-t0)/60:.1f} min ===")
    for cid, arch, axis, vl, rc in sorted(results, key=lambda r: (r[3] is None, r[3])):
        print(f"  {cid:16s} {arch:14s} [{axis}] val_loss={vl} {'' if rc==0 else f'(rc={rc})'}")
    print("\n=== screening Pareto (FLOPs↓ vs val_loss↓) ===")
    rows = reg.read_rows()
    front, _ = reg.pareto(rows, "flops", "val_loss", minimize_x=True, minimize_y=True)
    for r in sorted(front, key=lambda r: r.get("flops", 0)):
        print(f"  {r['id']:18s} {r['arch']:14s} {r.get('flops')} MFLOP  val_loss={r.get('val_loss')}")


if __name__ == "__main__":
    main()
