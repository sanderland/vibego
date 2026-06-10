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

from vibego.net.flops import arch_flops
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


def _arch_muon():
    # Stage-A round 2: arch axis under the Stage-A-winning optimizer (Muon lr 0.04). dw7+Muon is
    # already in the registry as t_muon_lr04, so it's the reference here. Includes the new global-
    # mixing blocks (rwkv/linattn) vs the gpool/nbt baselines — the "richer global connection" probe.
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [(f"m_{n}", a, "arch", m) for n, a in [
        ("old6b", "b6c96-gpool"), ("nbt6b", "b6c96nbt"), ("dw8", "b8c102nbt"),
        ("dw9", "b9c92nbt"), ("dw10", "b10c88nbt"), ("b10nbt", "b10c128nbt"),
        ("rwkv6", "b6c96-rwkv"), ("linat6", "b6c96-linat"),
        ("rwkv10", "b10c128-rwkv"), ("linat10", "b10c128-linat"),
    ]]


def _recipe():
    # Recipe FIRST (before arch): on the fixed champion arch dw7, line-search Muon LR past the a0
    # edge (0.04 won {0.01,0.02,0.04} -> extend up), + 2 targeted schedule probes. r_lr04 == a0's
    # t_muon_lr04 (already in registry), so not re-run. Not a grid: 1-D LR line + 2 schedule probes.
    def muon(lr, extra=()):
        return ["--optimizer", "muon", "--lr", str(lr), *extra]
    return [
        ("r_lr06", "b7c106nbt", "train", muon(0.06)),
        ("r_lr09", "b7c106nbt", "train", muon(0.09)),
        ("r_lr13", "b7c106nbt", "train", muon(0.13)),
        ("r_lr06_wu600", "b7c106nbt", "train", muon(0.06, ["--warmup", "600"])),
        ("r_lr06_wd0", "b7c106nbt", "train", muon(0.06, ["--lr-final-frac", "0.0"])),
    ]


def _arch_broad():
    # Broad arch screen under the recipe winner (Muon lr 0.04). dw7=t_muon_lr04 is the reference.
    # Same-FLOP A/Bs: globmod vs gpool (richer global conn), {gpool,nbt}-pat vs {gpool,nbt} (pattern
    # memory). rwkv/linat = the parked linear mixers (cheap keep/reject signal). 10b points for the
    # FLOP-scaling axis. Val-loss triage -> survivors go to Stage-B Elo (the verdict).
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [(f"g_{n}", a, "arch", m) for n, a in [
        ("gpool", "b6c96-gpool"), ("nbt", "b6c96nbt"),
        ("globmod", "b6c96-globmod"), ("gpoolpat", "b6c96-gpool-pat"), ("nbtpat", "b6c96nbt-pat"),
        ("rwkv", "b6c96-rwkv"), ("linat", "b6c96-linat"),
        ("globmod10", "b10c128-globmod"), ("b10nbt", "b10c128nbt"),
    ]]


def _scale():
    # Scaling run (more data + steps) on the conv-nbt set to test whether the b6c96 gap closes.
    # Includes the nbt-vs-nbtpat A/B at DATA SCALE (the only fair test of pattern-memory). Muon 0.04.
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [(f"s_{n}", a, "arch", m) for n, a in [
        ("nbt", "b6c96nbt"), ("dw7", "b7c106nbt"), ("nbtpat", "b6c96nbt-pat"), ("b10nbt", "b10c128nbt"),
    ]]


def _scale2():
    # Follow-ups to s0 at the same scale (300 shards, 30k steps — directly comparable rows):
    #  - pattern_embed on the frontier nets (s0: pattern was a decisive Elo win on 6b at ~0 cost)
    #  - the 19×19-only training arm of the data-filter A/B (control = s_dw7, already trained;
    #    compare BOTH post-hoc on a fixed 19×19 val via eval_loss --only-19x19, then Stage-B)
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [
        ("s1_dw7pat", "b7c106nbt-pat", "arch", m),
        ("s1_b10pat", "b10c128nbt-pat", "arch", m),
        ("s1_dw7f19", "b7c106nbt", "data", m + ["--only-19x19"]),
    ]


def _scale3():
    # Next data/steps scaling point on the expected-best arch (Elo-vs-scale curve hadn't bent at
    # 300sh/30k). Run with --data /workspace/distill/scale600 --steps 60000 (val shards 0-3 are
    # the same files as the scale/ subset, so val stays comparable).
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [("s2_dw7pat600", "b7c106nbt-pat", "data", m)]


def _seed_var():
    # Seed-variance replicates at scale (300sh/30k): quantifies run-to-run noise for the val/Elo
    # deltas we triage on (queued stats item), AND replicates the b10 pattern A/B at seed 2
    # (s1: +8.6±7.5 sL, suggestive only). Compare vs the seed-1 rows s_dw7 / s_b10nbt / s1_b10pat.
    m = ["--optimizer", "muon", "--lr", "0.04", "--seed", "2"]
    return [
        ("s3_dw7_sd2", "b7c106nbt", "train", m),
        ("s3_b10nbt_sd2", "b10c128nbt", "train", m),
        ("s3_b10pat_sd2", "b10c128nbt-pat", "train", m),
    ]


def _scale4():
    # Fourth scaling doubling on the champion arch (Elo curve −417/−112/−22 still unbent).
    # Run with --data /workspace/distill/scale1200 --steps 120000 (val shards 0–3 unchanged).
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [("s4_dw7pat1200", "b7c106nbt-pat", "data", m)]


def _scale5():
    # Fifth doubling on the champion + catch the 1567MF tier up to 1200sh. Launch with
    # --data ignored per-config via flags (each run names its own --data), steps from flags too.
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [
        ("s5_dw7pat2400", "b7c106nbt-pat", "data",
         m + ["--data", "/workspace/distill/scale2400", "--max-steps", "240000"]),
        ("s5_b10pat1200", "b10c128nbt-pat", "data",
         m + ["--data", "/workspace/distill/scale1200", "--max-steps", "120000"]),
    ]


def _explore_depth():
    # EXPLORE (user 06-10: "data doubling is exploit/final; explore is thin"): the ~760-MF depth
    # ladder at data scale (300sh/30k — the established good-signal point; 8k screens are known
    # to flip verdicts). Reference rows: s_dw7 / s3_dw7_sd2 (b7), a2's 8k-step verdicts.
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [(f"e0_{a}", a, "arch", m) for a in
            ["b9c92nbt", "b10c88nbt", "b12c78nbt", "b14c74nbt", "b16c68nbt"]]


def _explore_pattern_loss():
    # Explore round e1: (a) the 5×5 hashed pattern table at 6b (vs s_nbtpat's 3×3 = the A/B);
    # (b) loss-weight arms on the dw7 reference — does policy-heavier or ownership-free training
    # convert to Elo at scale? (References: s_dw7/s3_dw7_sd2.)
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [
        ("e1_pat5", "b6c96nbt-pat5", "arch", m),
        ("e1_wpol2", "b7c106nbt", "train", m + ["--w-policy", "2.0"]),
        ("e1_wown0", "b7c106nbt", "train", m + ["--w-ownership", "0.0"]),
    ]


def _explore_eramix():
    # Explore round e2: era-mixed data (stratified g170 b6c96/b10c128/b15c192-era sample, b18-
    # relabeled) appended to the kata1 scale set at ~10% / ~30% of shards, same 30k steps —
    # does era diversity beat pure recent-strong data? Controls: s_dw7/s3_dw7_sd2. Val shards
    # 0-3 are the same kata1 files in every arm (zmix_ names sort after shard_).
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [
        ("e2_mix10", "b7c106nbt", "data", m + ["--data", "/workspace/distill/mix10"]),
        ("e2_mix30", "b7c106nbt", "data", m + ["--data", "/workspace/distill/mix30"]),
    ]


def _explore_eramix2():
    # e3: the e2_mix30 Elo signal (best dw7-tier sL, −7.7±4.2, despite a −0.16 kata1-val hit)
    # is one seed and contradicts mix10's direction — replicate at seed 2 + extend the dose to
    # 50% before believing it.
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [
        ("e3_mix30sd2", "b7c106nbt", "data",
         m + ["--data", "/workspace/distill/mix30", "--seed", "2"]),
        ("e3_mix50", "b7c106nbt", "data", m + ["--data", "/workspace/distill/mix50"]),
    ]


def _explore_eramix3():
    # e4: era-mix30 CONFIRMED on dw7 (+13 sL pooled over 2 seeds, ~2.6σ, dose sweet spot at
    # ~30%) — does it transfer to the champion's tier? b10pat on mix30 vs the s1_b10pat
    # control (+1.7 ± 5.7 vs b6c96 anchor).
    m = ["--optimizer", "muon", "--lr", "0.04"]
    return [("e4_b10mix30", "b10c128nbt-pat", "data", m + ["--data", "/workspace/distill/mix30"])]


def _explore_attn_lr():
    # e5a: FAIR-SHOT recipe sweep for the attention/mixer trunks — the a2 screen used the
    # conv-tuned Muon 0.04 and 8k steps, which by our own sign-flip findings proves nothing.
    # Quick lr triage at 48sh/8k (relative lr ranking is what 8k CAN answer), winners go to
    # 300sh/30k + Elo in e5b. Transformers beat CNNs in chess (Lc0); Go at N=361 is open.
    out = []
    for arch, tag in [("b6c96-linat", "linat"), ("b6c96-rwkv", "rwkv"), ("b6c96-globmod", "globmod")]:
        for lr in ["0.01", "0.02", "0.04"]:
            out.append((f"e5a_{tag}_lr{lr.replace('0.', '')}", arch, "train",
                        ["--optimizer", "muon", "--lr", lr]))
    return out


def _explore_attn_scale():
    # e5b: the mixers' fair shot — best-lr arms (e5a: val nearly lr-flat, so lr was NOT what
    # buried them; lr02 best for rwkv/globmod) at data scale, then Stage-B Elo. This is the
    # test the a2 screen never ran; only Elo can acquit or convict (val ranks block types
    # wrongly — founding lesson).
    return [
        ("e5b_rwkv", "b6c96-rwkv", "arch", ["--optimizer", "muon", "--lr", "0.02"]),
        ("e5b_globmod", "b6c96-globmod", "arch", ["--optimizer", "muon", "--lr", "0.02"]),
    ]


BATCHES = {"a0": _train_axis() + _arch_axis(), "a1": _arch_muon(), "r0": _recipe(),
           "a2": _arch_broad(), "s0": _scale(), "s1": _scale2(), "s2": _scale3(),
           "s3": _seed_var(), "s4": _scale4(), "s5": _scale5(), "e0": _explore_depth(),
           "e1": _explore_pattern_loss(), "e2": _explore_eramix(), "e3": _explore_eramix2(),
           "e4": _explore_eramix3(), "e5a": _explore_attn_lr(), "e5b": _explore_attn_scale()}

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
    p.add_argument("--eval-interval", type=int, default=0, help="0 = steps//8; smaller = denser early curve")
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
               "--eval-interval", str(args.eval_interval or max(500, args.steps // 8)),
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
