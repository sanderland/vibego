"""Plot train + validation loss CURVES from screening run logs — the trajectory shows what a single
final number hides: still-descending (undertrained), spiking (LR too high), or val-vs-train gap
(overfit). Overlay several runs to compare LR/schedule/arch variants.

    uv run python scripts/curves.py --ids r_lr06,r_lr09,r_lr13 --out experiments/curves.png
    uv run python scripts/curves.py --glob 'm_*' --metric total
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/workspace/distill/runs"
TRAIN_RE = re.compile(r"step (\d+) lr=[\d.eE+-]+ (.*?) \(")          # "step N lr=.. total=.. ..."
EVAL_RE = re.compile(r"\[eval (?:step (\d+)|final)\]\s+(.*)")
KV_RE = re.compile(r"(\w+)=([\d.eE+-]+)")


def parse(log, metric):
    tr, va = [], []
    last_step = 0
    with open(log) as f:
        for line in f:
            m = TRAIN_RE.search(line)
            if m:
                last_step = int(m.group(1))
                kv = dict((k, float(v)) for k, v in KV_RE.findall(m.group(2)))
                if metric in kv:
                    tr.append((last_step, kv[metric]))
                continue
            e = EVAL_RE.search(line)
            if e:
                step = int(e.group(1)) if e.group(1) else last_step
                kv = dict((k, float(v)) for k, v in KV_RE.findall(e.group(2)))
                if metric in kv:
                    va.append((step, kv[metric]))
    return tr, va


MILESTONES = [250, 500, 1000, 2000, 4000, 6000, 8000, 12000, 20000]  # log-ish, dense early


def _at(series, step):
    """Nearest recorded (step,val) at or before `step` (else earliest)."""
    le = [v for s, v in series if s <= step]
    if le:
        return le[-1]
    return series[0][1] if series else None


def print_table(runs, metric):
    """runs: list of (name, train, val). Print val (and final train) at log-spaced steps — the
    numeric trajectory, denser early, <=~9 points."""
    maxstep = max((va[-1][0] for _, _, va in runs if va), default=0)
    cols = [m for m in MILESTONES if m <= maxstep][:9] or [maxstep]
    print(f"\n=== val {metric} at step milestones ===")
    print("run".ljust(16) + "".join(f"{c:>8}" for c in cols) + "   trainf")
    for name, tr, va in runs:
        cells = "".join(f"{(_at(va, c) if _at(va, c) is not None else float('nan')):>8.3f}" for c in cols)
        tf = tr[-1][1] if tr else float("nan")
        print(f"{name:16s}{cells}  {tf:7.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default=None, help="comma-separated run ids")
    p.add_argument("--glob", default=None, help="glob over run ids (e.g. 'm_*')")
    p.add_argument("--runs-dir", default=RUNS)
    p.add_argument("--metric", default="total", choices=["total", "policy", "value", "score", "ownership"])
    p.add_argument("--out", default="experiments/curves.png")
    args = p.parse_args()

    if args.ids:
        logs = [os.path.join(args.runs_dir, f"{i}.log") for i in args.ids.split(",")]
    else:
        logs = sorted(glob.glob(os.path.join(args.runs_dir, (args.glob or "*") + ".log")))
    logs = [l for l in logs if os.path.exists(l)]
    if not logs:
        print("no logs"); return

    runs = [(os.path.basename(l)[:-4], *parse(l, args.metric)) for l in logs]
    print_table(runs, args.metric)

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.tab10
    for i, (name, tr, va) in enumerate(runs):
        c = cmap(i % 10)
        if tr:
            ax.plot([s for s, _ in tr], [v for _, v in tr], "-", c=c, alpha=0.25, lw=1)
        if va:
            ax.plot([s for s, _ in va], [v for _, v in va], "o-", c=c, lw=1.8, ms=3,
                    label=f"{name} (val {va[-1][1]:.3f})")
    ax.set_xlabel("step"); ax.set_ylabel(f"{args.metric} loss")
    ax.set_title(f"train (faint) + val (bold) — {args.metric}")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out} ({len(logs)} runs)")


if __name__ == "__main__":
    main()
