"""Snapshot curated models + experiment artifacts into released/ for git-push survival —
the pod's /workspace dies with the credits; only the repo outlives it.

Selection policy (balance git growth vs coverage):
  - FULL checkpoints (optimizer state included → future --resume) for current champions only.
  - STRIPPED (model weights + config only, ~half size) for baselines/frontier points whose
    purpose is play/eval/reproduction, not continued training.
  - registry.jsonl + frontier.png snapshots (gitignored at their working paths).
Only rewrites a file when the source changed (mtime+size), so unchanged check-ins are no-ops.

    uv run python scripts/snapshot_release.py            # snapshot per the SELECTION below
"""
from __future__ import annotations

import os
import shutil
import sys

import torch

RUNS = "/workspace/distill/runs"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "released")

# (run id, mode) — mode "full" keeps optimizer (resume-capable), "model" strips to weights+meta
SELECTION = [
    ("s8_b10pat_final", "full"),      # overall champion (b10pat, era-mix from scratch, 14.8M pos)
    ("s6_b10pat_mixcont", "full"),    # prior champion (polish lineage exemplar)
    ("s4_dw7pat1200", "full"),        # 759 MF champion (until s7 supersedes)
    ("s5_b10pat1200", "model"),       # pre-polish champion (polish-delta baseline)
    ("s2_dw7pat600", "model"),        # dw7pat scaling-curve point
    ("s_nbtpat", "model"),            # 561 MF frontier point
    ("s_dw7", "model"),               # the universal dw7 control
]
KEEP_KEYS = ("model", "model_config", "spatial_subset", "global_subset", "step")


def fresh(src, dst):
    return not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst) + 1


def main():
    os.makedirs(OUT, exist_ok=True)
    for cid, mode in SELECTION:
        src = os.path.join(RUNS, f"{cid}.pt")
        if not os.path.exists(src):
            print(f"skip {cid}: no checkpoint")
            continue
        dst = os.path.join(OUT, f"{cid}{'_model' if mode == 'model' else ''}.pt")
        if not fresh(src, dst):
            continue
        if mode == "full":
            shutil.copy2(src, dst)
        else:
            ckpt = torch.load(src, map_location="cpu")
            torch.save({k: ckpt[k] for k in KEEP_KEYS if k in ckpt}, dst)
            shutil.copystat(src, dst)
        print(f"snapshot {os.path.basename(dst)} ({os.path.getsize(dst)/1e6:.1f}MB, {mode})")
    for src, name in [(os.path.join(REPO, "experiments/registry.jsonl"), "registry.jsonl"),
                      (os.path.join(REPO, "experiments/frontier.png"), "frontier.png")]:
        dst = os.path.join(OUT, name)
        if os.path.exists(src) and fresh(src, dst):
            shutil.copy2(src, dst)
            print(f"snapshot {name}")


if __name__ == "__main__":
    main()
