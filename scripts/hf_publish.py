"""Publish the study's artifacts to Hugging Face (user request 2026-06-12): collection
'vibego' under user sanderland — one DATASET repo per data subset, one MODEL repo per net,
data CC-BY-4.0, no code (that's a later GitHub release). Token via HF_TOKEN env only.

    uv run python scripts/hf_publish.py --datasets    # create+upload dataset repos (resumable)
    uv run python scripts/hf_publish.py --models      # create+upload model repos from released/
    uv run python scripts/hf_publish.py --collection  # create/refresh the collection
"""
from __future__ import annotations

import argparse
import os
import sys

from huggingface_hub import HfApi

USER = "sanderland"
REPO_URL = "https://github.com/sanderland/vibego"

DATA_HEADER = """---
license: cc-by-4.0
language: []
tags: [go, baduk, weiqi, katago, distillation]
---
"""

DATA_FORMAT = """
## Format

Each `.npz` holds training rows in KataGo-v7-compatible arrays:
- `binaryInputNCHWPacked` (N, 22, 46) uint8 — packed 19x19 binary input planes (v7 channel order)
- `globalInputNC` (N, 19) float32 — global features (komi at index 5 = selfKomi/20, side-to-move relative)
- `policyTargetsNCMove` (N, 1, 362) float32 — teacher policy target (361 moves + pass), to-move
- `globalTargetsNC` (N, 64) float32 — [0:3] win/loss/nr, [3] scoreLead (to-move), [27] ownership weight
- `valueTargetsNCHW` (N, 1, 19, 19) — ownership target, to-move-positive

Teacher: public `kata1-b18c384nbt` queried at the stated visits. Upstream positions from
katagoarchive.org (g170 self-play / kata1 daily training data) — credit to lightvector and
the KataGo distributed-training contributors. License: CC-BY-4.0.
"""

DATASETS = [
    ("vibego-distilled-g170mix", "/workspace/distill/distilled-g170mix",
     "Era-stratified g170 self-play positions (b6c96/b10c128/b15c192-era heavy, thin b20 tail), "
     "relabeled by b18 at visits 1. The data behind the study's era-diversity result: mixing "
     "~20-30% of this into recent kata1 data beat recent-only at fixed compute (+13 scoreLead, "
     "2 seeds), with the weak eras carrying the entire effect. ~3.8M positions, 471 shards."),
    ("vibego-replay-relabel-ab", None,  # two source dirs, special-cased below
     "Paired label-quality A/B sets: the SAME 450k replayed g170 positions (full move-history "
     "context, real history input planes) labeled two ways — `v1/` b18 raw prior at 1 visit, "
     "`v32p/` b18 32-visit searched policy (visit-count distribution) + searched value. "
     "Study result: searched labels only reach parity with the raw prior at 32x label cost "
     "(and are decisively harmful if relabeling lacks history context)."),
    ("vibego-distilled-b18-kata1", "/workspace/distill/distilled-b18",
     "Recent kata1 training positions (daily archives 2026-05-04..06, multi-board-size, "
     "randomized komi) relabeled by b18 at visits 1 — the study's base distillation set. "
     "~44M positions, 5379 shards, ~75GB. Champion nets trained on this + the g170mix set."),
]

# (repo_suffix, released file, blurb) — model repos, one per net
MODELS = [
    ("vibego-s11-b12c152nbt-pat", "s11_b12c152_cap.pt",
     "OVERALL STUDY CHAMPION. b12c152nbt-pat (4.21M params, 2686 MFLOP/eval, ~30.8 single-thread "
     "CPU-ms). 260k steps on the full pool (44M kata1 positions + weak-era x3, ~20% mix). "
     "Strength: BEATS g170e-b10c128 by +7.7 ± 2.4 judge scoreLead [+3.0, +12.5] over 192 paired "
     "games at 48 visits (b18 judge, 256v); h2h vs the 2.6M s10 +21.0 ± 7.8. The b10 tier was "
     "cleared by a capacity step — same data as s10 (which measured −9.5 sL), +17 sL from width "
     "alone. Full checkpoint (optimizer included, resume-capable)."),
    ("vibego-s9-b10c128nbt-pat", "s9_b10pat_parity.pt",
     "b10c128nbt-pat (2.57M params, 1567 MFLOP/eval, ~17.8 single-thread CPU-ms). 240k steps on "
     "23.5M positions (kata1-2400sh + full weak-era pool). −8.6 ± 2.4 judge scoreLead vs "
     "g170e-b10c128 over 192 paired games (48 visits, b18 judge 256v) at 0.7x the anchor's "
     "FLOPs; decisively above g170-b6c96. Full checkpoint (resume-capable)."),
    ("vibego-s10-b10c128nbt-pat-max", "s10_b10pat_max.pt",
     "Maximal-data candidate: b10c128nbt-pat on the FULL pool (44M kata1 positions + weak-era "
     "x3 oversample, ~20% mix), 320k steps. See collection notes for final Stage-B numbers."),
    ("vibego-s8-b10c128nbt-pat", "s8_b10pat_final.pt",
     "b10c128nbt-pat, 14.8M positions (kata1-1400sh + 23% era mix), 180k steps. −20.4 ± 3.7 sL "
     "vs g170e-b10c128. Full checkpoint."),
    ("vibego-s6-b10c128nbt-pat-polish", "s6_b10pat_mixcont_model.pt",
     "The cheap-polish exemplar: s5 resumed +60k steps on era-mixed data → +18.6 sL / +120 Elo "
     "over its base (h2h, decisive). Weights-only."),
    ("vibego-s5-b10c128nbt-pat", "s5_b10pat1200_model.pt",
     "b10c128nbt-pat, kata1-1200sh/120k (pre-polish baseline). Weights-only."),
    ("vibego-s4-b7c106nbt-pat", "s4_dw7pat1200.pt",
     "759-MFLOP tier champion (1.38M params, ~11.2 single-thread CPU-ms): first net of the study "
     "to decisively beat g170-b6c96 (+14.4 sL [2.5,26.3] / Elo +124 [40,227], 64 games, 48v). "
     "Full checkpoint."),
    ("vibego-s2-b7c106nbt-pat-600", "s2_dw7pat600_model.pt",
     "dw7pat scaling-curve point (600sh/60k): Elo −22 vs g170-b6c96. Weights-only."),
    ("vibego-b6c96nbt-pat", "s_nbtpat_model.pt",
     "561-MFLOP frontier point (0.9M params, ~8.2 CPU-ms) with the 3x3 dihedral pattern table "
     "(~+200 Elo at this size, ~0 FLOPs). Weights-only."),
    ("vibego-b7c106nbt", "s_dw7_model.pt",
     "The study's universal dw7 control net (b7c106nbt, 300sh/30k). Weights-only."),
]

MODEL_HEADER = """---
license: cc-by-4.0
tags: [go, baduk, weiqi, katago, mcts, distillation]
---
"""

MODEL_FOOTER = f"""
## Format / usage

PyTorch checkpoint: `{{'model': state_dict, 'model_config': dict, 'spatial_subset': [...],
'global_subset': [...], 'step': int}}` (full checkpoints also carry `optimizer`). Inputs are a
14-channel subset of KataGo v7 spatial features + 2 global features; the engine, training
pipeline, and evaluation harness will be released at {REPO_URL} (the study writeup lives there
under `experiments/WRITEUP.md`). Strength numbers are judge scoreLead / win-rate Elo from
paired color-reversed-opening matches at 48 visits/move with a kata1-b18 judge.

Distilled from the public `kata1-b18c384nbt` net over katagoarchive.org positions — credit to
lightvector and the KataGo distributed-training contributors.
"""


def publish_datasets(api: HfApi, only=None):
    for name, path, blurb in DATASETS:
        if only and name != only:
            continue
        repo = f"{USER}/{name}"
        api.create_repo(repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=(DATA_HEADER + f"# {name}\n\n{blurb}\n" + DATA_FORMAT).encode(),
                        path_in_repo="README.md", repo_id=repo, repo_type="dataset")
        if name == "vibego-replay-relabel-ab":
            # upload_large_folder doesn't take path_in_repo; these subdirs are small enough
            # (~600MB each) for plain upload_folder
            for sub, p in [("v1", "/workspace/distill/replay_v1"), ("v32p", "/workspace/distill/replay_v32p")]:
                api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=p,
                                  path_in_repo=sub, allow_patterns=["*.npz"])
        else:
            api.upload_large_folder(repo_id=repo, repo_type="dataset", folder_path=path,
                                    allow_patterns=["*.npz", "_manifest.txt"])
        print(f"dataset {repo} done")


def publish_models(api: HfApi):
    rel = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "released")
    for name, fname, blurb in MODELS:
        src = os.path.join(rel, fname)
        if not os.path.exists(src):
            print(f"skip {name}: {fname} not in released/")
            continue
        repo = f"{USER}/{name}"
        api.create_repo(repo, repo_type="model", exist_ok=True)
        api.upload_file(path_or_fileobj=(MODEL_HEADER + f"# {name}\n\n{blurb}\n" + MODEL_FOOTER).encode(),
                        path_in_repo="README.md", repo_id=repo, repo_type="model")
        api.upload_file(path_or_fileobj=src, path_in_repo=fname, repo_id=repo, repo_type="model")
        print(f"model {repo} done")


def make_collection(api: HfApi):
    from huggingface_hub import get_collection
    col = api.create_collection(
        title="vibego", namespace=USER, exists_ok=True,
        description="Tiny distilled KataGo-style Go nets (one GPU, public data) "
                    "+ the distillation datasets. Writeup in the repo.")
    items = [(f"{USER}/{n}", "model") for n, f, _ in MODELS
             if os.path.exists(os.path.join("released", f))]
    items += [(f"{USER}/{n}", "dataset") for n, _, _ in DATASETS]
    for rid, rtype in items:
        try:
            api.add_collection_item(col.slug, item_id=rid, item_type=rtype, exists_ok=True)
        except Exception as e:
            print(f"collection add {rid}: {e}")
    print(f"collection: https://huggingface.co/collections/{col.slug}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", action="store_true")
    p.add_argument("--only-dataset", default=None)
    p.add_argument("--models", action="store_true")
    p.add_argument("--collection", action="store_true")
    args = p.parse_args()
    api = HfApi()
    print("as:", api.whoami()["name"])
    if args.datasets or args.only_dataset:
        publish_datasets(api, only=args.only_dataset)
    if args.models:
        publish_models(api)
    if args.collection:
        make_collection(api)


if __name__ == "__main__":
    main()
