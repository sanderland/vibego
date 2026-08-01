# vibego

A flexible, hackable platform for **small-network Go experimentation** — an
agent-driven lab for trying out architectures, features, training recipes, and search on
KataGo-style nets, with the tooling to measure whether each change actually helped.

It's light enough to read end to end (PyTorch + numpy, one file per concept, in the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat)),
but the goal is **extensibility, not minimalism**: an architecture registry you extend by adding
one line, a feature set you ablate by editing one list, distillation from any KataGo teacher,
neutral-judge evaluation, arenas with proper Elo, a search-isolation proxy, an inference-speed
benchmark, and a dated lab notebook — all designed to be driven by a coding agent running and
recording experiments.


It trains on real [katagoarchive.org](https://katagoarchive.org/) self-play data (no self-play
generation of its own) and serves a JSON analysis engine that
[KaTrain](https://github.com/sanderland/katrain) can drive.

## Results: the June 2026 distillation study

A multi-day agent-driven study using this platform trained 0.8–2.6M-param nets purely by
distilling the public `kata1-b18c384nbt` net over public positions — **no self-play RL loop** —
on a single GPU:

- a **1.38M-param / 759 MFLOP net decisively beats `g170-b6c96`** (+14.4 judge scoreLead
  [2.5, 26.3], Elo +124 [40, 227]) at ~⅓ the anchor's inference cost;
- the champion **b12c152nbt-pat (4.2M params) beats `g170e-b10c128`** by +7.7 ± 2.4 scoreLead
  [+3.0, +12.5] over 192 paired games — reached by a capacity step (same data as the saturated
  2.6M net, +17 scoreLead from width alone);
- along the way: data **diversity beats data strength** at fixed compute (and in-domain val
  loss anti-correlates), searched/amplified distillation targets are a trap unless relabeling
  has full history context (and break even at best), Gumbel root search loses to PUCT for
  distilled nets at low visits, and screening verdicts routinely **flip sign with scale**.

**Read the full writeup: [`experiments/WRITEUP.md`](experiments/WRITEUP.md)** — methods,
mechanisms, and every number with CIs. The dated lab notebook is in
[`experiments/`](experiments/INDEX.md).

### Artifacts on Hugging Face

Everything is published in the
**[vibego collection](https://huggingface.co/collections/sanderland/vibego-6a2bd05f6853451f0d0fabf8)**:

- **Models** (one repo per net, full checkpoints resume-capable): champion
  [s11](https://huggingface.co/sanderland/vibego-s11-b12c152nbt-pat) ·
  [s9](https://huggingface.co/sanderland/vibego-s9-b10c128nbt-pat) ·
  [s10](https://huggingface.co/sanderland/vibego-s10-b10c128nbt-pat-max) ·
  [s8](https://huggingface.co/sanderland/vibego-s8-b10c128nbt-pat) ·
  [s6-polish](https://huggingface.co/sanderland/vibego-s6-b10c128nbt-pat-polish) ·
  [s5](https://huggingface.co/sanderland/vibego-s5-b10c128nbt-pat), the 759-MFLOP tier
  [s4](https://huggingface.co/sanderland/vibego-s4-b7c106nbt-pat) ·
  [s2](https://huggingface.co/sanderland/vibego-s2-b7c106nbt-pat-600), and baselines
  [b6c96nbt-pat](https://huggingface.co/sanderland/vibego-b6c96nbt-pat) ·
  [b7c106nbt](https://huggingface.co/sanderland/vibego-b7c106nbt).
- **Datasets** (CC-BY-4.0, ~92GB):
  [vibego-distilled-b18-kata1](https://huggingface.co/datasets/sanderland/vibego-distilled-b18-kata1)
  (44M b18-relabeled kata1 positions),
  [vibego-distilled-g170mix](https://huggingface.co/datasets/sanderland/vibego-distilled-g170mix)
  (the era-diversity set),
  [vibego-replay-relabel-ab](https://huggingface.co/datasets/sanderland/vibego-replay-relabel-ab)
  (paired label-quality A/B sets).

A small selection also lives in [`released/`](released/) in this repo.

## What's here

| Module | Role |
|--------|------|
| `vibego/go/board.py` | Go board: stones, liberties, captures, simple ko, history |
| `vibego/go/features.py` | A subset of KataGo's V7 input features, decoded from `.npz` *and* recomputed from a board (validated to match KataGo's own encoder) — ablate by editing `SPATIAL_SUBSET` / `GLOBAL_SUBSET` |
| `vibego/net/model.py` | Pre-activation ResNet trunk (regular / global-pooling / **nested-bottleneck `nbt`** blocks) + policy / value / score / ownership heads, selected from an **`ARCHS` registry** |
| `vibego/net/data.py` | Streaming shuffle-buffer loader over archive `.npz` files |
| `vibego/net/losses.py` | Multi-head loss (policy CE, value CE, score Huber, ownership CE) |
| `vibego/engine/search.py` | Batched `NNEvaluator` + leaf-parallel (virtual-loss) PUCT MCTS, search decoupled from inference |
| `vibego/engine/analysis.py` | KataGo-compatible JSON analysis protocol (streaming, concurrent queries) |
| `vibego/engine/proxy.py` | Run our MCTS on an **external** KataGo net — isolates search quality from net quality |
| `vibego/eval/` | `selfplay.py` (+ Tromp-Taylor scoring), `elo.py` (Bayesian Elo w/ credible intervals), `arena.py` (round-robin) |
| `vibego/katago/` | Open a **released** KataGo `.bin.gz` directly (format v8–17, incl. the v1.17 transformers): parse → edit → write back byte-exactly, params/FLOPs per block, and in-format structural pruning that the stock engine still loads |

### Experiment tooling (`scripts/`)

| Script | Use |
|--------|-----|
| `train.py` | Train any registry arch on archive or distilled data |
| `relabel.py` | Distill a stronger teacher's targets onto positions (logit forcing) |
| `run_engine.py` | Serve a checkpoint as a KataGo-protocol analysis engine |
| `match.py` / `vs.py` | Engine-vs-engine games with a neutral judge → mean scoreLead ± stderr + Elo ± CI |
| `arena.py` | Round-robin Bayesian-Elo tournament over checkpoints |
| `policy_eval.py` | Raw-net agreement (policy/value/score/ownership) vs a reference net — no search, no games |
| `move_eval.py` | Low-noise per-move "points conceded vs a reference" |
| `trace_search.py` | Deterministic batch-1 node-by-node search trace for harness validation/debugging |
| `bench_net.py` | Wall-clock inference speed (nnevals/s) per arch — compare nets **speed-matched**, not just size-matched |
| `kata_inspect.py` | What's actually inside a released KataGo net — arch, params, FLOPs/eval, per-block breakdown (no engine, no torch) |
| `kata_prune.py` | Structurally prune a released net (blocks / attention heads / FFN width) into a `.bin.gz` the stock engine loads — measurable immediately with the scripts above |
| `setup_katago.sh` / `download_data.py` | Fetch KataGo teacher/judge nets and training data |

`experiments/` is the **lab notebook**: one dated markdown per experiment (question → setup →
numbers with error bars → conclusion), plus `SUMMARY.md` (live digest + things-to-try) and
`INDEX.md`. The submodules `katago/`, `nanochat/`, `nanogpt/`, `katrain/` are references only.

## Quick start

```bash
uv sync --extra dev
uv run pytest                              # board + feature-consistency + model/arch tests

# Download one daily archive (~1.4GB extracted, ~2.5M positions / ~100k+ games)
uv run python scripts/download_data.py --n 1 --from 2021-06-01

# Train a network — pick any architecture from the ARCHS registry (see net/model.py)
uv run python scripts/train.py --data data --arch b6c96-gpool \
    --batch-size 256 --max-steps 30000 --out checkpoints/b6c96.pt

# Self-play sanity check
uv run python scripts/play_demo.py -model checkpoints/b6c96.pt --moves 40 --visits 50

# Serve as a JSON analysis engine
echo '{"id":"x","moves":[["B","Q16"]],"komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":100,"includePolicy":true,"includeOwnership":true}' \
  | uv run python scripts/run_engine.py -model checkpoints/b6c96.pt

# Rank checkpoints by Bayesian Elo (round-robin self-play)
uv run python scripts/arena.py --models checkpoints/b6c96.pt checkpoints/smoke.pt \
    --games 20 --visits 100 --board 19

# Head-to-head vs another engine, judged by a neutral strong net (any KataGo-protocol engine)
uv run python scripts/match.py --games 48 --visits 48 --workers 6 \
    --a "uv run python scripts/run_engine.py -model checkpoints/b6c96.pt" --a-name ours \
    --b "katago analysis -model models/g170-b6c96.bin.gz -config $KATA_CFG" --b-name b6c96 \
    --judge "katago analysis -model models/kata1-b18c384nbt.bin.gz -config $KATA_CFG" --judge-visits 256

# Compare architectures by inference speed (speed-matched vs size-matched)
uv run python scripts/bench_net.py --device mps --batch-sizes 1,16
```

## Architectures

Architectures live in the `ARCHS` registry in `vibego/net/model.py` and are selected by name
(`--arch`). Trunks are described by a list of `block_kinds` (`regular` / `gpool` / `nbt`), so new
block types drop in without new config flags. The current ladder spans classic ResNet
(`b6c96-gpool` … `b15c192-gpool`) and nested-bottleneck (`b6c96nbt` … `b15c192nbt`) families, plus
param-matched variants for fair comparisons. Adding an entry is the whole cost of trying a new size
or block type.

## Using it from KaTrain

Point KaTrain's engine at vibego using the **custom command** backend:

```
uv run --project /path/to/vibego python /path/to/vibego/scripts/run_engine.py -model /path/to/checkpoints/b6c96.pt
```

`run_engine.py` accepts (and ignores) the KataGo-style `analysis -model ... -config ...` flags
KaTrain passes, and honors `overrideSettings.reportAnalysisWinratesAs: "BLACK"`.

## Design notes

- **Feature subset.** We use only the V7 channels that are cheap and exact to recompute from a
  board state (on-board, stones, 1/2/3 liberties, simple ko, last-5 moves; globals: komi,
  pass-would-end). `tests/test_features.py` plays random games on both our board and KataGo's
  reference board and asserts the selected channels match exactly — a net trained on archive
  `.npz` sees identical inputs at play time. Ablate by editing the subset lists.
- **Heads.** policy, value (win/loss/no-result), score lead, and ownership.
- **Engine.** Search is decoupled from inference, KataGo-style: a batching `NNEvaluator` thread
  merges leaf evaluations from all concurrent queries (and from leaf-parallel search with virtual
  loss) into single forward passes. One search thread per query lets a ponder, an AI move, and node
  analysis run at once and share batches; results stream while a search runs and stop on `terminate`.
- **Distillation (logit forcing).** `scripts/relabel.py` rewrites positions' targets with a
  stronger teacher's policy/value/ownership (e.g. a KataGo b18), then `train.py --data distilled`
  trains the student to match them — the policy cross-entropy against the teacher's soft policy is
  the logit forcing. (The archive `.npz` targets are themselves distillation from kata1's search.)
- **Measure everything.** Single games swing ±30–40 pts, so claims come from the arena
  (`match.py`, Elo ± CI), the per-move tool (`move_eval.py`), or raw-net agreement vs a *neutral*
  judge (`policy_eval.py`) — never a single game. Validate the harness with `trace_search.py` before
  trusting an aggregate. Compare architectures **speed-matched** (`bench_net.py`), not just by params.
- **Reproduce / extend.** Add an arch to `ARCHS`, edit a feature subset, or drop a new script in
  `scripts/` — then write up the result in `experiments/`. The platform is built to be driven by an
  agent running this loop.
