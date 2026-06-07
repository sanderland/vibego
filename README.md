# nanogo

A minimal KataGo-style Go engine and training pipeline, in the spirit of
[nanoGPT](https://github.com/karpathy/nanoGPT) / [nanochat](https://github.com/karpathy/nanochat):
minimal dependencies (just PyTorch + numpy), maximal simplicity, and built for fast
experimentation and ablations on small networks.

It trains on real [katagoarchive.org](https://katagoarchive.org/) self-play data (no
self-play generation of its own) and serves a JSON analysis engine that
[KaTrain](https://github.com/sanderland/katrain) can drive.

## What's here

| Module | Role |
|--------|------|
| `nanogo/go/board.py` | Minimal Go board: stones, liberties, captures, simple ko, history |
| `nanogo/go/features.py` | A subset of KataGo's V7 input features, decoded from `.npz` *and* recomputed from a board (validated to match KataGo's own encoder) |
| `nanogo/net/model.py` | Small pre-activation ResNet trunk + policy / value / score / ownership heads |
| `nanogo/net/data.py` | Streaming shuffle-buffer loader over archive `.npz` files |
| `nanogo/net/losses.py` | Multi-head loss (policy CE, value CE, score Huber, ownership CE) |
| `nanogo/engine/search.py` | Batched `NNEvaluator` + leaf-parallel (virtual-loss) PUCT MCTS, search decoupled from inference |
| `nanogo/engine/analysis.py` | KataGo-compatible JSON analysis protocol (streaming, concurrent queries) |
| `nanogo/eval/selfplay.py` | Self-play between two nets + Tromp-Taylor area scoring |
| `nanogo/eval/elo.py` | Bayesian Elo (Bradley-Terry MAP with a Gaussian prior + credible intervals) |
| `nanogo/eval/arena.py` | Round-robin tournament driver |
| `scripts/` | `download_data.py`, `train.py`, `run_engine.py`, `play_demo.py`, `arena.py`, `vs.py` (engine-vs-engine + neutral judge), `relabel.py` (distill a teacher) |

The submodules `katago/`, `nanochat/`, `nanogpt/`, `katrain/` are references only.

## Quick start

```bash
uv sync --extra dev
uv run pytest                              # board + feature-consistency + model tests

# Download one daily archive (~1.4GB extracted, ~2.5M positions / ~100k+ games)
uv run python scripts/download_data.py --n 1 --from 2021-06-01

# Train a depth-6 network (--gpool adds KataGo-style global-pooling blocks, recommended)
uv run python scripts/train.py --data data --blocks 6 --channels 96 --gpool \
    --batch-size 256 --max-steps 30000 --out checkpoints/depth6.pt

# Self-play sanity check
uv run python scripts/play_demo.py -model checkpoints/depth6.pt --moves 40 --visits 50

# Serve as a JSON analysis engine
echo '{"id":"x","moves":[["B","Q16"]],"komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":100,"includePolicy":true,"includeOwnership":true}' \
  | uv run python scripts/run_engine.py -model checkpoints/depth6.pt

# Rank several checkpoints by Bayesian Elo (round-robin self-play)
uv run python scripts/arena.py --models checkpoints/depth6.pt checkpoints/smoke.pt \
    --games 20 --visits 100 --board 19

# Play two engines head-to-head, judged by a neutral strong net (any KataGo-protocol engine)
uv run python scripts/vs.py \
    --black "uv run python scripts/run_engine.py -model checkpoints/depth6_gpool.pt" \
    --white "uv run python scripts/run_engine.py -model checkpoints/depth6.pt" \
    --judge "katago analysis -model models/kata1-b18c384nbt.bin.gz -config katago/cpp/configs/analysis_example.cfg" \
    --visits 48 --max-moves 200 --judge-visits 256
```

## Using it from KaTrain

Point KaTrain's engine at nanogo using the **custom command** backend:

```
uv run --project /path/to/nanogo python /path/to/nanogo/scripts/run_engine.py -model /path/to/checkpoints/depth6.pt
```

`run_engine.py` accepts (and ignores) the KataGo-style `analysis -model ... -config ...`
flags KaTrain passes, and honors `overrideSettings.reportAnalysisWinratesAs: "BLACK"`.

## Design notes

- **Feature subset.** We use only the V7 channels that are cheap and exact to recompute
  from a board state (on-board, stones, 1/2/3 liberties, simple ko, last-5 moves; globals:
  komi, pass-would-end). No ladder search / territory / encore. `tests/test_features.py`
  plays random games on both our board and KataGo's reference board and asserts the selected
  channels match exactly — so a net trained on archive `.npz` sees identical inputs at play time.
- **Heads.** policy, value (win/loss/no-result), score lead, and ownership.
- **Engine.** Search is decoupled from inference, KataGo-style: a batching `NNEvaluator`
  thread merges leaf evaluations from all concurrent queries (and from leaf-parallel search
  with virtual loss) into single forward passes. One search thread per query lets a ponder,
  an AI move, and node analysis run at once and share batches; results stream while a search
  runs and stop on `terminate`.
- **Data.** Archive `.npz` files hold ~25 correlated rows each; the loader fills a shuffle
  buffer across many files before batching.
- **Ablations.** Change `SPATIAL_SUBSET` / `GLOBAL_SUBSET` in `features.py`, or add an entry to
  the `ARCHS` registry in `net/model.py` and pass `--arch <name>` — everything else follows.
- **Distillation (logit forcing).** `scripts/relabel.py` rewrites positions' targets with a
  stronger teacher's policy/value/ownership (e.g. a KataGo b18), then `train.py --data distilled`
  trains the student to match them — the policy cross-entropy against the teacher's soft policy
  is the logit forcing. (The archive `.npz` targets are themselves distillation from kata1's search.)
