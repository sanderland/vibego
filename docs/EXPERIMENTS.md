# nanogo experiments

A running log of training runs and what we learned. All strength numbers are from a neutral
**b18 judge** (`kata1-b18c384nbt`, 256 visits) scoring the final position of a 200-move game
played through `scripts/vs.py`. Scores are **points from Black's perspective** (positive = Black
ahead). Each cell is currently a **single game** unless noted — high variance; trust the
judge over rules-based area counting on unfinished games.

## Setup

- **Model**: ~1M-param ResNet, arch `b6c96-gpool` (6 blocks × 96ch, 2 global-pooling blocks),
  heads = policy / value (win-loss-noresult) / score / ownership.
- **Features**: a subset of KataGo's V7 input planes, validated channel-for-channel against
  KataGo's own Python encoder (incl. the ladder channels 14/17). Ladders come precomputed in
  the `.npz`, so they're free at training time.
- **Data**: kata1 self-play `.npz` from katagoarchive.org (2021, b40c256-generated). The policy
  /value/ownership targets are that run's MCTS output — i.e. training is already distillation
  from a ~b40-strength teacher.
- **Search**: leaf-parallel PUCT with virtual loss; a batching `NNEvaluator` decouples search
  from inference (KataGo-style). Engine speaks the KataGo JSON analysis protocol.

## Reference: KataGo g170 b6c96

The target to match. From its filename `g170-b6c96-s175395328-d26788732`:
**~26.8M unique positions, ~175M training samples (~6.5 epochs)**. Same capacity as ours.

## Runs

| net | features | data (positions / samples) | steps | policy | value | score | ownership |
|-----|----------|----------------------------|-------|--------|-------|-------|-----------|
| depth6 (v0) | 12-ch | 2.5M / 3.8M | 15k | 2.70 | 0.66 | ~50 | 0.38 |
| depth6_gpool | 12-ch + gpool | 7.4M / 7.7M | 30k | 2.52 | 0.63 | ~48 | 0.379 |
| depth6_v2 | 14-ch (ladders) + gpool, masked policy | 7.4M / 12.8M | 50k | 2.48 | 0.62 | ~33 | 0.374 |
| depth6_distill | b18 raw policy (logit forcing) | 0.10M / 12.8M | 15k | 2.52 | **0.55** | **9.0** | 0.374 |
| depth6_distill_1m | b18, ~1M positions | ~1.0M / TBD | TBD | _running_ | | | |

(score loss isn't comparable across supervised vs distill — different targets: noisy game
outcome vs b18's smooth scoreLead.)

## Strength vs KataGo b6c96 (b18 judge)

| net | as Black | as White | avg |
|-----|----------|----------|-----|
| depth6 (v0) | −106 | — | — |
| **depth6_v2** (supervised, 7.4M pos) | −70 | −125 | **−97** |
| **depth6_distill** (b18, 0.10M pos) | −59 | −70 | **−64** |
| **depth6_distill_1m** (b18, 0.997M pos) | −109 | −15 | **−62** |

`depth6_distill_1m` beat `depth6_distill` head-to-head (+30, one game), but its avg vs b6c96
is unchanged from the 10×-smaller distill set — and the ±50-point per-game swing for the *same*
net (−109 as B, −15 as W) shows single-game judging can't resolve differences this small.
10× more distillation data gave at best a marginal gain; at ~1M params with plain blocks we
look capacity/target-saturated. Next levers: a multi-game **arena** (to measure), and **nbt
blocks** (more strength per param) over more data.

Head-to-head `depth6_distill` vs `depth6_v2` (b18 judge): roughly even (−0.8 and +5.0 across
the two color assignments) — close in direct play, but distill is clearly better against the
fixed strong reference (b6c96) while using ~70× less data.

## Findings

1. **Distillation (logit forcing) is far more data-efficient.** 0.10M b18-labeled positions
   matched/beat 7.4M supervised positions vs b6c96, and gave much better-calibrated value/score
   (b18's winrate/scoreLead/ownership are smooth, accurate soft targets; the npz value/score are
   noisy single-game outcomes). This is the most promising path to match/beat b6c96.
2. **The gap to b6c96 is undertraining + data + target quality, not a capacity wall.** Same
   6×96 capacity; b6c96 saw ~13× more samples (~6.5 vs ~1.7 epochs).
3. **gpool + ladders + masked-policy helped modestly** (−106 → −97 avg; better calibration).
4. **Engine is Python/GIL-bound, not GPU-bound.** Tiny net does 4500+ pos/s batched on MPS;
   the costs are pure-Python legal-move/ladder/tree work. Ladder feature adds ~6 ms/node.
5. **Single-game judging is noisy** (large color asymmetries, area-vs-judge disagreements). A
   multi-game arena (Elo ± CI) is needed for a reliable number.

## Diagnostics: is the gap the net or the search?

Two tools decompose the gap (both sidestep noisy full games):

- **`scripts/policy_eval.py`** — raw policy/value agreement with a reference net (no search).
  vs **b18**, 300 positions:

  | net | top-1 | top-5 | value MAE |
  |-----|-------|-------|-----------|
  | ours (distill_1m) | 38.3% | 71.3% | 0.213 |
  | b6c96 | 33.0% | 63.7% | 0.226 |

  Our raw net matches b18 at least as well as b6c96 does (**biased** — we distilled from b18,
  the reference). Takeaway: the **network is not the bottleneck**.

- **`scripts/run_engine.py -proxy "<katago>"` + `nanogo/engine/proxy.py`** — run *our* MCTS on
  an external net (search-correctness test). our-search+b6c96 vs KataGo's own search, same
  b6c96 net, 48 visits (b18 judge): our side lost **both colors (−44, −165)**. Single-game
  noise inflates the 165, but losing both with the identical net shows **our search is
  materially weaker per visit** than KataGo's.

**Conclusion: the in-game gap to b6c96 is dominated by SEARCH, not the net.** Likely causes
(impact order): our Q is **win-rate only and ignores score** (KataGo's utility values score
margin); move selection by **raw visit count** (vs LCB/playSelectionValue); untuned cpuct/FPU.

## Open items / next

- **Improve the search (highest priority)**: add score to the PUCT utility, LCB/value move
  selection, tune cpuct/FPU — diagnostics say this is where most of the gap is.
- Expand features: **territory / pass-alive (18,19)** — cheap (reuse area flood-fill); ladder
  history (15,16) needs prev-board reconstruction in the engine.
- Multi-game **arena** (Elo ± CI) for final comparisons instead of single games.
- **Teacher ensembling** (average b18 + b28 + b40 policies, still 1 visit) as the next target-
  quality lever over searched policy.

## Validation / correctness

Feature encoding is validated against KataGo's reference encoder; a code review pass fixed:
off-board policy-CE masking, an evaluator deadlock on forward errors, a `terminate_all` race,
and double liberty-grid computation per leaf. 34 tests pass (`uv run pytest`).
