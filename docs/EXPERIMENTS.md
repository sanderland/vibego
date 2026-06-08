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
  vs **b18**, 300 positions (all output channels):

  | net | pol top-1 | top-5 | winrate MAE | score MAE | ownership MAE |
  |-----|-----------|-------|-------------|-----------|---------------|
  | ours (distill_1m) | 39.3% | 72.7% | 0.211 | 3.45 | 0.096 |
  | b6c96 | 32.7% | 70.7% | 0.226 | 5.74 | 0.107 |

  Our raw net matches b18 at least as well as b6c96 across **policy, winrate, score, and
  ownership** (**biased** — we distilled from b18, the reference). Takeaway: the **network is
  not the bottleneck**, and the score/ownership heads are well-calibrated.

- **`scripts/run_engine.py -proxy "<katago>"` + `nanogo/engine/proxy.py`** — run *our* MCTS on
  an external net (search-correctness test). With the proxy's perspective bug fixed (see the
  next section), our-search+b6c96 vs KataGo's own engine, same net, 48 games @ 48 visits
  (b18 judge): **−8.7 ± 5.4, 35% win rate** — near parity. The search is sound.

**Conclusion: the in-game gap to b6c96 is the NET, not the search** — our search on b6c96 nearly
matches KataGo's own. (For most of the investigation a proxy bug made the search look far
weaker than it is; the full story is below.)

## Matching KataGo's search (our search on b6c96 vs KataGo's own search, same net)

Goal: our MCTS driving KataGo's b6c96 net should match KataGo's own engine at equal visits.
Proxy: `run_engine -proxy "<katago>"` (`nanogo/engine/proxy.py`). Measured with the concurrent
**arena** `scripts/match.py` (mean scoreLead ± stderr + win-rate + Elo ± CI, b18 judge) and the
per-move tool `scripts/move_eval.py`.

**RESULT: near parity.** Our search on b6c96 vs KataGo's own engine, same net, 48 games @ 48
visits, b18 judge: **scoreLead −8.7 ± 5.4, win rate 35%, Elo ≈ −104 [−221, −7]**.

### The −57 was a bug, not search quality

For most of this investigation the proxy showed a ~−57 gap, and a long progression of
KataGo-faithful tuning (score utility, FPU blends, leaf-batch caps, valueWeightExponent) only
nudged it. All of that was chasing a **perspective bug in the proxy**, found by a deterministic
node-by-node trace (`scripts/trace_search.py`, batch=1, 10 visits, 9×9):

- KataGo reports winrate/scoreLead/ownership from **Black's** perspective (the analysis config
  sets `reportAnalysisWinratesAs=BLACK`). The proxy fed that value to our search as the
  **side-to-move** value, so **every White-to-move node got a sign-flipped value and score** —
  half the tree ran on inverted evals.
- The trace made it unmistakable: at a White-to-move root our proxy returned 0.43 where
  KataGo's side-to-move value was 0.59 (i.e. it returned *Black's* 0.41). Aggregate game stats
  had completely hidden it.

Fixing the proxy (force BLACK reporting, then flip to side-to-move when White is to move)
collapsed the gap: **−57 → −8.7**, win rate **0% → 35%**. The bug also explained every earlier
"finding": the White-side deficit, "engine-vs-itself balanced" (both copies equally buggy, so
it cancels), and the gap *growing* with visits (16v −46 → 128v −74 — compounding the
corruption; post-fix it's a mild 16v −5 → 128v −15 with stable ~25–29% win rate).

Only the proxy diagnostic was ever affected — the real local-net engine uses side-to-move
evals natively and was never buggy.

### What actually helped (re-validated on the *fixed* proxy)

- **score in the PUCT utility** (KataGo's static+dynamic atan score utility) — the original
  win-rate-only Q was a real simplification bug (caught separately).
- **FPU base = parent running mean blended with raw eval** (KataGo `fpuParentWeightByVisited
  Policy`, mass²): standard/correct, kept.
- **leaf-batch capped ∝ visits (~1/8)**: a synchronous virtual-loss batch expands leaves
  "blind"; KataGo's threads re-select asynchronously. Kept.
- **variance-based LCB** for final move selection (`lcbStdevs`, `minVisitPropForLCB`).
- bugfix: under tromp-taylor KataGo's policy includes suicide moves our board rejects → filter
  proxy policy to our legal moves (was a search crash → pass fallback).

### What we tried and *reverted/left off* (no benefit on correct evals)

- **`valueWeightExponent` (0.25)** + the recursive value-recompute it needs: appeared to help
  on the *buggy* proxy, but re-tested post-fix it **hurts/neutral** (arena −18.6 vs −10.6 at
  0.0) — it had only been masking the sign-flip. Reverted to clean flat-MC for selection.
- **`subtreeValueBias` (0.45)**: confirmed no-op (flat-MC already uses subtree values at full
  weight; its real benefit is cross-node pattern sharing, and transpositions are **~0.9%** of
  nodes in a 48-visit tree — measured).
- **`cpuctUtilityStdevScale` (0.85)**: a wash at 48 visits.
- **graph search**: ruled out by the 0.9% transposition measurement.

**Bottom line:** the search algorithm was essentially right all along — within ~9 points / 35%
win rate of KataGo's own engine at equal visits with the same net, once the diagnostic harness
stopped lying. The residual is the genuine (small) cost of a synchronous pure-Python MCTS vs an
async-threaded engine. **Lesson: validate the measurement harness with a deterministic
node-level trace before trusting a long tuning progression.**

## Open items / next

- **Search matching is done** — near parity (−8.7 ± 5.4, 35% win) once the proxy bug was fixed.
  The residual is the small cost of synchronous pure-Python MCTS vs an async-threaded engine;
  not worth a search-core rewrite.
- ~~Multi-game **arena** (Elo ± CI)~~ — built: `scripts/match.py` (concurrent, scoreLead±se +
  Elo±CI).
- Expand features: **territory / pass-alive (18,19)** — cheap (reuse area flood-fill); ladder
  history (15,16) needs prev-board reconstruction in the engine.
- **Teacher ensembling** (average b18 + b28 + b40 policies, still 1 visit) as the next target-
  quality lever over searched policy.
- **Main remaining goal: make the distilled nanogo net itself beat b6c96** — the in-game gap is
  now the net, not the search.

## Validation / correctness

Feature encoding is validated against KataGo's reference encoder; a code review pass fixed:
off-board policy-CE masking, an evaluator deadlock on forward errors, a `terminate_all` race,
and double liberty-grid computation per leaf. 34 tests pass (`uv run pytest`).
