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
  an external net (search-correctness test). our-search+b6c96 vs KataGo's own search, same
  b6c96 net, 48 visits (b18 judge): our side lost **both colors (−44, −165)**. Single-game
  noise inflates the 165, but losing both with the identical net shows **our search is
  materially weaker per visit** than KataGo's.

**Conclusion: the in-game gap to b6c96 is dominated by SEARCH, not the net.**

**Search pass (proxy diagnostic, our-search+b6c96 vs KataGo-b6c96 own search, same net, 48 visits):**

| search | as B | as W | avg | note |
|--------|------|------|-----|------|
| win-rate-only Q (original) | −44 | −165 | ~−104 | wild color asymmetry = instability |
| + score in utility | −38 | −128 | ~−83 | helps, not enough |
| + cpuct-scaling + LCB selection + adaptive leaf-batch | −63 | −61 | ~−62 | deficit ~halved, asymmetry gone |
| + KataGo's real params/formulas (stolen, not swept) | −25 | −45 | **~−35** | **score utility = (2/π)atan(score/(scale·√area)) static+dynamic; variance-based LCB; fpu 0.2; cpuct 1.0/0.45/500** |

Final params are **stolen from KataGo** (`cpp/search/searchparams.cpp` + `searchhelpers.cpp`),
not swept: score utility = `staticFactor·(2/π)atan(score/(2·√area)) + dynamicFactor·(2/π)
atan(score/(0.75·√area))` (factors 0.1 / 0.3), variance-based LCB (`lcbStdevs=5`,
`minVisitPropForLCB=0.15`), `fpu=0.2`, cpuct `1.0 + 0.45·log((N+500)/500)`. This beat the
hand-tuned version (~62 → ~35). The deficit fell ~104 → ~35; our search is now within ~35 pts
of KataGo's own at 48 visits with the same net. Remaining gap (smaller): we use center=0 (no
running score center), raw atan (no score-stdev smoothing), and no tree reuse. Precise gains
still want a multi-game arena over single games.

## Matching KataGo's search (our search on b6c96 vs KataGo's own search, same net)

Goal: our MCTS driving KataGo's b6c96 net should match KataGo's own engine at equal visits.
Measured with `scripts/vs.py --games N` (b18 judge) and tuned with the low-noise per-move
tool `scripts/move_eval.py`. Proxy: `run_engine -proxy "<katago>"` (`nanogo/engine/proxy.py`).

Full-game variance is huge (per-game sd ~20–42), so the **primary instrument is
`move_eval.py`** (points conceded per move on fixed b40 positions, both colors), with full
games only for confirmation. The "game gap" column below is mean judge scoreLead from our
side (negative = we trail KataGo), 8–16 games @ 48 visits.

| change | move_eval (pts/move conceded) | game gap | note |
|--------|-------------------------------|----------|------|
| win-rate-only Q (start) | — | ~−108 (lucky low-var sample) | huge variance, blunder games |
| + score in utility + pass guard + variance-LCB | ~5.2 (n=24) | ~−83 ± 7 | the big algorithmic win |
| + FPU base = parent-avg blend (KataGo `fpuParentWeightByVisitedPolicy`) | ~3.7 (n=24) | **−55 ± 5** | over-optimistic raw-eval base was hurting |
| + leaf-batch capped ∝ visits (~1/8) | **~3.2 (n=40)** | **−54 ± 8** | synchronous virtual-loss batch was too "blind" at low visits |

All formulas stolen from KataGo (`searchparams.cpp` / `searchexplorehelpers.cpp` /
`searchupdatehelpers.cpp`). The KataGo we benchmark against runs the analysis-mode **parse
defaults** (the cfg comments them out): `subtreeValueBiasFactor=0.45`,
`fpuParentWeightByVisitedPolicy=true` (pow 2), `valueWeightExponent=0.25`,
`cpuctUtilityStdevScale=0.85`. We now match the FPU blend exactly. `cpuctUtilityStdevScale=0.85`
re-tested with move_eval (n=40): a **net wash** (Black 5.1→4.2, White 3.1→4.1) → kept off.
Also fixed a real bug: under tromp-taylor KataGo's policy includes suicide moves our board
rejects → search crash → pass fallback.

**Two mechanisms we deliberately did *not* port** (they need replacing our flat
Monte-Carlo backup with KataGo's recursive node-stats recompute — a large complexity add
against this project's minimal mandate, for uncertain gain at 48 visits):
- **subtreeValueBias** (0.45): corrects each leaf eval toward its subtree average, *shared across
  repeated local move-patterns*. But our flat-MC backup already propagates each child's full
  subtree average to the parent (W sums all leaf utilities up every path) — i.e. we already use
  subtree values at weight 1.0, vs KataGo's partial 0.45 — and the cross-node pattern-sharing
  needs transpositions that barely occur in a 48-visit tree.
- **valueWeightExponent** (0.25) / noise-pruning: re-weights children toward the better ones
  (z-score CDF) when recomputing a parent's value, instead of a visit-weighted mean.

**Findings:** deficit cut ~−83 → ~−54 this pass (the honest pre-pass number is −83, not the
earlier lucky −42 single-sample). Our search is internally **correct** (engine vs itself is
balanced, −2.8 ± 4.6 — no perspective bug). The gap doesn't shrink with visits (48 vs 128 ≈
same), so it's genuine per-visit search quality. The remaining gap to parity is concentrated
in KataGo's recursive value-recompute machinery above — a real architecture-vs-simplicity
fork. A proper Elo arena (sd ~40/game means 8 games can't resolve sub-20-pt changes) is the
prerequisite for tuning the last stretch.

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
