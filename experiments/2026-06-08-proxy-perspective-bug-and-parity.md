# 2026-06-08 — Proxy perspective bug found by node-level trace → near parity

## Question

The aggregate proxy gap had stalled at ~−57 after porting every KataGo search param. Rather than
trust more aggregate tuning, **trace a tiny search node-by-node** and see exactly where our
selection/values diverge from KataGo's own at equal visits.

## Method

`scripts/trace_search.py` (new): deterministic run of our MCTS with **batch_size=1** (no virtual
loss, no concurrency → fully sequential & deterministic, matching KataGo single-threaded), on the
**same net** via the proxy. Per visit it prints the root PUCT table (N, V, Q, U, prior, chosen),
the selected path, the leaf eval, and the final root-child distribution. Compared move-for-move
against KataGo's own analysis (`config/analysis_1t.cfg`, 1 thread) at 10 visits on an **asymmetric**
9×9 position (B E5, W G3, B C4 → W to move) — asymmetric so KataGo's root **symmetry pruning**
(default on; it shares visits across symmetric moves) doesn't confound.

## The bug

Both engines explored the same top-2 moves but split visits differently (us 7/3, KataGo 5/4).
Checking the **raw evals** exposed why:

| position | KataGo raw (Black-persp) | our proxy returned | search interpreted as |
|----------|--------------------------|--------------------|-----------------------|
| root (W to move) | winrate_B 0.41, score_B −1.19 | 0.43 / −1.08 | side-to-move (W) value ← **wrong** |

Our proxy returned **0.43** — that's Black's value (0.41) — but the search treated it as the
**White-to-move** value. Root cause: the KataGo analysis config sets
`reportAnalysisWinratesAs = BLACK`, so KataGo reports winrate/scoreLead/ownership from **Black's**
perspective; the proxy passed that straight through as the side-to-move value. **Every
White-to-move node in the tree (half of it) was sign-flipped.**

## Fix

`nanogo/engine/proxy.py`: force `reportAnalysisWinratesAs=BLACK` in the query, then in `_parse`
convert to side-to-move (`persp = +1` if Black to move else `−1`; flip winrate→1−wr, score→−score,
ownership→−own). Verified by re-running the trace: our evals now match KataGo at **both** colors
(root W 0.578 ≈ KataGo's W 0.592; after-F6 B 0.509 ≈ KataGo's 0.512).

## Result (arena, our search on b6c96 vs KataGo's own engine, same net, b18 judge, 48v)

| | scoreLead | win rate | Elo |
|--|-----------|----------|-----|
| buggy proxy | −57 ± 4 | 0% | ≪ |
| **fixed proxy** | **−8.7 ± 5.4** | **35%** (17/48) | **−104 [−221, −7]** |

**Near parity.** The search algorithm was essentially right all along; the −57 was one sign flip
on half the tree. The bug also explained: the White-side deficit (the flipped half), "engine vs
itself is balanced" (both copies equally buggy → cancels), and the gap growing with visits
(compounding; post-fix it's a mild 16v −5 → 128v −15, win-rate stable ~25–29%).

Scope: only the proxy *diagnostic* was affected. The real local-net engine uses side-to-move
evals natively and was never buggy.

## Cleanup (re-validated on the fixed proxy)

- **Reverted** `valueWeightExponent` + its recursive value-recompute: on correct evals it
  **hurts** (arena −18.6 vs −10.6 at 0.0) — it had only masked the sign-flip. Engine back to
  clean flat-MC node values for selection.
- **`subtreeValueBias`**: confirmed no-op — measured **0.9%** of nodes in a 48-visit tree are
  transpositions (`/tmp` instrumentation), so there's nothing for its pattern table to share;
  this also **rules out graph search** as a lever at these visit counts.
- **Kept**: score utility, FPU parent-mean blend, variance-LCB, leaf-batch cap, suicide filter.

## Conclusions

- **Goal met: our search ≈ KataGo's own at equal visits** (−8.7 ± 5.4, 35% win, Elo ≈ −104).
- **The in-game gap to b6c96 is the NET, not the search.** → pivot to making the distilled net
  beat b6c96 (see SUMMARY "things to try").
- **Process lesson**: a node-level deterministic trace found in minutes what 200+ games of
  aggregate tuning hid. Validate the harness first.
