# SUMMARY

Short digest of what we've learned and what to do next. See `INDEX.md` for navigation and the
dated files for full detail.

## Key conclusions

- **Our search is at near-parity with KataGo's own engine** (same b6c96 net, equal visits):
  −8.7 ± 5.4 pts, 35% win rate, Elo ≈ −104 [−221, −7] (48 games @ 48 visits, b18 judge). The
  search algorithm was essentially right all along.
- **The headline lesson: a measurement-harness bug masqueraded as a search-quality gap.** The
  b6c96 proxy fed KataGo's *Black-perspective* evals to our *side-to-move* search → every
  White-to-move node sign-flipped. It read as a −57 gap and survived a long tuning progression.
  A deterministic node-by-node trace (`trace_search.py`) caught it in minutes. **Always validate
  the harness with a node-level trace before trusting an aggregate tuning curve.**
- **The in-game gap to b6c96 is the NET, not the search.** So the path to a stronger engine is a
  stronger net.
- **Distillation (logit-forcing from b18) is ~70× more data-efficient** than supervised training
  on npz game outcomes, and gives much better-calibrated value/score/ownership. 0.10M b18-labeled
  positions ≈ 7.4M supervised positions against b6c96.
- **Our ~1M-param net matches b18 at least as well as b6c96 does** on policy/winrate/score/
  ownership (biased — we distilled from b18) → capacity is not the wall; data/steps/targets are.

## Search: things that worked vs didn't (all KataGo-faithful, re-validated post-bug-fix)

- **Kept** (correct & helpful): score in the PUCT utility (atan static+dynamic), FPU base =
  parent running-mean blended with raw eval (`fpuParentWeightByVisitedPolicy`), variance-based
  LCB for move selection, leaf-batch capped ∝ visits (~1/8), suicide-move policy filter (bugfix).
- **Reverted / inert** (one line, ineffective): `valueWeightExponent` + recursive value-recompute
  (only masked the bug; hurts on correct evals → reverted to flat-MC); `subtreeValueBias`
  (no-op — transpositions are 0.9% of a 48-visit tree); `cpuctUtilityStdevScale` (wash at 48v);
  graph search (ruled out by the 0.9% transposition measurement).

## Things to try / implement (priority order)

1. **Make our own distilled net beat b6c96** — the main open goal now that search is solved.
   The in-game gap is the net. Levers below feed this.
2. **More distillation data + steps.** b6c96 saw ~13× more samples (~6.5 epochs) than our best
   run. Scale the b18-relabeled set well past 1M positions; train longer. (Open — most promising.)
3. **Teacher ensembling** — average b18 + b28 + b40 soft policies (still 1 visit) as a
   higher-quality distillation target than any single teacher's searched policy. (Open.)
4. **nbt blocks** (nested bottleneck) — more strength per parameter; add a `b6c128nbt`-style arch
   to the registry and compare at equal param count. (Open.)
5. **Features 18/19 (pass-alive / territory)** — cheap to add (reuse the area flood-fill), the
   one meaningful omission from our 14-channel subset. Ablate its effect on strength/calibration.
6. **Use `scripts/match.py` for net-vs-net Elo** too (it already does command-engine arenas);
   `scripts/arena.py` does checkpoint round-robins with Bayesian Elo.

## Tooling built (reusable)

- `scripts/match.py` — concurrent multi-game arena: scoreLead ± stderr + win-rate + Elo ± CI.
- `scripts/move_eval.py` — low-noise per-move "points conceded vs a reference" (both colors).
- `scripts/trace_search.py` — deterministic batch=1 node-by-node search trace for harness
  validation / debugging selection.
- `scripts/policy_eval.py` — raw net agreement (policy/winrate/score/ownership) vs a reference.
- `nanogo/engine/proxy.py` — run our MCTS on an external KataGo net (search-isolation test).
