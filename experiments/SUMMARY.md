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
- **The in-game gap to b6c96 is the NET, not the search.** Quantified: distill_1m vs b6c96 arena
  = **−33.4 ± 6.2, 18.8% win, Elo −255** — holding search ≈ constant (parity −104), the net costs
  ~150 Elo in games. This is the **acceptance baseline** the 75G distill net must beat.
- **policy_eval (raw-net agreement) does NOT predict game strength.** Against a *neutral* judge
  (zhizi/b40, fixing the old b18-teacher bias) our net's raw outputs ≈ b6c96 — yet it loses ~150
  Elo in games. Judge net quality by the **arena**, not policy_eval.
- **Project goal: the FLOPs↔Elo Pareto frontier of small nets in the b6–b10 range** ([ROADMAP](ROADMAP.md)),
  with single-thread CPU-ms as the second cost axis (in-browser engine target). Report Elo vs **both**.
  Current frontier ([2026-06-09-d](2026-06-09-scale-elo-and-cpu-frontier.md)): **dw7 (b7c106nbt) and
  b6c96nbt-pat tie at Elo −124** vs the b6c96 anchor (300sh/30k, paired scoreLead −25.8 / −32.6),
  pattern cheaper by 35% FLOPs; **s1 combos (dw7+pat, b10+pat) queued**.
- **pattern_embed (3×3 canonical lookup table) is a decisive Elo win at data scale** at ~0 FLOPs /
  ~2% CPU-ms (−48.6 → −32.6 paired scoreLead vs identical-FLOPs base). At 48-shard screen scale it
  looked like a −669 disaster — **small-data screens can flip the sign of memory-heavy archs**.
- **Data+steps scaling converts directly into Elo, no bend yet** (dw7 −417 → −124 going 48sh/8k →
  300sh/30k; only 2.5M of 44M positions used). Keep scaling before spending on new data (but see
  the off-policy gate in [2026-06-09-c](2026-06-09-data-distribution-and-stats.md)).
- **Muon (lr 0.04, the peak) beats AdamW by ~0.09 val-loss** arch-independently → screening recipe.
- **FLOPs→CPU-ms is tier-dependent** (clean min-of-iters bench, 1 thread): nbt's −28% FLOPs ≈ −2%
  CPU at 6b but a real −17% at 10b; depth costs wall-clock at matched FLOPs (the CPU frontier
  prefers shallower-wider than the FLOPs frontier); linat/rwkv are the fastest 6b trunks per ms
  (no val win though). Timing rule: **min-of-iters, never mean** — contention inflates mean ~2×.
- **Harness lesson from actually running the arena:** the real bug was **deterministic self-play**
  (genmove always plays the top move → identical games per colour) → fixed with **randomized
  openings** in `vs.py`/`match.py` (`tests/test_vs.py`). A separate "engine crash" during intrinsic
  turned out to be **operator error** (shell-quoting in the `policy_eval` invocation — built
  `--engine` args in a loop + `eval`, collapsing quotes), not a code bug; lesson: **inspect the
  actual failing command before theorizing** (chased deadlock/GPU/orphan ghosts first). Genuine
  fixes that fell out: `policy_eval` skips a failed engine; `TeacherEngine.close()` waits for teardown.
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

> **The canonical prioritized backlog is now [ROADMAP.md](ROADMAP.md)** (FLOPs↔Elo frontier goal,
> Parameter-Golf borrows, transformer/Muon/weight-tying, etc.). The list below is the older
> net-quality thread, still valid.
>
1. **Make our own distilled net beat b6c96** — ~~the main open goal~~ **parity reached 2026-06-09**:
   s_b10nbt (300sh/30k, Muon) = Elo 0 [−87,+87] vs the anchor at −30% FLOPs. Open: *beat* it
   decisively, and reach parity in the ≤800 MFLOP tier (best: −124, s1 combos running).
2. **More distillation data + steps.** b6c96 saw ~13× more samples (~6.5 epochs) than our best
   run. Scale the b18-relabeled set well past 1M positions; train longer. (Open — most promising.)
3. **Teacher ensembling** — average b18 + b28 + b40 soft policies (still 1 visit) as a
   higher-quality distillation target than any single teacher's searched policy. (Open.)
4. **nbt blocks** (nested bottleneck) — **DONE: `NBTResBlock` + a 6b/10b/15b ladder added to the
   registry** (`b6c96nbt` 0.80M, `b10c128nbt` 2.21M, `b15c192nbt` 7.41M; old `b15c192-gpool` 10.35M
   added too). Ready to train and compare at equal param count once the 75G data lands. Sequence
   *after* the data-scale result so arch doesn't confound it. **`b6c96nbt` is the in-browser/wasm
   pick** (0.56 GFLOP/eval, ~0.8 MB int8).
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
  Use the **neutral** zhizi/b40 as `--ref` (not b18, our teacher). NB: raw agreement ≠ game strength.
- `vibego/engine/proxy.py` — run our MCTS on an external KataGo net (search-isolation test).
