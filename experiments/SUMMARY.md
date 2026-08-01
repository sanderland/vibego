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
- **Project goal (updated 06-10): aim toward g170-b15c192-tier strength at much lower FLOPs** —
  beating b6c96 was the first milestone (done: s4_dw7pat1200, +124 Elo at 759 MFLOP). Single-thread
  CPU-ms is the second cost axis (in-browser engine target). Report Elo vs **both**. Re-anchor
  Stage-B at g170 b10c128, then b15c192, as champions pass each tier (b6c96 anchor is saturating).
  Current frontier ([2026-06-09-d](2026-06-09-scale-elo-and-cpu-frontier.md)): b6c96nbt-pat (561 MF,
  Elo −124) → dw7/dw7pat (759 MF, −112) → **s1_b10pat (b10c128nbt-pat, 1567 MF, Elo +44 [−41,+134]
  — first positive point estimate vs the anchor, at −30% FLOPs vs its shape)**.
- **pattern_embed (3×3 canonical lookup table, ~0 FLOPs / ~2% CPU-ms) is a decisive Elo win at 6b
  at data scale** (−48.6 → −32.6 paired scoreLead) but **6b only** — no effect at dw7, and the
  b10 seed-1 "+8.6" did not replicate at seed 2. Default-on at 6b. At 48-shard screen scale it
  looked like a −669 disaster — **small-data screens can flip the sign of memory-heavy archs**.
- **Train-seed variance: val ±0.003–0.011 (tiny) but Elo ±~10 scoreLead at 64 games (large)** —
  single-seed Stage-B deltas under ~15 sL are noise; pool seeds or use SPRT-scale game counts
  for close calls. The b10 tier is at-or-above anchor parity **robustly across seeds** (pooled
  b10pat +3.4 sL / b10nbt −0.5 sL; b10nbt_sd2 Elo +100 [16,199], first positive CI).
- **19×19 filtering is a wash** (fixed-19×19 val Δ ≤ 0.015 with sign flips, Elo −26.2 vs −25.8
  despite 1.45× more views per position) — keep the 31% non-19×19 data.
- **ERA-MIX 30% is a confirmed Elo win (+13 sL, 2 seeds, ~2.6σ, free)**: blending ~30% stratified
  g170 b6/b10/b15-era data (b18-relabeled) into recent kata1 beats recent-only at fixed compute —
  despite a kata1-val penalty (in-domain val cannot judge data composition). Sweet spot ~30%
  (10% noise, 50% back to control). Tier transfer to b10pat under test; fold into the final
  exploit run. Explored-and-flat axes so far: depth (760 MF), loss weights, 5×5 pattern (at
  300sh), 19×19 filter.
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

- **The v1.17 KataGo transformer nets carry 4.2-27% SUBNORMAL weights, and that is the whole CPU
  story** (2026-08-01 phase 4). x86 runs subnormal operands in microcode at ~100x the cost, and
  KataGo sets flush-to-zero nowhere in `cpp/`. Zeroing them is numerically inert and lives in the
  weight file, so on the **unmodified** engine at batch 1: `b10c384h6nbttflrs` **5.4x faster**,
  `b10c512h8nbt3tflrs` **13.7x faster**, outputs **bit-identical**. Every conv net we have
  (b18c384nbt, g170-b6c96, g170e-b10c128) has exactly 0.0000% - this arrived with the transformer
  recipe. `scripts/kata_prune.py --flush-subnormal`. **Check our own nets before trusting any
  CPU-ms number.**
- **A benchmark result that cannot be right is a finding, not noise.** An 11% FLOP cut appearing to
  buy 3.7x wall-clock is what led to the above. Ruled out in order: power-of-two stride (512->511
  moves it 6%, not 4x), matmul shape (a bare GEMM scales linearly in the width), batching (per-eval
  cost flat from batch 1 to 8).
- **Whiten the SVD.** Data-aware low-rank (minimize ||C^(1/2)(M - M')||_F with C = E[x x^T] at the
  layer input, not ||M - M'||_F) beat plain SVD by **7x** in damage at equal FLOPs. Far bigger than
  the 1.4-2.8x that activation-awareness bought for magnitude pruning.
- **Activation-aware selection halves pruning damage; it still does not make post-hoc pruning pay**
  (2026-08-01 phase 2). At equal FLOPs, activation-weighted FFN selection cuts |Δ scoreLead| from
  1.96 → 1.25 at −11% FLOPs and 6.12 → 3.01 at −22%, vs the weight-only criterion — so weight-norm
  screening is a real floor, not the answer. But 1.25 points for 11% of the FLOPs is still far too
  expensive. Mechanism: the quietest trunk block writes a residual **23%** the size of the stream
  (LLM depth-drop wants a few %), head importance spread is **1.76×** median (no dead heads), and
  **no trunk channel is idle** (287 of 384 needed for 90% of variance).
- **Take per-channel statistics BEFORE the norm.** Measuring the trunk residual stream after the
  tip RMSNorm + SiLU claimed 51% of channels were idle; pre-norm the figure is 0%. The norm's
  per-channel gamma and SiLU's squashing of negatives were being reported as properties of the
  stream. Applies to any diagnostic on a normed architecture, ours included.
- **The v1.17 KataGo transformer nets are dense — post-hoc compression without healing does not
  pay** (2026-08-01). `b10c384h6nbttflrs`: heads exactly tile the bottleneck (6×32=192), SwiGLU is
  at the standard 8/3 ratio, all 10 blocks cost the same FLOPs. Cheapest structural edit (−4.4%
  FLOPs) already costs 0.58 scoreLead; heads are the worst lever (−7.2% FLOPs → −7.2 sL), FFN
  width the best (−11% → −1.8 sL). Quantization is dropped for a deployment reason, not an
  accuracy one: **no KataGo backend has int8 kernels**, so there is no speedup to weigh the loss
  against.
- **Policy agreement / KL is a bad screen for compression damage** — a block-drop scored top-1
  0.90 and the *lowest* KL of six variants while losing 5.1 scoreLead. Quantization/pruning error
  is a deterministic function of the position, i.e. bias, and search does not average bias away.
  Screen on value/score, not on policy.
- **`b10c384h6nbttflrs` is a better teacher than our pinned `kata1-b18c384nbt`** — stronger per
  visit at 10.6M params / 9.56 GFLOP vs 26.4M / 18.9 GFLOP. Same relabel throughput, better
  targets. Costs an engine bump to v1.17.x and a re-baseline.

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
1. **Make our own distilled net beat b6c96** — **DONE DECISIVELY 2026-06-10**: s4_dw7pat1200
   (b7c106nbt-pat, 1200sh/120k, Muon) = **scoreLead +14.4 [2.5,26.3] / Elo +124 [40,227]** vs
   the anchor, at 759 MFLOP / 11.2 CPU-ms (~⅓ the anchor-shape FLOPs). Scaling did it:
   −417 → −112 → −22 → +124 across four data/steps doublings (9.8M of 44M positions used —
   curve still not bent; next: 2400sh, then the full set needs user OK).
   **CHAMPION UPDATE (06-10 evening): s5_b10pat1200 (1567 MF, val 2.712) beats s4 h2h +31.4 sL
   / Elo +156** — the tier ordering FLIPS with data scale (dw7≥b10 at 300sh, b10≫dw7 at 1200sh).
   **s6 (06-11): era-mix polish (+60k continue) beats s5 by +18.6 sL / Elo +120 (h2h, decisive).**
   **FINAL (06-12): s9_b10pat_parity (23.5M pos incl. all weak-era data, 240k steps) ends the
   study at **−8.6 ± 2.4 sL [−13.3,−4.0] vs g170e-b10c128 (192 games pooled)**** — per-doubling gains decayed to
   ~5-8 sL; parity needs the full 44M pool or a capacity step. Champion lineage s4→s5→s6→s8→s9
   all in released/. Community writeup: experiments/WRITEUP.md.
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
- `vibego/katago/` + `scripts/kata_inspect.py` / `kata_prune.py` — open a released KataGo
  `.bin.gz` directly (v8–17, incl. the v1.17 transformers), report params/FLOPs per block, and
  structurally prune it back into a file the **stock engine loads** — so pruned nets go straight
  into `policy_eval` / `match` / `arena` with no new inference code. Byte-exact round-trip is the
  contract, verified on 9 real nets.
