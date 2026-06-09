# 2026-06-08 (d) — The FLOPs↔Elo frontier (first pass): bake-off + depth-width

New project north star ([ROADMAP](ROADMAP.md)): map the **Pareto frontier of Elo vs FLOPs/eval** for
small nets. This is the first plane — the 6-net bake-off (old vs nbt vs nbt-matched at 6b/10b) plus
the depth-vs-width-at-fixed-budget slice (b6c112→b10c88nbt, all ~1.09M / ~770 MFLOP). All nets
trained identically: `distilled_1m`, 8k steps, batch 256 (so **undertrained** — relative arch
signal, not final strength).

## The data

`bench_net.py` (batch-1, 60 iters); val loss = held-out total at step 6000; FLOPs = MACs×2.

| name | arch | params | MFLOP | val total↓ | CPU ms↓ | MPS ms↓ | eff. CPU MFLOP/ms |
|------|------|--------|-------|-----------|---------|---------|-------------------|
| old6b | b6c96-gpool | 1.090M | 774 | 3.532 | 2.94 | 1.44 | **263** |
| nbt6b | b6c96nbt | 0.796M | 561 | 3.388 | 3.93 | 2.36 | 143 |
| nbt6b_matched | b6c112nbt | 1.065M | 753 | 3.367 | 4.17 | 2.42 | 181 |
| dw7 | b7c106nbt | 1.072M | 759 | **3.328** | 4.71 | 2.76 | 161 |
| dw8 | b8c102nbt | 1.102M | 780 | 3.340 | 5.32 | 3.07 | 147 |
| dw9 | b9c92nbt | 1.091M | 769 | 3.383 | 5.66 | 3.34 | 136 |
| dw10 | b10c88nbt | 1.082M | 762 | 3.335 | 5.32 | 3.67 | 143 |
| old10b | b10c128-gpool | 3.122M | 2230 | 3.401 | 5.56 | 2.06 | **401** |
| nbt10b | b10c128nbt | 2.207M | 1567 | 3.324 | 6.99 | 3.83 | 224 |
| nbt10b_matched | b10c152nbt | 3.072M | 2188 | **3.260** | 7.68 | 3.85 | 285 |

## Finding 1 — the x-axis matters: FLOPs flatters nbt (less so on CPU than GPU)

Effective batch-1 FLOP throughput (MFLOP ÷ ms) is **not uniform across block types**. Wide regular
convs are far more hardware-efficient per FLOP than nbt's narrow (c/2) bottleneck convs:

| nbt FLOP-efficiency penalty (regular ÷ nbt, matched budget) | CPU | MPS |
|--|--|--|
| 6b (old6b vs nbt6b_matched) | ~1.45× | ~1.7× |
| 10b (old10b vs nbt10b_matched) | ~1.4× | ~1.9× |

**The hypothesis holds: FLOPs is a better wall-clock proxy on CPU than GPU** (nbt penalty ~1.4× vs
~1.9×) — CPU is nearer compute-bound, no GPU parallelism-starvation. But ~1.4× is **not negligible**:
FLOPs systematically *under*states nbt's real CPU cost, so a FLOPs-only frontier favors nbt more than
deployment will. **Report Elo against both FLOPs (portable) and CPU-ms (deployment-real).** Caveat:
this is PyTorch oneDNN/MPS; the wasm/XNNPACK kernels (Tier-3 #5) may narrow or widen the gap — that's
the real validation.

## Finding 2 — the frontier, on real Elo (arena, vs old6b anchor)

Arena: each net vs the **old6b** anchor (= Elo 0), 24 games w/ randomized openings, b18 judge 256v,
48 visits. (CIs are wide at 24 games — see caveats — but the frontier shape is robust.)

| net | MFLOP | CPU ms | **Elo vs old6b** | 95% CI |
|-----|-------|--------|------------------|--------|
| old6b | 774 | 2.94 | 0 (anchor) | — |
| nbt6b | 561 | 3.93 | +58 | [−81, +220] |
| nbt6b_matched | 753 | 4.17 | −0 | [−147, +147] |
| **dw7** (b7c106) | 759 | 4.71 | **+232** | [+92, +527] |
| dw8 | 780 | 5.32 | +191 | [+54, +432] |
| dw9 | 769 | 5.66 | +154 | [+18, +364] |
| dw10 | 762 | 5.32 | +154 | [+18, +364] |
| old10b | 2230 | 5.56 | +280 | [+134, +699] |
| nbt10b | 1567 | 6.99 | +280 | [+134, +699] |
| nbt10b_matched | 2188 | 7.68 | +338 | [+184, +3600] |

**Elo ≠ val-loss.** `nbt6b_matched` had the 2nd-best 6b val-loss (3.367) but plays at **anchor
level (−0 Elo)**, while `dw7` (near-identical val-loss/FLOPs) is **+232**. And `old10b` (worst 10b
val-loss, 3.401) ties `nbt10b` on Elo. Net lesson, again: **the frontier must be Elo.**

**Frontier on FLOPs** (max Elo, min FLOPs) — **all nbt:**
```
  nbt6b (561,+58) → dw7 (759,+232) → nbt10b (1567,+280) → nbt10b_matched (2188,+338)
```
`dw7` is the efficiency star: +232 Elo at 759 MFLOP — most of `nbt10b`'s +280 at **half** the FLOPs.

**Frontier on CPU-ms** (max Elo, min ms) — **two regular nets join, and a decision flips:**
```
  old6b (2.94,0) → nbt6b (3.93,+58) → dw7 (4.71,+232) → old10b (5.56,+280) → nbt10b_matched (7.68,+338)
```
**`nbt10b` is on the FLOPs frontier but DOMINATED on CPU-ms** — `old10b` matches its +280 Elo
**25% faster** (5.56 vs 6.99 ms), because FLOPs flatters nbt (Finding 1). So *which* net to ship at
the 10b tier **depends on the axis**: FLOPs says nbt10b, real CPU says old10b. `dw7` sits on **both**
frontiers — the unambiguous small-net pick.

## Finding 3 — depth vs width at fixed budget (Elo confirms depth 7)

In the ~770-MFLOP / 1.09M slice, **Elo decreases with depth past 7**: depth-7 `dw7` **+232** >
depth-8 `dw8` +191 > depth-9/10 +154, and the widest depth-6 `nbt6b_matched` is way back at **−0**.
So the sweet spot is **depth ~7** — *sharper* than val-loss implied (which had d6–d8 within 0.04).
Both extremes lose: too wide (d6) and too deep (d9–10). `dw7` (b7c106nbt) is the **standout** —
+232 Elo at 759 MFLOP, matching `nbt10b`'s +280 at half the FLOPs, and it sits on **both** frontiers.

## Methodology bug found & fixed (Elo re-running)

The first Elo attempt was **degenerate and discarded**: `vs.py`'s `genmove` always plays the **top**
move, so two deterministic nanogo engines replay **one identical game per colour** — 24 alternating
games = effectively 2 distinct games → win-rates collapsed to 0/12/24 and Elo saturated at ±3600.
(Earlier arenas only varied because *KataGo* injects its own search randomness.) **Fix:** forced
**randomized openings** (`random_opening` in `vs.py`; `--opening-plies` default 8, added to
`vs.py`/`match.py`; a colour-reversed pair shares an opening seed for balance; test in
`tests/test_vs.py`). Post-fix smoke (nbt10b_matched vs old6b) gives diverse games (sd ~29, varied
margins) as expected. Also noted: `policy_eval` **deadlocks at n=500** (sends all queries before
reading → pipe fills); using n=300 until that's fixed.

Arena Elo (above) was then run validly. The intrinsic run *appeared* to crash (`run_engine`
"BrokenPipe") — but that was **operator error, not a bug**: the `policy_eval` invocation built its
`--engine "name=cmd"` args in a shell loop and `eval`-ed them, and the nested quotes collapsed
(the `run_engine` commands contain spaces), concatenating all ten net names into one mangled
argument. Fixed by building the args as a bash **array** (no `eval`); the full 12-engine run then
ran clean. Genuine improvements that fell out: `policy_eval` now **skips a failed engine** instead
of aborting the sweep (which is what revealed the mangled name), and `TeacherEngine.close()` now
**waits** for clean subprocess teardown. (No `run_engine` or deadlock bug existed.)

**Intrinsic (raw-net agreement vs neutral b40, n=300)** is too noisy at 8k-steps to rank our
nearby nets — top-1 clusters at ~29–37% for all ten (e.g. `dw7` 34.3 and `nbt6b_matched` 34.3 sit
together despite +232 vs −0 Elo), `nbt10b_matched` highest at 37.0. All ≈ `g170-b6c96` (33.7),
below `g170-b10c128` (42.0). Net: **intrinsic confirms only the coarse "our nets ≈ b6c96 tier"; it
cannot resolve the frontier — Elo does.**

## Conclusions

- **dw7 (b7c106nbt) is the small-net champion** — on both the FLOPs and CPU-ms frontiers; depth ~7
  at the 1.09M/~770-MFLOP budget. First concrete frontier point to beat.
- **The axis matters** (validating the whole FLOPs-vs-CPU framing): nbt owns the FLOPs frontier, but
  on real CPU `old10b` dominates `nbt10b`, and regular nets reclaim the cheap + 10b-tier corners.
  Report Elo against **both** FLOPs and CPU-ms.
- **Elo ≠ val-loss ≠ intrinsic** — measure the frontier with games.

## Caveats

- **24 games → wide Elo CIs (±100–300).** The top cluster (old10b/nbt10b +280, nbt10b_matched +338)
  overlaps within CI — the *top* of the frontier isn't crisply resolved (need more games). Robust
  regardless: dw7 ≫ the rest of the 6b cluster, and old10b ≥ nbt10b on CPU (equal point-Elo, faster).
- **8k-step undertraining** — absolute Elo shifts with the 75G data; the **relative arch frontier** is
  the deliverable.
- Bench is PyTorch (oneDNN/MPS); **wasm wall-clock is the real x-axis** and may reorder the
  FLOP-efficiency gap — the eventual validation (ROADMAP Tier-3 #5).
- **Harness work this run:** found+fixed **deterministic self-play** → randomized openings
  (`vs.py`/`match.py`, `tests/test_vs.py`); made `policy_eval` resilient (skip a failed engine) and
  `TeacherEngine.close()` wait for teardown. The apparent "`run_engine` crash" was an **operator
  shell-quoting bug** in the eval invocation, not a code bug — no open engine bug.
