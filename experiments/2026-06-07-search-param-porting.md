# 2026-06-07 — Porting KataGo search params against the b6c96 proxy (CAUTIONARY)

> **Read this together with [2026-06-08](2026-06-08-proxy-perspective-bug-and-parity.md).**
> Most of the "progression" below was chasing a measurement bug. It's kept as an honest record
> of how a broken harness produces a plausible, monotone-looking tuning curve that is mostly
> noise-compensation. The genuinely-correct changes survived re-validation; the rest were reverted.

## Question

Our MCTS driving KataGo's b6c96 net should match KataGo's own engine at equal visits. Where's
the gap, and which KataGo search params close it?

## Setup

- **Proxy** (`nanogo/engine/proxy.py`): our MCTS, but leaf evals come from an external KataGo
  engine running b6c96 at 1 visit (raw net). So search algorithm is isolated — identical net.
- **Measure**: `scripts/vs.py` / later `scripts/match.py` (b18 judge, 48 visits); per-move
  `scripts/move_eval.py`. Params stolen from `cpp/search/searchparams.cpp`,
  `searchexplorehelpers.cpp`, `searchupdatehelpers.cpp` (not swept).
- KataGo we benchmark against runs analysis-mode **parse defaults** (cfg comments them out):
  `subtreeValueBiasFactor=0.45`, `fpuParentWeightByVisitedPolicy=true` (pow 2),
  `valueWeightExponent=0.25`, `cpuctUtilityStdevScale=0.85`.

## The (confounded) progression — game gap, b18 judge

| change | game gap | status after bug-fix |
|--------|----------|----------------------|
| win-rate-only Q (start) | ~−108 | a real bug: Q must include score (caught separately) — **kept** |
| + score in utility + pass guard + variance-LCB | ~−83 | **kept** (correct) |
| + FPU base = parent-mean blend (`fpuParentWeightByVisitedPolicy`) | ~−55 | **kept** (correct/standard) |
| + leaf-batch capped ∝ visits (~1/8) | ~−63→−57 | **kept** (synchronous virtual-loss batch is "blind"; KataGo threads re-select async) |
| + `valueWeightExponent` 0.25 (recursive recompute) | ~−57 | **REVERTED** — only masked the bug |

Also re-tested and dropped at 48 visits: `cpuctUtilityStdevScale=0.85` (net wash — helped Black,
hurt White equally).

## What was real vs artifact

- **Real, kept**: score utility (`(2/π)·atan(score/(scale·√area))`, static factor 0.1 / dynamic
  0.3), variance-based LCB (`lcbStdevs`, `minVisitPropForLCB=0.15`), FPU base from the node's
  running mean blended with raw eval by `mass²`, leaf-batch cap, and a real bugfix: under
  tromp-taylor KataGo's policy includes suicide moves our board rejects → filter proxy policy to
  legal moves (was a crash → pass fallback).
- **Artifact**: the *magnitude* of every number here (−108…−57). The proxy was sign-flipping
  half the tree (next file). The "gap grows with visits" finding (16v −46 → 128v −74) was the
  bug compounding, not per-visit search quality.

## Lesson

A monotone-improving tuning curve is **not** evidence the harness is correct. Several of these
changes "helped" partly by dampening corrupted values. Validate the harness with a deterministic
node-level trace (see next) before trusting a progression like this.
