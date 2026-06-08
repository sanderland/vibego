# experiments/

The lab notebook for nanogo. The goal of the project is a tiny, hackable KataGo-style Go
engine + training pipeline (nanoGPT/nanochat spirit) that's easy to run ablations on.

## How this folder works

- **One dated markdown per experiment**, named `YYYY-MM-DD-short-slug.md`. Each is a
  self-contained record: the question, the setup (exact commands / configs / data), the
  results (tables, numbers with error bars), and a conclusion. Write it while you run it.
- **`SUMMARY.md`** — the digest. Key conclusions and a live "things to try / implement"
  list. Ineffective experiments collapse to a single line; promising/open ones get expanded.
  Append as you go; rewrite periodically so it stays short and current.
- **`INDEX.md`** (this file) — explains the approach and lists every experiment file with
  ~one line (setup + conclusion) for navigation.

Conventions: all strength numbers come from a neutral **b18 judge** (`kata1-b18c384nbt`,
256 visits). Game scores are points from Black's perspective unless stated. Prefer the
**arena** (`scripts/match.py`, mean scoreLead ± stderr + Elo ± CI) or the per-move tool
(`scripts/move_eval.py`) over single games — per-game sd is ~30–40 points.

## Experiments

| date | file | setup | conclusion |
|------|------|-------|------------|
| 2026-06-06 | [distillation-and-net-diagnosis](2026-06-06-distillation-and-net-diagnosis.md) | train depth-6 (~1M param) nets: supervised npz vs b18 logit-forcing; diagnose net vs b6c96 with `policy_eval` | distillation is ~70× more data-efficient; our net matches b18 at least as well as b6c96 (biased) → **the net is not the search bottleneck** |
| 2026-06-07 | [search-param-porting](2026-06-07-search-param-porting.md) | port KataGo search params one by one (score utility, FPU blend, leaf-batch cap, LCB, valueWeightExponent) against the b6c96 proxy | **cautionary**: a long −108→−57 "progression" that was mostly chasing a measurement bug (see next); the genuinely-correct pieces (score utility, FPU mean, LCB, batch cap) were kept |
| 2026-06-08 | [proxy-perspective-bug-and-parity](2026-06-08-proxy-perspective-bug-and-parity.md) | deterministic node-by-node trace (`trace_search.py`, batch=1, 10v, 9×9) to find where our search diverges from KataGo's | found a **perspective bug** in the proxy (Black-perspective evals fed as side-to-move); fixing it → **near parity (−8.7 ± 5.4, 35% win, Elo ≈ −104)**; reverted the recompute machinery (only masked the bug) |
