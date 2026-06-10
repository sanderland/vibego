# 2026-06-09 — data-scaling Elo (s0), pattern_embed pays at scale, real CPU-ms frontier

## Stage-B Elo at data scale (300 shards / 30k steps vs the 48-shard / 8k Stage-B1)

Same protocol both rows: 64 games (B1: 48) vs g170-b6c96 anchor @ 48 visits, b18 judge 256v,
paired scoreLead headline (color-reversed pairs share an opening seed).

| net | arch | MFLOP | B1 (48sh/8k) Elo | s0 (300sh/30k) scoreLead | s0 Elo [CI] |
|---|---|---|---|---|---|
| nbt | b6c96nbt | 561 | −374 | −48.6 ± 5.6 | −338 [−535,−234] |
| nbtpat | b6c96nbt-pat | 561 | −669 | **−32.6 ± 4.8** | −124 [−227,−40] |
| dw7 | b7c106nbt | 759 | −417 | **−25.8 ± 5.6** | −124 [−227,−40] |
| b10nbt | b10c128nbt | 1567 | −280 | **−6.9 ± 4.9** (not decisive) | **0 [−87,+87]** |

Findings:
- **MILESTONE: s_b10nbt reaches statistical parity with the real g170-b6c96 anchor** (paired
  scoreLead −6.9 ± 4.9, CI spans 0; Elo 0 [−87,+87]) — the first of our nets to close the
  "net gap" acceptance baseline (was −255 Elo for distill_1m, −280 for the 48-shard b10).
  It does so at 1567 MFLOP vs the anchor's ~2230 (b10c128-gpool shape) — i.e. **at parity
  while ~30% cheaper in FLOPs and ~17% in CPU-ms** (17.65 vs 21.20 ms).
- **Data+steps scaling converts directly into Elo** across the board (dw7: −417 → −124). The
  300 shards are ~2.5M of the 44M available positions (30k×256 ≈ 7.7M samples ≈ 3 epochs);
  the curve has not bent.
- **pattern_embed is a decisive Elo win at scale**: −48.6 → −32.6 paired scoreLead vs the
  identical-FLOPs base (non-overlapping CIs). The Stage-B1 −669 was a small-data overfit
  artifact — the A/B's verdict flips entirely at 300 shards. At ~0 FLOPs and ~2% CPU-ms,
  the 3×3 canonical-pattern table is a pure frontier shift. **Next: pattern on dw7 + b10 (s1).**
- dw7 ≈ nbtpat in win-rate Elo, dw7 better on scoreLead but it costs +35% FLOPs; both sit on
  the frontier. The combo (dw7+pat) should dominate.

## Real CPU-ms frontier (single-thread, the in-browser proxy)

`bench_net.py` fixes: (1) random gaussian inputs crashed PatternEmbed's table gather → bench now
feeds valid mutually-exclusive stone planes; (2) `--threads`; (3) **min-of-iters timing** — mean
was inflated up to ~2× by bursty contention from concurrent jobs and is not reproducible on a
shared box (b9c92: 15.6 → 24.1 between runs); min-of-50 repeats to ±1%. `bench_cpu4.log`:

| arch | MFLOP | b1 ms (1 thread) | note |
|---|---|---|---|
| b6c96-linat | 608 | **7.15** | fastest 6b, but no val win under conv recipe |
| b6c96-rwkv | 614 | 7.54 | |
| b6c96nbt | 561 | 7.98 | |
| b6c96-gpool | 774 | 8.13 | nbt's −28% FLOPs ≈ −2% CPU at 6b |
| b6c96nbt-pat | 561 | 8.15 | pattern ≈ free (+0.17 ms) |
| b6c112nbt | 753 | 10.00 | shallow-wide wins the dw study on CPU |
| b7c106nbt | 759 | 10.98 | |
| b9c92nbt | 769 | 11.44 | |
| b10c88nbt | 762 | 11.98 | |
| b8c102nbt | 780 | 12.47 | |
| b10c128nbt | 1567 | 17.65 | nbt's FLOP win is real at 10b (−17% CPU) |
| b10c128-gpool | 2230 | 21.20 | |

Findings:
- **FLOPs→CPU-ms is tier-dependent**: at 6b, nbt's FLOP saving disappears in sequential-layer
  overhead (7.98 vs 8.13 ms); at 10b it survives (17.65 vs 21.20). Keep reporting both axes.
- **Depth costs wall-clock at matched FLOPs**: the depth-width study is monotone-ish against
  depth on CPU (c112@b6 10.0 → c102@b8 12.5 ms at ~equal MFLOP), opposing the Elo ordering
  (dw7 > 6b). The CPU-ms frontier will prefer shallower-wider than the FLOPs frontier does.
- **Linear mixers are the fastest trunks per ms** — worth one Elo test someday under a tuned
  recipe, but priors still against at N=361 (no val win).
- Registry rows now carry `cpu_ms` (bench_cpu4 values); `pareto_plot.py --cpu` adds the
  CPU-ms panels.

## Off-policy diagnostic (review #6, gate for data capex) — gap confirmed, moderate

Chain: `match.py --save-games` (32 games s_dw7 vs anchor @48v) → `game_positions.py`
(stride 3 → 2048 positions, source-format npz) → `relabel.py` (b18, **visits 1, same as the
75G set**) → `eval_loss.py` (same net, same teacher both sides — isolates the distribution term).

Raw totals mislead: on-trajectory score-loss explodes (7.5 → 42.9) but that's mostly **target
scale** — KataGo's komi randomization keeps archive games balanced (|scoreLead| mean 3.3,
sd 6.6) while our match games are decided (mean 18.4, sd 31.5); value-loss is *lower*
on-policy (decided positions are easy winrate calls). The scale-free measure is the
**policy KL to the teacher** (CE − teacher entropy; teacher H = 1.71 archive / 1.60 onpolicy,
19×19 rows only on both sides):

| net | KL archive | KL own-trajectory | excess |
|---|---|---|---|
| s_dw7 | 0.74 | 0.98 | **+32%** |
| s_nbt | 0.78 | 1.02 | +31% |
| s_b10nbt | 0.67 | 0.90 | +33% |

**Verdict:** a real, consistent ~+32% off-policy policy-KL excess — but moderate, and the
data-scaling curve hasn't bent (2.5M of 44M positions used). **Decision: keep scaling first**
(s1/s2 running); revisit on-policy mixing (era-stratified g170 capex, or the DAgger one-shot
champion polish) when scaling bends. The +0.12 raw-CE gap also bundles komi-7.5 and
decided-position effects — by design: it measures "our play setup" vs the archive overall.

## 19×19-filter A/B (review queue item) — filtering is a WASH, keep all data

s1_dw7f19 (dw7, `--only-19x19`, 300sh/30k — sees each 19×19 position ~1.45× more often than
the control at equal steps) vs s_dw7 (all data), both evaluated on the SAME fixed 19×19 val
(scale shards 0–1 via `eval_loss --only-19x19`): total 3.040/3.069 vs 3.025/3.077 — sign flips
between shards, |Δ| ≤ 0.015, policy ~+0.005 for all-data. **Dropping the 31% non-19×19 data
does not help 19×19 val** — consistent with KataGo's multi-size-helps claim. Keep all data;
`--only-19x19` stays available for inference-time experiments only.

(Ops note: screen.py crashed appending the first `axis="data"` row — registry VALID_AXES now
includes "data"; the s1 trainings survived as orphans and s1_dw7f19's row was appended manually.)

## s1 results (Stage-B, 64 games each, same protocol as s0)

| net | arch | MFLOP | val | paired scoreLead | Elo | vs base |
|---|---|---|---|---|---|---|
| s1_dw7pat | b7c106nbt-pat | 759 | 2.952 | −27.0 ± 6.2 | −112 | s_dw7 −25.8 → **no effect** |
| s1_b10pat | b10c128nbt-pat | 1567 | 2.866 | **+1.7 ± 5.7** (nd) | **+44** [−41,+134] | s_b10nbt −6.9 → +8.6 ± 7.5 (~1.1σ) |
| s1_dw7f19 | b7c106nbt (19×19-only) | 759 | (3.043*) | −26.2 ± 4.8 | −112 | s_dw7 −25.8 → **wash in Elo too** |

\*filtered val, not comparable (see A/B section — the fixed-val comparison was a wash).

- **First positive point estimate vs the anchor**: s1_b10pat +1.7 scoreLead / Elo +44 (CI spans
  0 — at-or-above parity, not decisively ahead). New b10-tier champion by point estimate.
- **pattern_embed is tier-dependent, not universal**: 6b **+16 sL (significant)**, dw7 **−1
  (nothing)**, b10 +8.6 ± 7.5 (suggestive). Not a clean "saturates with depth" story (b10 is
  deeper than dw7). Resolving the dw7/b10 deltas properly needs seed replicates or bigger game
  counts (the queued round-robin/SPRT work). Either way pattern never *hurts* and is ~free →
  default-on for 6b, optional elsewhere.
- **19×19 filter confirmed a wash in Elo** (−26.2 vs −25.8), matching the fixed-val result.
  Keep all data. Closed.

## s1/s2 launched (20:32)

s1 = s1_dw7pat, s1_b10pat, s1_dw7f19 (300sh/30k, concurrency 3); s2 = s2_dw7pat600
(600sh/60k, `scale600/` symlink subset — val shards 0–3 identical to `scale/`). 6h check-in
cron armed: Stage-B on completion, registry/notes/plot updates, user report.

**s2 val landed (23:26, 5.8 it/s — 60k steps in 2.9h):** dw7pat 2.952 (300sh/30k) →
**2.879 (600sh/60k)**, same val shards — **scaling still not bent**, and a 759-MFLOP net now
matches the 1567-MFLOP s_b10nbt's val (2.878). s3 seed-variance batch (dw7/b10nbt/b10pat at
seed 2, 300sh/30k) training concurrently.

**s3 seed-variance (300sh/30k, seed 1 vs 2):** dw7 2.957/2.954, b10nbt 2.878/2.867, b10pat
2.866/2.862 — **seed noise on val ≈ 0.003–0.011**, so val deltas under ~0.015 are noise-level.
Pattern's val edge at b10 (0.012 / 0.005) is marginal but sign-consistent across seeds; the
seed-2 Stage-B (running) is the Elo replicate that decides it. Muon presumably contributes to
the low seed variance (whitened updates); nice property for screening either way.

**s4 val landed (06-10 ~03:15, 120k steps in 2.5h solo):** dw7pat 2.952 (300sh/30k) → 2.879
(600sh/60k) → **2.819 (1200sh/120k)** — per-doubling −0.073/−0.060, mild deceleration only,
and the 759-MFLOP net now has the **best val in the study** (b10pat 300sh: 2.862). Stage-B +
head-to-head vs s1_b10pat next; data used so far ≈ 9.8M of 44M positions.

**s4 Stage-B — THE STUDY GOAL, DECISIVELY: s4_dw7pat1200 beats the real g170-b6c96 anchor**
with paired scoreLead **+14.4 ± 6.1 [2.5, 26.3] (decisive)** and **Elo +124 [40, 227]** (both
CIs positive) at **759 MFLOP / 11.2 single-thread CPU-ms** — ~⅓ the FLOPs of the anchor's
arch shape. dw7pat Elo across data doublings: −417 → −112 → −22 → **+124**; val
3.04 → 2.952 → 2.879 → 2.819. The b6c96-class strength now costs ~b7c106nbt-pat inference.
(Seed-noise caveat: ±10 sL; even at the unlucky end the verdict stays positive.)

**Head-to-head (match.py direct, b18 judge, --early-stop's first live run):** s4_dw7pat1200
beats s1_b10pat **+13.7 [1.4, 26.0] paired scoreLead (decisive, 32 pairs)** — the 759 MF champion
dominates the 1567 MF tier outright (cheaper AND stronger head-on). Caveat: b10pat had 4× less
training (300sh/30k); s5_b10pat1200 gives the matched-scale rematch. Early-stop triggered at 30
pairs — direct paired h2h resolved at 64 games what two vs-anchor numbers (±10 sL seed noise +
two ±5 CIs) could not.

## Explore round e0 — the depth axis at data scale is CLOSED (val pays, Elo doesn't)

Pivot per user (06-10): "data doubling is exploit/final; explore is thin." First explore axis:
the ~750-MF FLOPs/param-matched depth ladder at 300sh/30k (new archs b12c78/b14c74/b16c68nbt).

| net | MFLOP | CPU ms | val | Stage-B sL / Elo (vs b6c96) |
|---|---|---|---|---|
| b7c106nbt (ref, 2 seeds) | 759 | 10.98 | 2.957/2.954 | −25.8 / −16.2 |
| e0_b9c92nbt | 769 | 11.44 | 2.979 | — |
| e0_b10c88nbt | 762 | 11.98 | 2.959 | — |
| e0_b12c78nbt | 745 | 12.24 | 2.944 | −24.2 / −203 |
| e0_b14c74nbt | 754 | 13.15 | **2.940** | −28.3 / −114 |
| e0_b16c68nbt | 749 | 14.08 | **2.940** | −16.7 / −54 |

- **Val improves monotonically with depth past b10, plateauing at b14–b16 (−0.017 vs b7)** —
  the 8k screen's "depth stops at b7" was wrong at scale (again: small-data screens mislead).
- **But the Elo is FLAT**: all three deep nets sit inside the dw7 seed band, and the decisive
  test — FLOPs-matched h2h **b14c74 vs s_dw7: −5.3 ± 6.1 [−17.3, +6.7], 64 games, no diff**
  (point estimate *against* depth). Third instance of **val≠Elo across an axis** (after
  block-type and pattern). Depth also costs **+0.5 CPU-ms/block** (b16 = +28% vs b7).
- **Verdict: b7 stays the ~760-MF pick on both cost axes; depth axis closed at this scale.**
  (Caveat: at much larger data the val trend hints deep nets could eventually convert — recheck
  only at the final exploit scale, not in screening.)

## Explore rounds e1/e2 — val reads (Elo pending; totals decoded per-head since loss weights differ)

**e1 (pattern dig + loss weights, 300sh/30k, refs s_dw7 2.957/pol 2.404, s_nbtpat pol 2.423):**
- `e1_pat5` (5×5 hashed table, 65k buckets): pol 2.451 ≈ no-pattern, worse than the 3×3.
  **No win at 300sh; parked** (recheck only at exploit scale — bigger memory is plausibly more
  data-hungry, the exact trap the 3×3 taught us, but it doesn't earn screen budget now).
- `e1_wpol2` (policy weight 2.0): pol **2.380** (−0.024 vs ref, value/own unchanged) — small
  real policy win; **Elo check queued** (policy is the Elo correlate).
- `e1_wown0` (no ownership aux): pol 2.406 ≈ ref — the ownership aux is val-neutral at this
  scale (KataGo's claim that aux helps isn't visible here; keep aux on, it's free).

**e2 (era mixing, kata1 300sh + stratified g170 b6/b10/b15-era data, same 30k steps):**
- `e2_mix10` (+33 g170sh): val 2.983 (−0.026 vs control); `e2_mix30` (+128): 3.119 (−0.16).
- **kata1-val punishes mixing dose-dependently — but val is kata1-domain-only**, so this is
  partly mechanical distribution shift, and this study's core lesson is val≠Elo. **Stage-B
  running** (sequential chain after the s5 matches; the burst of concurrent engine spawns
  earlier OOM-killed a match — sequence match jobs from now on).
- g170mix relabel capex: 352 shards / 2.9M positions / 13 GB at `distilled-g170mix/`.

## s5 + era-mix Elo (06-10 evening) — NEW CHAMPION, tier order flips with scale, mix30 signal

- **s5_b10pat1200 (b10c128nbt-pat, 1200sh/120k, val 2.712) beats s4 head-to-head:
  +31.4 sL [14.3, 48.6], 71% wr, Elo +156** (early-stop, 19 pairs). New overall champion.
  **The tier ordering flips with data scale** — at 300sh dw7-tier ≥ b10-tier; at 1200sh/120k
  the b10 tier wins decisively. Scale-dependent arch ranking is now an established phenomenon
  in this study (pattern, depth-val, now tier) — screen verdicts are provisional, period.
- Distance to goal: s5 vs g170e-b10c128 anchor = **−39.8 sL / Elo −291** (s4: −49.1/−372).
  FLOPs frontier: s4 holds 759 MF; s5 extends the strong end at 1567 MF / 17.75 CPU-ms.
- **ERA-MIX 30% CONFIRMED — the first data-composition Elo win.** e3_mix30sd2 = −7.71 ± 5.3,
  replicating e2_mix30's −7.7 ± 4.2 exactly; pooled (128 games, 2 seeds) **−7.7 ± 3.4 vs
  control −21 ± 3.9 → +13 sL at ~2.6σ**, at zero extra compute, *despite* a kata1-val penalty
  (val is domain-biased — never judge data composition by in-domain val). Dose-response is a
  sweet spot: 10% ≈ noise/worse, **30% best**, 50% back to control (−23.3 ± 5.3). Mix-arm
  kata1-val is also ~10× less seed-stable (3.119 vs 3.001 across seeds) — heterogeneous-data
  exposure order matters. e4 running: does mix30 transfer to the b10pat champion tier?
- e1_wpol2 Elo: **−18.4 ± 5.2 sL — inside the control band** (dw7 seeds −25.8/−16.2). The
  val-policy gain doesn't convert; **loss-weight axis closed** at single-seed resolution.

## Re-anchoring (06-10, goal updated: aim toward g170-b15c192-tier strength at low FLOPs)

Downloaded the **final** g170e anchors (`g170e-b10c128-s1141M`, `g170e-b15c192-s1672M` →
`models/g170-b10c128.bin.gz` / `g170-b15c192.bin.gz`). Champion distance to the next tier:
**s4_dw7pat1200 vs g170e-b10c128 = −49.1 sL [−60.9,−37.3], Elo −372** (early-stopped at 19
pairs, 26 games saved). So the ladder reads: +124 over b6c96, ~−370 to b10c128, b15 above
that. The b6c96→now climb cost 4 data doublings (~540 Elo); 2 more doublings remain before
the full-set OK gate — the quality levers (searched targets, teacher ensemble, era mix,
on-policy polish) will have to carry the rest. `stage_b.py` now takes `--anchor` (registry
rows + frontier are filtered per-anchor) and `--early-stop`.

**s3 seed-2 Stage-B (the Elo replicate):** dw7_sd2 −16.2±5.3 (Elo 0), b10nbt_sd2 **+5.8±6.0
(Elo +100 [16,199])**, b10pat_sd2 +5.0±5.7 (Elo +66). Three lessons:
1. **Train-seed variance in Elo ≈ ±10 scoreLead** (dw7 −25.8→−16.2, b10nbt −6.9→+5.8 across
   seeds) — as large as the 64-game CI. Single-seed single-run Stage-B differences under
   ~15 sL are NOT conclusions. (Val seed noise is 30× smaller — val is the stable metric,
   Elo the noisy-but-real one.) Multi-seed pooling or SPRT-style game counts needed for
   close calls; bumps the queued round-robin/Bayesian item.
2. **Pattern at b10 does NOT replicate** (seed-2 Δ = −0.8 sL vs seed-1's +8.6). Final verdict:
   pattern_embed is decisive at 6b, absent at dw7/b10. Default-on at 6b only.
3. **The b10 tier is at-or-above the anchor robustly**: pooled over seeds (128 games/config),
   b10nbt ≈ −0.5 sL, b10pat ≈ +3.4 sL; b10nbt_sd2 posts the study's first positive-CI Elo.

**s2 Stage-B: Elo −22 [−110,+64], paired scoreLead −10.5 ± 4.7** (decisive on points, near-
parity in Elo) at 759 MFLOP / 11.2 CPU-ms. The dw7pat data-scaling Elo curve: **−417 (48sh/8k)
→ −112 (300sh/30k) → −22 (600sh/60k)** — each doubling keeps paying; the frontier is now
nbtpat(561, −124) → dw7pat600(759, −22) → b10pat(1567, +44). Next scaling point (1200sh/120k,
~6h) is the obvious spend once s3 frees the GPU; per-position epoch count stays ~3 so the new
data is genuinely new.

## Next (s1 batch, queued)

`screen.py --batch s1 --data /workspace/distill/scale --steps 30000`: s1_dw7pat
(b7c106nbt-pat), s1_b10pat (b10c128nbt-pat), s1_dw7f19 (dw7 + `--only-19x19` — the filter A/B
arm; control = s_dw7, both evaluated post-hoc on a fixed 19×19 val via `eval_loss --only-19x19`).
