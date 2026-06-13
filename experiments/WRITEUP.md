# Distilling tiny KataGo-style nets on one GPU: what actually moved Elo, and what didn't

A community writeup of the **vibego** study (repo: [sanderland/vibego](https://github.com/sanderland/vibego), `dev` branch). Everything below comes
from the dated lab notes in `experiments/`; exact numbers, commands, and per-run registry rows
(`experiments/registry.jsonl`) are in the repo.

## TL;DR

We built a single-GPU pipeline that trains small (0.8–2.6M param) KataGo-style nets purely from
**public data**: katagoarchive.org self-play positions, relabeled by the public `kata1-b18c384nbt`
net at **1 visit** (soft policy/winrate/score/ownership logit-forcing), plus our own
KataGo-faithful PUCT search (validated at near-parity with KataGo's engine on the same net:
−8.7 ± 5.4 judge points, Elo −104 [−221, −7]). No self-play RL loop anywhere.

Headline results, all at 48 visits per move, b18 judge at 256 visits, paired color-reversed
openings, scoreLead ("sL") = mean judge points:

- **`s4_dw7pat1200` (1.38M params, 759 MFLOP/eval, 11.2 single-thread CPU-ms) beats `g170-b6c96`
  decisively: +14.4 sL [2.5, 26.3], Elo +124 [40, 227]** — at ~⅓ the FLOPs of the anchor's
  architecture shape. The climb was four data+steps doublings: Elo −417 → −112 → −22 → +124.
- **The final champion `s11_b12c152_cap` (4.21M params, 2686 MFLOP, 30.8 CPU-ms) BEATS the next
  anchor up, `g170e-b10c128`: +7.7 ± 2.4 sL [+3.0, +12.5]** (pooled 192 games; both blocks
  decisive-positive). It got there by a **capacity step**: on the *same* full-pool data, the
  2.6M-param `s10` measured −9.5 ± 2.7 sL against that anchor and the 4.2M-param `s11` measured
  +7.7 — a **~17 sL swing from width alone** (h2h s11 vs s10: +21.0 ± 7.8, decisive). The b10
  tier is cleared; the cost is comparable-to-higher than the anchor (2686 vs 2230 MFLOP,
  30.8 vs 21.2 CPU-ms), so this is "stronger via capacity," not "cheaper and stronger" — the
  cheaper-and-stronger wins are s4 (vs b6c96, ⅓ FLOPs) and the s9/s10 parity-at-0.7×-FLOPs.
- Individual training runs are hours, not weeks: 120k steps ≈ 2.5h on one GPU at this size; the
  whole champion path (relabeling included) is on the order of three days of single-GPU time.

That part is unsurprising — distillation is known to work. The interesting findings are the ones
below: **what we expected to help and didn't, what helped and shouldn't have, and how often a
verdict flipped with scale.**

**The capacity-bound finding (s10 → s11), the study's strongest single result.** Doubling data
on the 2.6M-param arch (`s10`: full 44M kata1 pool + weak-era ×3, 320k steps) gave the best val
of the study (2.666) but **zero Elo gain** over `s9` (−9.5 vs −8.6 sL, h2h flat) — the arch had
saturated ~9 sL short of `g170e-b10c128` regardless of data. Holding that same data fixed and
stepping width to 4.2M params (`s11`) jumped to **+7.7 sL — past the anchor**. So the wall was
capacity, not data or steps, and (yet again) **val loss did not see it**: s10 and s11 have
near-identical val (2.666 vs 2.662) but ~17 sL of Elo between them. When a small distilled net
plateaus, add parameters, not data.

The parity-attempt run (`s9_b10pat_parity`: 23.5M positions = kata1-2400sh + the full 471-shard
weak-era pool at 16% mix, 240k steps) finished the study as champion: **−8.6 ± 2.4 sL [−13.3, −4.0] vs `g170e-b10c128`** (pooled over 192 games; the extra 128
fresh-opening games alone measured −7.0 ± 2.8, Elo −112 [−180, −52]) — the best anchor result — while
h2h vs s8 was +5.1 ± 6.4 (indistinguishable, full 64). The per-doubling gain has decayed to
~5-8 sL: closing the last ~13 sL to b10c128 parity within this 2.6M-param arch looks like
1-2 more data doublings (the full 44M-position pool) or a capacity step.

## The non-obvious findings

### 1. Data diversity beats data strength and recency, at fixed compute

Blending ~30% of **old, weak-era g170 self-play data** (b18-relabeled, like everything else) into
the recent kata1 training mix beats recent-data-only at identical step budget:

- **Era-mix 30%: −7.7 ± 3.4 sL vs control −21 ± 3.9 (pooled, 128 games, 2 seeds) → +13 sL at
  ~2.6σ**, for zero extra training compute. Replicated exactly across seeds (−7.7 ± 4.2 / −7.71 ± 5.3).
- **Dose-response is a sweet spot**: 10% ≈ noise-or-worse, 30% best, 50% back to control
  (−23.3 ± 5.3).
- **It transfers across tiers**: on the 2.6M-param net, +11.2 ± 4.7 sL [2.1, 20.4] vs its
  +1.7 ± 5.7 control — same size and direction as the small-tier effect.
- **The weak eras carry the entire effect** (era-pure arms, single seeds each): b6-era data alone
  −5.1 ± 4.8 *at only 17% dose* ≈ b15-era −7.1 ± 6.5 ≈ the full mix, while **b10-era data is
  inert** (−21.7 ± 6.2 = control). Position diversity is the active ingredient — not label
  strength (labels are b18's everywhere), not recency; mid-strength b10-era positions plausibly
  overlap most with what kata1 data already covers.
- **In-domain validation loss anti-correlates with the gain.** The mix arms are *punished* on
  kata1-domain val dose-dependently (mix30: −0.16 val), and the s6 polish gained ~+220
  anchor-Elo at essentially unchanged val (2.704 vs 2.712). If you judge data composition by
  held-out loss on your existing distribution, you will conclude the opposite of the truth.

### 2. Searched / amplified distillation targets are a trap

The obvious "better targets" idea — relabel with the teacher at 32 visits instead of 1 — fails in
two separate ways:

- **History-less searched targets are decisively harmful.** Relabeling archive positions from
  stones only (no move history) makes ko/capture context approximate. A 1-visit eval inherits that
  error statically; a 32-visit search *explores through it and compounds it into the root values*.
  Search-improved value/score/ownership targets (policy unchanged): **h2h −38.7 sL [−58.6, −18.7],
  24% win rate** vs visits-1 labels on the same 490k positions. Making the searched targets
  self-consistent (searched policy + value) recovers most of it but still loses: −12.7 ± 8.4.
  (Instrumentation gotcha en route: KataGo analysis "policy" is the raw prior regardless of
  `maxVisits` — verified by identical target entropies; true searched policies need the
  `moveInfos` visit counts.)
- **Even done right, searched targets only reach parity — at 32× the label cost.** A full-history
  replay-relabel pipeline (walk the actual game records, query the teacher with the real move
  prefix, correct side-to-move and history channels) fixes the mechanism: searched policy+value
  with history = **+2.8 ± 5.6 [−8.2, +13.8] vs the raw prior — statistical parity**, for 32
  teacher visits per label instead of 1.

So the full progression is −38.7 (history-less, value-searched) → −12.7 (history-less,
consistent) → ±0 (context-correct) — context confirmed as the failure mechanism, and **visits-1
soft-prior distillation confirmed as the per-FLOP optimum**. A dark-knowledge reading fits:
searched policies are much sharper (entropy 1.093 vs 1.665 measured), i.e. they discard exactly
the soft tail of the teacher's distribution that makes distillation data-efficient in the first
place.

### 3. Gumbel root search loses to PUCT at low visits — for distilled nets

We implemented Gumbel-AlphaZero root search (gumbel-top-k + sequential halving + completed-Q)
expecting the published low-visit gains. After debugging, it still loses on the champion net at
32 visits: **m=4 −41 ± 14 sL vs PUCT; a leaf-batch-1 probe −19 ± 18; a prior-dominant σ-scale
arm −149 (≈ raw-policy level)**.

The proposed mechanism is specific to distilled nets: visits-1 distillation produces an
**unusually strong policy prior next to a comparatively noisy value head**, and Gumbel's
value-trusting completed-Q argmax is exactly the wrong selection rule for that profile.
Published low-visit Gumbel results come from RL-loop nets where policy and value quality are in
balance. KataGo-tuned PUCT+LCB stays the default (Gumbel remains behind flags).

Implementation pitfall worth broadcasting: the *first* A/B looked catastrophic (−36/−79/−86 sL
for m=16/8/4) for a different reason — **blind forced batches under virtual loss poison root Q**.
A forced batch wider than a candidate's established subtree dives past the refutation into
passive opponent replies; we measured a 0.009-prior move go from raw −1.09 to Q +0.02 in 12
wide-batch visits. Fix: cap forced width by established visits (1, 1, 2, 4, 8, …). A
**Q-equivalence test** (assert the search's backed-up Q matches an independently recomputed
average of the same leaf evals, bit-for-bit) pins this class of bug and is cheap to keep in CI.

### 4. Verdicts flip with scale — repeatedly

The single most recurring phenomenon in the study: a small-data screening verdict reversing at
larger data/steps.

- **pattern_embed** (finding 6 below) screened at **Elo −669** on a 48-shard subset (textbook
  overfit: train-val gap 0.146 vs ~0.05 baseline) and became a **decisive win** at 300 shards
  (−48.6 → −32.6 paired sL, non-overlapping CIs). Memory-heavy architectures *cannot* be screened
  on small data.
- **Tier ordering flipped at 4× data**: at 300sh/30k the 1.38M b7c106 net beat the 2.57M b10c128
  net head-to-head (+13.7 sL [1.4, 26.0]); at 1200sh/120k the b10 tier wins decisively
  (+31.4 sL [14.3, 48.6], Elo +156).
- **Depth**: an 8k-step screen said depth stops paying at b7; at 300sh/30k val improves
  monotonically to b14–b16 — but Elo stays flat (FLOPs-matched h2h b14c74 vs b7c106:
  −5.3 ± 6.1 [−17.3, +6.7]), the third instance of val ≠ Elo across an axis (after block type
  and pattern).
- **Capacity-bound vs data-bound regimes**: the era-mix "resume polish" (finding 6) is worth
  **+120 Elo at 2.6M params and ~0 at 1.4M** (s7 ties s4: −3.1 ± 5.9, single seeds each). The
  1.4M b7 net gained from *neither* pattern memory *nor* era diversity, while the 0.8M 6b tier
  gained from pattern (cheap capacity) and the 2.6M b10 tier gained from diversity and polish
  (better data). Working hypothesis: **≤1.5M-param nets are capacity-bound at this data scale —
  better data can't help them; ~2.6M is data-bound — diversity pays.** Consistent across three
  interventions, but single-seed; treat as a hypothesis.

Practical consequence: screen-stage rankings are provisional, period. Anything that looks like
memory or data-composition has to be re-judged at the exploit scale.

### 5. Methodology that made any of this resolvable

Per-game score sd is ~30–40 points; naive win-rate Elo at affordable game counts cannot resolve
most of the effects above.

- **Paired color-reversed openings scored by judge scoreLead** is the workhorse: each pair shares
  a randomized opening seed, the pair-mean cancels opening+color variance. It resolved search
  parity at −8.7 ± 5.4 points where 48-game win-rate Elo gave a ±214 CI.
- **Train-seed variance in Elo is ±~10 sL at 64 games** despite val-loss seed noise of only
  ±0.003–0.011 (dw7 −25.8 → −16.2, b10 −6.9 → +5.8 across seeds). Rule adopted: single-seed
  Stage-B deltas under ~15 sL are not conclusions — pool seeds or scale the game count. (This is
  also how a "+8.6 sL" pattern effect at b10 died at seed 2.)
- **Sequential early stopping** on the paired statistic: decisive head-to-heads resolved at 19–30
  pairs that two separate vs-anchor measurements (two ±5 CIs plus ±10 seed noise) could not
  resolve at all. Direct h2h beats anchor-differencing.
- **CPU timing: min-of-iters, never mean.** Mean wall-clock was inflated up to ~2× by bursty
  contention (same net: 15.6 → 24.1 ms between runs on a shared box); min-of-50 reproduces to ±1%.
- **FLOPs diverge from wall-clock tier-dependently**: nested-bottleneck's −28% FLOPs is worth
  −2% single-thread CPU at the 6b tier (7.98 vs 8.13 ms) but a real −17% at 10b (17.65 vs 21.20);
  depth costs ~+0.5 CPU-ms/block at matched FLOPs. A FLOPs-only frontier picks different nets
  than a deployment-CPU frontier; report both.

Two harness war stories that cost real days and are easy to inherit: (1) a **perspective bug** —
KataGo analysis reports Black-perspective values (`reportAnalysisWinratesAs=BLACK`); feeding them
to a side-to-move search sign-flips half the tree, and it masqueraded as a plausible
monotonically-improving search-tuning curve for a whole day. A deterministic batch-1 node-by-node
trace found it in minutes. (2) **deterministic self-play** — two deterministic engines replay one
identical game per color; without randomized openings a 24-game arena is effectively 2 games.

### 6. Cheap wins worth stealing

- **3×3 dihedral pattern table**: map each cell's D4-canonical 3×3 {empty/own/opp} neighborhood
  through a learned embedding table, added to the trunk input. ~0 FLOPs, +0.17 ms CPU (~2%), and
  at the 6b tier at data scale it's worth **Elo −338 → −124** (paired sL −48.6 → −32.6) — roughly
  +200 Elo for free. Tier-dependent: no effect at b7, did not replicate at b10 (seed 2). Default-on
  at 6b.
- **Resume + diversity-data polish**: take the trained champion (120k steps on recent data),
  resume for +60k steps on the era-mix data with a cosine warm restart (~0.013 → 0.004) — ⅓ the
  cost of the 180k-step from-scratch retrain on the mixed data.
  Worth **+18.6 sL [5.9, 31.3] / Elo +120 h2h** over its own starting checkpoint, and ~+220
  anchor-Elo — matching what a from-scratch retrain on the mixed data achieved (s8 vs s6 h2h:
  +14.2 ± 8.1 [−1.7, +30.0], not decisive). Only at the data-bound tier (see finding 4); single
  seed.
- **Muon (lr 0.04) over AdamW**: −0.09 val-loss at fixed arch/FLOPs, architecture-independent,
  and it appears to shrink seed variance (whitened updates). The biggest single training-recipe
  lever we found, and the cheapest.

## The recipe

To reproduce the current champion (s8-class, ~2.6M params, decisively above `g170-b6c96`,
~20 sL short of `g170e-b10c128`):

1. **Data.** Download kata1 self-play `.npz` from katagoarchive.org (~1400 shards used) plus
   g170-era self-play archives, **stratified by era and weighted toward the weak eras** (b6- and
   b15-era; skip b10-era — measured inert). Keep all board sizes (filtering to 19×19 was a
   measured wash) and KataGo's randomized komi (the net has a komi input).
2. **Relabel everything with `kata1-b18c384nbt` at visits 1** (`scripts/relabel.py`): raw soft
   policy + winrate + scoreLead + ownership. Do not spend teacher visits on labels (finding 2).
   Distilled positions are ~2KB each.
3. **Mix ~23–25% era-diverse data into the recent data** (sweet spot measured at ~30% with
   unstratified era data; s8 used 23% weak-era-weighted — 409 era shards against 1400 kata1
   shards, 14.8M positions total).
4. **Architecture: `b10c128nbt-pat`** — 10-block nested-bottleneck trunk, c=128, every 3rd block
   gpool, plus the 3×3 pattern-embed table. 2.57M params, 1567 MFLOP/eval, 17.65 ms
   single-thread CPU. (For a 759-MFLOP budget: `b7c106nbt-pat`, the s4 recipe, 1200sh/120k —
   beats b6c96 at +124 Elo.)
5. **Train: Muon, lr 0.04, batch 256, 180k steps** (`scripts/train.py`). At this size that's a
   few hours on one modern GPU (120k steps ≈ 2.5h in our runs). Alternatively train on recent
   data only and apply the +60k resume-on-mix polish (finding 6) — same endpoint, ⅓ the
   mixed-data training cost.
6. **Evaluate with games, not val loss**: `scripts/match.py` vs a fixed anchor at 48 visits both
   sides, b18 judge at 256 visits, paired color-reversed random openings, paired scoreLead as the
   headline statistic with early stopping. Expect ±10 sL train-seed noise at 64 games.

(The final champion `s9_b10pat_parity` used kata1-2400sh + all 471 weak-era shards (16% mix)
for 240k steps — same recipe otherwise.)

## What this is NOT: a fair fight with the RL run

Comparing our nets to the g170 anchors on "training rows" or "samples" flatters distillation
on three axes that deserve to be stated plainly:
- **Per-row supervision richness.** We regress b18's full soft policy distribution plus its
  calibrated value/score/ownership estimates — near-zero-variance targets. g170's nets trained
  on their own ~600-visit search visit-counts and **game outcomes** (binary value) — far
  noisier per row. Much of distillation's apparent sample efficiency is just this.
- **The teacher already paid for the knowledge.** b18's strength is the product of years of
  distributed RL; our training compresses it, it doesn't rediscover it. Every claim here is
  conditional on a strong public teacher existing.
- **Era advantage.** Nested-bottleneck blocks, the pattern table, and Muon are all post-g170
  advances; part of our FLOPs edge over the 2020-era anchor shapes is simply that.

The transferable contributions are the recipe findings (diversity > recency, raw soft prior >
searched labels, where capacity binds, the measurement methodology) — not an efficiency
victory over self-play RL.

## Caveats

- **One search implementation, one visit budget.** All strength numbers are our PUCT+LCB search
  at 48 visits/move. The search was validated at near-parity with KataGo's own engine at equal
  visits on an identical net, so net comparisons aren't search-confounded — but rankings could
  shift at much higher visit counts or under other engines. The Gumbel result in particular is a
  claim about *distilled* nets at *low* visits, not about Gumbel in general.
- **One teacher.** Every net is distilled from `kata1-b18c384nbt`; all share its blind spots, and
  any teacher-relative metric is biased (we used the unrelated zhizi b40c768 net as a neutral
  reference for raw-agreement checks, and real KataGo nets as opponents).
- **Judging**: b18 at 256 visits scoring games **capped at 200 moves** — long-endgame skill is
  underweighted, and "scoreLead" deltas are judge-model estimates, not final scores.
- **Rules/setting**: 19×19, komi 7.5, Tromp-Taylor area scoring (Chinese-like), simple ko in our
  board implementation.
- **Small nets only** (0.8–2.6M params). The capacity-bound/data-bound boundary, the
  searched-target verdict, and the Gumbel verdict are all claims at this scale and may not
  transfer upward — and several findings here flipped with scale *within* the study.
- **Single-seed results** (flagged inline above, listed for clarity): the era-source ablation
  arms (b6/b10/b15-era), the s6/s7 polish pair and s8, the depth-ladder Elo arms, the
  searched-target A/Bs, the Gumbel A/Bs, and the loss-weight probe. Multi-seed/pooled: era-mix
  30% (2 seeds, 128 games), the b10-tier-at-parity result (pooled seeds), pattern-at-b10's
  *non*-replication, and the seed-variance numbers themselves.
- Win-rate Elo CIs at these game counts are wide; the paired scoreLead numbers are the
  load-bearing statistics throughout.

## Released artifacts

`released/` in the repo ([sanderland/vibego](https://github.com/sanderland/vibego), `dev` branch):

- `s8_b10pat_final.pt` — current champion (b10c128nbt-pat, 2.57M params; kata1+era-mix from
  scratch, 180k steps).
- `s6_b10pat_mixcont.pt` — the resume-polish champion (same arch; best measured vs
  `g170e-b10c128`: −17.4 ± 5.2 sL).
- `s5_b10pat1200_model.pt`, `s4_dw7pat1200.pt` — the 1567-MF and 759-MF tier champions
  (s4 is the b6c96-beater at ~⅓ FLOPs).
- `s2_dw7pat600_model.pt`, `s_dw7_model.pt`, `s_nbtpat_model.pt` — scaling-curve and 6b
  baselines.
- `registry.jsonl` — one row per run: arch, params, FLOPs/eval, CPU-ms, val losses, and every
  Stage-B arena result (games, anchor, scoreLead ± CI, Elo ± CI).
- `frontier.png` — the Elo-vs-FLOPs / Elo-vs-CPU-ms frontier plot.

The full lab notebook (dated experiment files, including the failures and the harness bugs) is
`experiments/` on the same branch; `DISTILL.md` documents the cloud relabeling pipeline.
