# 2026-06-09 — data distribution audit, statistics fixes, on-policy plan

Acting on an external review (the statistical/distribution twin of our perspective-bug lesson).

## Data audit — we were NOT filtering for our play setup

Sampled 49k positions from the 75 G b18 set:
- **31% is NOT 19×19** (17², 18², 16², 13², 9² …). The relabel pipeline reads each position's actual
  board size and relabels at that size — no filter — so the set inherits kata1's multi-size mix.
- **Komi heavily randomized** — only **6% at 7.5** (lots of negative/odd komi). Probably *fine*: the
  net has a komi input and KataGo randomizes komi on purpose; we feed 7.5 at inference. Don't filter komi.

**Action:** added `--only-19x19` (loader filter, `data.read_batches(only_full_board=)`). Open question
(cheap A/B, queued): does dropping the 31% non-19×19 help our 19×19 play, or does multi-size training
help even single-size strength (KataGo's claim)? Train dw7 all-data vs 19×19-only, **same step budget,
eval on a fixed 19×19 val set** (else val-loss isn't comparable), + Stage-B Elo.

## Statistics — make Stage-B verdicts resolvable (review #1)

24–48 game win-rate-Elo CIs (~±70–200) were too wide for the conclusions drawn on them ("dw7 champion"
is within noise). Fixes landed:
- **Paired scoreLead** is now the headline Stage-B discriminator (`match.py`): color-reversed pairs
  share an opening seed, so the pair-mean cancels opening+color variance → far tighter CI, and
  scoreLead resolved parity (−8.7±5.4) where win-rate Elo gave ±214. `stage_b.py` records it + a
  `decisive` flag and draws the real frontier from it.
- **Still TODO (queued):** round-robin Bayesian Elo (every game constrains every rating) + SPRT
  (stop-when-decided) via `arena.py`; one repeated-seed run to quantify seed variance vs the 0.04
  val-loss gaps we triage on.

## Triage honesty (review #2)

Downgrade the arch-screen "rejected" verdicts: val-loss is known not to rank Elo *across block types*
(the founding frontier experiment), and the recipe was tuned on the conv incumbent — so globmod/rwkv/
linat are **"no val win at 8k under a conv-tuned recipe; not Elo-tested"**, not "rejected". (Priors
still against the linear mixers per the N=361 analysis.)

## On-policy / distribution (review #6) — refined plan

The −150 net gap with raw-agreement≈b6c96 is the off-policy distribution-shift signature. We already
have 44 M distilled positions (> b6c96's 27 M), so **volume isn't the wall — distribution is**. Plan,
cheapest-first (capex vs opex framing):
1. **Free diagnostic gate:** net error on the student's OWN arena-game positions vs archive positions.
   If much worse on its own trajectories → off-policy confirmed → spend; else skip.
2. **If confirmed — download + relabel a g170 b6c96/b15-era slice** (capex: label once, train forever,
   valid for every net). **Stratify by era, not volume** (archive volume ∝ compute ∝ strong-end);
   equal-positions-per-era, overweight ~student tier, drop degenerate near-random earliest eras, keep a
   thin weak-but-sane tail for far-ahead/behind calibration, ~zero from strong eras (the 75 G already is
   that). Apply the 19×19/komi filter. Size ~5–15% of the pool. Mix ratio screenable ({0,10,30}% arms).
3. **True DAgger** (student self-play, temperature-sampled to avoid the deterministic-collapse trap,
   relabeled by b18) held as a **reserve one-shot polish for the champion** — per-generation opex, not
   in the screening loop.
Plus: a small **searched-target subset** (b18 at 16–64 visits on high-disagreement positions) as a
target-quality lever.
