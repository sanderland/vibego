# 2026-06-09 — recipe + broad arch screen (Stage A, Muon)

Screening funnel, all on the 48-shard b18-distilled subset, 8k steps, val-loss (pre-filter; Elo is
the verdict). Recipe fixed first, then a broad arch sweep under it.

## Recipe (batch r0, dw7) — Muon lr 0.04 wins; it's the peak, not the edge

| recipe | val_loss |
|---|---|
| Muon **lr 0.04** | **3.122** |
| lr 0.06 / +warmup600 / +warmdown→0 | 3.128 / 3.123 / 3.124 |
| lr 0.09 / 0.13 | 3.147 / 3.156 |
| lr 0.02 / 0.01 / AdamW | 3.146 / 3.136 / 3.220 |

Extending the a0 sweep up resolved it: **0.04 is optimal**, higher LR monotonically worse; schedule
(warmup/warmdown) and EMA are within noise at 8k. lr 0.13 transiently blew up mid-warmup (caught by
a small smoke; grad-clip recovered). **Adopt Muon lr 0.04, default schedule.** Revisit schedule/EMA
only at Stage-C length.

## Broad arch (batch a2, Muon 0.04) — conv-nbt frontier holds; new ideas don't beat it on val

| arch | FLOP | val | train | train-val gap |
|---|---|---|---|---|
| b10c128nbt | 1567 | **3.067** | — | — (best val, 2× cost) |
| dw7 b7c106nbt | 759 | 3.122 | — | — |
| b6c96nbt | 561 | **3.173** | 3.127 | 0.046 (cheap champion) |
| b6c96nbt-pat | 561 | 3.165 | 3.019 | **0.146** |
| b6c96-rwkv | 614 | 3.210 | 3.141 | 0.069 |
| b6c96-gpool | 774 | 3.214 | 3.156 | 0.058 |
| b6c96-linat | 608 | 3.233 | 3.154 | 0.079 |
| b6c96-globmod | 761 | 3.241 | 3.157 | 0.084 |
| b6c96-gpool-pat | 774 | 3.232 | 3.077 | **0.155** |

**Triage:**
- **pattern_embed** (`-pat`): lowers *train* loss but **overfits** — train-val gap ~0.15 vs ~0.05 for
  baselines, no val gain (nbt) or worse (gpool). Its premise is *stored memory*, so the 48-shard
  subset is likely too small to fill the table usefully → **don't adopt yet; re-test at full data
  scale**, where it may pay off.
- **globmod / rwkv / linat**: all ≥ the gpool/nbt baseline val-loss; none beat plain **nbt**. The
  richer global connection (globmod FiLM) and the linear token-mixers gave **no val improvement** →
  **rejected** (consistent with the N=361 analysis: at 361 non-causal tokens the linear-mixer "cheaper
  attention" lever has no FLOP win, and here no quality win either).
- **conv-nbt remains the frontier.** Cheapest strong small net = `b6c96nbt` (561 MFLOP); `dw7` better
  at 759; scaling to `b10c128nbt` (1567) gives the best val but at 2× cost.

Next: Stage-B Elo on {b6c96nbt, dw7, b10c128nbt} (+ one `nbtpat` overfit check) vs the g170-b6c96
anchor — does Muon-trained conv-nbt close the net gap? Then the real lever is data/steps scaling (#3).
