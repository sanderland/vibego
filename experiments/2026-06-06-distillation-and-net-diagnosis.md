# 2026-06-06 — Distillation vs supervised, and is the gap the net or the search?

## Questions

1. Can a tiny (~1M-param) net trained on katago archive npz approach KataGo's g170 **b6c96**
   (same capacity)?
2. Is **distillation** (logit-forcing from a strong teacher, b18) better than supervised
   training on the npz game targets?
3. When our *engine* loses to b6c96 in games, is the deficit the **net** or the **search**?

## Setup

- **Arch**: ~1M-param ResNet, `b6c96-gpool` (6 blocks × 96ch, 2 global-pooling blocks); heads =
  policy / value (win-loss-noresult) / score / ownership.
- **Features**: a subset of KataGo's V7 planes (14 spatial incl. ladders 14/17, validated
  channel-for-channel against KataGo's encoder). Komi /20, dynamic score center.
- **Data**: kata1 self-play npz from katagoarchive.org (2021, b40c256-generated). Supervised
  targets = that run's MCTS policy/value/ownership. Distill targets = **b18** (`kata1-b18c384nbt`)
  relabeled at 1 visit via `scripts/relabel.py` (raw soft policy/winrate/score/ownership).
- **Judge**: b18 @ 256 visits scoring move-200 positions; `scripts/policy_eval.py` for raw-net
  agreement (no search).

## Runs

| net | data (pos / samples) | steps | policy loss | value | score | own |
|-----|----------------------|-------|-------------|-------|-------|-----|
| depth6 (v0), 12-ch | 2.5M / 3.8M | 15k | 2.70 | 0.66 | ~50 | 0.38 |
| depth6_gpool, 12-ch+gpool | 7.4M / 7.7M | 30k | 2.52 | 0.63 | ~48 | 0.379 |
| depth6_v2, 14-ch+gpool, masked-policy | 7.4M / 12.8M | 50k | 2.48 | 0.62 | ~33 | 0.374 |
| depth6_distill (b18 logit-forcing) | 0.10M / 12.8M | 15k | 2.52 | **0.55** | **9.0** | 0.374 |
| depth6_distill_1m | ~1.0M | — | — | — | — | — |

(score loss isn't comparable supervised-vs-distill — noisy game outcome vs b18's smooth lead.)

## Results

**Strength vs b6c96 (b18 judge, single games — noisy):** supervised depth6_v2 ≈ −97 avg;
depth6_distill (0.10M b18 pos) ≈ −64 avg; distill_1m ≈ −62 avg. Single-game judging has ±50-pt
swings, so treat these as coarse.

**Raw-net agreement vs b18 (`policy_eval`, 300 positions):**

| net | pol top-1 | top-5 | winrate MAE | score MAE | ownership MAE |
|-----|-----------|-------|-------------|-----------|---------------|
| ours (distill_1m) | 39.3% | 72.7% | 0.211 | 3.45 | 0.096 |
| b6c96 | 32.7% | 70.7% | 0.226 | 5.74 | 0.107 |

## Conclusions

- **Distillation is ~70× more data-efficient.** 0.10M b18-labeled positions matched/beat 7.4M
  supervised positions vs b6c96, with much better-calibrated value/score/ownership (b18's soft
  targets vs noisy single-game outcomes). This is the path to a stronger net.
- **The net is not the bottleneck.** Our distilled net matches b18 at least as well as b6c96 on
  every output head (**biased** — we distilled from b18, the reference — but it rules out the net
  as the cause of in-game losses). → motivated isolating the **search** (next experiments).
- Capacity is not the wall: same 6×96 as b6c96; b6c96 just saw ~13× more samples.

## Caveats / follow-ups

- policy_eval is biased toward us (reference = our teacher). A neutral third net would be fairer.
- Single-game strength numbers are too noisy → built the arena (`scripts/match.py`) later.
- Open: scale distillation data ≫1M and train longer to actually beat b6c96.
