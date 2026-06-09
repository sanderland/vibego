# 2026-06-09 — data-scaling Elo (s0), pattern_embed pays at scale, real CPU-ms frontier

## Stage-B Elo at data scale (300 shards / 30k steps vs the 48-shard / 8k Stage-B1)

Same protocol both rows: 64 games (B1: 48) vs g170-b6c96 anchor @ 48 visits, b18 judge 256v,
paired scoreLead headline (color-reversed pairs share an opening seed).

| net | arch | MFLOP | B1 (48sh/8k) Elo | s0 (300sh/30k) scoreLead | s0 Elo [CI] |
|---|---|---|---|---|---|
| nbt | b6c96nbt | 561 | −374 | −48.6 ± 5.6 | −338 [−535,−234] |
| nbtpat | b6c96nbt-pat | 561 | −669 | **−32.6 ± 4.8** | −124 [−227,−40] |
| dw7 | b7c106nbt | 759 | −417 | **−25.8 ± 5.6** | −124 [−227,−40] |
| b10nbt | b10c128nbt | 1567 | −280 | *(running)* | *(running)* |

Findings:
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

## Off-policy diagnostic plumbing (review #6, gate for data capex)

Built and smoke-tested end-to-end: `match.py --save-games` (JSONL game records) →
`game_positions.py` (records → source-format npz, full 22-ch packed planes) → `relabel.py`
(b18, **visits 1, same as the 75G set**) → `eval_loss.py` (same-net loss on archive-val vs
own-trajectory positions, same teacher both sides — isolates the distribution term).
Run after stage_b frees the GPU: 32 games s_dw7 vs anchor @48v, stride 3.

## Next (s1 batch, queued)

`screen.py --batch s1 --data /workspace/distill/scale --steps 30000`: s1_dw7pat
(b7c106nbt-pat), s1_b10pat (b10c128nbt-pat), s1_dw7f19 (dw7 + `--only-19x19` — the filter A/B
arm; control = s_dw7, both evaluated post-hoc on a fixed 19×19 val via `eval_loss --only-19x19`).
