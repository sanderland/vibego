# 2026-06-08 (c) — KataGo's own size ladder, intrinsic eval (the yardstick)

A reference scale for the [arch bake-off](2026-06-08-net-arena-baseline-and-nbt.md): how well do
**real KataGo nets of each size** agree with a strong neutral judge? This is the ruler we read our
own nets against on the same metric (raw-net agreement, no search, no games).

## Setup

`scripts/policy_eval.py`, 300 positions from `data/`, raw net (1 visit). Reference = **zhizi
`b40c768nbt`** (neutral: not in the g170 lineage). Engines = the final **g170** nets at each size
(`g170-b6c96`, `g170e-b10c128`, `g170e-b15c192`, from `katagoarchive.org/g170/neuralnets/`) plus
`kata1-b18c384nbt` as a top-end anchor. b6c96 already lived in `models/`; b10c128/b15c192 pulled
fresh.

## Result — agreement vs neutral b40

| net | params | pol top-1 | top-5 | winrate MAE | score MAE | ownership MAE |
|-----|--------|-----------|-------|-------------|-----------|---------------|
| g170-b6c96 | ~1.3M | 30.3% | 66.3% | 0.258 | 5.30 | 0.118 |
| g170-b10c128 | ~3.3M | 44.7% | 81.3% | 0.201 | 2.95 | 0.087 |
| g170-b15c192 | ~6.6M | 49.0% | 88.0% | 0.162 | 2.36 | 0.077 |
| kata1-b18c384nbt | ~24M | 65.0% | 97.3% | 0.098 | 1.44 | 0.048 |

For context, our distilled nets on the **same** neutral metric:

| net | params | pol top-1 | top-5 | winrate MAE | score MAE | ownership MAE |
|-----|--------|-----------|-------|-------------|-----------|---------------|
| ours `distill_1m` (b6c96-gpool, 40k steps) | 1.09M | 35.0% | 70.7% | 0.251 | 3.53 | 0.108 |

## Reading

- **The metric is monotonic in size** — b6c96 → b10c128 → b15c192 → b18 climbs cleanly on *every*
  channel. So raw-net agreement vs b40 is a meaningful axis to **rank architectures** (even though
  it does *not* predict in-game Elo — see the [arena note](2026-06-08-net-arena-baseline-and-nbt.md)).
  Use it to choose a block type / size, then confirm the winner in the arena.
- **Our 1M distilled net ≈ g170-b6c96, slightly better** (top-1 35.0 vs 30.3, score MAE 3.53 vs
  5.30) — sensible: same capacity class, but distilled from b18 so better-calibrated value/score.
  It is **well short of g170-b10c128** (top-1 44.7) — the jump from 6b to 10b is large and is what
  more capacity + the 75G data should buy.
- **Noise:** at n=300 the top-1 has ~±3–4 pt run-to-run wobble (an earlier run put g170-b6c96 at
  34.7 vs 30.3 here — argmax ties / engine nondeterminism). The ladder gaps (15+ pts/step) dwarf
  it, but for closer comparisons (bake-off nets) use **n≥500**.

## Use

When the bake-off nets finish, drop their `policy_eval` rows next to this table. Targets:
- a 6b-class net should aim to clear **g170-b6c96** (top-1 ~30, score MAE ~5.3) comfortably;
- a 10b-class net should head toward **g170-b10c128** (top-1 ~45, score MAE ~3.0).

## Command

```
ZHIZI="katago analysis -model models/zhizi.bin.gz -config <analysis_example.cfg>"
uv run python scripts/policy_eval.py --src data --n 300 --ref "$ZHIZI" \
  --engine "g170-b6c96=katago analysis -model models/g170-b6c96.bin.gz -config <cfg>" \
  --engine "g170-b10c128=katago analysis -model models/g170-b10c128.bin.gz -config <cfg>" \
  --engine "g170-b15c192=katago analysis -model models/g170-b15c192.bin.gz -config <cfg>" \
  --engine "kata1-b18=katago analysis -model models/kata1-b18c384nbt.bin.gz -config <cfg>"
```
