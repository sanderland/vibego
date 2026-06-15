# 2026-06-08 (b) — Neutral judge, net-vs-net arena baseline, and the nbt ladder

Follow-on to [the parity result](2026-06-08-proxy-perspective-bug-and-parity.md). Search is solved;
the only lever left is the net. Before the 75G distillation run lands, get a **trustworthy,
unbiased** way to answer "did the new net beat b6c96?" — and stand up the candidate architectures.

## 1. Neutral raw-net judge (fixing the policy_eval bias)

`policy_eval` previously used **b18** as the reference — our own distillation teacher, so the
"ours matches b18 as well as b6c96 does" claim was rigged in our favour. Re-ran it with a
**neutral third net**: **zhizi `b40c768nbt`** (`scripts/setup_katago.sh` URL), neither our teacher
nor the target. 300 positions, raw net (1 visit).

| engine | pol top-1 | top-5 | winrate MAE | score MAE | ownership MAE |
|--------|-----------|-------|-------------|-----------|---------------|
| ours (distill_1m) | 35.0% | **70.7%** | **0.251** | **3.53** | **0.108** |
| b6c96 | 34.7% | 65.3% | 0.260 | 5.21 | 0.119 |

→ Against an unbiased strong judge our net's **raw outputs** are comparable to (slightly better
than) b6c96 on every head. The 06-06 finding survives de-biasing. **But raw agreement turns out
not to predict game strength — see §2.**

## 2. Net-vs-net arena baseline (the real acceptance test)

`scripts/match.py`: **our real engine on our net** vs **KataGo on b6c96**, both at 48 visits,
b18 judge @ 256v, 48 games, colors balanced. This is the number the 75G net must beat.

| comparison | judge scoreLead | win rate | Elo |
|------------|-----------------|----------|-----|
| search only (b6c96 both sides, [prior](2026-06-08-proxy-perspective-bug-and-parity.md)) | −8.7 ± 5.4 | 35% | −104 [−221, −7] |
| **+ our net (distill_1m vs b6c96)** | **−33.4 ± 6.2** | **18.8%** (9/48) | **−255 [−431, −149]** |

**Decomposition:** holding search ≈ constant (parity = −104 Elo), swapping our net in for b6c96
costs another **~−150 Elo / ~−25 pts** in real games.

**The headline:** raw-net agreement (§1) said the nets were comparable, yet the net costs ~150 Elo
**in games**. So `policy_eval` is necessary but *not sufficient* — top-1/MAE on a fixed position
set misses the game-relevant deficits (calibration under search, tactical positions self-play
steers into). **Judge net quality by the arena, not by policy_eval.** The 75G distill run is the
right lever, and we now have a baseline + CI to measure it against.

## 3. The nbt ladder (candidate architectures, ready to train)

Added KataGo's **nested-bottleneck block** (`NBTResBlock`) to `vibego/net/model.py`: 1×1 bottleneck
c→c/2, two nested pre-act 3×3 residual blocks at c/2, 1×1 back to c, outer residual. Four
half-width 3×3 convs cost ≈ two full-width ones, so an nbt block buys more depth/param. Registered
a full ladder (every 3rd block a full-width gpool block, as in the regular archs). Measured:

| arch | params | MFLOP/eval | int8 MB | class |
|------|--------|-----------|---------|-------|
| b6c96-gpool | 1.09M | 774 | 1.1 | old 6b |
| b10c128-gpool | 3.12M | 2230 | 3.1 | old 10b |
| b15c192-gpool | 10.35M | 7417 | 10.4 | old 15b |
| b6c96nbt | 0.80M | 561 | 0.8 | nbt 6b |
| b10c128nbt | 2.21M | 1567 | 2.2 | nbt 10b |
| b15c192nbt | 7.41M | 5288 | 7.4 | nbt 15b |

At equal b/c label, nbt is ~25–30% **fewer** params (bottleneck) but deeper — the per-param
strength win. Tests in `tests/test_archs.py` (build, block placement, residual identity, bottleneck
width, nbt<regular params).

**wasm target:** for in-browser CPU (wasm-SIMD, ~1–5 GFLOP/s single-thread), the 6b class is the
sweet spot — **b6c96nbt** (0.80M, 0.56 GFLOP/eval, ~0.8 MB int8) gives interactive MCTS and the
smallest download; b10c128 only with WebGPU; b15 is desktop/server.

## Method / commands

```
# neutral judge
ZHIZI="katago analysis -model models/zhizi.bin.gz -config <analysis_example.cfg>"
uv run python scripts/policy_eval.py --src data --n 300 --ref "$ZHIZI" \
  --engine "ours=uv run python scripts/run_engine.py -model checkpoints/depth6_distill_1m.pt" \
  --engine "b6c96=katago analysis -model models/g170-b6c96.bin.gz -config <cfg>"
# net arena
uv run python scripts/match.py --games 48 --visits 48 --workers 6 \
  --a "uv run python scripts/run_engine.py -model checkpoints/depth6_distill_1m.pt -pos-len 19" --a-name distill_1m \
  --b "katago analysis -model models/g170-b6c96.bin.gz -config <cfg>" --b-name b6c96 \
  --judge "katago analysis -model models/kata1-b18c384nbt.bin.gz -config <cfg>" --judge-visits 256
```

## Conclusions

- **Unbiased net check in place** (zhizi/b40 neutral ref); our net's raw outputs ≈ b6c96.
- **Acceptance test in place**: distill_1m vs b6c96 = **−33.4 ± 6.2, Elo −255**. The net costs
  ~150 Elo in games; this is the baseline the 75G net must beat. Swap weights, re-run, read Elo.
- **policy_eval ≠ game strength** — judge net quality by the arena.
- **nbt ladder ready** (6b/10b/15b, old + nbt) for the post-75G arch ablation; b6c96nbt is the wasm pick.
