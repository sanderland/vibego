# 2026-08-01 — compressing the v1.17 KataGo transformer nets (branch open, phase 1)

KataGo v1.17.0 (2026-07-29) shipped its first transformer nets and the main run is switching to
them. Question for us: are they a target for **post-hoc compression — quantization, layer drops,
structural pruning — with no retraining**, reusing this repo's measurement harness?

This entry is phase 1: build the tooling, place the nets on our cost axes, and run the cheap
structural screen. **Verdict so far: the nets have no obvious structural slack, and weight-only
pruning is expensive — but the branch is worth continuing, for a reason that is not the one we
started with (see Conclusion).**

## What decides the whole approach: the model file format

Our eval scripts (`policy_eval.py`, `move_eval.py`, `match.py`, `arena.py`) all take **engine
command strings**. So anything expressible as a `.bin.gz` runs as `katago analysis -model X` and
drops into the existing harness with zero new code, at real backend speed. Anything else needs a
torch reimplementation served through `run_engine.py`. That line, not the compression literature,
decides what is cheap:

| method | in format v17? | real speedup on the stock engine? | verdict |
|---|---|---|---|
| per-block FFN hidden width | yes | yes, FLOP-proportional | **best lever** |
| attention head count | yes | yes | tried below |
| trunk block drop | yes | yes | tried below |
| inner attention/FFN pair drop (inside `nbt`) | yes | yes | tried below |
| trunk residual width | yes | yes | coupled across all blocks — one global mask, not done |
| low-rank / SVD factorization | **no** (needs a new layer type) | no | torch-only, or a C++ patch |
| unstructured / 2:4 sparsity | no | no | skip (also known-bad without retraining) |
| int8 / fp8 weights | no kernels anywhere | **no** | skip — see below |

**Why quantization is dropped for now.** KataGo's backends are fp32/fp16; there are no int8
kernels in CUDA/Metal/OpenCL. You can quantize weights and re-emit a valid `.bin.gz`, and the
engine will load it as floats and run at *exactly the same speed*. That is a size-matched
comparison, which this repo explicitly refuses to do. The real int8 win lives in TensorRT tensor
cores and requires calibration inside KataGo's C++ TRT backend — a KataGo PR, not a vibego
experiment. (Prior art: [KataGo#799](https://github.com/lightvector/KataGo/issues/799), int8 TRT
tried, "winrates a lot different from fp16", no methodology, no follow-up.)

This is *not* re-litigating the wasm-quantization triage in `ROADMAP.md` / `IDEAS.md` — that was
about our own 0.8–3M-param conv nets in a browser, and it still stands. Different nets, different
target, same conclusion by a different route.

## Tooling built

- `vibego/katago/binmodel.py` — reader/writer for KataGo's exported model format, versions 8–17,
  including the v17 transformer path (attention, SwiGLU FFN, learnable RoPE, trunk RMSNorm tip).
  Contract is **byte-exact round-trip**: verified against 9 real nets (the three v1.17
  transformers, `kata1-b18c384nbt`, `g170-b6c96`, `g170e-b10c128`, and the three bundled KataGo
  test nets). Its arch summariser reproduces KataGo's own names from file contents alone
  (`b10c384h6nbttflrs`, `b10c512h8nbt3tflrs`, `b18c384nbt`), which is the strongest cheap check
  that the parse is right.
- `vibego/katago/cost.py` + `scripts/kata_inspect.py` — params and FLOPs/eval (MACs×2, our usual
  axis) per trunk block, straight off the weight file. No engine, no torch.
- `vibego/katago/prune.py` + `scripts/kata_prune.py` — structural pruning that stays in-format:
  drop blocks, drop inner attention/FFN pairs, prune heads, narrow FFN. Output is an ordinary
  `.bin.gz`; **verified to load and evaluate in stock KataGo v1.17.1** (eigen/CPU build).

## The nets, on our axes

`kata_inspect`, 19×19, FLOPs = MACs×2:

| net | params | MFLOP/eval | attention share of FLOPs |
|---|---|---|---|
| `g170-b6c96` (our anchor) | 1.03 M | 724 | — |
| `g170e-b10c128` | 3.00 M | — | — |
| `kata1-b18c384nbt` (our pinned teacher) | 26.4 M | 18,859 | — |
| **`b10c384h6nbttflrs`** | **10.6 M** | **9,563** | 20.9% |
| `b10c512h8nbt3tflrs-fson-silu-rsnh` | 28.6 M | 24,518 | 16.3% |
| `b11c768h12nbt3tflrs-fson-silu` | 70.5 M | 57,286 | 11.5% |

The headline is the first transformer row against the teacher row: **b10c384h6nbttflrs is 2.5×
fewer params and 2.0× fewer FLOPs than `kata1-b18c384nbt`, and lightvector reports it as stronger
per visit.** That, not compression, is the immediately actionable item for this repo (see
Conclusion).

Structure of `b10c384h6nbttflrs`, read off the file: 10 identical `nbt` blocks, trunk c384,
bottleneck c192, each holding 2 attention/FFN pairs — attention h6 × q32/v32, SwiGLU FFN 192→512.
Every block is exactly 9.9% of total FLOPs; there is no cheap block by construction.

**Two structural facts that predict the screen results below:**
- `num_heads × q_head_dim = 6 × 32 = 192 = the bottleneck width` — exactly. The attention
  projections are square. The same holds for the other two nets (8×32=256, 12×32=384). There is
  no head-dimension over-parameterization to harvest.
- SwiGLU 192→512 is a 2.67× expansion — the standard 8/3 SwiGLU ratio, not a wide FFN.

## Structural screen (weight-only criterion, no healing)

N=10 positions sampled across one `g170-b6c96` self-play game, raw net at **maxVisits=1**, run
through stock KataGo. Deltas are against the *unpruned parent*, not ground truth. Unit selection is
a weight-only magnitude proxy (‖W_v[:,h]‖·‖W_out[h,:]‖ for heads; ‖W1[:,j]‖·‖W_gate[:,j]‖·‖W2[j,:]‖
for FFN units) — the zeroth-order criterion, no activations.

| variant | MFLOP | ΔFLOP | policy top-1 | top-5 | KL | Δwinrate | ΔscoreLead |
|---|---|---|---|---|---|---|---|
| drop 1 inner pair (block 9) | 9,143 | −4.4% | 0.90 | 1.00 | 0.33 | −0.020 | **−0.58** |
| heads 6→5 (all blocks) | 8,874 | −7.2% | 0.70 | 0.80 | 1.24 | −0.149 | **−7.23** |
| FFN 512→384 (all blocks) | 8,498 | −11.1% | 0.70 | 0.90 | 0.38 | −0.031 | **−1.76** |
| drop trunk block 9 | 8,617 | −9.9% | 0.90 | 1.00 | 0.32 | −0.125 | **−5.11** |
| heads 6→4 (all blocks) | 8,186 | −14.4% | 0.40 | 0.50 | 2.83 | −0.063 | −4.56 |
| FFN 512→256 (all blocks) | 7,433 | −22.3% | 0.30 | 0.80 | 1.12 | −0.054 | −4.60 |

Reading it:

1. **Nothing is free.** The mildest edit available (−4.4% FLOPs) already costs 0.58 points of
   scoreLead. There is no slack to harvest with a weight-only criterion.
2. **Heads are the worst thing to prune** — −7.2% FLOPs for −7.2 scoreLead, the worst ratio in the
   table, and the h6→h4 row has top-1 agreement of 0.40. Consistent with the square-projection
   observation above: the heads exactly tile the bottleneck.
3. **FFN width is the most forgiving per FLOP**, as predicted from locality — −11% FLOPs for −1.8
   scoreLead, ~3× better than heads per FLOP removed.
4. **Policy agreement is a bad proxy, confirmed on real data.** `drop block 9` has top-1 0.90,
   top-5 1.00 and the *lowest* KL in the table (0.32) while losing 5.1 points of scoreLead —
   worse than `FFN 512→384`, which has a *higher* KL (0.38) and loses 1.8. The policy survives
   while the value/score head drifts, exactly the bias-not-noise failure mode. **Do not screen
   compression candidates on policy agreement or KL.**
5. Depth damage is not monotone in FLOPs: dropping one *inner pair* (−4.4%) is far cheaper per
   FLOP than dropping a whole *block* (−9.9%), so the fine-grained depth knob is the right one.

**Caveats, stated plainly.** N=10 from a single game is a smoke test, not a measurement — no error
bars are quoted because none would be meaningful. Raw net at 1 visit; MCTS may recover some of
this, and that is exactly what `move_eval.py` / `match.py` are for. The criterion is weight-only,
which is the floor of what pruning can do, not the ceiling. And no healing was applied, which is
the constraint the question asked for — and, on this evidence, the constraint that kills it.

## Conclusion / next

- **Post-hoc compression of these nets, with no retraining, does not look promising.** The nets
  are dense: heads tile the bottleneck exactly, the FFN sits at the standard SwiGLU ratio, and
  every trunk block costs the same. Cheap structural pruning buys ~10% FLOPs for multiple points
  of scoreLead.
- **Quantization is dropped** on deployment grounds, not accuracy grounds: no int8 kernels exist
  in any KataGo backend, so there is no speed win to measure against the accuracy loss.
- **The branch is still worth continuing, reframed.** The interesting comparison is
  **compress-down vs train-up on one frontier plot**: we already have the train-up curve (0.8–4M
  params distilled from b18, fixed anchor gauntlet, b18 judge). Pruning b10c384h6nbttflrs toward
  the same FLOP budgets gives a compress-down curve on identical axes. Three arms, and the third
  is nearly free given `relabel.py` and the 44M-position set: (A) train-up from scratch,
  (B) prune-down no heal, (C) prune-down + a short distill heal. The LLM prior is that B loses to
  A except at mild compression and C beats both; where the crossover sits for Go — and whether
  MCTS moves it — is open and nobody has published it.
- **The highest-value item found on this branch has nothing to do with compression:**
  `b10c384h6nbttflrs` is a strictly better teacher than our pinned `kata1-b18c384nbt` — stronger
  per visit at 2× fewer FLOPs, so same relabel throughput, better targets. Costs: engine bump to
  v1.17.x, and a re-baseline, since the teacher is deliberately pinned for comparability.
- Second-highest: the loader gives us a reference implementation of `nbttflrs` and **learnable
  non-axis-aligned RoPE**, which the discord notes call the clearest single architecture win and
  which ROADMAP #6/Tier-4 already wants.

### Open cross-check before compress-down and train-up share a plot

`kata_inspect` puts `g170-b6c96` at **723.6 MFLOP/eval @19×19** under this repo's own MACs×2
convention (hand-check: 6 blocks × 2 convs × 3×3×96×96 × 361 × 2 ≈ 719 MFLOP, plus heads). But
`README.md` and `SUMMARY.md` describe the 759-MFLOP `s4` net as running at "~⅓ the anchor's
inference cost" / "~⅓ the anchor-shape FLOPs", which implies an anchor near 2,300 MFLOP. Those two
numbers cannot both be on the same axis. Worth resolving before compressed nets are plotted
against the existing train-up points — if the anchor's x-position is wrong, so is the frontier.

### Next steps, in order

1. **Activation-aware selection.** Everything above uses a weight-only criterion. Calibration
   positions through a torch forward (Wanda-style, or Taylor on the distillation loss) is the
   standard upgrade and should strictly beat it. Needs the `.bin.gz` → torch model (loader is
   done; the forward pass is not).
2. **The four diagnostics**, which need that same torch forward: trunk effective rank (is c384
   really 384 dims of variance?), per-block angular distance, per-head importance by masking,
   weight singular-value spectra. These say whether *any* criterion can do better, or whether the
   net is simply dense.
3. **Heal.** Take the FFN-pruned net (best ratio) and run a short distill heal from the parent.
   That measures the thing actually worth knowing: how much of the loss is recoverable cheaply.
4. Only then spend games: `move_eval.py`, then `match.py` at fixed visits with a neutral judge.

## Repro

```bash
# what's in a released net
uv run python scripts/kata_inspect.py models/b10c384h6nbttflrs.bin.gz --blocks

# a pruned net the stock engine loads
uv run python scripts/kata_prune.py models/b10c384h6nbttflrs.bin.gz \
    --ffn-keep 0.75 --out models/ffn384.bin.gz

# measure it with the existing harness
CFG=katago/cpp/configs/analysis_example.cfg
uv run python scripts/policy_eval.py --src data --n 400 \
  --ref    "katago analysis -config $CFG -model models/kata1-b18c384nbt.bin.gz" \
  --engine "full=katago analysis -config $CFG -model models/b10c384h6nbttflrs.bin.gz" \
  --engine "ffn384=katago analysis -config $CFG -model models/ffn384.bin.gz"
```
