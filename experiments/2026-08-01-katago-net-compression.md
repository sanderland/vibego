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

---

# Phase 2 — the torch forward, the four diagnostics, and activation-aware selection

Phase 1's screen used a weight-only pruning criterion, which is the *floor* of what pruning can
do. Phase 2 builds the forward pass that lets us look at activations, checks it against the stock
engine, and then answers the two questions that were open: **is there any structural slack in
these nets at all**, and **does a better criterion rescue post-hoc pruning?**

## The forward pass, and what it took to validate it

`vibego/katago/torchmodel.py` runs a parsed `.bin.gz` in torch — nbt blocks, learnable 2D RoPE,
SwiGLU FFN, per-position and spatial trunk RMSNorm, board masking. `scripts/kata_torch_check.py`
compares it against the stock engine.

Getting a clean comparison surfaced two things worth recording, both of which look exactly like a
bug in your forward if you don't know about them:

1. **`nnRandomize = true` is on in `analysis_example.cfg`.** It applies a random dihedral symmetry
   to every evaluation. The nets are only approximately symmetry-equivariant, so this alone moved
   policy top-1 agreement from 0.94 to 0.39 on near-symmetric positions. Any raw-net comparison
   against the engine needs `-override-config nnRandomize=false`.
2. **The engine forces no-result probability to zero** under superko + area scoring and
   renormalizes, and scales `lead` by `(1 - P(no result))`. Reproducing its winrate means doing
   the same (`nneval.cpp`).

The other lesson is about the *data*: reconstructing an analysis query from a training row is
inherently lossy. Training data is deliberately rule- and komi-randomized (54 distinct rule
combinations in one 1024-row shard, komi from −79 to +77, half of them off the half-integer grid
the analysis API accepts), and a query built from `initialStones` carries no move history. So the
check has a second mode using positions whose V7 features are determined by inspection — isolated
stones on star points, where the liberty, ladder and pass-alive planes are all provably zero.

On those, against the engine with symmetry randomization off:

| net | policy KL median | \|Δwinrate\| mean | \|ΔscoreLead\| mean | scoreLead corr |
|---|---|---|---|---|
| `b10c384h6nbttflrs` (v15) | **0.00025** | **0.00042** | 0.061 | 0.99975 |
| `g170e-b10c128` (v8) | 0.00014 | 0.0022 | 1.81 | 0.9941 |
| `g170-b6c96` (v8) | 0.0022 | 0.011 | 0.46 | 0.9973 |

The target net matches to ~1e-4 on policy and winrate. The old v8 nets match on policy and value
but their `scoreLead` drifts by 0.5–2 points; their misc-value head is only four channels wide and
the engine post-processes it differently. Not chased — v8 is not the target, and the diagnostics
below compare our forward against *itself*.

## [1] The trunk is correlated but not narrowable -- and a measurement artifact nearly said otherwise

Residual stream at the trunk tip, c384, 48 real positions. Two summaries, because they answer
different questions: the **PCA** spectrum is rotation-invariant and says whether a low-rank
*projection* would do, while the **channel** basis is the only one that channel pruning can act on.

| basis | participation ratio | dims for 90% / 99% of variance | dims under 1% of the busiest |
|---|---|---|---|
| PCA (rotation-invariant) | 17.9 of 384 (4.7%) | 112 / 299 | — |
| **channel** (what pruning sees) | **176.7 of 384 (46%)** | **287 / 371** | **0%** |

**Read the second row first: trunk width pruning is dead.** No channel is idle -- not one of the
384 carries under 1% of the busiest channel's variance -- and you need 287 of them for even 90% of
the variance. There is nothing to delete.

The PCA row says the stream's channels are strongly *correlated* (participation ratio 17.9 against
the channel basis's 176.7), so a few rotated directions dominate. But the tail is heavy: 99% of
variance still needs 299 of 384 principal dimensions. That is real structure, not a dramatic one.

**The measurement artifact is worth recording,** because the first version of this table was much
more exciting and wrong. Capturing the stream *after* the trunk-tip RMSNorm and SiLU gave a channel
participation ratio of 40 and claimed **51% of channels sat under 1% of the busiest**. Both were
artifacts: the tip norm's per-channel `gamma` rescales the stream, and SiLU squashes negatives, so
post-norm per-channel variance describes the norm's parameters as much as the stream's use of its
channels. Reading the pre-norm tensor moves "half the trunk is idle" to "none of it is". Any
per-channel statistic in a normed architecture has to be taken before the norm.

**And even a narrowable trunk would barely pay.** Only **12.3%** of this net's FLOPs depend on
trunk width at all: per nbt block the two 1×1 bottleneck convs are 106.5 of 945.4 MFLOP, plus the
54.9 MFLOP stem and 53.4 in the heads. So c384 → c310 would buy ~2.4% of total FLOPs and c384 →
c256 ~4.0%. The `nbt` design deliberately puts the compute in the bottleneck, which means trunk
redundancy -- had there been any -- would have been worth almost nothing on our axis.

## [2] No block is coasting — the LLM depth-drop premise does not hold here

Per-block cosine between the residual-stream input and output, and the relative size of the
residual each block writes:

| block | cos(in, out) | ‖res‖ / ‖in‖ |
|---|---|---|
| `blocks.3` (quietest) | 0.9805 | 0.235 |
| `blocks.4` | 0.9773 | 0.258 |
| `blocks.5` | 0.9712 | 0.263 |
| ... | | |
| `blocks.0.blockstack.1` (loudest) | 0.3427 | 3.437 |

In the LLM depth-pruning literature a droppable layer has cos > 0.99 and writes a residual a few
percent the size of the stream. **The quietest block here writes a residual 23% the size of the
stream.** There is no coasting block to delete — which is exactly what phase 1's −5.1 scoreLead
for one dropped block was telling us, now explained rather than just measured.

The inner sub-blocks are the opposite: the first attention block inside an `nbt` block writes a
residual **3.4×** the size of its input, i.e. it essentially rewrites the bottleneck
representation. The `nbt` bottleneck is not a lightly-perturbed residual stream at all.

## [3] There are no dead heads, and the weight-only criterion was measuring the wrong thing

- Within a block, the ratio of most- to least-important head (activation-weighted) has **median
  1.76**, worst 7.81. The NLP head-pruning results that motivate this lever report spreads of 10×
  and up. Six heads that exactly tile a 192-dim bottleneck are all doing work.
- Only **7.6%** of SwiGLU hidden units fall below 10% of their block's median importance.
- **Rank correlation between the activation-aware and weight-only head rankings: +0.26.** The two
  criteria mostly disagree, so phase 1 was ranking heads close to arbitrarily.

## [4] Attention projections are substantially low-rank; the FFN is not

Rank needed for 99% of the spectral energy, one representative trunk block:

| matrix | shape | r90 | r99 | r99 / full |
|---|---|---|---|---|
| `k_proj` | (192, 192) | 33 | 76 | **0.40** |
| `q_proj` | (192, 192) | 44 | 96 | 0.50 |
| `v_proj` | (192, 192) | 58 | 122 | 0.64 |
| `out_proj` | (192, 192) | 78 | 131 | 0.68 |
| `linear1` | (192, 512) | 117 | 170 | 0.89 |
| `linear_gate` | (192, 512) | 119 | 171 | 0.89 |
| `linear2` | (512, 192) | 137 | 180 | **0.94** |

Consistent with [1]: the compressible structure is in the attention projections and is *low-rank*,
not *low-width*. Spectral energy is a weak proxy for functional equivalence, so this is a
plausibility check, not a promise — but it points the same direction as the trunk PCA.

## [5] Activation-aware selection roughly halves FFN damage -- and still is not enough

Both criteria pruned to identical FLOPs, damage measured against the unpruned parent on 96
**held-out** positions (calibration used a disjoint 96):

| variant | ΔFLOP | policy top-1 | KL | \|Δ lead\| | \|Δ winrate\| |
|---|---|---|---|---|---|
| FFN keep 0.75, weight-only | −11.1% | 0.59 | 0.371 | 1.96 | 0.123 |
| FFN keep 0.75, **activation-aware** | −11.1% | **0.71** | **0.241** | **1.25** | **0.076** |
| FFN keep 0.50, weight-only | −22.3% | 0.32 | 1.410 | 6.12 | 0.230 |
| FFN keep 0.50, **activation-aware** | −22.3% | **0.40** | **0.826** | **3.01** | **0.195** |
| heads keep 0.75, weight-only | −14.4% | 0.20 | 2.034 | 17.60 | 0.271 |
| heads keep 0.75, activation-aware | −14.4% | 0.28 | 1.817 | 12.18 | 0.317 |
| heads keep 0.50, weight-only | −21.6% | 0.22 | 2.211 | 14.70 | 0.259 |
| heads keep 0.50, activation-aware | −21.6% | 0.12 | 2.343 | 12.20 | 0.272 |

Reading it:

1. **The weight-only floor was a real floor, for FFN width.** Activation-aware selection cuts lead
   damage by **36%** at keep 0.75 and **51%** at keep 0.50, at identical FLOPs. Phase 1 understated
   what pruning can do; that is the honest correction to make to it.
2. **It is still not enough.** The best criterion available loses **1.25 points of scoreLead for
   11% of the FLOPs**. For scale, the train-up study fought over margins of 5--15 points, so this
   is not a rounding error.
3. **For heads the criterion barely matters, because the net is already broken.** At keep 0.50 the
   activation-aware variant is *worse* on KL (2.34 vs 2.21) and top-1 (0.12 vs 0.22). Damage is
   also non-monotone in the amount removed (17.6 points at −14.4%, 14.7 at −21.6%): once six heads
   that exactly tile the bottleneck become three, the outputs are no longer a perturbation of the
   parent's and the deltas stop tracking anything. **Do not prune heads in these nets**, and do
   not read fine distinctions off a destroyed net.
4. FFN damage is superlinear in the amount removed (1.25 → 3.01 for 11% → 22%), so there is no
   "shave a little everywhere" budget that stays cheap.

**Caveats.** 96 held-out positions; damage is measured against the parent net in our torch forward,
not in games -- this ranks criteria, it does not measure Elo. No healing, which remains the
constraint the original question imposed and, on this evidence, the constraint that decides it.

## Phase-2 conclusion

The phase-1 verdict survives contact with better tools, and now has a mechanism rather than just a
number attached:

- **These nets are dense where it counts.** No coasting block (quietest writes a 23% residual), no
  dead heads (median importance spread 1.76×), 92% of FFN units carrying real signal. The
  structural facts read off the file in phase 1 -- heads exactly tiling the bottleneck, SwiGLU at
  the standard 8/3 ratio, all blocks costing the same -- are borne out by the activations.
- **Better selection helps but does not rescue it.** Halving the damage still leaves 1.25 points of
  scoreLead for 11% of the FLOPs. Post-hoc structural pruning without healing is not a route to a
  better point on the FLOPs↔Elo frontier for these nets.
- **The redundancy that does exist is in the wrong basis and in the wrong place.** No trunk channel
  is idle, so channel pruning has nothing to take; the trunk's channels are correlated, and the
  attention projections need only 40--68% of their rank for 99% of spectral energy, but exploiting
  either needs a low-rank *factorization*, which format v17 cannot express. And trunk width is only
  12.3% of this net's FLOPs anyway -- the `nbt` design puts the compute in the bottleneck -- so even
  a freely narrowable trunk would have been worth ~2--4%.

### What would actually be worth doing next, in order

1. **Take the teacher upgrade.** Still the highest value/effort item on this branch and unrelated
   to compression: `b10c384h6nbttflrs` at 10.6M params / 9.56 GFLOP versus the pinned
   `kata1-b18c384nbt` at 26.4M / 18.9 GFLOP, reported stronger per visit. Same relabel throughput,
   better targets. Costs an engine bump to v1.17.x and a re-baseline.
2. **Prune-down + distill heal, as the third arm of compress-down vs train-up.** Everything above
   is arm B (no heal) and it loses. Arm C is cheap here given `relabel.py` and the 44M-position
   set, and it is the arm the LLM literature says wins. Start from the activation-aware FFN-pruned
   net, since that is the best-behaved lever.
3. **Steal the architecture rather than the weights.** Learnable non-axis-aligned RoPE and the
   `nbttflrs` block shape are now readable from the file and implementable in our registry
   (ROADMAP #6/Tier-4). The transferable question is whether the transformer edge survives at
   1--4M params speed-matched on CPU, where the discord numbers say it mostly does not.
4. Low-rank factorization of the attention projections is the only in-net lever with headroom, and
   it needs a new layer type in KataGo's C++ -- a KataGo PR, not a vibego experiment.

## Repro (phase 2)

```bash
# validate the torch forward against the engine (note: symmetry randomization must be off)
uv run python scripts/kata_torch_check.py --synthetic \
    --model models/b10c384h6nbttflrs.bin.gz \
    --katago ./katago --config katago/cpp/configs/analysis_example.cfg

# diagnostics + criterion head-to-head
uv run python scripts/kata_diagnose.py --model models/b10c384h6nbttflrs.bin.gz \
    --npz katago/python/testdata/benchmark_data_1024.npz --n 96 --eval-n 96 --keep 0.75 0.5
```

---

# Phase 3 — a correction: low-rank *is* expressible in format v17

Phase 1's format table put "low-rank / SVD factorization" in the **not expressible** row, on the
reasoning that a factorization `W ≈ AB` needs two matmuls where the format has one slot. That is
right in general and **wrong for the attention block**, which is the one place it matters — because
the format already stores both of the block's paths in factored form:

- **The value path.** `v_proj` maps `c_in → heads·v_head_dim`, `out_proj` maps
  `heads·v_head_dim → c_out`, and `v_head_dim` is a plain header field. Each head's composed map
  `M = W_v W_out` has rank at most `v_head_dim`; **shrinking the dimension between them *is* its
  low-rank approximation**, in-format, no new layer type, and the stock engine loads it.
- **The query path.** RoPE rotates interleaved channel pairs `(2p, 2p+1)` by position-dependent
  angles, so an arbitrary change of basis would break the correspondence between a dimension and
  its learned frequency — a free SVD is genuinely not available here. But each pair is an
  independent additive term in the logit, so *selecting* pairs is free, and a dropped pair takes
  its two columns of `q_proj`/`k_proj` and its row of `rope_freqs` with it. `q_head_dim` is also a
  header field.

Phase 2 concluded "the only lever with headroom needs a KataGo C++ change". That conclusion was
wrong on the mechanism, and this is the correction.

**A second correction, to how phase 2 read its own spectra.** Those numbers — `k_proj` needing 40%
of its rank for 99% of spectral energy, against 94% for the FFN's `linear2` — are for the
**concatenated multi-head** matrices, all 192×192. Compressing *those* would mix heads together,
which is not expressible in the format and is not what the levers below do. The quantity that
actually governs the value lever is each head's composed map `M = W_v W_out`, and that is
**already rank-capped at `v_head_dim` = 32 by construction**. So the encouraging spectra were
answering a different question than the one that decides this, and should not have been read as
predicting headroom.

Cost-wise the attention block is also where the FLOPs are. Per `nbt` block: the four projections
are 213 MFLOP and the two N² attention matmuls 100 MFLOP, against 213 for the SwiGLU FFN and 106
for the bottleneck convs — so **attention is 66% of a block**, and `v_head_dim` and `q_head_dim`
between them scale all of it.

**Selection method.** For the value path, the best rank-r approximation *under the input
distribution*: with `C = E[x xᵀ]` measured at the block input over calibration positions,
minimize `‖C^½(M − M′)‖_F` rather than `‖M − M′‖_F`, so rank is not spent on directions the data
never visits. For the query path, keep the pairs with the highest `E[|q_p|²]·E[|k_p|²]`, both
computable from the same `C`. Unlike head pruning, nothing is discarded outright — every head
keeps a smaller subspace — which matters given the phase-2 finding that no head is idle.

## [5b] Head-to-head at equal FLOPs, including the low-rank levers

96 held-out positions, damage against the unpruned parent:

| variant | ΔFLOP | policy top-1 | KL | \|Δ lead\| |
|---|---|---|---|---|
| FFN keep 0.75, activation-aware | −11.1% | 0.71 | **0.208** | **1.27** |
| FFN keep 0.50, activation-aware | −22.3% | 0.40 | 0.814 | 2.84 |
| FFN keep 0.25, activation-aware | −33.4% | 0.16 | 1.814 | 7.38 |
| v_dim keep 0.75, plain SVD | −5.4% | 0.50 | 1.182 | 7.52 |
| v_dim keep 0.75, **data-aware** | −5.4% | 0.70 | **0.160** | **1.11** |
| v_dim keep 0.50, plain SVD | −10.8% | 0.29 | 1.853 | 18.79 |
| v_dim keep 0.50, **data-aware** | −10.8% | 0.54 | 0.590 | 2.69 |
| v_dim keep 0.25, plain SVD | −16.2% | 0.10 | 2.688 | 34.53 |
| v_dim keep 0.25, **data-aware** | −16.2% | 0.33 | 1.618 | 5.74 |

1. **The low-rank hypothesis was wrong.** At matched FLOPs the value-path factorization loses to
   FFN narrowing: at ~−11% it costs 2.69 points against FFN's 1.27. Per FLOP removed, FFN is about
   1.8× more efficient. The reason is the one that has explained everything on this branch — each
   head's value map is *already* rank-capped at `v_head_dim`, so there is no spare rank to take.
2. **But whitening the SVD by the input covariance is worth 7×.** 7.52 → 1.11 points at keep 0.75,
   18.79 → 2.69 at keep 0.50. The data-aware objective matters far more for a low-rank
   approximation than activation-awareness did for magnitude pruning (1.4–2.8× there). If you take
   one method away from this entry, take that one.
3. Ranking the in-format levers by lead damage per % of FLOPs removed, each with its best
   criterion: **FFN width (0.11) ≈ inner-pair drop (0.13) > low-rank value path (0.21) > whole
   block drop (0.52) > heads (0.85)**.

---

# Phase 4 — the actual CPU result: 4.2–27% of these nets' weights are subnormal

Benchmarking the FFN-pruned net against its parent on CPU produced a number that could not be
right: **an 11% FLOP cut buying a 3.7× wall-clock speedup** in the stock engine. Chasing that is
what produced the useful finding, and it is not about compression at all.

Ruling things out: it is not a power-of-two stride pathology (FFN 512 → 511 changes wall-clock by
6%, not 4×). It is not the matmul shape (a bare `(361×192)@(192×h)` scales *linearly* in h — 604 µs
at 512, 458 µs at 384). It is not batching (per-eval cost is unchanged from batch 1 to batch 8).

It is **denormals**. `b10c384h6nbttflrs` carries **4.22% subnormal weights** (444,703 of 10.5M), and
x86 handles subnormal operands in microcode at roughly two orders of magnitude the cost of normal
arithmetic. Importance-based FFN pruning had been speeding the net up mostly by deleting the
smallest-magnitude units — which is exactly where the subnormals live (4.22% → 0.74% at keep 0.75
→ 0.00% at keep 0.50).

Confirmed directly, single-thread batch-1 torch, same weights: **2551 ms/eval with denormals
enabled, 205 ms/eval with `torch.set_flush_denormal(True)` — 12.5×, changing nothing.**

**KataGo never sets flush-to-zero.** There is no `_MM_SET_FLUSH_ZERO_MODE`, no `-ffast-math`,
nothing setting MXCSR anywhere in `cpp/` outside vendored third-party code. So its CPU backend pays
this on every subnormal it meets.

## The fix, and what it is worth

Zeroing subnormal weights is numerically inert — nothing below 1.2e-38 can affect a net whose
activations are order 1 — and it is expressible in the weight file, so it works on the
**unmodified** engine today. `scripts/kata_prune.py --flush-subnormal`.

Stock KataGo v1.17.1, eigen (CPU) backend, raw evals at batch 1, single analysis thread:

| net | subnormal weights | parent | subnormals→0 | speedup |
|---|---|---|---|---|
| `b10c384h6nbttflrs` | 4.22% | 4.04 s/eval | 0.74 s/eval | **5.4×** |
| `b10c512h8nbt3tflrs` | 15.94% | 25.9 s/eval | 1.89 s/eval | **13.7×** |

Outputs are **bit-identical** — over 40 positions, max |Δpolicy| = 0, max |Δwinrate| = 0,
max |ΔscoreLead| = 0, top-1 identical on 100%. This is not a tradeoff.

## It is specific to the new transformer nets, and it scales with size

| net | params | subnormal weights |
|---|---|---|
| `b10c384h6nbttflrs` (v15) | 10.5 M | **4.22%** |
| `b10c512h8nbt3tflrs` (v17) | 28.5 M | **15.94%** |
| `b11c768h12nbt3tflrs` (v17) | 70.4 M | **27.23%** |
| `kata1-b18c384nbt` (v14, conv-nbt) | 26.3 M | 0.0000% |
| `g170e-b10c128` (v8) | 3.0 M | 0.0000% |
| `g170-b6c96` (v8) | 1.0 M | 0.0000% |

Every conv net has exactly zero. Every transformer net is affected, and the largest is over a
quarter subnormal. Whatever produces it — the transformer training recipe, weight decay against
RMSNorm's scale invariance — it arrived with the new architecture.

## What this changes

- **The phase-2/3 verdict was a GPU verdict.** "Post-hoc compression does not pay" holds on FLOPs.
  On CPU the largest available win is not compression at all, costs nothing, and is available today
  by rewriting the weight file.
- **Worth reporting upstream.** Setting FTZ/DAZ once at startup in KataGo's CPU backends would give
  every CPU user of the v1.17 nets a multiple-times speedup with no downside, and would not need
  the weight-file workaround.
- **For this repo's own frontier:** the CPU-ms column is not a refinement of the FLOPs axis, it is
  a different axis. Here it disagreed by 13×, and no amount of care with FLOPs would have found it.
  Worth checking our own trained nets for subnormal weights before trusting any CPU-ms number.

## Repro (phase 4)

```bash
# how many subnormals does a net carry, and what does removing them cost? (nothing)
uv run python scripts/kata_prune.py models/b10c384h6nbttflrs.bin.gz --flush-subnormal \
    --out models/b10c384h6nbttflrs-ftz.bin.gz

# time it in the stock engine, batch 1
katago analysis -model models/b10c384h6nbttflrs.bin.gz -config CFG \
    -override-config numAnalysisThreads=1,nnRandomize=false < queries.jsonl
```
