# Experiment roadmap — toward a small/fast/strong (in-browser) Go net

**Objective — the FLOPs↔Elo Pareto frontier.** Map and push the **upper envelope of Elo vs
FLOPs/eval** for small nets: for each inference-cost budget, find the strongest net. Every
experiment is a new point on the plane or a shift of the frontier.

- **x-axis = FLOPs/eval (MACs×2)** — hardware-independent, and a *better wall-clock proxy on our
  CPU/wasm target than on GPU*: CPU is closer to compute-bound at these sizes (no GPU
  parallelism-underutilization or kernel-launch artifacts), and conv/GEMM nets vectorize comparably
  so SIMD just sets the FLOPs/s constant. Imperfect — equal-FLOP nets still vary ~10–30% in CPU
  wall-clock from arithmetic intensity (nbt's 1×1 bottlenecks), channel/SIMD-width alignment, and
  runtime fusion (our smoke: nbt ≈ 65% of a regular block's effective CPU FLOP-throughput) — so keep
  **real CPU/wasm ms as a validation column**, not the primary axis.
- **y-axis = Elo**, arena-measured on **one fixed scale**: every net plays a **fixed anchor gauntlet**
  (e.g. `g170-b6c96` at fixed visits, b18 judge) so points are comparable as the set grows.
  Intrinsic `policy_eval` vs neutral b40 is the **cheap pre-filter** to rank candidates before
  spending arena games — but the frontier itself is **Elo** (we showed intrinsic ≠ game strength).
- **Fix search visits** when comparing nets (Elo depends on visits). Total move cost = visits ×
  FLOPs/eval, so the low-visit regime (Gumbel, #4) shifts the whole frontier — report the frontier
  at the deployment visit budget.
- **Results registry:** one row per net — `(arch, params, FLOPs/eval, CPU/wasm ms, Elo±CI)` — plotted
  as the envelope. Download size (fp16, param-driven) is a secondary axis, not the primary one.

**Why not quantization (as a wasm lever).** Our nets are already tiny (0.8–3 M params ≈ 1.5–6 MB
fp16), so there's no hard size cap to fight; and in-browser runtimes don't reliably accelerate int8
(WebGPU is fp16-centric; wasm int8 carries quantize/dequantize overhead that often erases the gain
on small conv nets). So int8/GPTQ/QAT is dropped; **fp16** is the trivial default. The size lever we
*do* take is **weight tying / depth recurrence** (item 5b), which shrinks params at any precision.

Sources feeding this list: our own bake-off (nbt sweeps val-loss at 6b/10b but runs ~1.5× slower
wall-clock at equal params), the [Go-AI survey](../docs/go-ai-survey.pdf), and the
[KataGo dev discord summary](../docs/discord_experiments_summary.md).

Each item: **why** (evidence) · **run** (concrete) · **cost** · **confidence**. Rejected / deferred /
low-priority ideas live in **[IDEAS.md](IDEAS.md)** (don't re-litigate them without new evidence).

---

## Tier 0 — plot the first frontier (in flight; finish before anything else)

**0. Place the current bake-off + depth-width nets on the FLOPs↔Elo plane.**
- *Why:* this is the first frontier and the baseline every later experiment must beat. The
  depth-width series (b6c112→b10c88nbt, all ~1.09M, ~760–780 MFLOP) is a near-vertical slice of the
  plane — same FLOPs, varying depth/width — so it directly shows **where on the frontier the best
  depth/width balance sits** at that budget.
- *Run (post-training, free GPU):* (a) **FLOPs/eval** per net — already computed; (b) `bench_net.py`
  (cpu + mps, batch 1/16) for the **wall-clock validation column**; (c) **intrinsic pre-filter** —
  one `policy_eval` n=500 vs neutral b40 to rank all 10; (d) **arena Elo** for the frontier
  candidates (per-class best + the full depth-width slice) vs the fixed anchor `g170-b6c96` (b18
  judge, fixed visits); (e) plot **Elo vs FLOPs/eval** and record the registry rows. *(scheduled)*
- *Cost:* low (arena games for ~6–8 nets). *Confidence:* high — this is the plane everything plugs into.

---

## Tier 1 — cheap, high-confidence, in-browser-aligned (do next; each an ablation vs the Tier-0 winner)

**1. RMSNorm in place of BatchNorm.**
- *Why:* discord — equal-or-better loss; and BatchNorm couples across the batch and needs
  running-stat folding, whereas RMSNorm is clean at **batch=1** (our in-browser latency case) and
  **quantizes better**. A direct deployment win, not just a wash.
- *Run:* swap block + trunk-end norms; retrain 1–2 Tier-0-winner-sized nets; compare loss + neutral
  eval + speed. *Cost:* low. *Confidence:* high.

**2. Input representation: add the omitted features (pass-alive / territory, the "feat 18/19" set).**
- *Why:* AlphaVile's chess result (survey) — *improving the input representation beat switching the
  backbone*. We deliberately run a 14-channel subset; pass-alive is the one meaningful omission and
  is cheap (reuse the area flood-fill). Likely better strength-per-ms than any block change.
- *Run:* add the channels to `features.py` (keep the exact-recompute invariant + its test), retrain,
  ablate strength/calibration; watch the small stem-FLOP cost. *Cost:* low–med. *Confidence:* med–high.

**3. Distillation scaling + better targets (power lever; partly in flight on the remote 75G run).**
- *Why:* our own finding — distillation ~70× more sample-efficient; the in-game gap to b6c96 is the
  net, closed by more data/steps. Teacher **ensembling** (b18+b28+b40 soft targets, à la Rapfi
  distillation) raises target quality further.
- *Run:* when 75G lands, retrain the Tier-0 winner → arena vs b6c96 (beat the −104 search-parity
  bar). Then an ensembled-relabel pass. *Cost:* data high (remote), train med. *Confidence:* high.

---

## Tier 2 — search for the low-visit regime (high ROI for in-browser; architecture-independent)

**4. Gumbel root selection at low visits.**
- *Why:* survey (Gumbel MuZero, MiniZero) — sound policy improvement and markedly better play at
  **few** simulations, exactly the in-browser regime where each visit is an expensive forward pass.
  Strength-per-visit is a multiplier that stacks on top of any net.
- *Run:* implement Gumbel (sample-without-replacement at the root + Sequential Halving) in
  `engine/search.py`; arena our net at 8/16/32 visits, Gumbel vs current PUCT+noise. *Cost:* med.
  *Confidence:* high at low visits.

---

## Tier 3 — the export/deploy path (the actual deliverable; without it "in-browser optimal" is unprovable)

**5. ONNX/WebGPU export + real in-browser latency (fp16).**
- *Why:* we cannot claim "optimal in-browser" without measuring in a browser. ONNX export +
  onnxruntime-web (wasm-SIMD + WebGPU) latency bench at **fp16** — the real target hardware, closing
  the loop on Tier 0's MPS/CPU proxies. (No int8/GPTQ — see "why not quantization" up top.)
- *Run:* export the Tier-0 winner; measure ms/eval in-browser at batch 1/16; confirm fp16 strength
  delta is negligible. *Cost:* med (new tooling). *Confidence:* high relevance.

**5b. Weight tying / depth recurrence — the size lever (Parameter-Golf borrow).**
- *Why:* the Parameter Golf leaderboard's main *size* trick (`MiniDepthRecurrence`,
  `ProgressiveRecurrence`, `3LayerRecur`, `Loop45x2`): **share weights across repeated blocks** for
  more depth/compute at fewer *unique* params → smaller download at any precision, no reliance on
  in-browser int8. Replaces quantization as our download-shrink lever.
- *Run:* add a "tied/looped" trunk option (a block group applied N times with shared weights);
  compare to the untied net of equal *compute* — does looping cost much strength for the param
  saving? Try **progressive** recurrence (introduce the loop mid-training) and **partial untie** of
  one sub-layer. *Cost:* med. *Confidence:* med — it trades params (download) for nothing in compute,
  so the question is purely how much strength weight-sharing costs.

---

## Tier 4 — the transformer frontier (bigger bet; uncertain at tiny scale, but the long-range + robustness path)

**6. Transformer block → learnable RoPE → nbt-transformer.**
- *Why:* discord + survey — `nbt` + **learnable non-axis-aligned RoPE** is the accepted frontier win
  ("clearest single improvement"); ResTNet reports hybrids help the long-range patterns (ladders,
  cyclic groups) pure-CNNs miss **and** are less cyclic-attack-vulnerable. *Caveat:* discord warns
  small transformers struggle to beat small conv nets at equal **inference time**, and the gains
  lean on Muon — i.e. our small/in-browser scale is where transformers are *weakest*. So:
  exploratory, judged strictly speed-matched.
- *Run:* add a `tfrs` block to the registry; speed-match vs nbt; add learnable RoPE (`tflrs`); then
  `nbttflrs`. Pair with Muon (#7). Decide per ms. *Cost:* high. *Confidence:* med (high upside).
- *Defer from this family:* `qkn`/`trb` (discord: need fused kernels or they're slower) and `ireg16`
  inline registers (natural only once a transformer/sequence trunk exists).

**7. Training-recipe borrows from Parameter Golf (optimizer + EMA + schedule) — promote.**
- *Why:* **Muon is the single most-used lever on the entire Parameter Golf leaderboard** (`MuonWD`,
  `MuonEqR`, `NeoMuon`, `ParallelMuon`), corroborating the discord's "strong early-training speed."
  It's architecture-independent, so it lifts *every* net (conv-nbt included), not just transformers.
- *Run:* Muon for matrix params (Adam for scalars/embeddings), with the top entry's knobs:
  **momentum 0.99 warmed up from 0.92 over ~1500 steps**, **per-param-type LRs** (embeddings ~0.03 /
  matrices ~0.02 / scalars ~0.02), warmdown LR tail. Add **EMA of weights** (cheap, often a free
  bump) and try **orthogonal/spectral init**. Compare loss-vs-steps on a fixed arch. *Cost:* low–med.
  *Confidence:* high (Muon/EMA), med (init/schedule). Also enables #6.
- *Also cheap:* **parallel residuals** (attention/MLP — or our conv branches — in parallel, per
  `ParallelResiduals`/`ParResid`): a slightly faster trunk, worth a speed-matched A/B.

---

## Tier 5 — robustness (strategic differentiator; later)

**8. Cyclic-attack susceptibility, and whether the hybrid helps.**
- *Why:* the survey's central open problem; a small **and robust** fast engine would be a real
  differentiator. ResTNet claims the transformer hybrid is less vulnerable.
- *Run:* probe our nets on known cyclic positions; if #6 is built, compare CNN vs hybrid robustness.
  *Cost:* med–high. *Confidence:* low for the speed goal, but strategically interesting.

---

## Sequencing

Tier 0 (now) → front-load the cheap, **architecture-independent** wins in parallel: **#7 Muon+EMA**
(high-confidence, lifts *every* net — likely the biggest single win here), **#1 RMSNorm**,
**#2 features**, and **#4 Gumbel** (search side) → **#5b weight-tying** for download size + **#5
export** to get real browser numbers → **#3 distillation** when 75G lands → the **#6 transformer +
RoPE** bet only if the conv-nbt frontier plateaus. Robustness (#8) rides on #6. Quantization: dropped.

**Anti-confound rule (from the bake-off lessons):** change one lever at a time, always re-baseline
against the current frontier, and report every net as an **(FLOPs/eval, Elo±CI)** point on one
scale (intrinsic eval + val-loss are pre-filters only, never the verdict). A change "works" only if
it pushes the upper envelope — being better at equal *params* doesn't count if it costs FLOPs.
