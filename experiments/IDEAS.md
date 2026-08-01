# IDEAS — experiment pipeline

A staging area for **speculative architecture/method ideas** worth exploring, before they're ripe
enough to schedule, plus a **triage bin** for ideas that have been rejected/deferred/down-ranked.
Distinct from [ROADMAP.md](ROADMAP.md) (the prioritized near-term backlog): ideas graduate from here
into the roadmap once there's a concrete hypothesis and a way to measure it; killed ideas stay here
with their reason so they aren't re-litigated. Everything is judged by the project goal — pushing the
**FLOPs↔Elo frontier** of small, fast, in-browser-capable Go nets (low FLOPs/eval, low batch-1
latency, small params).

Each idea: **what** · **why for us** · **open questions / how to test** · status.

---

## Candidate trunks (alternatives to conv-nbt / softmax-transformer)

These feed ROADMAP Tier-4 (the transformer bet) — cheaper global-mixing primitives that might beat
softmax attention on the FLOPs↔Elo frontier at small scale. **Status: low priority / speculative** —
keep parked until the conv-nbt frontier plateaus and #6 is actually pursued; their own caveats below
show no clear win at our N=361, non-causal setting, so they're not near-term roadmap.

> NB: the *near-term* "richer global connection than gpool" path is in **ROADMAP #6** — cheap
> content-dependent routing first (**register tokens**, then **axial attention**), full softmax last,
> all judged speed-matched on ms (at N=361 attention is ~conv-FLOPs; wall-clock is the gate). These
> linear-attention trunks only become interesting if N is *raised* (e.g. `ireg16` register tokens).

### RWKV-style linear token-mixing
- **Status: IMPLEMENTED as a registry block kind (`rwkv`)** — `RWKVResBlock` in `vibego/net/model.py`,
  archs `b6c96-rwkv` / `b10c128-rwkv` (regular conv backbone, every-3rd block = the rwkv mixer,
  swapping gpool's slot). Vision-RWKV-style (Duan et al. 2024): omnidirectional **token shift**
  (`_q_shift`) + a **non-causal global WKV** (per-channel softmax-over-board weighting of v by
  exp(k) with a learned self-bonus `u`) + a **receptance** gate, then RWKV's squared-ReLU channel
  mix. **Spatial decay dropped** (a board has no canonical 1-D order). All 1×1 convs, stays in
  (B,C,H,W). Tests in `tests/test_archs.py`. **Next:** bench FLOPs/CPU-ms (`bench_net.py`, now in
  DEFAULT) + train on RunPod, plot Elo-vs-cost against gpool/nbt. Open: does the global softmax-pool
  capture long-range relations (ladders) without softmax attention's pairwise structure?
- **What:** RWKV ("Receptance Weighted Key Value") — a linear-attention architecture trainable in
  parallel like a transformer but with RNN-like O(1)-per-token inference and **no KV cache**. The
  token-mixing is a cheap linear recurrence; channel-mixing is a gated FFN.
- **Why for us:** the token-mixer is a **low-FLOP global-mixing primitive** — a candidate to replace
  softmax attention in a Go trunk at lower cost, which is exactly the frontier lever. Gated/linear
  mixing may also be friendlier to wasm/WebGPU kernels than softmax.
- **Open questions / caveats:**
  - Go is **non-causal, single-pass over 361 fixed tokens** — RWKV's headline win (KV-cache-free
    *autoregressive* generation) does **not** transfer. The interesting part is only the linear
    token-mixer as a global op. Need a 2D/bidirectional adaptation (RWKV's recurrence is 1D/ordered;
    a board has no canonical order — try bidirectional or a 2D scan, or treat it as set-mixing).
  - Does linear mixing capture long-range Go relations (ladders, cyclic groups) as well as softmax?
  - Test: drop an RWKV-style mixer into the registry as a block kind; **speed-match** vs nbt and vs a
    softmax-transformer block; plot Elo vs FLOPs/eval and CPU-ms.

### Linearized / kernel attention (Performer, linear attention, etc.)
- **Status: IMPLEMENTED as a registry block kind (`linattn`)** — `LinAttnResBlock` in
  `vibego/net/model.py`, archs `b6c96-linat` / `b10c128-linat`. Multi-head linear attention with the
  φ=elu+1 feature map (Katharopoulos et al. 2020), pre-norm attention sublayer + squared-ReLU FFN,
  both residual; **no internal positional encoding** (permutation-equivariant → interspersed with
  conv blocks that supply position). Tests in `tests/test_archs.py`. **Next:** the honesty check
  below — measure FLOPs/eval at N=361 with `bench_net.py` *before* trusting any "cheaper" claim.
- **What:** replace softmax attention's O(N²·d) with linear attention O(N·d²) via kernel feature
  maps (FAVOR+/elu+1/etc.).
- **Why for us:** sub-quadratic attention is the standard "cheaper transformer" lever.
- **Open questions / caveats (important honesty check):**
  - **At N=361 the FLOP win is marginal.** Softmax ≈ N²·d; linear ≈ N·d². For d≈384 these are
    *comparable* (50M vs 53M) — linear only clearly wins when N ≫ d, which a 19×19 board is not.
    So the motivation can't be "fewer FLOPs" by default — **verify the FLOP math at our N first.**
  - Real reasons it might still help: no softmax (cheaper/quantization-friendlier kernels), a
    different inductive bias, or pairing with **inline registers** (extra tokens) to raise N.
  - Test: implement as a registry block kind; **measure FLOPs/eval at our board size** (it may not
    beat softmax here), then Elo-vs-cost only if the cost actually drops.

---

## Borrowed from chess engines + memory-arch survey (2026-06-09)

A subagent dig through **Lc0**, **Stockfish-NNUE**, other open engines (Berserk/Seer/Caissa/…), and
"params-as-memory" LLM archs (Gemma 3n PLE, product-key memory, MoE), filtered for transfer to *our*
setting: a small Go net + PUCT MCTS on the **FLOPs↔Elo / low-visit / wasm** frontier. Chess-only
machinery (alpha-beta pruning, tablebases, draws/contempt, single-piece incrementality) was dropped.
Ranked within each bucket; everything here is **speculative until A/B'd** per the test-ablations rule.

### Arch — cheap global mixing & memory-for-FLOPs (highest value)

#### Smolgen-style low-rank global modulation (Lc0) — a new global-mixing primitive vs gpool
- **Status: IMPLEMENTED (`globmod`)** — `GlobalModBlock` in `vibego/net/model.py`, archs
  `b6c96-globmod` / `b10c128-globmod` (every-3rd slot, swapping gpool). A global board summary +
  per-cell 1×1 conv produce a content-dependent, spatially-varying **FiLM** (γ,β) modulation — richer
  than gpool's additive rank-0 bias, at ~the same cost (`b6c96-globmod` 1.109M / **760.8 MFLOP** vs
  gpool 1.090M / 774). Tests + 120-step CPU smoke pass (eval total 5.02→4.24). **Next:** speed-matched
  arena vs `gpool`/`nbt` on RunPod. (This is the *cheap, low-rank* realization — NOT the dense 361²
  attention-bias, which doesn't scale; see caveat.)
- **External evidence (go-ai-survey):** RestNet finds conv+attention **hybrids help specifically on
  long-range patterns** (`restnet2025`) — validates the interspersed-global-block design (globmod/
  attention among conv blocks, vs pure-attention) and says *where* to look for the win in the arena:
  ladders / long-range life-and-death, not the average position.
- **What:** Lc0's *smolgen* compresses the whole board to a small global summary vector, then a tiny
  per-block dense map turns that into a **content-dependent additive bias on the attention/mixing**
  ("plays ~50% larger for ~10% throughput"). The transferable core: a 361→d bottleneck summary that
  broadcasts a *learned, position-conditioned* interaction back to every cell — strictly richer than
  KataGo gpool's per-channel mean/max (which is content-independent and rank-0).
- **Why for us:** exactly the cheap-global-mixing lever; could beat the gpool block at equal CPU-ms.
- **Test:** add a `smolgen`/`globmod` block — global-pool → dense → **low-rank** per-cell gains/shifts
  (NOT a dense 361² bias), speed-matched A/B vs `gpool` in the arena.
- **Caveat:** chess uses a dense 64×64 bias (64 tokens flatters it); at N=361 the dense pairwise form
  is ~31× bigger and breaks the FLOP budget — **must stay low-rank / 2D-separable**. Source: lczero.org/blog/2024/02/transformer-progress.

#### Local-pattern lookup embedding — "store Go patterns, don't recompute them" (the params-as-memory bet)
- **Status: IMPLEMENTED, step 1 (`pattern_embed`)** — `PatternEmbed` + `ModelConfig.pattern_embed`
  in `vibego/net/model.py`; archs `b6c96-gpool-pat` / `b6c96nbt-pat`. The exact **dihedral(D4)-
  canonical 3×3** {empty/own/opp} table (**K=2862** distinct shapes — own/opp kept distinct, ~2× the
  classical colour-symmetric ~1.4k), scatter-added to the stem, zero-init (no-op until trained).
  **The thesis is confirmed by the FLOP counter:** it adds **+0.275M params (~0.5 MB) at exactly 0
  added FLOPs** (`b6c96-gpool-pat` 1.365M / **774.0** MFLOP = base's 774.0; nbt-pat 1.071M / 561.1 =
  base's 561.1). Tests (canon invariance, zero-init, local sensitivity) + 120-step CPU smoke pass
  (eval total 4.71→4.08). **Next:** arena A/B vs the base arch (does the stored pattern memory buy
  Elo?); then step 2 = **5×5/diamond hashed** window (where it should beat the multi-conv receptive
  field). off-board is padded as empty in this v1 (edge info still comes from the on-board channel).
- **External evidence (go-ai-survey):** AlphaVile finds **representation matters more than the
  backbone** (`alphavile2024`) — the strongest outside support for spending effort on *what the net
  sees* (this lookup, the feature subset) over trunk-block A/Bs, and a caution on how much Elo to
  expect from globmod/linat/rwkv vs gpool. Treat this as the highest-priority of the implemented archs.
- **What:** per cell, map its local N×N (or diamond) configuration to an id and look up a learned
  d-dim embedding; scatter-add into the stem. This is NNUE's first layer (sparse-feature → weight
  column = a memory gather, ~0 MACs) and the classical MoGo/Zen 3×3-pattern table, generalized.
  Param-vs-FLOP math: a *symmetry-canonicalized* 3×3 table ≈ **1.4K rows × C (~0.1 MB), ~0 FLOPs**;
  hashed 5×5/diamond → 2¹⁸ rows (~16 MB fp16). Cost moves to **memory (cheap on wasm; we have
  headroom)**, not FLOPs — directly the user's idea and on the frontier.
- **Why for us:** buys Elo at ≈0 added FLOPs; natural home is the **early layers** (local shape).
- **Test (cheapest-first):** (1) exact dihedral-canonicalized **3×3** embedding (~1.4K rows, hash-free,
  interpretable) scatter-added to the stem, A/B Elo at equal FLOPs; (2) only if it helps, scale the
  *window* to **5×5/diamond hashed** — where a conv would need 2–3 stacked layers (real FLOPs) to see
  that receptive field; (3) `ProductKeyMemory` block only if a learned vocab / >5×5 context is needed.
- **Caveat:** a 3×3 lookup partly duplicates what the first conv already computes ~free — the real win
  is at **5×5+** receptive fields, early but wider-than-one-conv. Global/high layers are the *wrong*
  place (astronomically many configs; that's a compute problem). Sources: NNUE docs; Lample 2019
  (arXiv 1907.05242) + Meta "Memory Layers at Scale" (arXiv 2412.09764); Bouzy 3×3 pattern counts.

#### SCReLU / transcendental-free trunk (Stockfish) — cheap wasm-SIMD polish
- **What:** modern NNUE nets use clipped-ReLU / **squared-clipped-ReLU** — bounded, exp/softmax-free,
  auto-vectorizes cleanly under wasm-SIMD and often *gains* capacity-per-FLOP. Lesson: keep the trunk
  **transcendental-free** (one softmax at the policy head is fine).
- **Test:** activation ablation (GELU/SiLU → ReLU → SCReLU) measuring batch-1 wasm latency + Elo.
- **Caveat:** SF's specific tricks are int8-driven (ignore); we're fp16 — clamp bounds the squared
  range. Polish lever, not a frontier-mover.

#### Phase-bucketed heads (Stockfish "LayerStacks") — more params, ~same FLOPs
- **What:** share the trunk, but keep K small head variants and select one by a cheap scalar (SF: piece
  count). K× head *params*, ~1× per-eval *FLOPs* (only one bucket runs).
- **Why for us:** Go analogue = bucket value/score heads by **game phase / stones-on-board**; cheap-
  per-eval capacity, fits the wasm memory budget.
- **Test:** K=3–4 head buckets keyed by move number, A/B vs single head at matched FLOPs.
- **Caveat:** "stones on board" is a noisier regime selector than chess material; only pays if phases
  genuinely want different weights — verify first.

### Heads

#### Key-dot (attention) policy head (Lc0) — low-FLOP global policy
- **What:** policy logitᵢ = q·kᵢ where kᵢ = dense(cellᵢ) and q = dense(global/smolgen summary); pass
  from dense(q). A cheap, naturally-global head; Lc0 got ~270 Elo of policy over a conv head (chess).
- **Why for us:** better priors matter most at **low visits** (we expand few nodes); pairs with smolgen.
- **Test:** swap the conv policy head for key-dot, speed-matched; measure policy top-1/KL vs teacher +
  arena Elo at 50–150 visits. **Caveat:** drop chess's from→to/promotion machinery (Go move = a point);
  a single global query may underfit local tactics — may need a few query channels or a residual conv logit.
- **External evidence (go-ai-survey):** Chessformer / chessbench train **transformer policies by pure
  supervised learning on static game positions** (`chessformer2024`, `chessbench2024`) — i.e. exactly
  our distillation regime (SL from a static teacher). De-risks both this head and the transformer-trunk
  path (ROADMAP #6): no online self-play needed to train an attention policy well.

#### Score-gated auxiliary selection utility (Lc0 moves-left analogue)
- **What:** Lc0's moves-left head adds a Q-gated utility ("when clearly winning, prefer shorter lines").
  Transferable mechanism (not "moves-left" itself): when `|Q|>τ`, bias child selection using our
  **existing score head** — push margin when winning / complicate (high score-variance) when losing.
- **Why for us:** cleans up Go's shuffling/dame-filling-when-ahead without retraining the net.
- **Test:** add a small `scoreUtility` PUCT term gated on value, magnitude from score; confirm strength
  within arena CI while the score-margin distribution shifts. **Caveat:** keep the effect small.

### Search at low visits (our regime — the richest vein)

#### Dynamic search stopping / adaptive visit budget — Go-validated, top latency lever
- **Status: IMPLEMENTED, v1 (locked-winner stop)** — `MCTS(early_stop=, early_stop_min_frac=)` +
  `_winner_locked` in `vibego/engine/search.py`; reachable via `run_engine.py -early-stop` (so any
  arena/match can use it). Parameter-free and **provably safe**: stops once the top root child's visit
  lead exceeds the visits left to spend (no remaining visit can change the visit-winner). Unit-tested
  + integration-tested. **Next on RunPod:** arena at 16/32/48/64 visits measuring Elo vs *mean* visits
  (latency win). **Still open (higher-variance):** KLD-gain stop and a trained DS-MCTS uncertainty head
  — both stop *earlier* than the safe lock but can mis-stop in sharp life-death (gate on disagreement).
- **What:** stop (or reallocate) search per-position by convergence/uncertainty instead of a fixed cap:
  **KLD-gain** stop (Lc0: halt when the root visit distribution stops moving), **DS-MCTS** ("Learning
  to Stop", AAAI 2021 — *demonstrated on 19×19 Go*, ~2.5× speedup at equal win-rate), **Virtual
  Expansions** (NeurIPS 2022, match strength at <50% search). Cheapest signal needs no net: top1−top2
  visit margin, or policy entropy.
- **Why for us:** *the* batch-1 latency lever, and validated on Go; spends our tiny visit budget where
  it matters → higher Elo-per-eval (the north-star). Cross-links **ROADMAP #4 (Gumbel low-visit)**.
- **Test:** stop rule on (top1−top2) visits / KLD-gain; measure visits-saved at matched 48-visit
  strength; allocate a per-*game* visit pool by a difficulty score. **Caveat:** naive margins stop too
  early in sharp life-death — exempt high-disagreement nodes (see correction-history idea); test in full
  games, not puzzles.

#### Low-visit re-tuning bundle (policy temperature + FPU)
- **What:** the selection knobs that *dominate* at tens of visits, where most children are never
  expanded: **policy-softmax temperature** (Lc0 default 2.2 — a sharp distilled prior over-concentrates
  and starves alternatives) and **FPU** strategy/constant (parent-blend vs Lc0 absolute-pessimistic).
- **Why for us:** cheap, net-agnostic sweeps; optimal values differ from KataGo's high-visit tuning.
- **Test:** sweep policy-temperature × cpuct × FPU specifically at 16/32/48/64 visits. **Caveat:** Lc0's
  numbers are 8×8-specific (don't copy); Go's ~250 branching may make absolute FPU too pessimistic —
  measure. Relates to deferred **`cpuctUtilityStdevScale`** (also a low-visit-only re-test).

#### Small-net → big-net confidence cascade (Stockfish dual-net)
- **What:** run a tiny net on most leaves, **escalate to a bigger net only on decision-critical leaves**
  (revisited / near-root / value near 0.5). Drops *average* leaf FLOPs toward the small net while
  keeping accuracy where it matters — orthogonal to quantization.
- **Why for us:** a per-leaf-cost lever directly on the FLOPs↔Elo frontier.
- **Test:** route small-net-by-default, big-net if `|v_small−0.5|<τ` or visits>k; plot Elo vs avg-FLOPs/leaf.
- **Caveat:** chess gates on cheap material eval; Go has **no cheap reliable static gate**, so we'd gate
  on the small net's own (maybe miscalibrated) uncertainty; adds hot-loop branching.

### Training / distillation (low-risk wins)

#### λ-blend teacher value × game outcome (NNUE "lambda")
- **What:** value target = `λ·v_teacher + (1−λ)·z` (game result), loss in win-prob space — the universal
  NNUE lever. We currently match teacher Q only; blending in the actual outcome is a principled
  regularizer, and λ scheduling trades calibration vs sharpness for low-visit play.
- **Test:** sweep λ∈{1.0,0.8,0.5}+anneal on value/score targets; measure 48-visit strength + calibration.
- **Caveat:** needs real game outcomes (not just static teacher positions) to use the z term.

#### Distillation data filtering + margin balancing (NNUE data curation)
- **What:** drop/down-weight positions where the target is unreliable — operationalized for us as
  **high teacher value-variance** or large **raw-net-vs-search disagreement** (ladders/semeai/ko) — and
  **balance the score-margin distribution** (Go nets miscalibrate badly when far ahead/behind).
- **Test:** filter on `|v_raw−v_search|>τ` + resample to a margin histogram; compare student value/score
  MAE + calibration. **Caveat:** thresholds are chess-specific (retune); over-filtering removes the hard
  positions the student most needs — test both directions.

#### Reanalysis — re-label the distill set as the teacher/student improves (EfficientZero/ReZero)
- **What:** MuZero-lineage sample-efficiency lever (go-ai-survey): **EfficientZero**'s self-supervised
  consistency loss and **ReZero**'s backward-view *reanalysis* both squeeze more out of fixed data by
  recomputing targets (`efficientzero2021`, `rezero2024`). Our analogue: as the teacher improves (or
  the 75G relabel finishes / a stronger teacher lands), **re-relabel** existing positions rather than
  only collecting new ones — and consider a consistency loss between a position and its post-move
  successor's predicted value.
- **Why for us:** distillation is already our most data-efficient lever (~70× vs npz outcomes); reanalysis
  compounds it — more Elo per GB of relabel compute, which is the actual bottleneck on RunPod.
- **Test:** relabel a slice with teacher vN+1, retrain, compare to training on the vN labels at equal
  positions. **Caveat:** reanalysis pays off most when targets are still moving (early teacher / active
  self-play); for a frozen strong teacher it reduces to "use the best teacher once."

#### Terminal-anchored targets (Seer's EGTB idea, Go-native)
- **What:** near game end, blend in **exact** ground truth instead of teacher logits — Tromp-Taylor
  score + Benson-unconditional life/death for ownership — a noise-free signal precisely where teacher
  value is noisiest and low-visit search most often misreads dead/alive groups.
- **Test:** for near-terminal / few-empty positions, set λ→1 toward exact score+ownership; ablate value/
  ownership L2 on a held-out endgame set. **Caveat:** Go has no full-board exact tablebase, so only the
  *endgame anchoring* transfers (not Seer's retrograde back-up loop); bounded to the scoring phase.

---

## Rejected (evidence-backed; don't redo without a new reason)

- **Quantization (int8 / GPTQ / QAT) as a wasm lever.** Nets are already 0.8–3M params (~1.5–6 MB
  fp16); no hard size cap, and in-browser runtimes don't reliably accelerate int8 (WebGPU is
  fp16-centric; wasm int8 quantize/dequantize overhead erases the gain on small conv nets). fp16 is
  the default; the size lever we keep is weight-tying (ROADMAP #5b).
- **Quantization of *released KataGo* nets (the v1.17 transformers), as a speed lever.** Separate
  question from the wasm one above, same answer by a different route: KataGo's CUDA/Metal/OpenCL
  backends have no int8 kernels, so a quantized `.bin.gz` loads as floats and runs at *identical*
  speed -- accuracy loss with no speed win to weigh it against. The int8 win lives in TensorRT
  tensor cores and needs calibration inside KataGo's C++ backend (a KataGo PR, not a vibego
  experiment); a prior attempt is [KataGo#799](https://github.com/lightvector/KataGo/issues/799)
  ("winrates a lot different from fp16", no methodology, no follow-up). See
  [2026-08-01-katago-net-compression](2026-08-01-katago-net-compression.md).
- **Weight-only structural pruning of the v1.17 transformer nets, without healing.** Measured:
  heads exactly tile the bottleneck (6x32 = 192 = c_mid), the SwiGLU FFN is at the standard 8/3
  ratio, and all 10 trunk blocks cost identical FLOPs -- the nets are dense by construction. The
  cheapest available edit (-4.4% FLOPs) already costs 0.58 scoreLead; heads are the worst lever,
  FFN width the best. Not dead, but only worth revisiting with **activation-aware selection plus a
  distill heal** -- which is a different experiment (compress-down vs train-up), not this one.
- **Graph search / transposition sharing**, and **`subtreeValueBias`.** Measured only **0.9%
  transpositions** in a 48-visit tree → not worth the complexity / a no-op at our tree sizes.
- **`valueWeightExponent` + recursive value recompute.** Only masked the perspective bug; *hurts* on
  correct evals → reverted to flat-MC.
- **NNUE-style incremental leaf eval** (reuse the parent's first-layer activation, update only the
  changed inputs when a child adds one stone). Verdict from the SF dig: **doesn't transfer to a conv
  trunk** — (1) conv receptive-field spread dirties a growing patch each layer (and *all* pooled head
  stats), so "one stone = local delta" dies after layer 1; (2) a Go capture changes a whole region
  (captured group + neighbors' liberties), not chess's single-piece delta; (3) only the (cheapest)
  first layer is updatable anyway, and MCTS re-evaluates every child's full net regardless. *Cheap
  falsification if ever revisited:* a ~1h numpy check measuring what fraction of activations actually
  differ between a parent and a one-stone child (no-capture + capture) — if mid-block dirtiness is
  >~15–20% it's dead on arrival. Only worth reconsidering with a *sparse-linear* (non-conv) front-end,
  which would forfeit the conv inductive bias we rely on for per-FLOP strength.

## Deferred (revisit when the prerequisite lands)

- **`cpuctUtilityStdevScale`.** A wash at 48 visits — re-test only at a very different visit budget
  (e.g. the Gumbel low-visit regime, ROADMAP #4).
- **Transformer `qkn` / `trb` blocks.** KataGo discord: need fused kernels or they're slower than
  conv at equal inference time. Revisit only with a fused-kernel path.
- **`ireg16` inline registers.** Natural only once a transformer/sequence trunk exists (after #6) —
  and the lever that could make linear attention (above) worthwhile by raising N.

## Low priority (plausible, not worth the current budget)

- **Cyclic-attack robustness (ROADMAP #8).** Strategically interesting (a small *and* robust engine
  is a differentiator) but low relevance to the speed/Elo goal; rides on the transformer hybrid (#6).
- **Parallel residuals** (attn/MLP or conv branches in parallel): a slightly faster trunk — keep as a
  cheap speed-matched A/B *inside* the Muon/arch work (ROADMAP #7), not its own line.
- **Per-cell MoE / expert routing.** Memory-not-FLOPs in principle, but at **batch-1, 361 tokens,
  single-thread wasm** the routing softmax + scatter/gather + loss of dense-conv vectorization likely
  costs *more* wall-clock than it saves, and a 1–3M-param net has too little to shard. If ever tried,
  use *fixed position-based* routing (corner/edge/center weights — no learned router, no FLOPs) rather
  than true MoE. (From the Gemma/MoE dig.)
- **Gemma 3n Per-Layer Embeddings (PLE) as-such.** Its whole point is the phone VRAM/CPU-RAM split —
  **irrelevant on wasm's single heap**. The transferable sub-case (a per-layer pattern-embedding table)
  is just the local-pattern lookup idea applied at multiple depths; pursue that, not PLE.
- **MatFormer / nested elastic submodels** (Gemma 3n): off the memory-for-FLOPs thesis, but a genuine
  *separate* idea — train one checkpoint from which a **family of nets at different FLOP budgets** can be
  extracted (one train, ship-many across the Pareto frontier). Park as its own line; only if we want a
  single checkpoint spanning the frontier.

---

## How to add an idea

Append a short entry (what / why-for-us / open questions / status). When an idea has a concrete,
measurable hypothesis and a registry-able implementation, promote it into [ROADMAP.md](ROADMAP.md)
with a tier and graduate it off this list. If you kill an idea, move it to Rejected/Deferred with the
reason rather than deleting it.
