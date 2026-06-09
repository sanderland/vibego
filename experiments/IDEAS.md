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

## Rejected (evidence-backed; don't redo without a new reason)

- **Quantization (int8 / GPTQ / QAT) as a wasm lever.** Nets are already 0.8–3M params (~1.5–6 MB
  fp16); no hard size cap, and in-browser runtimes don't reliably accelerate int8 (WebGPU is
  fp16-centric; wasm int8 quantize/dequantize overhead erases the gain on small conv nets). fp16 is
  the default; the size lever we keep is weight-tying (ROADMAP #5b).
- **Graph search / transposition sharing**, and **`subtreeValueBias`.** Measured only **0.9%
  transpositions** in a 48-visit tree → not worth the complexity / a no-op at our tree sizes.
- **`valueWeightExponent` + recursive value recompute.** Only masked the perspective bug; *hurts* on
  correct evals → reverted to flat-MC.

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

---

## How to add an idea

Append a short entry (what / why-for-us / open questions / status). When an idea has a concrete,
measurable hypothesis and a registry-able implementation, promote it into [ROADMAP.md](ROADMAP.md)
with a tier and graduate it off this list. If you kill an idea, move it to Rejected/Deferred with the
reason rather than deleting it.
