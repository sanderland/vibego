# IDEAS — experiment pipeline

A staging area for **speculative architecture/method ideas** worth exploring, before they're ripe
enough to schedule. Distinct from [ROADMAP.md](ROADMAP.md) (the prioritized near-term backlog):
ideas graduate from here into the roadmap once there's a concrete hypothesis and a way to measure
it. Everything is judged by the project goal — pushing the **FLOPs↔Elo frontier** of small,
fast, in-browser-capable Go nets (so: low FLOPs/eval, low batch-1 latency, small params).

Each idea: **what** · **why for us** · **open questions / how to test** · status.

---

## Candidate trunks (alternatives to conv-nbt / softmax-transformer)

These feed ROADMAP Tier-4 (the transformer bet) — cheaper global-mixing primitives that might beat
softmax attention on the FLOPs↔Elo frontier at small scale.

### RWKV-style linear token-mixing
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

## How to add an idea

Append a short entry (what / why-for-us / open questions / status). When an idea has a concrete,
measurable hypothesis and a registry-able implementation, promote it into [ROADMAP.md](ROADMAP.md)
with a tier and graduate it off this list.
