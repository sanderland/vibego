# KataGo dev-channel experiment notes (Discord summary)

Distilled from KataGo Discord discussion (≈Aug 2024 → Jun 2026), mainly **lightvector** (David Wu,
KataGo author) and **hzy_sigmoid** (trainer of the strong `zhizi`/`fdx6d` nets), with CCY and
others. Captures the architecture frontier *past* the nbt nets that `vibego` currently mirrors:
the move toward **transformers**, **learnable RoPE**, and **nbt-transformer hybrids**. Treat
numbers as informal lab notes, not a controlled paper.

---

## Net naming cheat-sheet

`b<N>` blocks · `c<N>` trunk channels · `h<N>` attention heads. Suffix flags stack:

| token | meaning |
|-------|---------|
| `tf` / `tfrs` | transformer block (with RoPE positional encoding) |
| `tflrs` | transformer block with **learnable, non-axis-aligned RoPE** (`lr` = learnable rope) |
| `nbt` | **nested bottleneck**: pairs of blocks wrapped in 1×1 convs that **double the trunk channels** at ~equal inference cost |
| `cnorm` | RMSNorm (channel-wise) at end of trunk |
| `bnorm` | BatchNorm at end of trunk |
| `qkn` / `qk` | query-key norm · `trb` trunk residual backout · `ireg16` inline registers (+16 board positions) |
| `kv3 qk32 v16`, `fson`, `silu`, `rsnh`, `bnh` | attention-shape / activation / norm variants on test models |

Example: `b9c768h12nbttflrs` = 9 nbt-grouped transformer blocks, 768 trunk channels, 12 heads,
learnable RoPE.

---

## The big picture

- The architecture has moved **conv-nbt → transformer (`tfrs`) → nbt-transformer (`nbttflrs`)**.
  The current sweet spot lightvector lands on is **`b9c768h12nbttflrs`** (≈ identical in perf to
  `b15c512h8nbttflrs`); it learns slower early but overtakes. Equivalent to taking
  `b21c384h12tfrs`, adding learnable RoPE, then **sacrificing 3 blocks to double trunk channels**
  at the same inference cost.
- **Two ideas are accepted as clear wins: `nbt` and learnable RoPE.** Several others tested at
  small scale did *not* survive or were deferred.
- hzy_sigmoid's stance is stronger: *"preliminary experiments suggest transformer-based models are
  stronger than nbt."* lightvector is more cautious — at equal **inference cost** the gain is
  smaller and is mostly nbt + learnable RoPE rather than "transformer" per se.
- Community mood (ksnznz): after nbt-transformers, new-architecture gains are hard to find;
  **scaling size** looks like the main lever going forward.

---

## What survived vs. what was deferred (lightvector, Jun 2026)

**Accepted:**
- **nbt (nested bottleneck).** Doubles trunk channels — the key feature. Intuition: *"trunk
  channels are a contested resource"* and Go nets are *"hungry for parameter space in the main
  net,"* so doubling trunk width helps a lot. In the rope-vs-nbt breakdown: **nbt drives most of
  the value-loss improvement**, less of policy.
- **Learnable non-axis-aligned RoPE.** The clearest single improvement. Old RoPE was axis-aligned
  (queries could only be grid/box-shaped, a product of X and Y queries); learnable RoPE lets each
  head pick arbitrary frequencies *and directions*. **Drives the larger share of policy-loss
  improvement.** Nearly free on CUDA if cos/sins are inlined and reads coalesced — apparent costs
  are framework graph-optimization artifacts, not fundamental.

**Deferred (help a little, but need fused kernels or they're slow):**
- **qkn (query-key norm)** and **trb (trunk residual backout)** — slightly better even at the cost
  of a block (15→14), but without fused kernels the global-memory round-trip makes them easy to
  end up *slower*.

**Worth more experimentation:**
- **ireg16 (inline registers)** — add extra non-board positions (361 → 377 on 19×19). Worth ~1
  extra block on **value loss**, and easy to implement efficiently (kernels just see seq-len 377).

**Normalization:**
- **RMSNorm can replace BatchNorm** at the trunk end. Works fine, slightly better policy loss;
  worse value loss very early but catches up/surpasses once LR shrinks (early gap is a BatchNorm
  mean-centering transient). lightvector dislikes BatchNorm and is happy to drop it.

---

## Transformer vs. nbt — strength & speed (hzy_sigmoid data points)

> Informal, partly un-controlled; mixes Muon-trained transformers vs SGD-trained CNN controls.

**Strength at fixed parameter count** (Muon-trained transformers):

| params | transformer | CNN/nbt control |
|--------|-------------|-----------------|
| 1.3M (`b11c96h3tfrs`) | ~12,200 Elo | ~12,100 |
| 6M (`b14c192h6tfrs`) | ~13,300 Elo | ~13,000–13,100 |
| 35M (`b18c384h12tfrs`) | ~14,050–14,100 (WIP, est. 14,200–14,300 done) | — |

- hzy: *"transformers significantly outperform nbt at the same parameter count; ~half the params
  can match an nbt."* The 35M transformer should beat the official **73M `b28c512nbt`**, and runs
  10–20% faster — but no Muon-trained b28 control exists, so a Muon b28 might close much of the gap.
- **Caveat lightvector/hzy both flag:** much of the early transformer edge was the **Muon
  optimizer** (strong early progress), not the architecture. At **equal inference time** the
  transformer is only *slightly* stronger (≈0–200 Elo).

**Speed (same param count):**
- `b28c512nbt` ≈ **4200 nnevals/s** on RTX 5090.
- Transformer ≈ **60%** of nbt throughput at equal params (≈1.6× slower); a 40-layer/384-dim/4-head
  RoPE transformer ≈ **50%** of `b28c512nbt` speed (both ~70M).

So the honest framing: at **equal params** transformers win clearly; at **equal inference cost**
(what actually matters for play) the advantage shrinks to "nbt + learnable RoPE is a clear win, the
rest is marginal."

---

## Sizing methodology notes (important for fair comparisons)

- To test **nbt alone**, you must **re-equalize compute** by adjusting blocks/channels — otherwise
  you're comparing different FLOP budgets. (`b18nbt` was deliberately sized slightly *faster* than
  `b40`, so naïvely adding mish to b40 would hand it an unfair compute advantage.)
- C++-backend timings matter: the apples-to-apples non-nbt match for `b9c768h12nbttflrs` is
  `b21c384h12tfrs`, **not** `b20c384h12tfrs` (the latter was a pytorch-compile timing artifact).
  nbt+rope still wins by more than one block could close.
- Meta-pattern (lightvector): *"dilute cost by amortizing across more stuff"* has paid off twice —
  bottleneck overhead amortized by more layers per bottleneck (nbt), and permute overhead
  amortized by more convs per permute (dilated convs).

---

## Engineering / versioning (this is the actionable part for downstream code)

- **Model file format bumped to version 17.** Transformers and any net with the "RMSNorm tip"
  now export as **v17**; the C++ loader was updated to read/handle v17 and **distinguish it from
  the old version**, and the Python export scripts were updated to match. A v17 re-export of the
  transformer is available (`b10c384h6nbttflrs-v17.bin.gz`).
- **Eigen reference-output generation** in progress for these test models:
  - `tests/models/b7c96h3tfrs-test5-cnorm.bin.gz`
  - `tests/models/b4c256h4nbttflrs-fson-silu-rsnh.bin.gz`
  - `tests/models/b7c96h6kv3qk32v16tflrs-fson-bnh.bin.gz`
- Reference: `model_pytorch.py` (block modules) and `modelconfigs.py` (note: contains both
  in-use *and* experimental/discarded configs) are the source of truth for what each suffix means.

---

## References

- nbt definition — **KataGoMethods.md**, "Nested Bottleneck Residual Nets" section
  (`github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md`).
- **Attention residuals paper**, arXiv:2603.15031 — cited re: trunk channels as a contested
  resource (lets blocks attend directly to earlier blocks); in lightvector's tests too expensive
  for these small nets.
- Sayuri uses **Gumbel policy optimization** for low-visit self-play targets (more principled than
  plain MCTS + noise) — relevant to cheap/low-visit training.
- Training cost (rough, Fulgore): full `b28c512nbt` run ≈ **4×10²⁰ FLOPs**.

---

## Relevance to `vibego`

- Our bake-off confirms the **nbt direction** at small scale (every nbt net beat its regular
  counterpart) — consistent with lightvector's "accept nbt." The doubled-trunk-channel intuition
  ("trunk channels are contested") is *why*, and matches our depth-vs-width study's premise.
- **Two cheap, high-value things to consider adding** to vibego's registry next, both flagged here
  as wins that are easy to implement: **learnable RoPE** (needs a transformer block first) and
  **inline registers / extra tokens** (worth ~1 block of value-loss for trivial cost).
- **RMSNorm-over-BatchNorm** is a free swap worth trying — vibego currently uses BatchNorm
  throughout; lightvector reports RMSNorm is equal-or-better and avoids BatchNorm's batch coupling.
- For honest comparisons, copy their **equal-inference-cost (not equal-param)** discipline and use
  **C++/real-backend timings**, not framework forward-pass times.
- Bigger picture: a full **transformer + learnable-RoPE + nbt** trunk is the current KataGo
  frontier; if vibego ever moves past conv-nbt, `b9c768h12nbttflrs`-style is the reference recipe —
  but note the gains over nbt are modest at equal inference cost and lean heavily on the **Muon
  optimizer** for early-training speed.
