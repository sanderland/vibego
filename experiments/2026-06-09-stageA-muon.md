# 2026-06-09 — Stage-A screening: Muon beats AdamW (training axis)

First run of the screening funnel (`scripts/screen.py` batch `a0`): 9 configs trained concurrently
(8-way GPU-pack → ~97% util vs ~33% for one ~1M net), 8k steps, batch 256, on a fixed 48-shard
subset of the b18-distilled 75G set (`/workspace/distill/screen`), val-loss as the pre-filter.

## Result — Muon is a clean win (ROADMAP #7 confirmed)

All four Muon variants beat both AdamW variants on the **same arch (b7c106nbt / dw7) and FLOPs**:

| config | optimizer | val_loss |
|---|---|---|
| t_muon_lr04 | Muon lr 0.04 | **3.122** |
| t_muon_ema  | Muon lr 0.02 + EMA | 3.124 |
| t_muon_lr01 | Muon lr 0.01 | 3.136 |
| t_muon      | Muon lr 0.02 | 3.146 |
| t_adamw_ema | AdamW + EMA | 3.207 |
| t_adamw     | AdamW lr 2e-3 | 3.220 |

- **~0.09 val-loss from the optimizer alone**, architecture-independent → adopt **Muon (lr 0.04)** as
  the default for all subsequent runs. Biggest single lever so far.
- **EMA ≈ neutral at 8k steps** (3.124 vs 3.122; 0.999 decay ≈ 1k-step window — marginal at this
  length). Revisit EMA only for the longer Stage-C runs.

## Caveats / next

- **Val-loss, not Elo.** Per the project lesson (policy_eval ≠ game strength), screening only ranks
  candidates; the frontier verdict is Stage-B arena Elo.
- Arch axis in `a0` ran on AdamW; re-screening the arch ladder + the new `rwkv`/`linattn` global-
  mixing blocks **under Muon** is batch `a1` (in flight).
- All 8k-step undertrained — relative screening, not absolute strength.
