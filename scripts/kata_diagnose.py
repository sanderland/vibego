"""Does a released KataGo net have any structural slack, and does knowing the activations help?

Four diagnostics plus one head-to-head, all on real positions from a training-data shard (exact V7
features, so nothing is reconstructed):

  1. trunk effective rank   -- is the c-wide residual stream really c-dimensional? If yes, trunk
                               width pruning is dead at any quality, whatever the criterion.
  2. per-block angular      -- how far each block rotates the residual stream. The LLM depth-drop
     distance                  literature drops the blocks that rotate it least; this says whether
                               any block here qualifies.
  3. head importance        -- per-head activation-weighted magnitude, and how unequal it is.
  4. weight spectra         -- singular values of each weight matrix: is low-rank factorization
                               even plausible (which would mean leaving the file format)?

  5. criterion head-to-head -- prune to the same FLOP budget with the weight-only criterion and
                               with the activation-aware one, then measure the damage each does
                               against the *parent net* on held-out positions. This is the number
                               that says whether better selection rescues post-hoc pruning.

    uv run python scripts/kata_diagnose.py --model models/b10c384h6nbttflrs.bin.gz \
        --npz katago/python/testdata/benchmark_data_1024.npz --n 128 --eval-n 128
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.katago.binmodel import NestedBottleneckBlock, read_model  # noqa: E402
from vibego.katago.calibrate import (  # noqa: E402
    axis_aligned_concentration,
    effective_rank,
    load_calibration,
    run_calibration,
    weighted_ffn_importance,
    weighted_head_importance,
)
from vibego.katago.cost import model_cost  # noqa: E402
from vibego.katago.prune import (  # noqa: E402
    drop_rope_pairs_everywhere,
    ffn_importance,
    head_importance,
    iter_attention_blocks,
    iter_ffn_blocks,
    low_rank_value_everywhere,
    narrow_ffn_everywhere,
    prune_heads_everywhere,
)
from vibego.katago.torchmodel import KataTorchModel  # noqa: E402


def evaluate(net: KataTorchModel, spatial, glob, batch: int):
    policies, leads, winrates = [], [], []
    with torch.no_grad():
        for i in range(0, len(spatial), batch):
            out = net(spatial[i:i + batch], glob[i:i + batch])
            policies.append(net.policy(out).numpy())
            leads.append(net.score_lead(out).numpy())
            winrates.append(net.winrate(out).numpy())
    return np.concatenate(policies), np.concatenate(leads), np.concatenate(winrates)


def damage(parent, child) -> dict:
    """How far a pruned net has moved from its parent, on the axes that matter."""
    p_pol, p_lead, p_win = parent
    c_pol, c_lead, c_win = child
    p = np.clip(p_pol, 1e-12, None)
    c = np.clip(c_pol, 1e-12, None)
    kl = (p * np.log(p / c)).sum(axis=1)
    return {
        "top1": float(np.mean(p_pol.argmax(1) == c_pol.argmax(1))),
        "kl": float(np.mean(kl)),
        "d_lead_abs": float(np.mean(np.abs(c_lead - p_lead))),
        "d_lead_mean": float(np.mean(c_lead - p_lead)),
        "d_winrate_abs": float(np.mean(np.abs(c_win - p_win))),
    }


def spectra_report(model) -> None:
    """Singular-value decay of the biggest weight matrices, as an energy-based rank."""
    print("\n[4] weight matrix spectra -- rank needed for 90/99% of the spectral energy")
    print(f"  {'matrix':<34}{'shape':>14}{'r90':>7}{'r99':>7}{'r99/full':>10}")
    rows = []
    for block in model.trunk.blocks:
        subs = block.blocks if isinstance(block, NestedBottleneckBlock) else [block]
        for sub in subs:
            for label in ("q_proj", "k_proj", "v_proj", "out_proj", "linear1", "linear_gate",
                          "linear2"):
                layer = getattr(sub, label, None)
                if layer is None or not hasattr(layer, "weight") or layer.weight.ndim != 2:
                    continue
                rows.append((f"{sub.name.split('.', 2)[-1]}.{label}", layer.weight))
        break  # one representative trunk block is enough; they are structurally identical
    for name, w in rows:
        sv = np.linalg.svd(w.astype(np.float64), compute_uv=False)
        energy = np.cumsum(sv ** 2) / (sv ** 2).sum()
        r90 = int(np.searchsorted(energy, 0.90) + 1)
        r99 = int(np.searchsorted(energy, 0.99) + 1)
        full = min(w.shape)
        print(f"  {name:<34}{str(w.shape):>14}{r90:>7}{r99:>7}{r99 / full:>9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--n", type=int, default=128, help="calibration positions")
    ap.add_argument("--eval-n", type=int, default=128, help="held-out positions for the head-to-head")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--skip-heads", action="store_true",
                    help="omit the head-pruning arm (already shown to be hopeless)")
    ap.add_argument("--keep", type=float, nargs="*", default=[0.75, 0.5],
                    help="keep fractions to test in the criterion head-to-head")
    args = ap.parse_args()

    model = read_model(args.model)
    net = KataTorchModel(model).eval()
    base_cost = model_cost(model)
    print(f"{os.path.basename(args.model)}: {base_cost.total.params:,} params, "
          f"{base_cost.total.flops / 1e6:,.0f} MFLOP/eval")

    spatial, glob = load_calibration(args.npz, args.n + args.eval_n, seed=0)
    cal_spatial, cal_glob = spatial[:args.n], glob[:args.n]
    ev_spatial, ev_glob = spatial[args.n:], glob[args.n:]
    print(f"calibration {len(cal_spatial)} positions, held-out {len(ev_spatial)} positions")

    cal = run_calibration(net, cal_spatial, cal_glob, batch=args.batch)

    # 1 -------------------------------------------------------------------------------
    er = effective_rank(cal.trunk_singular_values)
    print(f"\n[1] trunk residual stream: {er['dims']} channels")
    print(f"  participation ratio      {er['participation_ratio']:.1f}"
          f"   ({100 * er['participation_ratio'] / er['dims']:.0f}% of full width)")
    print(f"  dims for 90 / 99 / 99.9% of variance   "
          f"{er['dims_for_90pct']} / {er['dims_for_99pct']} / {er['dims_for_999pct']}")
    ax = axis_aligned_concentration(cal.trunk_channel_variance)
    print(f"  in the CHANNEL basis (the only one pruning can use):")
    print(f"    participation ratio    {ax['participation_ratio']:.1f}"
          f"   ({100 * ax['participation_ratio'] / ax['channels']:.0f}% of full width)")
    print(f"    channels for 90 / 99% of variance   "
          f"{ax['channels_for_90pct']} / {ax['channels_for_99pct']}")
    print(f"    channels under 1% of the busiest    "
          f"{100 * ax['frac_below_1pct_of_max']:.1f}%")

    # 2 -------------------------------------------------------------------------------
    print("\n[2] per-block residual-stream rotation (1.000 = block changes nothing)")
    ordered = sorted(cal.block_angles.items(), key=lambda kv: -kv[1])
    print(f"  {'block':<40}{'cos(in, out)':>14}{'||res||/||in||':>16}")
    for name, cos in ordered[:6]:
        print(f"  {name:<40}{cos:>14.4f}{cal.block_relative_norm[name]:>16.4f}")
    print("  ...")
    for name, cos in ordered[-3:]:
        print(f"  {name:<40}{cos:>14.4f}{cal.block_relative_norm[name]:>16.4f}")
    cosines = np.array([c for _, c in ordered])
    print(f"  range {cosines.min():.4f} .. {cosines.max():.4f}   "
          f"(a block worth dropping would sit near 1.0)")

    # 3 -------------------------------------------------------------------------------
    print("\n[3] attention head importance (activation-weighted), per block")
    act_heads = weighted_head_importance(model, cal)
    spreads, agreements = [], []
    for _, block in iter_attention_blocks(model):
        act = act_heads[block.name]
        wgt = head_importance(block)
        spreads.append(act.max() / act.min())
        agreements.append(np.corrcoef(np.argsort(np.argsort(act)),
                                      np.argsort(np.argsort(wgt)))[0, 1])
    spreads = np.array(spreads)
    print(f"  max/min head importance within a block: median {np.median(spreads):.2f}, "
          f"worst {spreads.max():.2f}")
    print(f"  rank correlation, activation-aware vs weight-only: "
          f"median {np.median(agreements):+.2f}")
    ffn_spreads = []
    act_ffn = weighted_ffn_importance(model, cal)
    for _, block in iter_ffn_blocks(model):
        a = act_ffn[block.name]
        ffn_spreads.append(float(np.sum(a < 0.1 * np.median(a)) / len(a)))
    print(f"  FFN hidden units below 10% of their block's median importance: "
          f"{100 * np.mean(ffn_spreads):.1f}%")

    # 4 -------------------------------------------------------------------------------
    spectra_report(model)

    # 5 -------------------------------------------------------------------------------
    print(f"\n[5] criterion head-to-head on {len(ev_spatial)} held-out positions "
          f"(damage vs the unpruned parent)")
    parent = evaluate(net, ev_spatial, ev_glob, args.batch)
    print(f"  {'variant':<28}{'MFLOP':>9}{'dFLOP':>8}{'top1':>7}{'KL':>9}"
          f"{'|d lead|':>10}{'|d winrate|':>13}")
    variants = []
    for keep in args.keep:
        variants.append((f"ffn {keep:.2f} weight-only", "ffn", keep, None))
        variants.append((f"ffn {keep:.2f} activation", "ffn", keep, act_ffn))
    for keep in args.keep:
        # The in-format low-rank lever: v_head_dim is a header field, so shrinking the dimension
        # between v_proj and out_proj IS a low-rank factorization of each head's value->output map.
        variants.append((f"v_dim {keep:.2f} plain-SVD", "vdim", keep, None))
        variants.append((f"v_dim {keep:.2f} data-aware", "vdim", keep, cal.input_moment))
        # The query path's counterpart: keep the highest-energy RoPE frequency pairs.
        variants.append((f"q_dim {keep:.2f} data-aware", "qdim", keep, cal.input_moment))
        variants.append((f"q+v {keep:.2f} data-aware", "qv", keep, cal.input_moment))
    if not args.skip_heads:
        for keep in args.keep:
            variants.append((f"heads {keep:.2f} weight-only", "heads", keep, None))
            variants.append((f"heads {keep:.2f} activation", "heads", keep, act_heads))

    for label, kind, keep, importance in variants:
        pruned = copy.deepcopy(model)
        if kind == "ffn":
            narrow_ffn_everywhere(pruned, keep, importance)
        elif kind == "vdim":
            low_rank_value_everywhere(pruned, keep, importance)
        elif kind == "qdim":
            drop_rope_pairs_everywhere(pruned, keep, importance)
        elif kind == "qv":
            drop_rope_pairs_everywhere(pruned, keep, importance)
            low_rank_value_everywhere(pruned, keep, importance)
        else:
            prune_heads_everywhere(pruned, keep, importance)
        cost = model_cost(pruned)
        child = evaluate(KataTorchModel(pruned).eval(), ev_spatial, ev_glob, args.batch)
        d = damage(parent, child)
        print(f"  {label:<28}{cost.total.flops / 1e6:>9,.0f}"
              f"{100 * (cost.total.flops / base_cost.total.flops - 1):>7.1f}%"
              f"{d['top1']:>7.2f}{d['kl']:>9.4f}{d['d_lead_abs']:>10.2f}"
              f"{d['d_winrate_abs']:>13.4f}")


if __name__ == "__main__":
    main()
