"""Structurally prune a released KataGo net and write out a net the stock engine still loads.

Everything here stays inside the model file format, so the output is measurable immediately with
the existing harness -- no torch, no new inference path, and the FLOPs genuinely go away:

    KATA="katago analysis -config $CFG -model"
    uv run python scripts/kata_prune.py models/b10c384h6nbttflrs.bin.gz \
        --heads-keep 0.67 --out models/pruned-h4.bin.gz
    uv run python scripts/policy_eval.py --src data --n 400 \
        --ref  "$KATA models/kata1-b18c384nbt.bin.gz" \
        --engine "full=$KATA models/b10c384h6nbttflrs.bin.gz" \
        --engine "pruned=$KATA models/pruned-h4.bin.gz"

Levers (combinable in one pass; applied in the order listed):

    --drop-blocks 3,7        remove whole trunk blocks             (coarsest: 10% of depth each)
    --drop-inner-pairs 4:1   remove 1 attention+FFN pair from nbt block 4
    --heads-keep 0.67        keep the top 2/3 of attention heads   (per block, uncoupled)
    --ffn-keep 0.75          keep the top 3/4 of FFN hidden units  (finest-grained)

Which units to drop is chosen by a weight-only magnitude proxy (see `vibego/katago/prune.py`);
that is the floor of what pruning can do, not the ceiling. `--dry-run` reports the cost delta
without writing anything.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.katago.binmodel import read_model, write_model  # noqa: E402
from vibego.katago.cost import arch_summary, model_cost  # noqa: E402
from vibego.katago.prune import (  # noqa: E402
    PruneError,
    drop_blocks,
    drop_inner_pairs,
    narrow_ffn_everywhere,
    prune_heads_everywhere,
    sanitize_name,
)


def _int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="input .bin.gz")
    ap.add_argument("--out", help="output .bin.gz (required unless --dry-run)")
    ap.add_argument("--drop-blocks", type=_int_list, default=[],
                    help="comma-separated trunk block indices to remove")
    ap.add_argument("--drop-inner-pairs", action="append", default=[], metavar="BLOCK:N",
                    help="remove N attention/FFN pairs from nbt block BLOCK (repeatable)")
    ap.add_argument("--heads-keep", type=float,
                    help="fraction of attention heads to keep in every attention block")
    ap.add_argument("--ffn-keep", type=float,
                    help="fraction of FFN hidden units to keep in every FFN block")
    ap.add_argument("--name-suffix", help="suffix for the model name recorded in the file "
                                          "(default: derived from the pruning options)")
    ap.add_argument("--board", type=int, default=19, help="board size for the FLOPs report")
    ap.add_argument("--dry-run", action="store_true", help="report the cost delta, write nothing")
    args = ap.parse_args()

    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")
    if not (args.drop_blocks or args.drop_inner_pairs or args.heads_keep or args.ffn_keep):
        ap.error("nothing to do: pass at least one of --drop-blocks/--drop-inner-pairs/"
                 "--heads-keep/--ffn-keep")

    model = read_model(args.model)
    before = model_cost(model, board=args.board)
    print(f"in:  {arch_summary(model)}")
    print(f"     {before.total.params:,} params, {before.total.flops / 1e6:,.1f} MFLOP/eval "
          f"@{args.board}x{args.board}")

    records = []
    try:
        # Depth first: dropping a block makes any later per-block index meaningless otherwise.
        for spec in args.drop_inner_pairs:
            block_str, _, count_str = spec.partition(":")
            records.append(drop_inner_pairs(model, int(block_str), int(count_str or 1)))
        if args.drop_blocks:
            records.append(drop_blocks(model, args.drop_blocks))
        if args.heads_keep:
            records += prune_heads_everywhere(model, args.heads_keep)
        if args.ffn_keep:
            records += narrow_ffn_everywhere(model, args.ffn_keep)
    except PruneError as e:
        sys.exit(f"error: {e}")

    after = model_cost(model, board=args.board)
    print(f"\napplied {len(records)} edits:")
    shown = records[:6]
    for rec in shown:
        print(f"  {rec.op:<18} {rec.detail}")
    if len(records) > len(shown):
        print(f"  ... and {len(records) - len(shown)} more of the same kind")

    d_params = 100.0 * (after.total.params / before.total.params - 1)
    d_flops = 100.0 * (after.total.flops / before.total.flops - 1)
    print(f"\nout: {arch_summary(model)}")
    print(f"     {after.total.params:,} params ({d_params:+.1f}%), "
          f"{after.total.flops / 1e6:,.1f} MFLOP/eval ({d_flops:+.1f}%)")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    suffix = args.name_suffix or _default_suffix(args)
    model.name = sanitize_name(model.name, suffix)
    write_model(model, args.out)
    print(f"\nwrote {args.out} (model name: {model.name})")
    print("Strength is NOT implied by these numbers -- measure it: policy_eval.py for raw "
          "agreement, then match.py at fixed visits with a neutral judge.")


def _default_suffix(args) -> str:
    bits = []
    if args.drop_blocks:
        bits.append("drop" + "-".join(str(i) for i in args.drop_blocks))
    if args.drop_inner_pairs:
        bits.append("pairs" + "-".join(s.replace(":", "x") for s in args.drop_inner_pairs))
    if args.heads_keep:
        bits.append(f"h{int(round(args.heads_keep * 100))}")
    if args.ffn_keep:
        bits.append(f"ffn{int(round(args.ffn_keep * 100))}")
    return "-".join(bits) or "pruned"


if __name__ == "__main__":
    main()
