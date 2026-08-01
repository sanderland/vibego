"""Open a KataGo `.bin.gz` and report what is actually inside it: architecture, params, and
FLOPs/eval -- with a per-block breakdown, which is what says where compression could pay.

No engine subprocess, no torch: this reads the weight file directly (`vibego/katago/binmodel.py`),
so it works on any released net including the v1.17 transformers.

    uv run python scripts/kata_inspect.py models/b10c384h6nbttflrs.bin.gz
    uv run python scripts/kata_inspect.py models/*.bin.gz --board 9
    uv run python scripts/kata_inspect.py models/b10c384h6nbttflrs.bin.gz --blocks
    uv run python scripts/kata_inspect.py models/*.bin.gz --json > registry.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.katago.binmodel import read_model  # noqa: E402
from vibego.katago.cost import arch_summary, model_cost  # noqa: E402


def _mflop(x: int) -> str:
    return f"{x / 1e6:,.1f}"


def report(path: str, board: int, show_blocks: bool) -> None:
    model = read_model(path)
    cost = model_cost(model, board=board)
    t = model.trunk

    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  name              {model.name}")
    print(f"  model version     {model.version}")
    print(f"  arch              {arch_summary(model)}")
    print(f"  input features    {model.num_bin_features} spatial / "
          f"{model.num_global_features} global"
          + (f" / meta v{model.meta_encoder_version}" if model.meta_encoder_version else ""))
    print(f"  trunk             {t.num_blocks} blocks x c{t.trunk_channels}, "
          f"tip norm = {'rmsnorm' if t.norm_kind else 'batchnorm/bias'}, "
          f"tip act = {t.tip_act.kind or 'relu (implicit, pre-v11)'}")
    print(f"  policy outputs    {model.policy_out_channels()}"
          + ("" if model.version >= 17 else f" (implied by model version {model.version})"))
    print(f"  params            {cost.total.params:,}")
    print(f"  FLOPs/eval @{board}x{board}  {_mflop(cost.total.flops)} MFLOP")
    share = 100.0 * cost.attention_flops / max(cost.total.flops, 1)
    print(f"    stem {_mflop(cost.stem.flops)} | trunk {_mflop(cost.trunk.flops)} "
          f"| policy {_mflop(cost.policy_head.flops)} | value {_mflop(cost.value_head.flops)}")
    if cost.attention_flops:
        print(f"    of which attention matmuls (N^2): {_mflop(cost.attention_flops)} MFLOP "
              f"({share:.1f}% of total) -- this is the part that grows with board area squared")

    if show_blocks:
        print(f"\n  {'#':>3}  {'kind':<26} {'detail':<40} {'params':>12} {'MFLOP':>10}  {'%':>5}")
        for b in cost.blocks:
            pct = 100.0 * b.cost.flops / max(cost.total.flops, 1)
            print(f"  {b.index:>3}  {b.kind:<26} {b.detail:<40} "
                  f"{b.cost.params:>12,} {_mflop(b.cost.flops):>10}  {pct:>4.1f}%")


def as_json(path: str, board: int) -> dict:
    model = read_model(path)
    cost = model_cost(model, board=board)
    return {
        "file": os.path.basename(path),
        "name": model.name,
        "model_version": model.version,
        "arch": arch_summary(model),
        "board": board,
        "num_blocks": model.trunk.num_blocks,
        "trunk_channels": model.trunk.trunk_channels,
        "params": cost.total.params,
        "flops": cost.total.flops,
        "attention_flops": cost.attention_flops,
        "blocks": [
            {"index": b.index, "kind": b.kind, "detail": b.detail,
             "params": b.cost.params, "flops": b.cost.flops}
            for b in cost.blocks
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="paths to .bin.gz (or .bin) KataGo model files")
    ap.add_argument("--board", type=int, default=19, help="board size for the FLOPs count")
    ap.add_argument("--blocks", action="store_true", help="per-trunk-block breakdown")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    if args.json:
        print(json.dumps([as_json(p, args.board) for p in args.models], indent=2))
        return
    for path in args.models:
        report(path, args.board, args.blocks)


if __name__ == "__main__":
    main()
