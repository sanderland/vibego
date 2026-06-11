"""Serve a vibego checkpoint as a KataGo-compatible JSON analysis engine.

KaTrain launches it like KataGo:  run_engine.py analysis -model CKPT -config CFG ...
so we accept (and ignore) the KataGo-style flags we don't need.

Standalone:
    uv run python scripts/run_engine.py -model checkpoints/depth6.pt
    echo '{"id":"x","moves":[["B","Q16"]],"komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":50,"includePolicy":true,"includeOwnership":true,"overrideSettings":{"reportAnalysisWinratesAs":"BLACK"}}' | uv run python scripts/run_engine.py -model checkpoints/depth6.pt

KaTrain config: set the engine command (custom backend / altcommand) to:
    uv run --project /path/to/vibego python /path/to/vibego/scripts/run_engine.py -model /path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.common import get_device
from vibego.engine.analysis import AnalysisEngine
from vibego.net.model import Model, ModelConfig
from vibego.engine.search import NNEvaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("subcommand", nargs="?", default="analysis")  # KataGo-style "analysis"
    p.add_argument("-model", default=None)
    p.add_argument("-config", default=None)            # ignored
    p.add_argument("-override-config", default=None)    # ignored
    p.add_argument("-human-model", default=None)        # ignored
    p.add_argument("-device", default=None)
    p.add_argument("-default-visits", type=int, default=100)
    p.add_argument("-leaf-batch", type=int, default=16, help="leaves per search step (1=sequential)")
    p.add_argument("-lcb-stdevs", type=float, default=1.0, help="move-selection LCB width (0=mean value)")
    p.add_argument("-cpuct", type=float, default=1.0, help="PUCT exploration constant")
    p.add_argument("-cpuct-log", type=float, default=0.45, help="cpuct growth ~ log((N+base)/base)")
    p.add_argument("-pos-len", type=int, default=19)
    p.add_argument("-proxy", default=None,
                   help="run OUR search on an external engine's net (KataGo cmd) — search diagnostic")
    p.add_argument("-early-stop", action="store_true",
                   help="stop search once the leading move's visit lead is unbeatable (low-visit latency lever)")
    p.add_argument("-early-stop-min-frac", type=float, default=0.5,
                   help="don't early-stop before this fraction of the visit budget is spent")
    p.add_argument("-gumbel", action="store_true",
                   help="Gumbel-AlphaZero root search (Danihelka et al. 2022): sample top-m root "
                        "moves by log-prior + Gumbel noise, sequential halving over the visit "
                        "budget; built for low visits. Deterministic per move (seed, turn).")
    p.add_argument("-gumbel-m", type=int, default=16,
                   help="number of root candidates sampled without replacement (gumbel mode)")
    p.add_argument("-gumbel-seed", type=int, default=0,
                   help="base seed for the per-move Gumbel noise (mixed with the turn number)")
    p.add_argument("-gumbel-c-scale", type=float, default=1.0,
                   help="sigma(q) scale on raw search utility (paper's c_scale; decisions "
                        "between sampled candidates are Q-driven at 1.0 — measured best)")
    args, _ignored = p.parse_known_args()
    return args


def load_evaluator(path, device):
    """Load a checkpoint into an NNEvaluator using the feature subset it was trained with."""
    ckpt = torch.load(path, map_location=device)
    config = ModelConfig(**ckpt["model_config"])
    model = Model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ev = NNEvaluator(model, device,
                     spatial_subset=ckpt["spatial_subset"],
                     global_subset=ckpt["global_subset"])
    return ev, config


def main():
    args = parse_args()
    if args.proxy:
        from vibego.engine.proxy import KataGoEvaluator
        evaluator = KataGoEvaluator(args.proxy)
        pos_len = args.pos_len
        sys.stderr.write(f"vibego: OUR search on proxy net [{args.proxy}]\n")
    else:
        device = get_device(args.device)
        evaluator, config = load_evaluator(args.model, device)
        pos_len = config.pos_len
        sys.stderr.write(f"vibego: loaded {args.model} ({config}) on {device}\n")
    sys.stderr.flush()
    mcts_kwargs = {"c_puct": args.cpuct, "c_puct_log": args.cpuct_log,
                   "early_stop": args.early_stop, "early_stop_min_frac": args.early_stop_min_frac,
                   "gumbel_root": args.gumbel, "gumbel_m": args.gumbel_m,
                   "gumbel_seed": args.gumbel_seed, "gumbel_c_scale": args.gumbel_c_scale}
    engine = AnalysisEngine(evaluator, pos_len, default_visits=args.default_visits,
                            leaf_batch=args.leaf_batch, lcb_stdevs=args.lcb_stdevs,
                            mcts_kwargs=mcts_kwargs)
    engine.run()


if __name__ == "__main__":
    main()
