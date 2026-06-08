"""Serve a nanogo checkpoint as a KataGo-compatible JSON analysis engine.

KaTrain launches it like KataGo:  run_engine.py analysis -model CKPT -config CFG ...
so we accept (and ignore) the KataGo-style flags we don't need.

Standalone:
    uv run python scripts/run_engine.py -model checkpoints/depth6.pt
    echo '{"id":"x","moves":[["B","Q16"]],"komi":7.5,"boardXSize":19,"boardYSize":19,"maxVisits":50,"includePolicy":true,"includeOwnership":true,"overrideSettings":{"reportAnalysisWinratesAs":"BLACK"}}' | uv run python scripts/run_engine.py -model checkpoints/depth6.pt

KaTrain config: set the engine command (custom backend / altcommand) to:
    uv run --project /path/to/nanogo python /path/to/nanogo/scripts/run_engine.py -model /path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogo.common import get_device
from nanogo.engine.analysis import AnalysisEngine
from nanogo.net.model import Model, ModelConfig
from nanogo.engine.search import NNEvaluator


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
    p.add_argument("-pos-len", type=int, default=19)
    p.add_argument("-proxy", default=None,
                   help="run OUR search on an external engine's net (KataGo cmd) — search diagnostic")
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
        from nanogo.engine.proxy import KataGoEvaluator
        evaluator = KataGoEvaluator(args.proxy)
        pos_len = args.pos_len
        sys.stderr.write(f"nanogo: OUR search on proxy net [{args.proxy}]\n")
    else:
        device = get_device(args.device)
        evaluator, config = load_evaluator(args.model, device)
        pos_len = config.pos_len
        sys.stderr.write(f"nanogo: loaded {args.model} ({config}) on {device}\n")
    sys.stderr.flush()
    engine = AnalysisEngine(evaluator, pos_len, default_visits=args.default_visits,
                            leaf_batch=args.leaf_batch)
    engine.run()


if __name__ == "__main__":
    main()
