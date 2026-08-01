"""Validate `vibego/katago/torchmodel.py` against the stock KataGo engine on real positions.

The forward pass is only useful if it computes what KataGo computes, and the diagnostics that
depend on it are worthless otherwise. So: take rows of real training data (exact V7 input features,
as KataGo itself encoded them), run them through our torch forward, reconstruct the same positions
as analysis queries, and compare the raw net outputs.

The comparison is not exact by construction, and the reason is worth being precise about. A query
built from `initialStones` carries **no move history**, so the engine sees zeros in the history
planes (9-13) and no ko point (plane 6), while the data rows have them filled in. We therefore zero
those planes on our side too, and drop rows whose remaining confounders differ (recent-pass globals,
playout-doubling advantage, non-full boards). What survives is a like-for-like comparison; the
residual disagreement is the "wave" komi-wobble global and rules-encoding differences, which is why
the bar below is "very high agreement", not "bitwise equal".

    uv run python scripts/kata_torch_check.py --model models/b10c384h6nbttflrs.bin.gz \
        --npz katago/python/testdata/benchmark_data_1024.npz \
        --katago ./katago --config analysis_example.cfg --n 64
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vibego.go.board import xy_to_gtp  # noqa: E402
from vibego.katago.binmodel import read_model  # noqa: E402
from vibego.katago.torchmodel import KataTorchModel  # noqa: E402

# Input planes/globals that encode move history rather than board state. A position rebuilt from
# stones alone cannot carry them, so they are zeroed on both sides.
HISTORY_PLANES = [6, 9, 10, 11, 12, 13]
HISTORY_GLOBALS = [0, 1, 2, 3, 4]
PDA_GLOBALS = [15, 16]   # playout doubling advantage: our queries never set it
ENCORE_GLOBALS = [12, 13]  # encore phases: not reachable through the analysis API


def rules_from_globals(g) -> dict:
    """Invert KataGo's rules -> global-feature encoding (nninputs.cpp, rowGlobal[6..11], [17])
    so the query asks for the ruleset the data row was actually generated under. Getting this
    wrong shifts score and ownership, which would otherwise look like a bug in our forward."""
    if g[6] == 0:
        ko = "SIMPLE"
    else:
        ko = "POSITIONAL" if g[7] > 0 else "SITUATIONAL"
    if g[10] == 0:
        tax = "NONE"
    else:
        tax = "ALL" if g[11] != 0 else "SEKI"
    return {
        "ko": ko,
        "scoring": "TERRITORY" if g[9] != 0 else "AREA",
        "tax": tax,
        "suicide": bool(g[8] != 0),
        "hasButton": bool(g[17] != 0),
        "whiteHandicapBonus": "0",
    }


def komi_wave(self_komi: float, board_area: int) -> float:
    """global[18]: the triangle wave over komi parity that KataGo derives from selfKomi and board
    area. It is a deterministic function of the komi, so once we round the komi to something the
    analysis API accepts we must recompute it rather than reuse the row's value."""
    drawable_even = board_area % 2 == 0
    if drawable_even:
        komi_floor = math.floor(self_komi / 2.0) * 2.0
    else:
        komi_floor = math.floor((self_komi - 1.0) / 2.0) * 2.0 + 1.0
    delta = min(max(self_komi - komi_floor, 0.0), 2.0)
    if delta < 0.5:
        return delta
    if delta < 1.5:
        return 1.0 - delta
    return delta - 2.0


def synthetic_rows():
    """Positions whose V7 features are fully determined by inspection, so the comparison against
    the engine is exact rather than best-effort.

    Every stone is isolated on a star point with four liberties, so the liberty planes (3-5), the
    ladder planes (14-17) and the pass-alive planes (18-21) are all zero; there is no history, no
    ko, and no capture. That leaves only the mask and the two stone planes -- a feature vector we
    can write down with certainty. If our forward disagrees with the engine *here*, the forward is
    wrong; if it agrees here but not on reconstructed training rows, the reconstruction is what is
    approximate.
    """
    stars = [(3, 3), (3, 9), (3, 15), (9, 3), (9, 15), (15, 3), (15, 9), (15, 15)]
    layouts = [
        ([], []),                                        # empty board
        ([stars[0]], []),                                # one black stone
        ([stars[0]], [stars[7]]),                        # one each, far apart
        (stars[:2], stars[6:8]),
        (stars[:3], stars[5:8]),
        (stars[:4], stars[4:8]),
    ]
    komis = [7.5, 0.5, -3.5]
    spatial, glob, meta = [], [], []
    for black, white in layouts:
        for komi in komis:
            sp = np.zeros((22, 19, 19), dtype=np.float32)
            sp[0] = 1.0                                   # full board mask
            for (x, y) in black:
                sp[1, y, x] = 1.0                         # black to move, so "own" is black
            for (x, y) in white:
                sp[2, y, x] = 1.0
            g = np.zeros(19, dtype=np.float32)
            g[5] = -komi / 20.0                           # selfKomi from Black's perspective
            g[6], g[7] = 1.0, 0.5                         # positional superko
            g[18] = komi_wave(-komi, 19 * 19)
            spatial.append(sp)
            glob.append(g)
            meta.append((black, white, komi))
    print(f"{len(spatial)} synthetic positions (isolated star-point stones, "
          f"all derived feature planes zero)")
    return np.stack(spatial), np.stack(glob), meta


def synthetic_query(black, white, komi, qid):
    stones = [["B", xy_to_gtp((x, y), 19)] for (x, y) in black]
    stones += [["W", xy_to_gtp((x, y), 19)] for (x, y) in white]
    return {
        "id": qid, "komi": komi, "boardXSize": 19, "boardYSize": 19,
        "rules": {"ko": "POSITIONAL", "scoring": "AREA", "tax": "NONE", "suicide": False,
                  "hasButton": False, "whiteHandicapBonus": "0"},
        "initialStones": stones, "initialPlayer": "B", "moves": [],
        "maxVisits": 1, "includePolicy": True,
        "overrideSettings": {"reportAnalysisWinratesAs": "BLACK"},
    }


def load_rows(npz_path: str, n: int, strict: bool = True):
    data = np.load(npz_path)
    packed = data["binaryInputNCHWPacked"]
    glob = data["globalInputNC"].astype(np.float32)
    spatial = np.unpackbits(packed, axis=2)[:, :, :361]
    spatial = spatial.reshape(-1, 22, 19, 19).astype(np.float32)

    full_board = spatial[:, 0].reshape(len(spatial), -1).min(axis=1) > 0
    no_history_globals = np.all(glob[:, HISTORY_GLOBALS] == 0, axis=1)
    no_pda = np.all(glob[:, PDA_GLOBALS] == 0, axis=1)
    no_encore = np.all(glob[:, ENCORE_GLOBALS] == 0, axis=1)
    usable = full_board & no_history_globals & no_pda & no_encore
    if strict:
        # This is rule- and komi-randomized *training* data. Territory scoring depends on encore
        # machinery a stone-only query cannot express, and a randomized komi outside the normal
        # range (or off the half-integer grid the analysis API accepts) cannot be asked for
        # exactly. Restrict to rows a query can actually reproduce.
        self_komi = glob[:, 5] * 20.0
        usable &= glob[:, 9] == 0                                    # area scoring
        usable &= np.abs(self_komi * 2 - np.round(self_komi * 2)) < 1e-4  # half-integer komi
        usable &= np.abs(self_komi) <= 20.0                          # normal komi range
    keep = np.flatnonzero(usable)[:n]
    print(f"{len(keep)} usable rows of {len(spatial)} "
          f"(full board {full_board.sum()}, no recent pass {no_history_globals.sum()}, "
          f"no pda {no_pda.sum()}, no encore {no_encore.sum()}, strict={strict})")

    spatial, glob = spatial[keep].copy(), glob[keep].copy()
    spatial[:, HISTORY_PLANES] = 0.0

    # The analysis API only accepts integer/half-integer komi, so the query cannot ask for the
    # row's randomized komi. Round it, then rewrite the two globals that are functions of komi so
    # both sides see the same value.
    komis = np.array([round(float(g[5]) * 20.0 * 2) / 2 for g in glob], dtype=np.float32)
    glob[:, 5] = komis / 20.0
    glob[:, 18] = [komi_wave(float(k), 19 * 19) for k in komis]
    return spatial, glob


def build_query(spatial_row, glob_row, qid: str) -> dict:
    stones = []
    own, opp = spatial_row[1], spatial_row[2]
    for y in range(19):
        for x in range(19):
            if own[y, x]:
                stones.append(["B", xy_to_gtp((x, y), 19)])
            elif opp[y, x]:
                stones.append(["W", xy_to_gtp((x, y), 19)])
    self_komi = float(glob_row[5]) * 20.0  # global[5] = selfKomi/20, and we report as Black
    return {
        "id": qid, "rules": rules_from_globals(glob_row), "komi": -self_komi,
        "boardXSize": 19, "boardYSize": 19, "initialStones": stones,
        "initialPlayer": "B", "moves": [], "maxVisits": 1, "includePolicy": True,
        "overrideSettings": {"reportAnalysisWinratesAs": "BLACK"},
    }


def run_engine(katago: str, config: str, model: str, queries: list[dict]) -> dict:
    proc = subprocess.run(
        [katago, "analysis", "-model", model, "-config", config,
         # nnRandomize applies a random dihedral symmetry to each evaluation. The nets are only
         # approximately symmetry-equivariant, so leaving it on injects a difference that looks
         # exactly like a bug in our forward.
         "-override-config", "numAnalysisThreads=1,nnRandomize=false"],
        input="\n".join(json.dumps(q) for q in queries) + "\n",
        capture_output=True, text=True, timeout=3600,
    )
    out = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue  # the AppImage wrapper can print non-JSON noise to stdout
        r = json.loads(line)
        if "policy" in r:
            out[r["id"]] = r
    if not out:
        sys.exit(f"engine produced no usable output. stderr tail:\n{proc.stderr[-2000:]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--npz", help="training-data npz (omit with --synthetic)")
    ap.add_argument("--katago", required=True, help="path to the katago binary")
    ap.add_argument("--config", required=True, help="analysis config")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true",
                    help="use hand-determined positions instead of training rows (exact check)")
    ap.add_argument("--loose", action="store_true",
                    help="skip the area-scoring / normal-komi filter (expect worse agreement)")
    args = ap.parse_args()

    if args.synthetic:
        spatial, glob, meta = synthetic_rows()
        queries = [synthetic_query(*meta[i], f"r{i}") for i in range(len(spatial))]
    else:
        if not args.npz:
            ap.error("--npz is required unless --synthetic")
        spatial, glob = load_rows(args.npz, args.n, strict=not args.loose)
        queries = None
    net = KataTorchModel(read_model(args.model)).eval()

    policies, winrates, leads = [], [], []
    with torch.no_grad():
        for i in range(0, len(spatial), args.batch):
            out = net(torch.from_numpy(spatial[i:i + args.batch]),
                      torch.from_numpy(glob[i:i + args.batch]))
            policies.append(net.policy(out).numpy())
            winrates.append(net.winrate(out).numpy())
            leads.append(net.score_lead(out).numpy())
    policies = np.concatenate(policies)
    winrates, leads = np.concatenate(winrates), np.concatenate(leads)

    if queries is None:
        queries = [build_query(spatial[i], glob[i], f"r{i}") for i in range(len(spatial))]
    responses = run_engine(args.katago, args.config, args.model, queries)

    top1, kl, dw, dl, ref_lead, our_lead = [], [], [], [], [], []
    for i in range(len(spatial)):
        r = responses.get(f"r{i}")
        if r is None:
            continue
        pe = np.array(r["policy"], dtype=np.float64)
        valid = pe >= 0
        pe = np.clip(pe[valid], 1e-12, None)
        pe /= pe.sum()
        po = np.clip(policies[i][valid], 1e-12, None)
        po /= po.sum()
        top1.append(int(pe.argmax() == po.argmax()))
        kl.append(float((pe * np.log(pe / po)).sum()))
        dw.append(abs(float(r["rootInfo"]["winrate"]) - float(winrates[i])))
        dl.append(abs(float(r["rootInfo"]["scoreLead"]) - float(leads[i])))
        ref_lead.append(float(r["rootInfo"]["scoreLead"]))
        our_lead.append(float(leads[i]))

    n = len(top1)
    print(f"\ncompared {n} positions (raw net, maxVisits=1)")
    print(f"  policy top-1 agreement   {np.mean(top1):.3f}")
    print(f"  policy KL(engine||torch) mean {np.mean(kl):.5f}   median {np.median(kl):.5f}")
    print(f"  |winrate| diff           mean {np.mean(dw):.5f}   max {np.max(dw):.5f}")
    print(f"  |scoreLead| diff         mean {np.mean(dl):.4f}    max {np.max(dl):.4f}")
    print(f"  scoreLead correlation    {np.corrcoef(ref_lead, our_lead)[0, 1]:.6f}")


if __name__ == "__main__":
    main()
