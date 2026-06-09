"""Plot the study's two Pareto frontiers from the registry: screening (val-loss vs FLOPs, lower
is better) and real (Elo vs FLOPs, higher is better). Saves a 1- or 2-panel PNG. Used for the
periodic / on-breakthrough reports.

    uv run python scripts/pareto_plot.py --out experiments/frontier.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import registry as reg


def _panel(ax, rows, ykey, ylabel, maximize_y, xkey="flops", xlabel="FLOPs/eval (MFLOP)"):
    pts = [r for r in rows if isinstance(r.get(xkey), (int, float)) and isinstance(r.get(ykey), (int, float))]
    if not pts:
        ax.set_title(f"{ylabel}: no data yet"); return
    front, _ = reg.pareto(pts, xkey, ykey, minimize_x=True, minimize_y=not maximize_y)
    fids = {id(r) for r in front}
    for r in pts:
        on = id(r) in fids
        ax.scatter(r[xkey], r[ykey], s=40 if on else 22,
                   c="tab:blue" if on else "lightgray", zorder=3 if on else 2,
                   edgecolors="k" if on else "none", linewidths=0.5)
        ax.annotate(r.get("id", ""), (r[xkey], r[ykey]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="black" if on else "gray")
    fr = sorted(front, key=lambda r: r[xkey])
    ax.plot([r[xkey] for r in fr], [r[ykey] for r in fr], "-", c="tab:blue", lw=1.2, zorder=2)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    ax.set_title(f"{ylabel} vs {xkey}  (frontier: {len(front)}/{len(pts)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="experiments/frontier.png")
    p.add_argument("--title", default="vibego frontier")
    p.add_argument("--cpu", action="store_true",
                   help="add a second row of panels with x=cpu_ms (single-thread, the browser-"
                        "target cost axis; diverges from FLOPs — nbt loses its edge there)")
    args = p.parse_args()
    rows = reg.read_rows()
    has_elo = any(isinstance(r.get("elo"), (int, float)) for r in rows)
    n = 2 if has_elo else 1
    nrows = 2 if args.cpu else 1
    fig, axes = plt.subplots(nrows, n, figsize=(6.2 * n, 5.0 * nrows), squeeze=False)
    _panel(axes[0][0], rows, "val_loss", "val_loss (lower better)", maximize_y=False)
    if has_elo:
        _panel(axes[0][1], rows, "elo", "Elo (higher better)", maximize_y=True)
    if args.cpu:
        xl = "CPU ms/eval (1 thread)"
        _panel(axes[1][0], rows, "val_loss", "val_loss (lower better)", maximize_y=False,
               xkey="cpu_ms", xlabel=xl)
        if has_elo:
            _panel(axes[1][1], rows, "elo", "Elo (higher better)", maximize_y=True,
                   xkey="cpu_ms", xlabel=xl)
    fig.suptitle(args.title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}  ({len(rows)} rows, elo_panel={has_elo})")


if __name__ == "__main__":
    main()
