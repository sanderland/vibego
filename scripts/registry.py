#!/usr/bin/env python3
"""Append-only results registry + Pareto-frontier tool for the vibego study.

Pure bookkeeping. No training is run here. Stdlib + numpy only; matplotlib is
optional and only used (guarded) when --plot is passed.

Data lives in an append-only JSONL file:
    experiments/registry.jsonl
One JSON object per line. Schema (all optional except id/axis/arch):
    id (str, unique)          axis (str: train|arch|search|baseline)
    arch (str)                params (float, millions)
    flops (float, MFLOP/eval) cpu_ms (float)
    val_loss (float)          val_policy / val_value (float)
    intrinsic (float)         elo (float)  elo_lo / elo_hi (float)
    stage (str: A|B|C)        data (str)   steps (int)  seed (int)
    extra (object)            ts (str ISO | null) -- the CALLER passes ts

History is append-only: on read we dedup by id keeping the LAST occurrence.

Module API (importable, no shelling out):
    append_row(dict)
    read_rows()                       -> deduped list (last wins)
    read_rows(dedup=False)            -> raw list in file order
    pareto(rows, x, y, minimize_x=True, minimize_y=True)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP_DIR = os.path.join(ROOT, "experiments")
REGISTRY_PATH = os.path.join(EXP_DIR, "registry.jsonl")

# Known fields (for ordering / validation). Extra keys are allowed and kept.
KNOWN_FIELDS = [
    "id", "axis", "arch", "params", "flops", "cpu_ms",
    "val_loss", "val_policy", "val_value", "intrinsic",
    "elo", "elo_lo", "elo_hi", "stage", "data", "steps", "seed",
    "extra", "ts",
]
REQUIRED_FIELDS = ("id", "axis", "arch")
VALID_AXES = ("train", "arch", "search", "baseline", "data")
VALID_STAGES = ("A", "B", "C")


# --------------------------------------------------------------------------- #
# Module-level API
# --------------------------------------------------------------------------- #
def append_row(row: dict, path: str = REGISTRY_PATH) -> dict:
    """Append one row (a dict) to the JSONL registry.

    Validates required fields. Does NOT enforce id uniqueness (history is
    append-only); callers should warn if they care. Returns the row written.
    """
    for f in REQUIRED_FIELDS:
        if row.get(f) in (None, ""):
            raise ValueError(f"missing required field: {f!r}")
    if row["axis"] not in VALID_AXES:
        raise ValueError(f"axis must be one of {VALID_AXES}, got {row['axis']!r}")
    if row.get("stage") not in (None, "") and row["stage"] not in VALID_STAGES:
        raise ValueError(f"stage must be one of {VALID_STAGES}, got {row['stage']!r}")
    if "ts" not in row:
        row["ts"] = None  # caller passes ts; never call datetime.now() here.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=False) + "\n")
    return row


def read_rows(path: str = REGISTRY_PATH, dedup: bool = True) -> list[dict]:
    """Read all rows. If dedup, keep the LAST occurrence per id (latest wins),
    preserving the order in which ids first appeared."""
    if not os.path.exists(path):
        return []
    raw: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed line {ln}: {e}", file=sys.stderr)
    if not dedup:
        return raw
    order: list[str] = []
    latest: dict[str, dict] = {}
    for r in raw:
        rid = r.get("id")
        if rid not in latest:
            order.append(rid)
        latest[rid] = r
    return [latest[i] for i in order]


def _num(row: dict, field: str):
    """Return float(row[field]) or None if absent / non-numeric / null."""
    v = row.get(field, None)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def pareto(rows: list[dict], x: str, y: str,
           minimize_x: bool = True, minimize_y: bool = True):
    """Compute the 2-D Pareto frontier over rows on fields (x, y).

    Only rows that have BOTH x and y numeric are considered ("eligible").

    A point P is DOMINATED if some other point Q is at least as good as P on
    both axes AND strictly better on at least one. P is on the frontier iff it
    is not dominated by any other eligible point.

    "Better" depends on minimize_*: smaller is better when minimize is True,
    larger is better when False.

    Returns (frontier, eligible) where both are lists of rows sorted ascending
    by x. frontier is the subset of eligible on the Pareto front.
    """
    eligible = []
    for r in rows:
        xv, yv = _num(r, x), _num(r, y)
        if xv is None or yv is None:
            continue
        eligible.append((xv, yv, r))

    # Orient so that LARGER is always better internally.
    def gx(xv):  # "goodness" along x
        return -xv if minimize_x else xv

    def gy(yv):
        return -yv if minimize_y else yv

    frontier = []
    for i, (xi, yi, ri) in enumerate(eligible):
        gxi, gyi = gx(xi), gy(yi)
        dominated = False
        for j, (xj, yj, rj) in enumerate(eligible):
            if i == j:
                continue
            gxj, gyj = gx(xj), gy(yj)
            # j dominates i: j >= i on both, and strictly > on at least one.
            if gxj >= gxi and gyj >= gyi and (gxj > gxi or gyj > gyi):
                dominated = True
                break
        if not dominated:
            frontier.append((xi, yi, ri))

    frontier.sort(key=lambda t: t[0])
    eligible.sort(key=lambda t: t[0])
    return ([t[2] for t in frontier], [t[2] for t in eligible])


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _print_table(rows: list[dict], cols: list[str]):
    if not rows:
        print("(no rows)")
        return
    table = [[_fmt(r.get(c)) for c in cols] for r in rows]
    widths = [max(len(cols[i]), *(len(row[i]) for row in table)) for i in range(len(cols))]
    header = "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(header)
    print("  ".join("-" * widths[i] for i in range(len(cols))))
    for row in table:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(cols))))


# --------------------------------------------------------------------------- #
# CLI subcommands
# --------------------------------------------------------------------------- #
def _coerce_row_from_flags(args) -> dict:
    """Build a row dict from individual --flags (only those provided)."""
    row: dict = {}
    flag_fields = {
        "id": str, "axis": str, "arch": str, "params": float, "flops": float,
        "cpu_ms": float, "val_loss": float, "val_policy": float,
        "val_value": float, "intrinsic": float, "elo": float, "elo_lo": float,
        "elo_hi": float, "stage": str, "data": str, "steps": int, "seed": int,
        "ts": str,
    }
    for f, caster in flag_fields.items():
        v = getattr(args, f, None)
        if v is not None:
            row[f] = caster(v)
    if getattr(args, "extra", None):
        row["extra"] = json.loads(args.extra)
    return row


def cmd_add(args):
    if args.json:
        row = json.loads(args.json)
        # Individual flags, if also provided, fill in / override the JSON blob.
        row.update(_coerce_row_from_flags(args))
    else:
        row = _coerce_row_from_flags(args)

    existing_ids = {r.get("id") for r in read_rows(dedup=False)}
    if row.get("id") in existing_ids:
        print(f"warning: id {row.get('id')!r} already exists; appending anyway "
              f"(append-only, latest wins on read)", file=sys.stderr)
    append_row(row)
    print(f"appended id={row.get('id')!r} axis={row.get('axis')!r} -> {REGISTRY_PATH}")


def cmd_list(args):
    rows = read_rows()
    if args.axis:
        rows = [r for r in rows if r.get("axis") == args.axis]
    if args.stage:
        rows = [r for r in rows if r.get("stage") == args.stage]
    cols = ["id", "axis", "arch", "stage", "params", "flops", "cpu_ms",
            "val_loss", "intrinsic", "elo", "steps", "seed"]
    _print_table(rows, cols)
    print(f"\n{len(rows)} row(s).")


def _run_pareto(rows, x, y, minimize_x, minimize_y, plot=False, label=None):
    frontier, eligible = pareto(rows, x, y, minimize_x, minimize_y)
    fset = {id(r) for r in frontier}
    xdir = "min" if minimize_x else "max"
    ydir = "min" if minimize_y else "max"
    title = label or f"Pareto: x={x} ({xdir}) vs y={y} ({ydir})"
    print(f"=== {title} ===")
    if not eligible:
        print(f"(no rows have both {x!r} and {y!r})\n")
        return frontier
    cols = ["on_front", "id", "arch", x, y]
    disp = []
    for r in eligible:
        d = {"on_front": "*" if id(r) in fset else ".",
             "id": r.get("id"), "arch": r.get("arch"),
             x: r.get(x), y: r.get(y)}
        disp.append(d)
    # sort by x ascending for display
    disp.sort(key=lambda d: (d[x] is None, d[x]))
    _print_table(disp, cols)
    print(f"frontier: {len(frontier)} / {len(eligible)} eligible  "
          f"(* = on frontier, . = dominated)\n")

    if plot:
        _try_plot(eligible, frontier, x, y, minimize_x, minimize_y, title)
    return frontier


def _try_plot(eligible, frontier, x, y, minimize_x, minimize_y, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - env dependent
        print(f"note: matplotlib unavailable, skipping plot ({e})", file=sys.stderr)
        return
    fset = {id(r) for r in frontier}
    ex = [_num(r, x) for r in eligible]
    ey = [_num(r, y) for r in eligible]
    colors = ["tab:red" if id(r) in fset else "tab:gray" for r in eligible]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ex, ey, c=colors, s=40, zorder=3)
    for r in eligible:
        ax.annotate(str(r.get("id")), (_num(r, x), _num(r, y)),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    fx = [_num(r, x) for r in frontier]
    fy = [_num(r, y) for r in frontier]
    ax.plot(fx, fy, color="tab:red", lw=1.2, zorder=2)
    ax.set_xlabel(f"{x} ({'min' if minimize_x else 'max'})")
    ax.set_ylabel(f"{y} ({'min' if minimize_y else 'max'})")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    os.makedirs(EXP_DIR, exist_ok=True)
    out = os.path.join(EXP_DIR, f"pareto_{x}_vs_{y}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote plot -> {out}")


def cmd_pareto(args):
    rows = read_rows()
    # Default is minimize on both axes; --maximize-* flips a given axis.
    minimize_x = not args.maximize_x
    minimize_y = not args.maximize_y
    _run_pareto(rows, args.x, args.y, minimize_x, minimize_y, plot=args.plot)


def cmd_frontiers(args):
    rows = read_rows()
    print("Canonical frontiers for the vibego study.\n")
    _run_pareto(rows, "flops", "val_loss", minimize_x=True, minimize_y=True,
                plot=args.plot,
                label="SCREENING frontier: flops (min) vs val_loss (min)")
    _run_pareto(rows, "flops", "elo", minimize_x=True, minimize_y=False,
                plot=args.plot,
                label="REAL frontier: flops (min) vs elo (max)")


# --------------------------------------------------------------------------- #
# argparse wiring
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="registry.py",
        description="Append-only results registry + Pareto-frontier tool.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("add", help="append one row")
    pa.add_argument("--json", help="full row as a JSON object")
    pa.add_argument("--id")
    pa.add_argument("--axis", choices=VALID_AXES)
    pa.add_argument("--arch")
    pa.add_argument("--params", type=float)
    pa.add_argument("--flops", type=float)
    pa.add_argument("--cpu-ms", dest="cpu_ms", type=float)
    pa.add_argument("--val-loss", dest="val_loss", type=float)
    pa.add_argument("--val-policy", dest="val_policy", type=float)
    pa.add_argument("--val-value", dest="val_value", type=float)
    pa.add_argument("--intrinsic", type=float)
    pa.add_argument("--elo", type=float)
    pa.add_argument("--elo-lo", dest="elo_lo", type=float)
    pa.add_argument("--elo-hi", dest="elo_hi", type=float)
    pa.add_argument("--stage", choices=VALID_STAGES)
    pa.add_argument("--data")
    pa.add_argument("--steps", type=int)
    pa.add_argument("--seed", type=int)
    pa.add_argument("--extra", help="JSON object for the extra field")
    pa.add_argument("--ts", help="ISO timestamp (caller-supplied; not auto-set)")
    pa.set_defaults(func=cmd_add)

    pl = sub.add_parser("list", help="print a table of rows (last wins per id)")
    pl.add_argument("--axis", choices=VALID_AXES)
    pl.add_argument("--stage", choices=VALID_STAGES)
    pl.set_defaults(func=cmd_list)

    pp = sub.add_parser("pareto", help="compute a Pareto frontier on two fields")
    pp.add_argument("--x", required=True, help="x field (e.g. flops, cpu_ms)")
    pp.add_argument("--y", required=True, help="y field (e.g. val_loss, elo)")
    gx = pp.add_mutually_exclusive_group()
    gx.add_argument("--minimize-x", dest="minimize_x", action="store_true",
                    help="smaller x is better (default)")
    gx.add_argument("--maximize-x", dest="maximize_x", action="store_true",
                    help="larger x is better")
    gy = pp.add_mutually_exclusive_group()
    gy.add_argument("--minimize-y", dest="minimize_y", action="store_true",
                    help="smaller y is better (default)")
    gy.add_argument("--maximize-y", dest="maximize_y", action="store_true",
                    help="larger y is better")
    pp.add_argument("--plot", action="store_true",
                    help="also write a PNG under experiments/ (needs matplotlib)")
    pp.set_defaults(func=cmd_pareto)

    pf = sub.add_parser("frontiers",
                        help="print the two canonical frontiers in one go")
    pf.add_argument("--plot", action="store_true")
    pf.set_defaults(func=cmd_frontiers)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
