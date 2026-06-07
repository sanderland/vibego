"""Integration test: drive run_engine.py as a subprocess exactly the way KaTrain does,
send the query shape KaTrain builds, and assert the response satisfies KaTrain's parser
(katrain/core/game_node.py set_analysis / update_move_analysis).
"""
import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from nanogo.go import features as F
from nanogo.net.model import Model, ModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    cfg = ModelConfig.plain(8, 2)
    model = Model(cfg)
    path = str(tmp_path_factory.mktemp("ckpt") / "tiny.pt")
    torch.save({"model": model.state_dict(), "optimizer": {}, "model_config": asdict(cfg),
                "step": 0, "spatial_subset": F.SPATIAL_SUBSET, "global_subset": F.GLOBAL_SUBSET}, path)
    return path


def _run_query(ckpt, query):
    # KaTrain launches:  <exe> analysis -model M -config C -override-config ...
    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_engine.py"),
           "analysis", "-model", ckpt, "-config", "dummy.cfg",
           "-override-config", "homeDataDir=/tmp"]
    proc = subprocess.run(cmd, input=json.dumps(query) + "\n",
                          capture_output=True, text=True, timeout=120, cwd=ROOT)
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    assert lines, f"no output. stderr:\n{proc.stderr}"
    return json.loads(lines[-1])


def test_katrain_contract(tiny_ckpt):
    # The exact query KaTrain's engine builds (engine.py:469-485).
    query = {
        "rules": "japanese",
        "priority": 0,
        "analyzeTurns": [2],
        "maxVisits": 12,
        "komi": 6.5,
        "boardXSize": 19,
        "boardYSize": 19,
        "includeOwnership": True,
        "includeMovesOwnership": True,
        "includePolicy": True,
        "initialStones": [],
        "initialPlayer": "B",
        "moves": [["B", "Q16"], ["W", "D4"]],
        "overrideSettings": {"reportAnalysisWinratesAs": "BLACK", "wideRootNoise": 0.0},
        "id": "QUERY:1",
    }
    r = _run_query(tiny_ckpt, query)

    assert r["id"] == "QUERY:1"
    assert r["turnNumber"] == 2

    root = r["rootInfo"]
    for k in ("winrate", "scoreLead", "visits", "currentPlayer"):
        assert k in root, f"rootInfo missing {k}"
    assert 0.0 <= root["winrate"] <= 1.0
    assert root["visits"] == 12
    assert root["currentPlayer"] == "B"  # black to move after B,W

    assert r["moveInfos"], "no candidate moves"
    seen_orders = []
    for mi in r["moveInfos"]:
        for k in ("move", "visits", "winrate", "scoreLead", "order", "pv", "prior"):
            assert k in mi, f"moveInfo missing {k}"
        assert 0.0 <= mi["winrate"] <= 1.0
        assert isinstance(mi["pv"], list) and mi["pv"]
        seen_orders.append(mi["order"])
    assert seen_orders == sorted(seen_orders)  # ordered by visits
    assert sum(mi["visits"] for mi in r["moveInfos"]) == 12

    # arrays in the documented shapes
    assert len(r["policy"]) == 19 * 19 + 1
    assert len(r["ownership"]) == 19 * 19
    assert all(-1.0 <= o <= 1.0 for o in r["ownership"])


def test_query_version(tiny_ckpt):
    r = _run_query(tiny_ckpt, {"id": "v", "action": "query_version"})
    assert r["id"] == "v"
    assert "version" in r


def test_concurrent_query_during_ponder(tiny_ckpt):
    """A normal query must be answered while a never-ending ponder is still running."""
    import threading
    import time

    cmd = [sys.executable, os.path.join(ROOT, "scripts", "run_engine.py"), "-model", tiny_ckpt]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
    results = {"ponder_partials": 0, "q2_final": None}
    done = threading.Event()

    def reader():
        for line in p.stdout:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "terminateId" in r:
                continue
            if r["id"] == "PONDER" and r.get("isDuringSearch"):
                results["ponder_partials"] += 1
            if r["id"] == "Q2" and not r.get("isDuringSearch"):
                results["q2_final"] = r
                done.set()

    threading.Thread(target=reader, daemon=True).start()
    base = {"komi": 7.5, "boardXSize": 19, "boardYSize": 19,
            "overrideSettings": {"reportAnalysisWinratesAs": "BLACK"}}
    # never-ending ponder
    p.stdin.write(json.dumps({**base, "id": "PONDER", "moves": [], "maxVisits": 10_000_000,
                              "reportDuringSearchEvery": 0.05}) + "\n")
    p.stdin.flush()
    time.sleep(0.3)  # let the ponder stream a few partials first
    # a normal query, WITHOUT terminating the ponder
    p.stdin.write(json.dumps({**base, "id": "Q2", "moves": [["B", "Q16"]], "maxVisits": 40}) + "\n")
    p.stdin.flush()

    got = done.wait(timeout=60)
    p.terminate()
    assert got, "normal query was not answered while ponder ran (blocked!)"
    assert results["q2_final"]["rootInfo"]["visits"] == 40
    assert results["ponder_partials"] >= 1  # ponder was streaming concurrently
