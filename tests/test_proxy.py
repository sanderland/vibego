"""The KataGoEvaluator proxy lets our MCTS run on an external engine's net. We test it using
vibego's own engine as the 'external' target (it speaks the same protocol), so no KataGo needed."""
import os
import sys
from dataclasses import asdict

import pytest
import torch

from vibego.engine.proxy import KataGoEvaluator
from vibego.engine.search import MCTS
from vibego.go import features as F
from vibego.go.board import Board
from vibego.net.model import Model, ModelConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    cfg = ModelConfig.plain(8, 2)
    path = str(tmp_path_factory.mktemp("ckpt") / "tiny.pt")
    torch.save({"model": Model(cfg).state_dict(), "optimizer": {}, "model_config": asdict(cfg),
                "step": 0, "spatial_subset": F.SPATIAL_SUBSET, "global_subset": F.GLOBAL_SUBSET}, path)
    return path


def test_proxy_drives_our_search(tiny_ckpt):
    cmd = (f"{sys.executable} {os.path.join(ROOT, 'scripts', 'run_engine.py')} "
           f"-model {tiny_ckpt} -device cpu")
    ev = KataGoEvaluator(cmd)
    board = Board(9)
    board.play(1, (2, 2))
    board.play(2, (6, 6))
    root = MCTS(ev, komi=5.5, pos_len=19).run(board, visits=6, batch_size=3)
    assert root.N >= 6
    assert any(c.N > 0 for c in root.children)  # search actually ran on the proxy net
    ev.proc.terminate()
