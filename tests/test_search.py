"""Tests for the batching evaluator's failure handling."""
import numpy as np
import pytest
import torch

from nanogo.engine.search import NNEvaluator
from nanogo.go.features import NUM_GLOBAL, NUM_SPATIAL


class _BoomModel(torch.nn.Module):
    def forward(self, spatial, glob):
        raise RuntimeError("boom")


def test_evaluator_surfaces_errors_instead_of_hanging():
    # If the model raises, infer() must raise (not block its caller forever).
    ev = NNEvaluator(_BoomModel(), "cpu")
    feats = [(np.zeros((NUM_SPATIAL, 19, 19), dtype=np.float32),
              np.zeros((NUM_GLOBAL,), dtype=np.float32))]
    with pytest.raises(RuntimeError, match="boom"):
        ev.infer(feats)
