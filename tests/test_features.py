"""Cross-check our feature subset against KataGo's own pure-Python V7 encoder.

We play identical random (non-pass) games on our Board and on KataGo's reference
GameState, then assert our selected channels match the reference exactly.
"""
import random

import numpy as np
import pytest

from vibego.go import features as F
from vibego.go.board import BLACK, WHITE, Board, opp

katago_board = pytest.importorskip("katago.game.board")
katago_features = pytest.importorskip("katago.game.features")
katago_gamestate = pytest.importorskip("katago.game.gamestate")

REF_CONFIG = {"version": 15}
KOMI = 7.5
POS_LEN = 19


def _ref_features(gs):
    feats = katago_features.Features(REF_CONFIG, POS_LEN)
    return gs.get_input_features(feats)  # (bin (1,22,P,P), global (1,19))


@pytest.mark.parametrize("size,seed", [(19, 0), (19, 1), (9, 2), (13, 3)])
def test_feature_subset_matches_reference(size, seed):
    rng = random.Random(seed)
    my = Board(size)
    rules = dict(katago_gamestate.GameState.RULES_CHINESE)
    rules["whiteKomi"] = KOMI
    gs = katago_gamestate.GameState(size, rules)

    player = BLACK
    for _ in range(60):
        legal = [m for m in my.legal_moves(player)]
        if not legal:
            break
        x, y = rng.choice(legal)
        my.play(player, (x, y))
        gs.play(player, gs.board.loc(x, y))

        # Compare at this position (now it's the opponent's turn).
        spatial, glob = F.encode_board(my, KOMI, POS_LEN)
        ref_bin, ref_glob = _ref_features(gs)
        ref_spatial = ref_bin[0][F.SPATIAL_SUBSET]
        np.testing.assert_array_equal(
            spatial, ref_spatial, err_msg=f"spatial mismatch size={size} seed={seed}"
        )
        # global[5] = selfKomi/20; global[14] = passWouldEndPhase (0 here, no passes).
        np.testing.assert_allclose(glob, ref_glob[0][F.GLOBAL_SUBSET], rtol=0, atol=1e-6)

        player = opp(player)


def test_ladder_features_match_reference():
    """Constructed position with a real ladder: our channels 14/17 must match KataGo and fire."""
    rules = dict(katago_gamestate.GameState.RULES_CHINESE)
    rules["whiteKomi"] = KOMI
    gs = katago_gamestate.GameState(19, rules)
    my = Board(19)
    for p, (x, y) in [(BLACK, (1, 0)), (WHITE, (1, 1)), (BLACK, (0, 1)), (WHITE, (10, 10))]:
        my.play(p, (x, y))
        gs.play(p, gs.board.loc(x, y))

    spatial, _ = F.encode_board(my, KOMI, 19)
    ref_bin, _ = _ref_features(gs)
    ch14 = spatial[F.SPATIAL_SUBSET.index(14)]
    ch17 = spatial[F.SPATIAL_SUBSET.index(17)]
    np.testing.assert_array_equal(ch14, ref_bin[0][14])
    np.testing.assert_array_equal(ch17, ref_bin[0][17])
    assert ch14.sum() > 0  # the ladder feature actually fires here (non-vacuous)


def test_decode_npz_roundtrip_shapes():
    # Pack a synthetic on-board mask through the same bit path as real data.
    N, pos_len = 3, 19
    full = np.zeros((N, 22, pos_len * pos_len), dtype=np.uint8)
    full[:, 0, :] = 1  # on-board everywhere
    full[:, 1, 0] = 1  # a stone at pos 0
    packed = np.packbits(full, axis=2)
    glob = np.zeros((N, 19), dtype=np.float32)
    glob[:, 5] = 0.3
    spatial, g = F.decode_npz(packed, glob, pos_len)
    assert spatial.shape == (N, F.NUM_SPATIAL, pos_len, pos_len)
    assert g.shape == (N, F.NUM_GLOBAL)
    assert spatial[:, 0].all()  # on-board channel
    assert spatial[0, 1, 0, 0] == 1.0  # the stone
    assert np.allclose(g[:, 0], 0.3)  # global komi channel
