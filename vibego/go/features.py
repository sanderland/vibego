"""Input-feature encoding: a *subset* of KataGo's V7 features, chosen so the engine can
recompute them exactly from a board state with no ladder search / territory / encore logic.

Two paths that must agree:
  - decode_npz: select our channels out of katagotraining .npz rows (training).
  - encode_board: compute the same channels from a live Board (engine / self-play).

Reference: katago/python/katago/game/features.py (the V7 encoder, version >= 10).

Selected spatial channels (original V7 indices):
  0  on-board mask
  1  own stone           2  opponent stone
  3  1 liberty           4  2 liberties        5  3 liberties
  6  simple ko point
  9..13  location of the last 1..5 moves (strict B/W alternation, passes excluded)

Selected global channels (original V7 indices):
  5   selfKomi / 20
  14  passWouldEndPhase
"""
from __future__ import annotations

import numpy as np

from .board import BLACK, EMPTY, PASS, WHITE, Board, opp

SPATIAL_SUBSET = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 17]
GLOBAL_SUBSET = [5, 14]
NUM_SPATIAL = len(SPATIAL_SUBSET)
NUM_GLOBAL = len(GLOBAL_SUBSET)
HISTORY_CHANNELS = [9, 10, 11, 12, 13]  # last 1..5 moves


def decode_npz(binary_packed: np.ndarray, global_nc: np.ndarray, pos_len: int):
    """Unpack katagotraining rows and select our subset.

    binary_packed: (N, 22, ceil(pos_len^2/8)) uint8
    global_nc:     (N, 19) float32
    returns spatial (N, NUM_SPATIAL, pos_len, pos_len) float32, global (N, NUM_GLOBAL) float32
    """
    bin_nchw = np.unpackbits(binary_packed, axis=2)
    bin_nchw = bin_nchw[:, :, : pos_len * pos_len]
    bin_nchw = bin_nchw.reshape(bin_nchw.shape[0], bin_nchw.shape[1], pos_len, pos_len)
    spatial = bin_nchw[:, SPATIAL_SUBSET].astype(np.float32)
    glob = global_nc[:, GLOBAL_SUBSET].astype(np.float32)
    return spatial, glob


def encode_board(board: Board, komi: float, pos_len: int,
                 spatial_subset=SPATIAL_SUBSET, global_subset=GLOBAL_SUBSET):
    """Compute features for `board` with `board.to_move` to play, selecting the given channels.

    The subset defaults to the module's, but callers (e.g. the engine loading a checkpoint)
    pass the subset the model was trained with, so old and new nets both work. The expensive
    ladder search runs only if channels 14/17 are actually requested.
    """
    pla = board.to_move
    other = opp(pla)
    H, W = board.y_size, board.x_size
    # Full V7-indexed spatial buffer, we slice the subset at the end.
    bin_full = np.zeros((22, pos_len, pos_len), dtype=np.float32)
    libs = board.liberty_grid()

    bin_full[0, :H, :W] = 1.0  # on-board
    grid = board.grid
    bin_full[1, :H, :W] = (grid == pla)
    bin_full[2, :H, :W] = (grid == other)
    stone = grid != EMPTY
    bin_full[3, :H, :W] = stone & (libs == 1)
    bin_full[4, :H, :W] = stone & (libs == 2)
    bin_full[5, :H, :W] = stone & (libs == 3)

    if board.simple_ko_point is not None:
        kx, ky = board.simple_ko_point
        bin_full[6, ky, kx] = 1.0

    # Ladder channels 14 (stones in a laddered group) and 17 (working capture moves for
    # opponent 2-liberty laddered groups). Solve once per group via a min-stone key.
    if 14 in spatial_subset or 17 in spatial_subset:
        solved: dict = {}
        for y in range(H):
            for x in range(W):
                c = grid[y, x]
                if c == EMPTY or libs[y, x] not in (1, 2):
                    continue
                key = min(board._group_and_libs(x, y)[0])
                if key not in solved:
                    if libs[y, x] == 1:
                        solved[key] = (board.is_ladder_captured(x, y, True), [])
                    else:
                        wm = board.ladder_working_moves(x, y)
                        solved[key] = (len(wm) > 0, wm)
                laddered, wm = solved[key]
                if laddered:
                    bin_full[14, y, x] = 1.0
                    if c == other and libs[y, x] == 2:  # opponent group we can ladder
                        for wx, wy in wm:
                            bin_full[17, wy, wx] = 1.0

    # History: walk back requiring strict opp, pla, opp, pla, opp alternation.
    expected = [other, pla, other, pla, other]
    hist = board.move_history
    for i, ch in enumerate(HISTORY_CHANNELS):
        idx = len(hist) - 1 - i
        if idx < 0:
            break
        mv_player, mv = hist[idx]
        if mv_player != expected[i]:
            break
        if mv is not PASS:
            mx, my = mv
            bin_full[ch, my, mx] = 1.0

    glob_full = np.zeros(19, dtype=np.float32)
    self_komi = komi if pla == WHITE else -komi
    b_area = W * H
    self_komi = max(-b_area - 1, min(b_area + 1, self_komi))
    glob_full[5] = self_komi / 20.0
    # passWouldEndPhase under area scoring with no encore: the previous move was a pass.
    glob_full[14] = 1.0 if (hist and hist[-1][1] is PASS) else 0.0

    spatial = bin_full[list(spatial_subset)]
    glob = glob_full[list(global_subset)]
    return spatial, glob
