"""Self-play between two networks, plus Tromp-Taylor area scoring to decide the winner."""
from __future__ import annotations

import random

from ..engine.search import MCTS, NNEvaluator, rank_children
from ..go.board import BLACK, EMPTY, PASS, WHITE, Board, xy_to_gtp


def area_score(board: Board, komi: float) -> float:
    """Tromp-Taylor area score from Black's perspective: black area - white area - komi.

    Each empty region bordered by only one color is that color's territory; regions touching
    both colors (dame) count for neither. Black wins iff the result is > 0.
    """
    grid = board.grid
    black = int((grid == BLACK).sum())
    white = int((grid == WHITE).sum())
    seen = set()
    bt = wt = 0
    for y in range(board.y_size):
        for x in range(board.x_size):
            if grid[y, x] != EMPTY or (x, y) in seen:
                continue
            region, borders, stack = [], set(), [(x, y)]
            seen.add((x, y))
            while stack:
                cx, cy = stack.pop()
                region.append((cx, cy))
                for nx, ny in board.neighbors(cx, cy):
                    v = grid[ny, nx]
                    if v == EMPTY:
                        if (nx, ny) not in seen:
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                    else:
                        borders.add(int(v))
            if borders == {BLACK}:
                bt += len(region)
            elif borders == {WHITE}:
                wt += len(region)
    return (black + bt) - (white + wt) - komi


def _choose(root, move_idx, opening_moves, temperature, rng):
    children = [c for c in root.children if c.N > 0]
    if not children:
        return PASS
    if move_idx < opening_moves and temperature > 0:  # opening: sample ~ visit counts
        weights = [c.N ** (1.0 / temperature) for c in children]
        total = sum(weights)
        r = rng.random() * total
        acc = 0.0
        for c, w in zip(children, weights):
            acc += w
            if acc >= r:
                return c.move
        return children[-1].move
    return rank_children(children)[0].move  # otherwise pick the LCB-best move


def play_game(eval_black: NNEvaluator, eval_white: NNEvaluator, pos_len: int,
              board_size: int = 19, komi: float = 7.5, visits: int = 100,
              opening_moves: int = 8, temperature: float = 1.0, c_puct: float = 1.0,
              leaf_batch: int = 16, max_moves: int | None = None, seed: int = 0) -> dict:
    """Play one game; black uses eval_black, white uses eval_white. Returns winner + score."""
    rng = random.Random(seed)
    board = Board(board_size, board_size)
    if max_moves is None:
        max_moves = board_size * board_size * 2
    evals = {BLACK: eval_black, WHITE: eval_white}
    passes = 0
    sgf_moves = []
    for move_idx in range(max_moves):
        mcts = MCTS(evals[board.to_move], komi, pos_len, c_puct=c_puct)
        root = mcts.run(board, visits, batch_size=leaf_batch)
        mv = _choose(root, move_idx, opening_moves, temperature, rng)
        sgf_moves.append((board.to_move, xy_to_gtp(mv, board_size)))
        board.play(board.to_move, mv)
        passes = passes + 1 if mv is PASS else 0
        if passes >= 2:
            break
    score = area_score(board, komi)
    winner = "B" if score > 0 else ("W" if score < 0 else "D")
    return {"winner": winner, "score_black": score, "moves": len(sgf_moves), "sgf_moves": sgf_moves}
