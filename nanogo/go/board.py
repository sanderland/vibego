"""A minimal Go board: stones, liberties, captures, simple ko, move history.

Coordinates are (x, y) with x the column (0 = left) and y the row (0 = top), matching
KataGo's tensor convention `pos = y * width + x` and the analysis-engine ownership/policy
ordering (row-major from top-left A19 to bottom-right T1).

This intentionally implements only *simple ko* (not positional/situational superko) and
treats single/multi-stone suicide as illegal. That exactly mirrors KataGo's pure-Python
reference board (`katago/python/katago/game/board.py`) for the feature subset we use, so
the recomputed input features match the training data.
"""
from __future__ import annotations

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
PASS = None  # a pass move is represented by None

COLS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # GTP columns skip "I"


def opp(player: int) -> int:
    return 3 - player


def gtp_to_xy(s: str, y_size: int) -> tuple[int, int] | None:
    """Parse a GTP coordinate like 'Q4' or '(0,13)'. Returns (x, y) or None for pass."""
    s = s.strip()
    if s.lower() == "pass":
        return PASS
    if s.startswith("(") and s.endswith(")"):
        x, y = s[1:-1].split(",")
        return int(x), int(y)
    col = s[0].upper()
    x = COLS.index(col)
    row = int(s[1:])  # GTP rows count from 1 at the bottom
    y = y_size - row
    return x, y


def xy_to_gtp(move: tuple[int, int] | None, y_size: int) -> str:
    if move is PASS:
        return "pass"
    x, y = move
    return f"{COLS[x]}{y_size - y}"


class Board:
    def __init__(self, x_size: int = 19, y_size: int | None = None):
        self.x_size = x_size
        self.y_size = y_size if y_size is not None else x_size
        self.grid = np.zeros((self.y_size, self.x_size), dtype=np.int8)
        self.to_move = BLACK
        self.simple_ko_point: tuple[int, int] | None = None
        # move_history: list of (player, move) where move is (x, y) or PASS
        self.move_history: list[tuple[int, tuple[int, int] | None]] = []
        self._libs: np.ndarray | None = None  # cached liberty grid, invalidated on play()

    def copy(self) -> "Board":
        b = Board(self.x_size, self.y_size)
        b.grid = self.grid.copy()
        b.to_move = self.to_move
        b.simple_ko_point = self.simple_ko_point
        b.move_history = list(self.move_history)
        return b

    def on_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.x_size and 0 <= y < self.y_size

    def neighbors(self, x: int, y: int):
        if x > 0:
            yield x - 1, y
        if x < self.x_size - 1:
            yield x + 1, y
        if y > 0:
            yield x, y - 1
        if y < self.y_size - 1:
            yield x, y + 1

    def _group_and_libs(self, x: int, y: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
        """Flood-fill the group at (x, y); return its stones and its set of liberties."""
        color = self.grid[y, x]
        stack = [(x, y)]
        seen = {(x, y)}
        group = []
        libs: set[tuple[int, int]] = set()
        while stack:
            cx, cy = stack.pop()
            group.append((cx, cy))
            for nx, ny in self.neighbors(cx, cy):
                v = self.grid[ny, nx]
                if v == EMPTY:
                    libs.add((nx, ny))
                elif v == color and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        return group, libs

    def num_liberties(self, x: int, y: int) -> int:
        if self.grid[y, x] == EMPTY:
            return 0
        return len(self._group_and_libs(x, y)[1])

    def liberty_grid(self) -> np.ndarray:
        """Per-stone liberty count for the whole board (one flood-fill per group). Cached;
        the cache is invalidated by play(). encode_board and legal_moves both call this on the
        same unchanged board, so caching saves a full O(area) pass on every evaluated leaf."""
        if self._libs is not None:
            return self._libs
        grid = self.grid
        libs = np.zeros(grid.shape, dtype=np.int32)
        seen = np.zeros(grid.shape, dtype=bool)
        for y in range(self.y_size):
            for x in range(self.x_size):
                if grid[y, x] == EMPTY or seen[y, x]:
                    continue
                group, liberties = self._group_and_libs(x, y)
                n = len(liberties)
                for gx, gy in group:
                    libs[gy, gx] = n
                    seen[gy, gx] = True
        self._libs = libs
        return libs

    def is_legal(self, player: int, move: tuple[int, int] | None) -> bool:
        if move is PASS:
            return True
        x, y = move
        if not self.on_board(x, y) or self.grid[y, x] != EMPTY:
            return False
        if self.simple_ko_point == (x, y):
            return False
        # Simulate to check for suicide.
        self.grid[y, x] = player
        legal = self._has_liberty_after(x, y, player)
        self.grid[y, x] = EMPTY
        return legal

    def _has_liberty_after(self, x: int, y: int, player: int) -> bool:
        # Captures any adjacent enemy group with no liberties -> move is legal.
        for nx, ny in self.neighbors(x, y):
            if self.grid[ny, nx] == opp(player):
                if len(self._group_and_libs(nx, ny)[1]) == 0:
                    return True
        # Otherwise legal iff our own resulting group has a liberty (no suicide).
        return len(self._group_and_libs(x, y)[1]) > 0

    def play(self, player: int, move: tuple[int, int] | None) -> None:
        self._libs = None  # board changes -> invalidate cached liberty grid
        if move is PASS:
            self.move_history.append((player, PASS))
            self.simple_ko_point = None
            self.to_move = opp(player)
            return
        x, y = move
        if not self.is_legal(player, move):
            raise ValueError(f"Illegal move {move} for player {player}")
        self.grid[y, x] = player
        captured: list[tuple[int, int]] = []
        for nx, ny in self.neighbors(x, y):
            if self.grid[ny, nx] == opp(player):
                group, libs = self._group_and_libs(nx, ny)
                if len(libs) == 0:
                    captured.extend(group)
                    for gx, gy in group:
                        self.grid[gy, gx] = EMPTY
        # Simple ko: capturing exactly one stone with a single-stone, single-liberty group.
        self.simple_ko_point = None
        if len(captured) == 1:
            group, libs = self._group_and_libs(x, y)
            if len(group) == 1 and len(libs) == 1:
                self.simple_ko_point = captured[0]
        self.move_history.append((player, move))
        self.to_move = opp(player)

    def legal_moves(self, player: int) -> list[tuple[int, int]]:
        """All legal points for `player`. Uses one liberty pass instead of a flood-fill per
        point: an empty point is legal unless it is the ko point or a suicide (every neighbor
        is the edge, a friendly group with only this liberty, or an enemy group with >1 liberty)."""
        grid = self.grid
        libs = self.liberty_grid()
        other = opp(player)
        ko = self.simple_ko_point
        out = []
        for y in range(self.y_size):
            for x in range(self.x_size):
                if grid[y, x] != EMPTY or (x, y) == ko:
                    continue
                legal = False
                for nx, ny in self.neighbors(x, y):
                    v = grid[ny, nx]
                    if v == EMPTY or (v == player and libs[ny, nx] > 1) or (v == other and libs[ny, nx] == 1):
                        legal = True
                        break
                if legal:
                    out.append((x, y))
        return out

    # ---- ladder search (for KataGo feature channels 14 and 17) ----
    # Ported from katago/python/katago/game/board.py searchIsLadderCaptured /
    # searchIsLadderCapturedAttackerFirst2Libs. Recursive on board copies. Does not implement
    # the rare double-ko heuristic; matches the reference on ordinary ladders.

    def _is_adjacent(self, a, b) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _count_immediate_liberties(self, pt) -> int:
        x, y = pt
        return sum(1 for nx, ny in self.neighbors(x, y) if self.grid[ny, nx] == EMPTY)

    def _liberty_gaining_captures(self, x: int, y: int) -> list[tuple[int, int]]:
        """Points where the group at (x,y) can capture an adjacent enemy group in atari."""
        color = self.grid[y, x]
        enemy = opp(color)
        group, _ = self._group_and_libs(x, y)
        caps, seen = set(), set()
        for gx, gy in group:
            for nx, ny in self.neighbors(gx, gy):
                if self.grid[ny, nx] == enemy and (nx, ny) not in seen:
                    egroup, elibs = self._group_and_libs(nx, ny)
                    seen.update(egroup)
                    if len(elibs) == 1:
                        caps.add(next(iter(elibs)))
        return list(caps)

    def is_ladder_captured(self, x: int, y: int, defender_first: bool) -> bool:
        if self.grid[y, x] == EMPTY:
            return False
        libs = self.num_liberties(x, y)
        if libs > 2 or (defender_first and libs > 1):
            return False
        b = self.copy()
        if defender_first:
            b.simple_ko_point = None  # assume all kos work for the defender at the root
        return b._ladder(x, y, int(self.grid[y, x]), defender_first)

    def _ladder(self, lx: int, ly: int, defender: int, defender_to_move: bool) -> bool:
        if self.grid[ly, lx] != defender:
            return True  # defender group was captured -> attacker wins
        attacker = opp(defender)
        _group, libs = self._group_and_libs(lx, ly)
        nlibs = len(libs)
        if defender_to_move:
            if nlibs >= 2:
                return False  # escaped
            if self.simple_ko_point is not None:
                return False  # ko escape; don't count ko-dependent ladders
            moves = self._liberty_gaining_captures(lx, ly) + list(libs)
            for mv in moves:
                if not self.is_legal(defender, mv):
                    continue
                nb = self.copy()
                nb.play(defender, mv)
                if not nb._ladder(lx, ly, defender, False):
                    return False  # found an escape
            return True
        # attacker to move
        if nlibs <= 1:
            return True
        if nlibs >= 3:
            return False
        m0, m1 = list(libs)
        moves = [m0, m1]
        if not self._is_adjacent(m0, m1):
            l0, l1 = self._count_immediate_liberties(m0), self._count_immediate_liberties(m1)
            if l0 >= 3 and l1 >= 3:
                return False
            elif l0 >= 3:
                moves = [m0]
            elif l1 >= 3:
                moves = [m1]
        for mv in moves:
            if not self.is_legal(attacker, mv):
                continue
            nb = self.copy()
            nb.play(attacker, mv)
            if nb._ladder(lx, ly, defender, True):
                return True
        return False

    def ladder_working_moves(self, x: int, y: int) -> list[tuple[int, int]]:
        """For a 2-liberty group, the attacker first-moves that start a capturing ladder."""
        color = self.grid[y, x]
        if color == EMPTY:
            return []
        _group, libs = self._group_and_libs(x, y)
        if len(libs) != 2:
            return []
        attacker = opp(color)
        working = []
        for mv in list(libs):
            if not self.is_legal(attacker, mv):
                continue
            nb = self.copy()
            nb.play(attacker, mv)
            if nb.is_ladder_captured(x, y, True):
                working.append(mv)
        return working

    def __str__(self) -> str:
        chars = {EMPTY: ".", BLACK: "X", WHITE: "O"}
        rows = []
        for y in range(self.y_size):
            rows.append(" ".join(chars[self.grid[y, x]] for x in range(self.x_size)))
        return "\n".join(rows)
