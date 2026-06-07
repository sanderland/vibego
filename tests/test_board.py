from nanogo.go.board import BLACK, EMPTY, WHITE, Board, gtp_to_xy, opp, xy_to_gtp


def test_capture_single_stone():
    b = Board(5)
    # Black surrounds the white stone at (1,1) on all four sides.
    for p, m in [(BLACK, (1, 0)), (WHITE, (1, 1)), (BLACK, (0, 1)),
                 (WHITE, (4, 4)), (BLACK, (2, 1)), (WHITE, (4, 3))]:
        b.play(p, m)
    assert b.grid[1, 1] == WHITE
    b.play(BLACK, (1, 2))  # last liberty -> capture
    assert b.grid[1, 1] == EMPTY


def test_suicide_is_illegal():
    b = Board(3)
    # Black holds the four cross points around the center; white center is suicide.
    for m in [(1, 0), (0, 1), (2, 1), (1, 2)]:
        b.play(BLACK, m)
    assert not b.is_legal(WHITE, (1, 1))


def test_simple_ko():
    # Standard ko: black plays (2,1) capturing the single white stone at (1,1),
    # leaving a single black stone at (2,1) with one liberty -> white may not retake.
    #   col:  0 1 2 3
    #   row0: . B W .
    #   row1: B W . W      (white (1,1) to be captured; white wall (3,1),(2,0),(2,2))
    #   row2: . B W .
    b = Board(5)
    for m in [(3, 1), (2, 0), (2, 2)]:
        b.play(WHITE, m)
    for m in [(0, 1), (1, 0), (1, 2)]:
        b.play(BLACK, m)
    b.play(WHITE, (1, 1))
    b.play(BLACK, (2, 1))  # captures white (1,1)
    assert b.grid[1, 1] == EMPTY
    assert b.simple_ko_point == (1, 1)
    assert not b.is_legal(WHITE, (1, 1))  # immediate recapture forbidden
    b.play(WHITE, None)  # pass clears the ko ban
    assert b.is_legal(WHITE, (1, 1))


def test_pass_clears_ko():
    b = Board(5)
    b.simple_ko_point = (2, 2)
    b.to_move = WHITE
    b.play(WHITE, None)
    assert b.simple_ko_point is None


def test_gtp_roundtrip():
    for s in ["A1", "Q4", "T19", "K10"]:
        x, y = gtp_to_xy(s, 19)
        assert xy_to_gtp((x, y), 19) == s
    assert gtp_to_xy("pass", 19) is None
    x, _ = gtp_to_xy("J1", 19)  # 'I' is skipped
    assert x == 8


def test_opp():
    assert opp(BLACK) == WHITE
    assert opp(WHITE) == BLACK
