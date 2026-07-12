"""HalfKAv2_hm sparse features (Pikafish xiangqi, king + attack buckets + mid mirror).

Ported from official-pikafish/Pikafish ``src/nnue/features/half_ka_v2_hm.{h,cpp}``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mycchess_sf.fen_parse import parse_fen_board

PS_NB = 689
ATTACK_BUCKET_NB = 4
N_FEATURES = 6 * ATTACK_BUCKET_NB * PS_NB  # 16536

WHITE = 0
BLACK = 1

NO_PIECE = 0
W_ROOK, W_ADVISOR, W_CANNON, W_PAWN, W_KNIGHT, W_BISHOP, W_KING = range(1, 8)
B_ROOK, B_ADVISOR, B_CANNON, B_PAWN, B_KNIGHT, B_BISHOP, B_KING = range(9, 16)

PIECE_NB = 16

ALL_PIECES = (
    W_ROOK,
    W_ADVISOR,
    W_CANNON,
    W_PAWN,
    W_KNIGHT,
    W_BISHOP,
    W_KING,
    B_ROOK,
    B_ADVISOR,
    B_CANNON,
    B_PAWN,
    B_KNIGHT,
    B_BISHOP,
    B_KING,
)

FEN_CHAR_TO_PIECE = {
    "R": W_ROOK,
    "A": W_ADVISOR,
    "C": W_CANNON,
    "P": W_PAWN,
    "N": W_KNIGHT,
    "B": W_BISHOP,
    "K": W_KING,
    "r": B_ROOK,
    "a": B_ADVISOR,
    "c": B_CANNON,
    "p": B_PAWN,
    "n": B_KNIGHT,
    "b": B_BISHOP,
    "k": B_KING,
}

BALANCE_ENCODING = 0xA4A92A74E989D3A7

FILE_E = 4
RANK_0 = 0
RANK_9 = 9


def _make_square(file_idx: int, rank: int) -> int:
    return rank * 9 + file_idx


def _file_of(sq: int) -> int:
    return sq % 9


def _rank_of(sq: int) -> int:
    return sq // 9


def _flip_rank(sq: int) -> int:
    return _make_square(_file_of(sq), RANK_9 - _rank_of(sq))


def _flip_file(sq: int) -> int:
    return _make_square(8 - _file_of(sq), _rank_of(sq))


def _bb_from_u128(lo: int, hi: int) -> tuple[int, int]:
    return lo & ((1 << 64) - 1), hi & ((1 << 64) - 1)


def _bb_test(lo: int, hi: int, sq: int) -> bool:
    if sq < 64:
        return bool(lo & (1 << sq))
    return bool(hi & (1 << (sq - 64)))


def _bb_or(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] | b[0], a[1] | b[1]


def _bb_and(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] & b[0], a[1] & b[1]


def _bb_not(a: tuple[int, int]) -> tuple[int, int]:
    mask_lo = (1 << 64) - 1
    mask_hi = (1 << (90 - 64)) - 1
    return (~a[0]) & mask_lo, (~a[1]) & mask_hi


def _bb_shift(a: tuple[int, int], n: int) -> tuple[int, int]:
    lo, hi = a
    mask_lo = (1 << 64) - 1
    mask_hi = (1 << (90 - 64)) - 1
    if n <= 0:
        return lo, hi
    if n >= 128:
        return 0, 0
    if n >= 64:
        return 0, (lo << (n - 64)) & mask_hi
    new_hi = ((hi << n) | (lo >> (64 - n))) & mask_hi
    new_lo = (lo << n) & mask_lo
    return new_lo, new_hi


def _bb_to_squares(bb: tuple[int, int]) -> frozenset[int]:
    lo, hi = bb
    return frozenset(s for s in range(90) if _bb_test(lo, hi, s))


FILE_A_BB = _bb_from_u128(0x8040201008040201, 0x20100)
FILE_B_BB = _bb_shift(FILE_A_BB, 1)
FILE_C_BB = _bb_shift(FILE_A_BB, 2)
FILE_D_BB = _bb_shift(FILE_A_BB, 3)
FILE_E_BB = _bb_shift(FILE_A_BB, 4)
FILE_F_BB = _bb_shift(FILE_A_BB, 5)
FILE_G_BB = _bb_shift(FILE_A_BB, 6)
FILE_H_BB = _bb_shift(FILE_A_BB, 7)
FILE_I_BB = _bb_shift(FILE_A_BB, 8)

RANK_0_BB = _bb_from_u128(0x1FF, 0)
RANK_1_BB = _bb_shift(RANK_0_BB, 9)
RANK_2_BB = _bb_shift(RANK_0_BB, 18)
RANK_3_BB = _bb_shift(RANK_0_BB, 27)
RANK_4_BB = _bb_shift(RANK_0_BB, 36)
RANK_5_BB = _bb_shift(RANK_0_BB, 45)
RANK_6_BB = _bb_shift(RANK_0_BB, 54)
RANK_7_BB = _bb_shift(RANK_0_BB, 63)
RANK_8_BB = _bb_shift(RANK_0_BB, 72)
RANK_9_BB = _bb_shift(RANK_0_BB, 81)

HalfBB_WHITE = _bb_or(
    RANK_0_BB,
    _bb_or(RANK_1_BB, _bb_or(RANK_2_BB, _bb_or(RANK_3_BB, RANK_4_BB))),
)
HalfBB_BLACK = _bb_or(
    RANK_5_BB,
    _bb_or(RANK_6_BB, _bb_or(RANK_7_BB, _bb_or(RANK_8_BB, RANK_9_BB))),
)

PawnFileBB = _bb_or(
    FILE_A_BB,
    _bb_or(FILE_C_BB, _bb_or(FILE_E_BB, _bb_or(FILE_G_BB, FILE_I_BB))),
)
# Pikafish: PawnBB[2] = { HalfBB[BLACK]|Rank3|4, HalfBB[WHITE]|Rank5|6 } but
# ValidBB uses PawnBB[WHITE] for W_PAWN and PawnBB[BLACK] for B_PAWN — the array
# index follows Color enum, not the HalfBB color used in each initializer entry.
PawnBB_WHITE = _bb_or(
    HalfBB_BLACK,
    _bb_and(_bb_or(RANK_3_BB, RANK_4_BB), PawnFileBB),
)
PawnBB_BLACK = _bb_or(
    HalfBB_WHITE,
    _bb_and(_bb_or(RANK_6_BB, RANK_5_BB), PawnFileBB),
)

Palace = _bb_from_u128(0xE07038, 0x70381C)


def _build_valid_bb() -> tuple[frozenset[int], ...]:
    table: list[frozenset[int]] = [frozenset() for _ in range(16)]
    table[W_ROOK] = _bb_to_squares(_bb_or(HalfBB_WHITE, HalfBB_BLACK))
    table[W_ADVISOR] = _bb_to_squares(
        _bb_or(
            _bb_and(_bb_or(RANK_0_BB, RANK_2_BB), _bb_or(FILE_D_BB, FILE_F_BB)),
            _bb_and(RANK_1_BB, FILE_E_BB),
        )
    )
    table[W_CANNON] = table[W_ROOK]
    table[W_PAWN] = _bb_to_squares(PawnBB_WHITE)
    table[W_KNIGHT] = table[W_ROOK]
    table[W_BISHOP] = _bb_to_squares(
        _bb_or(
            _bb_and(_bb_or(RANK_0_BB, RANK_4_BB), _bb_or(FILE_C_BB, FILE_G_BB)),
            _bb_and(RANK_2_BB, _bb_or(FILE_A_BB, _bb_or(FILE_E_BB, FILE_I_BB))),
        )
    )
    table[W_KING] = _bb_to_squares(
        _bb_and(_bb_and(HalfBB_WHITE, Palace), _bb_not(FILE_F_BB))
    )
    table[B_ROOK] = table[W_ROOK]
    table[B_ADVISOR] = _bb_to_squares(
        _bb_or(
            _bb_and(_bb_or(RANK_7_BB, RANK_9_BB), _bb_or(FILE_D_BB, FILE_F_BB)),
            _bb_and(RANK_8_BB, FILE_E_BB),
        )
    )
    table[B_CANNON] = table[W_ROOK]
    table[B_PAWN] = _bb_to_squares(PawnBB_BLACK)
    table[B_KNIGHT] = table[W_ROOK]
    table[B_BISHOP] = _bb_to_squares(
        _bb_or(
            _bb_and(_bb_or(RANK_5_BB, RANK_9_BB), _bb_or(FILE_C_BB, FILE_G_BB)),
            _bb_and(RANK_7_BB, _bb_or(FILE_A_BB, _bb_or(FILE_E_BB, FILE_I_BB))),
        )
    )
    table[B_KING] = _bb_to_squares(_bb_and(HalfBB_BLACK, Palace))
    return tuple(table)


VALID_BB = _build_valid_bb()

# PSQ_OFFSETS built lazily after table self-check in _ensure_psq_offsets().
PSQ_OFFSETS: tuple[tuple[int, ...], ...] | None = None


def _build_king_buckets_raw() -> list[int]:
    """Per-square king bucket nibble + mirror flag (Pikafish KingBuckets[])."""
    def m(s: int) -> int:
        return (1 << 3) | s

    rows = [
        [0, 0, 0, 0, 1, m(0), 0, 0, 0],
        [0, 0, 0, 2, 3, m(2), 0, 0, 0],
        [0, 0, 0, 4, 5, m(4), 0, 0, 0],
        [0] * 9,
        [0] * 9,
        [0] * 9,
        [0] * 9,
        [0, 0, 0, 4, 5, m(4), 0, 0, 0],
        [0, 0, 0, 2, 3, m(2), 0, 0, 0],
        [0, 0, 0, 0, 1, m(0), 0, 0, 0],
    ]
    out: list[int] = []
    for rank in range(10):
        for file_idx in range(9):
            out.append(rows[rank][file_idx])
    assert len(out) == 90
    return out


def _build_king_buckets() -> tuple[tuple[tuple[tuple[int, bool], ...], ...], ...]:
    king_buckets_raw = _build_king_buckets_raw()

    table: list[list[list[tuple[int, bool]]]] = [
        [[(0, False) for _ in range(2)] for _ in range(90)] for _ in range(90)
    ]
    for ksq in range(90):
        for oksq in range(90):
            for midm in (0, 1):
                king_bucket_ = king_buckets_raw[ksq]
                king_bucket = king_bucket_ & 0x7
                oking_bucket = king_buckets_raw[oksq] & 0x7
                mirror = bool(
                    (king_bucket_ >> 3)
                    or (
                        (king_bucket & 1)
                        and (
                            (king_buckets_raw[oksq] >> 3)
                            or (bool(oking_bucket & 1) and midm)
                        )
                    )
                )
                table[ksq][oksq][midm] = (king_bucket, mirror)
    return tuple(tuple(tuple(row) for row in plane) for plane in table)


KING_BUCKETS = _build_king_buckets()


def _build_index_map() -> tuple[tuple[tuple[int, ...], ...], ...]:
    table: list[list[list[int]]] = [[[0] * 90 for _ in range(2)] for _ in range(2)]
    for mirror in (0, 1):
        for rotate in (0, 1):
            for sq in range(90):
                ss = sq
                if mirror:
                    ss = _flip_file(ss)
                if rotate:
                    ss = _flip_rank(ss)
                table[mirror][rotate][sq] = ss
    return tuple(tuple(tuple(row) for row in plane) for plane in table)


INDEX_MAP = _build_index_map()


def _build_mid_mirror_encoding() -> tuple[tuple[int, ...], ...]:
    shifts = {
        0: (0, 0),
        1: (44, 0),  # ROOK
        2: (60, 36),  # ADVISOR
        3: (47, 7),  # CANNON
        4: (53, 21),  # PAWN
        5: (50, 14),  # KNIGHT
        6: (57, 29),  # BISHOP
        7: (0, 0),  # KING
    }
    encodings = [[0] * 90 for _ in range(PIECE_NB)]
    for color in (WHITE, BLACK):
        for pt in range(1, 8):  # ROOK..KING
            for r in range(10):
                for f in range(9):
                    encoding = 0
                    if f != FILE_E and pt != 7:
                        r_ = r if color == WHITE else RANK_9 - r
                        f_ = f if f < FILE_E else 8 - f
                        s1, s2 = shifts[pt]
                        encoding = (1 << s1) | (((3 - f_) * 10 + r_) << s2)
                        if f >= FILE_E:
                            encoding = (-encoding) & 0xFFFFFFFFFFFFFFFF
                    elif f != FILE_E and pt == 7:
                        encoding = 1 << 63
                    piece = (color << 3) + pt
                    sq = _make_square(f, r)
                    encodings[piece][sq] = encoding & 0xFFFFFFFFFFFFFFFF
    return tuple(tuple(row) for row in encodings)


MID_MIRROR_ENCODING = _build_mid_mirror_encoding()


def _build_psq_offsets() -> tuple[tuple[int, ...], ...]:
    offsets = [[0] * 90 for _ in range(PIECE_NB)]
    cumulative = 0
    for pc in ALL_PIECES:
        for sq in range(90):
            if sq in VALID_BB[pc]:
                offsets[pc][sq] = cumulative
                cumulative += 1
    if cumulative != PS_NB:
        raise ValueError(f"PS_NB mismatch: {cumulative} != {PS_NB}")
    return tuple(tuple(row) for row in offsets)


def _ensure_psq_offsets() -> tuple[tuple[int, ...], ...]:
    global PSQ_OFFSETS
    if PSQ_OFFSETS is None:
        PSQ_OFFSETS = _build_psq_offsets()
    return PSQ_OFFSETS


def _build_attack_bucket() -> tuple[tuple[tuple[int, ...], ...], ...]:
    table = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(3)]
    for rook in range(3):
        for knight in range(3):
            for cannon in range(3):
                table[rook][knight][cannon] = int(bool(rook)) * 2 + bool(knight + cannon)
    return tuple(tuple(tuple(row) for row in plane) for plane in table)


ATTACK_BUCKET = _build_attack_bucket()


def _fen_row_to_pikafish_rank(row_y: int) -> int:
    """MyCChessSF FEN row 0 = Pikafish rank 9 (first FEN rank)."""
    return RANK_9 - row_y


def _parse_board(fen: str) -> tuple[list[int], int, int]:
    board, red_to_move = parse_fen_board(fen)
    pieces = [NO_PIECE] * 90
    counts = [[0] * 8 for _ in range(2)]

    for y in range(10):
        rank = _fen_row_to_pikafish_rank(y)
        for x in range(9):
            ch = board[y, x]
            if not ch:
                continue
            sq = _make_square(x, rank)
            pc = FEN_CHAR_TO_PIECE[ch]
            pieces[sq] = pc
            color = pc >> 3
            pt = pc & 7
            counts[color][pt] += 1

    perspective = WHITE if red_to_move else BLACK
    return pieces, perspective, counts


def _king_square(pieces: list[int], color: int) -> int:
    king = W_KING if color == WHITE else B_KING
    for sq, pc in enumerate(pieces):
        if pc == king:
            return sq
    raise ValueError(f"Missing king for color {color}")


def _mid_encoding(pieces: list[int], color: int) -> int:
    enc = BALANCE_ENCODING
    for sq, pc in enumerate(pieces):
        if pc == NO_PIECE:
            continue
        if (pc >> 3) == color:
            enc = (enc + MID_MIRROR_ENCODING[pc][sq]) & 0xFFFFFFFFFFFFFFFF
    return enc


def requires_mid_mirror(pieces: list[int], perspective: int) -> bool:
    enc_us = _mid_encoding(pieces, perspective)
    enc_them = _mid_encoding(pieces, perspective ^ 1)
    return bool(
        ((1 << 63) & enc_us & enc_them)
        and (
            enc_us < BALANCE_ENCODING
            or (enc_us == BALANCE_ENCODING and enc_them < BALANCE_ENCODING)
        )
    )


def make_attack_bucket(counts: list[list[int]], perspective: int) -> int:
    c = perspective
    return ATTACK_BUCKET[counts[c][1]][counts[c][5]][counts[c][3]]


def make_feature_bucket(
    pieces: list[int], counts: list[list[int]], perspective: int
) -> tuple[int, bool, int]:
    ksq = _king_square(pieces, perspective)
    oksq = _king_square(pieces, perspective ^ 1)
    midm = int(requires_mid_mirror(pieces, perspective))
    king_bucket, mirror = KING_BUCKETS[ksq][oksq][midm]
    attack_bucket = make_attack_bucket(counts, perspective)
    bucket = king_bucket * ATTACK_BUCKET_NB + attack_bucket
    return bucket, mirror, attack_bucket


def make_index(perspective: int, sq: int, pc: int, bucket: int, mirror: bool) -> int:
    mapped_sq = INDEX_MAP[int(mirror)][int(perspective == BLACK)][sq]
    if perspective == BLACK:
        pc = pc ^ 8
    offsets = _ensure_psq_offsets()
    return offsets[pc][mapped_sq] + PS_NB * bucket


def active_indices_for_perspective(
    pieces: list[int], counts: list[list[int]], perspective: int
) -> list[int]:
    bucket, mirror, _attack = make_feature_bucket(pieces, counts, perspective)
    out: list[int] = []
    for sq, pc in enumerate(pieces):
        if pc == NO_PIECE:
            continue
        if sq not in VALID_BB[pc]:
            continue
        out.append(make_index(perspective, sq, pc, bucket, mirror))
    return sorted(out)


def fen_to_feature_indices(fen: str) -> np.ndarray:
    """Return active HalfKAv2_hm indices for side-to-move perspective."""
    pieces, perspective, counts = _parse_board(fen)
    indices = active_indices_for_perspective(pieces, counts, perspective)
    return np.array(indices, dtype=np.int64)


def fen_to_feature_indices_both_perspectives(fen: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (stm, other) perspective indices — useful for Pikafish parity checks."""
    pieces, perspective, counts = _parse_board(fen)
    stm = active_indices_for_perspective(pieces, counts, perspective)
    other = active_indices_for_perspective(pieces, counts, perspective ^ 1)
    return np.array(stm, dtype=np.int64), np.array(other, dtype=np.int64)
