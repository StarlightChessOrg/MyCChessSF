#pragma once
// Portable types/constants extracted from XQWL06.CPP (no Win32 GUI / search)
#include <cstdint>
#include <cstring>

#ifndef TRUE
#define TRUE true
#endif
#ifndef FALSE
#define FALSE false
#endif

using BYTE = std::uint8_t;
using BOOL = bool;
using DWORD = std::uint32_t;
using WORD = std::uint16_t;

// Board coordinate bounds (256-square mailbox)
const int RANK_TOP = 3;
const int RANK_BOTTOM = 12;
const int FILE_LEFT = 3;
const int FILE_RIGHT = 11;

// Piece type indices (0..6)
const int PIECE_KING = 0;
const int PIECE_ADVISOR = 1;
const int PIECE_BISHOP = 2;
const int PIECE_KNIGHT = 3;
const int PIECE_ROOK = 4;
const int PIECE_CANNON = 5;
const int PIECE_PAWN = 6;

const int MAX_GEN_MOVES = 128;
const int MAX_MOVES = 1024;  // Raised from 256 for long game records
const int LIMIT_DEPTH = 64;
const int MATE_VALUE = 10000;
const int BAN_VALUE = MATE_VALUE - 100;
const int WIN_VALUE = MATE_VALUE - 200;
const int DRAW_VALUE = 20;
const int ADVANCED_VALUE = 3;
const int NULL_OKAY_MARGIN = 200;
const int NULL_SAFE_MARGIN = 400;

// Search constants (XQWL 0.6)
const int RANDOM_MASK = 7;
const int NULL_DEPTH = 2;
const int HASH_SIZE = 1 << 20;
const int HASH_ALPHA = 1;
const int HASH_BETA = 2;
const int HASH_PV = 3;
const int ASPIRATION_DELTA_INIT = 16;
const int ASPIRATION_DELTA_MAX = 200;
const int ASPIRATION_MAX_FAILS = 6;
const int ASPIRATION_MIN_DEPTH = 4;
const int LMR_MOVE_THRESHOLD = 3;
const int LMR_MIN_DEPTH = 3;
const int QUIESCENCE_TT_DEPTH = 0;
const int NULL_ATTACKING_MIN = 3; // ElephantEye: unverified null move when attack pieces >= this
const int IID_MIN_DEPTH = 4;
const int SINGULAR_MIN_DEPTH = 6;
const int SINGULAR_MARGIN = 12;
const int BOOK_MAX_ENTRIES = 5000000; // max opening-book lines loaded into RAM

// Dynamic PST blend (XQWL cucvlPiecePos only; see xqwl_preeval.inc)
const int PST_MIDGAME_TOTAL = 66;
const int PST_ATTACK_CAP = 8;
const int PST_KNIGHT_END_PCT = 98; // endgame PST as % of XQWL mid table (same shape)
const int PST_ROOK_END_PCT = 90;
const int PST_CANNON_END_PCT = 95;
const int PST_ADV_BISH_PENALTY = 2;  // subtract per opponent major-attack point
const int PST_ADV_LEAK_BONUS = 60;   // bonus when opponent major attack is low
