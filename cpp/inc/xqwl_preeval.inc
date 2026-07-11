#ifndef XQWL_PREEVAL_INC
#define XQWL_PREEVAL_INC

#include "xqwl_preeval_tables.inc"

// Dynamic PST refresh: opening/middle/endgame material mix + attack posture blend.
// Inspired by ElephantEye PreEvaluate, adapted to our mailbox board and score layout.

namespace xqwl_pst {

inline BYTE lerp(int hi, int lo, int wHi, int total) {
  return static_cast<BYTE>((hi * wHi + lo * (total - wHi)) / total);
}

inline int clampInt(int v, int lo, int hi) {
  if (v < lo) {
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return v;
}

static int midgameWeight(const PositionStruct &pos) {
  constexpr int kTotal = 66; // 4 rooks*6 + 8 knights/cannons*3 + 18 others*1
  int raw = 0;
  for (int sq = 0; sq < 256; ++sq) {
    if (!IN_BOARD(sq)) {
      continue;
    }
    const BYTE pc = pos.ucpcSquares[sq];
    if (pc == 0) {
      continue;
    }
    const int kind = (pc < 16) ? pc - 8 : pc - 16;
    switch (kind) {
      case PIECE_ROOK:
        raw += 6;
        break;
      case PIECE_KNIGHT:
      case PIECE_CANNON:
        raw += 3;
        break;
      default:
        raw += 1;
        break;
    }
  }
  raw = (2 * kTotal - raw) * raw / kTotal;
  return clampInt(raw, 0, kTotal);
}

static int majorValue(const PositionStruct &pos, int sd) {
  int v = 0;
  for (int sq = 0; sq < 256; ++sq) {
    if (!IN_BOARD(sq)) {
      continue;
    }
    const BYTE pc = pos.ucpcSquares[sq];
    if (pc == 0) {
      continue;
    }
    const BOOL ours = (pc < 16) ? (sd == 0) : (sd == 1);
    if (!ours) {
      continue;
    }
    const int kind = (pc < 16) ? pc - 8 : pc - 16;
    if (kind == PIECE_ROOK) {
      v += 2;
    } else if (kind == PIECE_KNIGHT || kind == PIECE_CANNON) {
      v += 1;
    }
  }
  return v;
}

static int attackWeight(const PositionStruct &pos, int sd) {
  constexpr int kCap = 8;
  int n = 0;
  for (int sq = 0; sq < 256; ++sq) {
    if (!IN_BOARD(sq)) {
      continue;
    }
    const BYTE pc = pos.ucpcSquares[sq];
    if (pc == 0) {
      continue;
    }
    const BOOL ours = (pc < 16) ? (sd == 0) : (sd == 1);
    if (!ours || !AWAY_HALF(sq, sd)) {
      continue;
    }
    const int kind = (pc < 16) ? pc - 8 : pc - 16;
    if (kind == PIECE_KNIGHT || kind == PIECE_ROOK) {
      n += 2;
    } else if (kind == PIECE_CANNON || kind == PIECE_PAWN) {
      n += 1;
    }
  }
  const int selfMajor = majorValue(pos, sd);
  const int oppMajor = majorValue(pos, 1 - sd);
  if (selfMajor > oppMajor) {
    n += (selfMajor - oppMajor) * 2;
  }
  return clampInt(n, 0, kCap);
}

inline int pieceSquareValue(const Context &ctx, int pc, int sq) {
  const int pt = (pc < 16) ? pc - 8 : pc - 16;
  const int sqBase = (pc < 16) ? sq : SQUARE_FLIP(sq);
  switch (pt) {
    case PIECE_KING:
      return ctx.king[sqBase];
    case PIECE_ADVISOR:
    case PIECE_BISHOP: {
      const int oppAtk = (pc < 16) ? ctx.blackAttack : ctx.whiteAttack;
      return lerp(kAdvBishThreat[sqBase], cucvlPiecePos[pt][sqBase], oppAtk, 8);
    }
    case PIECE_KNIGHT:
      return ctx.knight[sqBase];
    case PIECE_ROOK:
      return ctx.rook[sqBase];
    case PIECE_CANNON:
      return ctx.cannon[sqBase];
    case PIECE_PAWN: {
      const int ownAtk = (pc < 16) ? ctx.whiteAttack : ctx.blackAttack;
      return lerp(ctx.pawnAtt[sqBase], ctx.pawnLess[sqBase], ownAtk, 8);
    }
    default:
      return 0;
  }
}

inline void refresh(PositionStruct &pos) {
  constexpr int kMidTotal = 66;
  constexpr int kAtkTotal = 8;
  constexpr int kAdvLeak = 80;

  Context &ctx = pos.pst;
  ctx.midgame = midgameWeight(pos);
  ctx.whiteAttack = attackWeight(pos, 0);
  ctx.blackAttack = attackWeight(pos, 1);
  ctx.vlAdvanced = (ADVANCED_VALUE * ctx.midgame + kMidTotal / 2) / kMidTotal;

  for (int sq = 0; sq < 256; ++sq) {
    if (!IN_BOARD(sq)) {
      continue;
    }
    ctx.king[sq] = lerp(kKingPawnMidAtt[sq], kKingPawnEndAtt[sq], ctx.midgame, kMidTotal);
    ctx.knight[sq] = lerp(cucvlPiecePos[PIECE_KNIGHT][sq], kKnightEnd[sq], ctx.midgame, kMidTotal);
    ctx.rook[sq] = lerp(cucvlPiecePos[PIECE_ROOK][sq], kRookEnd[sq], ctx.midgame, kMidTotal);
    ctx.cannon[sq] = lerp(cucvlPiecePos[PIECE_CANNON][sq], kCannonEnd[sq], ctx.midgame, kMidTotal);
    ctx.pawnAtt[sq] = ctx.king[sq];
    ctx.pawnLess[sq] = lerp(kKingPawnMidLess[sq], kKingPawnEndLess[sq], ctx.midgame, kMidTotal);
  }

  pos.vlWhite = kAdvLeak * (kAtkTotal - ctx.blackAttack) / kAtkTotal;
  pos.vlBlack = kAdvLeak * (kAtkTotal - ctx.whiteAttack) / kAtkTotal;
  for (int sq = 0; sq < 256; ++sq) {
    if (!IN_BOARD(sq)) {
      continue;
    }
    const BYTE pc = pos.ucpcSquares[sq];
    if (pc == 0) {
      continue;
    }
    if (pc < 16) {
      pos.vlWhite += pieceSquareValue(ctx, pc, sq);
    } else {
      pos.vlBlack += pieceSquareValue(ctx, pc, sq);
    }
  }
}

} // namespace xqwl_pst

#endif
