#ifndef XQWL_PREEVAL_INC
#define XQWL_PREEVAL_INC

// Dynamic PST: XQWL cucvlPiecePos is the sole matrix source.
// Endgame shape = uniform per-piece scale of the same table (no ElephantEye imports).
// King/pawn stay static; advisor/bishop get a small major-attack penalty only.

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
  raw = (2 * PST_MIDGAME_TOTAL - raw) * raw / PST_MIDGAME_TOTAL;
  return clampInt(raw, 0, PST_MIDGAME_TOTAL);
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

/** Crossed-river major attack index (no pawns; avoids pawn-rush feedback). */
static int majorAttackWeight(const PositionStruct &pos, int sd) {
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
    } else if (kind == PIECE_CANNON) {
      n += 1;
    }
  }
  const int selfMajor = majorValue(pos, sd);
  const int oppMajor = majorValue(pos, 1 - sd);
  if (selfMajor > oppMajor) {
    n += (selfMajor - oppMajor);
  }
  return clampInt(n, 0, PST_ATTACK_CAP);
}

static BYTE scaledEndgame(int midVal, int endPct) {
  return static_cast<BYTE>((midVal * endPct) / 100);
}

inline int pieceSquareValue(const Context &ctx, int pc, int sq) {
  const int pt = (pc < 16) ? pc - 8 : pc - 16;
  const int sqBase = (pc < 16) ? sq : SQUARE_FLIP(sq);
  switch (pt) {
    case PIECE_KING:
    case PIECE_PAWN:
      return cucvlPiecePos[pt][sqBase];
    case PIECE_ADVISOR:
    case PIECE_BISHOP: {
      const int base = cucvlPiecePos[pt][sqBase];
      const int oppAtk = (pc < 16) ? ctx.blackMajorAttack : ctx.whiteMajorAttack;
      const int penalty = oppAtk * PST_ADV_BISH_PENALTY;
      return base > penalty ? base - penalty : 0;
    }
    case PIECE_KNIGHT:
      return ctx.knight[sqBase];
    case PIECE_ROOK:
      return ctx.rook[sqBase];
    case PIECE_CANNON:
      return ctx.cannon[sqBase];
    default:
      return 0;
  }
}

inline void refresh(PositionStruct &pos) {
  Context &ctx = pos.pst;
  ctx.midgame = midgameWeight(pos);
  ctx.whiteMajorAttack = majorAttackWeight(pos, 0);
  ctx.blackMajorAttack = majorAttackWeight(pos, 1);
  ctx.vlAdvanced = (ADVANCED_VALUE * ctx.midgame + PST_MIDGAME_TOTAL / 2) / PST_MIDGAME_TOTAL;

  for (int sq = 0; sq < 256; ++sq) {
    if (!IN_BOARD(sq)) {
      continue;
    }
    const int knMid = cucvlPiecePos[PIECE_KNIGHT][sq];
    const int rkMid = cucvlPiecePos[PIECE_ROOK][sq];
    const int cnMid = cucvlPiecePos[PIECE_CANNON][sq];
    ctx.knight[sq] = lerp(scaledEndgame(knMid, PST_KNIGHT_END_PCT), knMid, ctx.midgame, PST_MIDGAME_TOTAL);
    ctx.rook[sq] = lerp(scaledEndgame(rkMid, PST_ROOK_END_PCT), rkMid, ctx.midgame, PST_MIDGAME_TOTAL);
    ctx.cannon[sq] = lerp(scaledEndgame(cnMid, PST_CANNON_END_PCT), cnMid, ctx.midgame, PST_MIDGAME_TOTAL);
  }

  pos.vlWhite = PST_ADV_LEAK_BONUS * (PST_ATTACK_CAP - ctx.blackMajorAttack) / PST_ATTACK_CAP;
  pos.vlBlack = PST_ADV_LEAK_BONUS * (PST_ATTACK_CAP - ctx.whiteMajorAttack) / PST_ATTACK_CAP;
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
