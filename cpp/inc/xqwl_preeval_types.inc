#ifndef XQWL_PREEVAL_TYPES_INC
#define XQWL_PREEVAL_TYPES_INC

// Per-position PST context (red-oriented tables; black uses SQUARE_FLIP at lookup).
namespace xqwl_pst {

struct Context {
  int vlAdvanced = ADVANCED_VALUE;
  int midgame = 0;
  int whiteAttack = 0;
  int blackAttack = 0;
  BYTE king[256];
  BYTE knight[256];
  BYTE rook[256];
  BYTE cannon[256];
  BYTE pawnAtt[256];
  BYTE pawnLess[256];
};

} // namespace xqwl_pst

#endif
