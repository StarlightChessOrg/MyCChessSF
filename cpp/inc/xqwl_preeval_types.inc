#ifndef XQWL_PREEVAL_TYPES_INC
#define XQWL_PREEVAL_TYPES_INC

// Blended PST built only from XQWL static cucvlPiecePos (see xqwl_preeval.inc).
namespace xqwl_pst {

struct Context {
  int vlAdvanced = ADVANCED_VALUE;
  int midgame = 0;
  int whiteMajorAttack = 0; // crossed major pieces; for advisor leak / penalty only
  int blackMajorAttack = 0;
  BYTE knight[256];
  BYTE rook[256];
  BYTE cannon[256];
};

} // namespace xqwl_pst

#endif
