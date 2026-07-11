// Inline FEN encoder; include after PositionStruct is defined (via xqwl_extract.inc).
#ifndef XQWL_FEN_INC
#define XQWL_FEN_INC

#include <string>

namespace xqwl_fen_detail {

inline int coord_xy_iccs(int x, int y) { return (x + 3) + ((y + 3) << 4); }

inline char piece_to_fen_char(BYTE pc) {
  if (!pc)
    return '\0';
  static const char *RED = "KABNRCP";
  static const char *BLK = "kabnrcp";
  if (pc < 16) {
    const int t = int(pc) - 8;
    if (t < 0 || t > 6)
      return '?';
    return RED[t];
  }
  const int t = int(pc) - 16;
  if (t < 0 || t > 6)
    return '?';
  return BLK[t];
}

inline std::string position_to_fen(const PositionStruct &pst) {
  std::string fen;
  fen.reserve(80);
  for (int y = 0; y < 10; ++y) {
    int empty = 0;
    for (int x = 0; x < 9; ++x) {
      const int sq = coord_xy_iccs(x, y);
      const BYTE pc = pst.ucpcSquares[sq];
      if (!pc) {
        ++empty;
        continue;
      }
      if (empty) {
        fen += char('0' + empty);
        empty = 0;
      }
      fen += piece_to_fen_char(pc);
    }
    if (empty)
      fen += char('0' + empty);
    if (y < 9)
      fen += '/';
  }
  fen += (pst.sdPlayer == 0) ? " w - - 0 1" : " b - - 0 1";
  return fen;
}

} // namespace xqwl_fen_detail

inline std::string xqwl_position_to_fen(const PositionStruct &pst) {
  return xqwl_fen_detail::position_to_fen(pst);
}

#endif // XQWL_FEN_INC
